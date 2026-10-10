"""app.memory.auto_resolve — resolve_ticker (moved from app.memory.__main__,
2026-08-03) and resolve_all_due (new automatic scheduler). Pool/conn/LLM all
faked, no real DB/network."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from app.clock import FakeClock, local_today, set_clock
from app.memory import auto_resolve, store

# Captured before any fixture patches auto_resolve.set_horizon_outcome to a no-op (see
# `_stub_set_horizon_outcome` below) — one test needs the REAL function to prove its UPDATE
# never touches `status`, and must explicitly restore this reference to do so.
_REAL_SET_HORIZON_OUTCOME = auto_resolve.set_horizon_outcome


class _AcquireCM:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


class FakeConn:
    def __init__(self, rows):
        self._rows = rows

    async def fetch(self, query, *args):
        return self._rows


class FakePool:
    def __init__(self, rows):
        self._conn = FakeConn(rows)

    def acquire(self):
        return _AcquireCM(self._conn)


@pytest.fixture(autouse=True)
def _stub_reflect_and_clock():
    set_clock(FakeClock(datetime(2026, 7, 16)))
    yield
    from app.clock import LiveClock

    set_clock(LiveClock())


@pytest.fixture(autouse=True)
def _stub_reflect(monkeypatch):
    async def fake_reflect(*args, **kwargs):
        return "reflection text"

    monkeypatch.setattr(auto_resolve, "reflect", fake_reflect)


@pytest.fixture(autouse=True)
def _stub_set_horizon_outcome(monkeypatch):
    """Default `set_horizon_outcome` to a no-op DB write (Opus review N3,
    2026-09-28): most tests here never patch it because their fixture price
    series are too short to reach the F6 opportunistic horizon path
    (_horizon_eval_date returns None), so it is never actually called today —
    but a fixture change that lengthens a series would otherwise hit the
    REAL production DB silently, with the surrounding try/except (mirrors
    resolve_ticker's own "best-effort, must never abort resolution") quietly
    swallowing the failure. A test that specifically wants to assert on this
    call still overrides it with monkeypatch.setattr AFTER this fixture runs,
    same convention as `_stub_reflect_and_clock`/`_stub_session_in_progress`."""
    async def noop_set_horizon_outcome(*args, **kwargs):
        pass

    monkeypatch.setattr(auto_resolve, "set_horizon_outcome", noop_set_horizon_outcome)


@pytest.fixture(autouse=True)
def _stub_session_in_progress(monkeypatch):
    """Most tests in this file don't exercise Fix 2 (the intraday-snapshot
    guard) — default `session_in_progress` to always False so every
    unrelated test doesn't have to reason about it. The dedicated Fix 2
    tests below override this per-test."""
    monkeypatch.setattr(auto_resolve, "session_in_progress", lambda ticker, row_date: False)


def _pending_row(trade_date, decision_id=1, rating="Buy", held_at_decision=None, held_at_decision_inferred=None):
    return {
        "id": decision_id,
        "trade_date": trade_date,
        "decision_text": "Buy — strong case",
        "rating": rating,
        "held_at_decision": held_at_decision,
        "held_at_decision_inferred": held_at_decision_inferred,
    }


# ---------------------------------------------------------------------------
# resolve_ticker
# ---------------------------------------------------------------------------


async def test_resolve_ticker_scopes_pending_query_by_account_mode_and_user_id(monkeypatch):
    # Fix 1: without this, real-account resolution could pick up (or
    # resolve against) paper-account pending rows sharing the same ticker,
    # or vice versa — every other query in this module already scopes by
    # user_id; this one previously had neither filter at all.
    trade_date = date(2026, 7, 1)
    captured = {}

    class SpyConn(FakeConn):
        async def fetch(self, query, *args):
            captured["query"] = query
            captured["args"] = args
            return self._rows

    class SpyPool:
        def __init__(self, rows):
            self._conn = SpyConn(rows)

        def acquire(self):
            return _AcquireCM(self._conn)

    async def fake_get_pool():
        return SpyPool([])

    monkeypatch.setattr(auto_resolve, "get_pool", fake_get_pool)

    count = await auto_resolve.resolve_ticker("AAPL", benchmark_ticker="SPY")
    assert count == 0
    assert "account_mode = $3" in captured["query"]
    assert "user_id = $4" in captured["query"]
    assert captured["args"][2] == "paper"
    assert captured["args"][3] == "default"


async def test_resolve_ticker_skips_zero_close_without_crashing(monkeypatch):
    # Regression: a vendor data glitch (literal 0.0 close) used to raise
    # ZeroDivisionError, aborting the run entirely.
    trade_date = date(2026, 7, 1)
    today = date(2026, 7, 16)

    async def fake_get_pool():
        return FakePool([_pending_row(trade_date)])

    async def fake_get_price_series(ticker, start, end):
        return {trade_date: 0.0, today: 155.0}

    async def should_not_be_called(*args, **kwargs):
        raise AssertionError("resolve_outcome must not run for an unsafe return calc")

    monkeypatch.setattr(auto_resolve, "get_pool", fake_get_pool)
    monkeypatch.setattr(auto_resolve, "get_price_series", fake_get_price_series)
    monkeypatch.setattr(auto_resolve, "resolve_outcome", should_not_be_called)

    count = await auto_resolve.resolve_ticker("AAPL")
    assert count == 0


async def test_resolve_ticker_skips_missing_benchmark_date(monkeypatch):
    trade_date = date(2026, 7, 1)
    today = date(2026, 7, 16)

    async def fake_get_pool():
        return FakePool([_pending_row(trade_date)])

    async def fake_get_price_series(ticker, start, end):
        if ticker == "AAPL":
            return {trade_date: 150.0, today: 155.0}
        return {today: 405.0}  # benchmark missing trade_date entirely

    async def should_not_be_called(*args, **kwargs):
        raise AssertionError("resolve_outcome must not run when benchmark data is missing")

    monkeypatch.setattr(auto_resolve, "get_pool", fake_get_pool)
    monkeypatch.setattr(auto_resolve, "get_price_series", fake_get_price_series)
    monkeypatch.setattr(auto_resolve, "resolve_outcome", should_not_be_called)

    count = await auto_resolve.resolve_ticker("AAPL", benchmark_ticker="SPY")
    assert count == 0


async def test_resolve_ticker_fetches_price_series_once_for_multiple_pending_rows(monkeypatch):
    # Fix 2: two pending rows for the same ticker used to trigger 2 separate
    # get_price_series calls PER row (4 total: ticker+benchmark x 2 rows).
    # They all need the exact same series (earliest trade_date -> today), so
    # it must be fetched once and reused — mirrors backfill_horizon_
    # returns's own identical pattern.
    trade_date_1 = date(2026, 7, 1)
    trade_date_2 = date(2026, 7, 3)
    today = local_today("overseas")
    fetch_calls = []

    async def fake_get_pool():
        return FakePool([_pending_row(trade_date_1, decision_id=1), _pending_row(trade_date_2, decision_id=2)])

    async def fake_get_price_series(ticker, start, end):
        fetch_calls.append((ticker, start, end))
        if ticker == "AAPL":
            return {trade_date_1: 150.0, trade_date_2: 152.0, today: 165.0}
        return {trade_date_1: 400.0, trade_date_2: 401.0, today: 404.0}

    async def fake_resolve_outcome(*args, **kwargs):
        pass

    monkeypatch.setattr(auto_resolve, "get_pool", fake_get_pool)
    monkeypatch.setattr(auto_resolve, "get_price_series", fake_get_price_series)
    monkeypatch.setattr(auto_resolve, "resolve_outcome", fake_resolve_outcome)

    count = await auto_resolve.resolve_ticker("AAPL", benchmark_ticker="SPY")
    assert count == 2
    assert [c[0] for c in fetch_calls].count("AAPL") == 1  # not once per row
    assert [c[0] for c in fetch_calls].count("SPY") == 1
    # Fetched from the EARLIEST pending row's trade_date, not the latest.
    assert fetch_calls[0][1] == trade_date_1


async def test_resolve_ticker_returns_zero_without_fetching_prices_when_nothing_pending(monkeypatch):
    async def fake_get_pool():
        return FakePool([])

    async def fail_if_called(*args, **kwargs):
        raise AssertionError("must not fetch a price series when there is nothing pending")

    monkeypatch.setattr(auto_resolve, "get_pool", fake_get_pool)
    monkeypatch.setattr(auto_resolve, "get_price_series", fail_if_called)

    count = await auto_resolve.resolve_ticker("AAPL", benchmark_ticker="SPY")
    assert count == 0


async def test_resolve_ticker_resolves_when_all_prices_valid(monkeypatch):
    trade_date = date(2026, 7, 1)
    # AAPL is overseas — resolve_ticker now derives "today" from
    # local_today("overseas") (US Eastern), not the frozen KST instant
    # directly, so this must match the same derivation rather than
    # hardcoding the fixture's raw KST date.
    today = local_today("overseas")
    resolved = {}

    async def fake_get_pool():
        return FakePool([_pending_row(trade_date)])

    async def fake_get_price_series(ticker, start, end):
        if ticker == "AAPL":
            return {trade_date: 150.0, today: 165.0}
        return {trade_date: 400.0, today: 404.0}

    async def fake_resolve_outcome(
        decision_id, ticker, trade_date_, raw_return, alpha_return, holding_days, reflection_text,
        benchmark_ticker=None, *, resolved_on=None,
    ):
        resolved["decision_id"] = decision_id
        resolved["raw_return"] = raw_return
        resolved["alpha_return"] = alpha_return
        resolved["resolved_on"] = resolved_on

    monkeypatch.setattr(auto_resolve, "get_pool", fake_get_pool)
    monkeypatch.setattr(auto_resolve, "get_price_series", fake_get_price_series)
    monkeypatch.setattr(auto_resolve, "resolve_outcome", fake_resolve_outcome)

    count = await auto_resolve.resolve_ticker("AAPL", benchmark_ticker="SPY")
    assert count == 1
    assert resolved["decision_id"] == 1  # _pending_row's default decision_id
    assert resolved["raw_return"] == pytest.approx((165.0 - 150.0) / 150.0)
    assert resolved["alpha_return"] == pytest.approx(resolved["raw_return"] - (404.0 - 400.0) / 400.0)
    # 2026-09-01 PIT fix: resolved_on is the eval_date the outcome was
    # actually graded on (here, `today`) — the same date get_similar_context's
    # backfill reconstructs as trade_date + holding_days.
    assert resolved["resolved_on"] == today


# ---------------------------------------------------------------------------
# resolve_ticker — per-market benchmark auto-selection (2026-08-12)
#
# Bug this replaces: resolve_ticker/resolve_all_due used to default to a
# single shared 'SPY' regardless of the ticker's market — every domestic
# decision got scored against the US index. Confirmed live: all 207
# already-resolved domestic memory_entries were mis-scored this way while
# KOSPI sat ~31% below its high and SPY sat near an all-time high, making
# domestic alpha figures uninterpretable (see project memory: ai-cycle-
# audit-2026-08-10's benchmark-mismatch finding).
# ---------------------------------------------------------------------------


async def test_resolve_ticker_auto_selects_kospi_for_a_domestic_ticker(monkeypatch):
    trade_date = date(2026, 7, 1)
    today = local_today("domestic")
    requested_tickers = []

    async def fake_get_pool():
        return FakePool([_pending_row(trade_date)])

    async def fake_get_price_series(ticker, start, end):
        requested_tickers.append(ticker)
        if ticker == "005930.KS":
            return {trade_date: 70_000.0, today: 65_000.0}
        return {trade_date: 3_000.0, today: 2_700.0}  # benchmark

    async def fake_resolve_outcome(*args, **kwargs):
        pass

    monkeypatch.setattr(auto_resolve, "get_pool", fake_get_pool)
    monkeypatch.setattr(auto_resolve, "get_price_series", fake_get_price_series)
    monkeypatch.setattr(auto_resolve, "resolve_outcome", fake_resolve_outcome)

    count = await auto_resolve.resolve_ticker("005930.KS")  # no explicit benchmark
    assert count == 1
    assert "^KS11" in requested_tickers  # KOSPI, never SPY, for a .KS ticker
    assert "SPY" not in requested_tickers


async def test_resolve_ticker_auto_selects_spy_for_an_overseas_ticker(monkeypatch):
    trade_date = date(2026, 7, 1)
    today = local_today("overseas")
    requested_tickers = []

    async def fake_get_pool():
        return FakePool([_pending_row(trade_date)])

    async def fake_get_price_series(ticker, start, end):
        requested_tickers.append(ticker)
        return {trade_date: 150.0, today: 155.0}

    async def fake_resolve_outcome(*args, **kwargs):
        pass

    monkeypatch.setattr(auto_resolve, "get_pool", fake_get_pool)
    monkeypatch.setattr(auto_resolve, "get_price_series", fake_get_price_series)
    monkeypatch.setattr(auto_resolve, "resolve_outcome", fake_resolve_outcome)

    count = await auto_resolve.resolve_ticker("AAPL")  # no explicit benchmark
    assert count == 1
    assert "SPY" in requested_tickers
    assert "^KS11" not in requested_tickers


async def test_resolve_ticker_explicit_benchmark_overrides_auto_selection(monkeypatch):
    """An explicit value (the CLI's own --benchmark flag) must still win —
    only the DEFAULT changed, not the ability to force a specific index."""
    trade_date = date(2026, 7, 1)
    today = local_today("domestic")
    requested_tickers = []

    async def fake_get_pool():
        return FakePool([_pending_row(trade_date)])

    async def fake_get_price_series(ticker, start, end):
        requested_tickers.append(ticker)
        return {trade_date: 100.0, today: 105.0}

    async def fake_resolve_outcome(*args, **kwargs):
        pass

    monkeypatch.setattr(auto_resolve, "get_pool", fake_get_pool)
    monkeypatch.setattr(auto_resolve, "get_price_series", fake_get_price_series)
    monkeypatch.setattr(auto_resolve, "resolve_outcome", fake_resolve_outcome)

    await auto_resolve.resolve_ticker("005930.KS", benchmark_ticker="SPY")
    assert "SPY" in requested_tickers
    assert "^KS11" not in requested_tickers


# ---------------------------------------------------------------------------
# resolve_all_due
# ---------------------------------------------------------------------------


async def test_resolve_all_due_resolves_every_ticker_returned(monkeypatch):
    async def fake_tickers_with_pending_resolution(cutoff):
        return {"AAPL", "005930.KS"}

    resolved_tickers = []

    async def fake_resolve_ticker(ticker, benchmark_ticker="SPY"):
        resolved_tickers.append(ticker)
        return 1

    async def fail_if_called(*args, **kwargs):
        raise AssertionError("backfill_horizon_returns must be stubbed — it hits the real DB/Yahoo otherwise")

    monkeypatch.setattr(auto_resolve, "tickers_with_pending_resolution", fake_tickers_with_pending_resolution)
    monkeypatch.setattr(auto_resolve, "resolve_ticker", fake_resolve_ticker)
    # Regression (2026-09-28): this test used to leave the F6 backfill pass unpatched, so it
    # silently hit the real production DB and Yahoo every run (~280s each) instead of the fakes above.
    monkeypatch.setattr(auto_resolve, "backfill_horizon_returns", fail_if_called)

    total = await auto_resolve.resolve_all_due()
    assert total == 2
    assert set(resolved_tickers) == {"AAPL", "005930.KS"}


async def test_resolve_all_due_continues_after_one_ticker_fails(monkeypatch):
    async def fake_tickers_with_pending_resolution(cutoff):
        return {"AAPL", "005930.KS"}

    async def fake_resolve_ticker(ticker, benchmark_ticker="SPY"):
        if ticker == "AAPL":
            raise RuntimeError("price fetch failed")
        return 1

    async def fail_if_called(*args, **kwargs):
        raise AssertionError("backfill_horizon_returns must be stubbed — it hits the real DB/Yahoo otherwise")

    monkeypatch.setattr(auto_resolve, "tickers_with_pending_resolution", fake_tickers_with_pending_resolution)
    monkeypatch.setattr(auto_resolve, "resolve_ticker", fake_resolve_ticker)
    # Regression (2026-09-28): same unpatched-backfill issue as the test above.
    monkeypatch.setattr(auto_resolve, "backfill_horizon_returns", fail_if_called)

    total = await auto_resolve.resolve_all_due()  # must not raise
    assert total == 1  # only 005930.KS's resolution counted


async def test_resolve_all_due_returns_zero_when_listing_pending_fails(monkeypatch):
    async def broken_tickers_with_pending_resolution(cutoff):
        raise RuntimeError("DB unavailable")

    async def fail_if_called(ticker, benchmark_ticker="SPY"):
        raise AssertionError("must not attempt to resolve any ticker if listing failed")

    monkeypatch.setattr(auto_resolve, "tickers_with_pending_resolution", broken_tickers_with_pending_resolution)
    monkeypatch.setattr(auto_resolve, "resolve_ticker", fail_if_called)

    total = await auto_resolve.resolve_all_due()  # must not raise
    assert total == 0


async def test_resolve_all_due_returns_zero_when_nothing_pending(monkeypatch):
    async def fake_tickers_with_pending_resolution(cutoff):
        return set()

    async def fail_if_called(ticker, benchmark_ticker="SPY"):
        raise AssertionError("must not attempt to resolve any ticker when none are due")

    async def fail_if_backfill_called(*args, **kwargs):
        raise AssertionError("backfill_horizon_returns must be stubbed — it hits the real DB/Yahoo otherwise")

    monkeypatch.setattr(auto_resolve, "tickers_with_pending_resolution", fake_tickers_with_pending_resolution)
    monkeypatch.setattr(auto_resolve, "resolve_ticker", fail_if_called)
    # Regression (2026-09-28): resolve_all_due calls the F6 backfill pass unconditionally, even
    # with zero pending tickers — this test used to leave it unpatched and hit the real prod
    # DB/Yahoo every run instead.
    monkeypatch.setattr(auto_resolve, "backfill_horizon_returns", fail_if_backfill_called)

    total = await auto_resolve.resolve_all_due()
    assert total == 0


# ---------------------------------------------------------------------------
# resolve_ticker — evaluation-date selection (2026-08-13)
#
# Regression suite for the silent stall that left 25 live decisions pending
# for 8+ days: resolution used to require a bar on `today` specifically, and a
# KRX daily bar does not exist while the session is still open, so an intraday
# cycle hit a bare `continue` and repeated it forever.
# ---------------------------------------------------------------------------


def test_usable_rejects_nan_zero_and_missing():
    import math

    assert auto_resolve._usable(150.0)
    assert not auto_resolve._usable(None)
    assert not auto_resolve._usable(0.0)
    assert not auto_resolve._usable(-1.0)
    assert not auto_resolve._usable(float("nan"))
    assert not auto_resolve._usable(math.inf)


def test_common_eval_date_picks_latest_date_usable_in_both_series():
    d = date
    prices = {d(2026, 7, 10): 10.0, d(2026, 7, 13): 11.0, d(2026, 7, 14): 12.0}
    benchmark = {d(2026, 7, 10): 100.0, d(2026, 7, 13): 101.0}
    # 07-14 exists for the ticker but not the benchmark — measuring the two
    # over different windows would not be alpha, so the shared 07-13 wins.
    assert auto_resolve._common_eval_date("AAPL", "SPY", prices, benchmark, d(2026, 7, 6)) == d(2026, 7, 13)


def test_common_eval_date_skips_nan_and_respects_not_before():
    d = date
    prices = {d(2026, 7, 10): 10.0, d(2026, 7, 14): float("nan")}
    benchmark = {d(2026, 7, 10): 100.0, d(2026, 7, 14): 101.0}
    assert auto_resolve._common_eval_date("AAPL", "SPY", prices, benchmark, d(2026, 7, 6)) == d(2026, 7, 10)
    # Nothing at or after the minimum holding date.
    assert auto_resolve._common_eval_date("AAPL", "SPY", prices, benchmark, d(2026, 7, 20)) is None


def test_common_eval_date_skips_a_date_still_in_progress_and_falls_back(monkeypatch):
    """The latest shared date is intraday/still-forming (today, market still
    open) — must fall back to the next-most-recent usable date rather than
    treat that in-progress bar as a settled close."""
    d = date
    prices = {d(2026, 7, 10): 10.0, d(2026, 7, 13): 11.0}
    benchmark = {d(2026, 7, 10): 100.0, d(2026, 7, 13): 101.0}

    def fake_session_in_progress(ticker, row_date):
        return row_date == d(2026, 7, 13)

    monkeypatch.setattr(auto_resolve, "session_in_progress", fake_session_in_progress)
    assert auto_resolve._common_eval_date("AAPL", "SPY", prices, benchmark, d(2026, 7, 6)) == d(2026, 7, 10)


def test_common_eval_date_returns_none_when_everything_is_in_progress_or_unusable(monkeypatch):
    d = date
    prices = {d(2026, 7, 13): 11.0}
    benchmark = {d(2026, 7, 13): 101.0}
    monkeypatch.setattr(auto_resolve, "session_in_progress", lambda ticker, row_date: True)
    assert auto_resolve._common_eval_date("AAPL", "SPY", prices, benchmark, d(2026, 7, 6)) is None


async def test_resolve_ticker_falls_back_to_last_available_close(monkeypatch):
    """Today's bar is missing (intraday run / vendor lag) — the outcome is
    measured at the most recent real bar instead of never resolving."""
    trade_date = date(2026, 7, 1)
    today = local_today("overseas")
    last_bar = today - timedelta(days=1)
    resolved = {}

    async def fake_get_pool():
        return FakePool([_pending_row(trade_date)])

    async def fake_get_price_series(ticker, start, end):
        if ticker == "AAPL":
            return {trade_date: 150.0, last_bar: 165.0}
        return {trade_date: 400.0, last_bar: 404.0}

    async def fake_resolve_outcome(
        decision_id, ticker, trade_date_, raw_return, alpha_return, holding_days, reflection_text,
        benchmark_ticker=None, *, resolved_on=None,
    ):
        resolved["raw_return"] = raw_return
        resolved["holding_days"] = holding_days

    monkeypatch.setattr(auto_resolve, "get_pool", fake_get_pool)
    monkeypatch.setattr(auto_resolve, "get_price_series", fake_get_price_series)
    monkeypatch.setattr(auto_resolve, "resolve_outcome", fake_resolve_outcome)

    count = await auto_resolve.resolve_ticker("AAPL", benchmark_ticker="SPY")
    assert count == 1
    assert resolved["raw_return"] == pytest.approx((165.0 - 150.0) / 150.0)
    # holding_days records the window actually measured, not the nominal one.
    assert resolved["holding_days"] == (last_bar - trade_date).days


async def test_resolve_ticker_never_stores_a_nan_return(monkeypatch):
    """A NaN close on the evaluation date must not be written — Postgres sorts
    NaN above every real number, so a stored NaN alpha counted as a WIN."""
    trade_date = date(2026, 7, 1)
    today = local_today("overseas")

    async def fake_get_pool():
        return FakePool([_pending_row(trade_date)])

    async def fake_get_price_series(ticker, start, end):
        if ticker == "AAPL":
            return {trade_date: 150.0, today: float("nan")}
        return {trade_date: 400.0, today: 404.0}

    async def should_not_be_called(*args, **kwargs):
        raise AssertionError("resolve_outcome must not run with a NaN close")

    monkeypatch.setattr(auto_resolve, "get_pool", fake_get_pool)
    monkeypatch.setattr(auto_resolve, "get_price_series", fake_get_price_series)
    monkeypatch.setattr(auto_resolve, "resolve_outcome", should_not_be_called)

    count = await auto_resolve.resolve_ticker("AAPL", benchmark_ticker="SPY")
    assert count == 0


async def test_resolve_ticker_uses_earlier_bar_when_latest_close_is_nan(monkeypatch):
    """NaN on the newest bar falls back to the previous good one rather than
    skipping the decision entirely."""
    trade_date = date(2026, 7, 1)
    today = local_today("overseas")
    good_bar = today - timedelta(days=2)
    resolved = {}

    async def fake_get_pool():
        return FakePool([_pending_row(trade_date)])

    async def fake_get_price_series(ticker, start, end):
        if ticker == "AAPL":
            return {trade_date: 150.0, good_bar: 160.0, today: float("nan")}
        return {trade_date: 400.0, good_bar: 402.0, today: 404.0}

    async def fake_resolve_outcome(
        decision_id, ticker, trade_date_, raw_return, alpha_return, holding_days, reflection_text,
        benchmark_ticker=None, *, resolved_on=None,
    ):
        resolved["raw_return"] = raw_return

    monkeypatch.setattr(auto_resolve, "get_pool", fake_get_pool)
    monkeypatch.setattr(auto_resolve, "get_price_series", fake_get_price_series)
    monkeypatch.setattr(auto_resolve, "resolve_outcome", fake_resolve_outcome)

    count = await auto_resolve.resolve_ticker("AAPL", benchmark_ticker="SPY")
    assert count == 1
    assert resolved["raw_return"] == pytest.approx((160.0 - 150.0) / 150.0)


async def test_resolve_ticker_will_not_evaluate_before_min_holding_days(monkeypatch):
    """The only available bar is younger than memory_min_holding_days — the
    fallback must not shorten the measurement window to get a number."""
    from app.config import settings

    today = local_today("overseas")
    trade_date = today - timedelta(days=settings.memory_min_holding_days + 3)
    too_early = trade_date + timedelta(days=1)

    async def fake_get_pool():
        return FakePool([_pending_row(trade_date)])

    async def fake_get_price_series(ticker, start, end):
        base = 150.0 if ticker == "AAPL" else 400.0
        return {trade_date: base, too_early: base * 1.05}

    async def should_not_be_called(*args, **kwargs):
        raise AssertionError("resolve_outcome must not run inside the minimum holding period")

    monkeypatch.setattr(auto_resolve, "get_pool", fake_get_pool)
    monkeypatch.setattr(auto_resolve, "get_price_series", fake_get_price_series)
    monkeypatch.setattr(auto_resolve, "resolve_outcome", should_not_be_called)

    count = await auto_resolve.resolve_ticker("AAPL", benchmark_ticker="SPY")
    assert count == 0


async def test_resolve_ticker_logs_instead_of_silently_skipping(monkeypatch, caplog):
    """The stall was invisible because this path had no log line at all."""
    trade_date = date(2026, 7, 1)

    async def fake_get_pool():
        return FakePool([_pending_row(trade_date)])

    async def fake_get_price_series(ticker, start, end):
        return {trade_date: 150.0}  # no bar at/after the holding cutoff

    monkeypatch.setattr(auto_resolve, "get_pool", fake_get_pool)
    monkeypatch.setattr(auto_resolve, "get_price_series", fake_get_price_series)

    with caplog.at_level("WARNING", logger="app.memory.auto_resolve"):
        count = await auto_resolve.resolve_ticker("AAPL", benchmark_ticker="SPY")
    assert count == 0
    assert any("will retry next cycle" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# _horizon_eval_date (Stage 1, F6, 2026-08-13)
#
# Fixed trading-day-count horizon for a second, statistically-comparable
# return figure alongside the existing variable-window resolution above —
# see settings.memory_fixed_horizon_trading_days's own comment for why the
# variable window (5 to 112+ calendar days across live rows) can't feed a
# confidence-interval calculation on its own.
# ---------------------------------------------------------------------------


def test_horizon_eval_date_returns_the_nth_trading_day():
    d = date
    trade_date = d(2026, 7, 1)
    prices = {trade_date: 100.0}
    benchmark = {trade_date: 3000.0}
    # 5 trading days after trade_date, both series present on each.
    for i in range(1, 6):
        day = trade_date + timedelta(days=i)
        prices[day] = 100.0 + i
        benchmark[day] = 3000.0 + i
    result = auto_resolve._horizon_eval_date("AAPL", "SPY", prices, benchmark, trade_date, 5)
    assert result == (trade_date + timedelta(days=5), 5)


def test_horizon_eval_date_skips_a_date_still_in_progress_and_pushes_the_count_out(monkeypatch):
    d = date
    trade_date = d(2026, 7, 1)
    prices = {trade_date: 100.0}
    benchmark = {trade_date: 3000.0}
    for i in range(1, 4):
        day = trade_date + timedelta(days=i)
        prices[day] = 100.0 + i
        benchmark[day] = 3000.0 + i
    in_progress_day = trade_date + timedelta(days=2)

    def fake_session_in_progress(ticker, row_date):
        return row_date == in_progress_day

    monkeypatch.setattr(auto_resolve, "session_in_progress", fake_session_in_progress)
    # Day 2's bar is still forming — must not count as one of the N days;
    # the 2nd VALID trading day is pushed out to day 3, same as a NaN close.
    result = auto_resolve._horizon_eval_date("AAPL", "SPY", prices, benchmark, trade_date, 2)
    assert result == (trade_date + timedelta(days=3), 2)


def test_horizon_eval_date_returns_none_when_fewer_than_n_trading_days():
    d = date
    trade_date = d(2026, 7, 1)
    prices = {trade_date: 100.0, trade_date + timedelta(days=1): 101.0}
    benchmark = {trade_date: 3000.0, trade_date + timedelta(days=1): 3010.0}
    # Only 1 trading day available, need 10.
    assert auto_resolve._horizon_eval_date("AAPL", "SPY", prices, benchmark, trade_date, 10) is None


def test_horizon_eval_date_excludes_dates_missing_from_the_benchmark():
    d = date
    trade_date = d(2026, 7, 1)
    prices = {trade_date: 100.0}
    benchmark = {trade_date: 3000.0}
    for i in range(1, 4):
        day = trade_date + timedelta(days=i)
        prices[day] = 100.0 + i
        benchmark[day] = 3000.0 + i
    # A 4th day exists for the ticker but NOT the benchmark — must not count
    # toward the trading-day tally (measuring against a missing benchmark
    # bar would not be alpha).
    ticker_only_day = trade_date + timedelta(days=4)
    prices[ticker_only_day] = 999.0
    assert auto_resolve._horizon_eval_date("AAPL", "SPY", prices, benchmark, trade_date, 4) is None
    assert auto_resolve._horizon_eval_date("AAPL", "SPY", prices, benchmark, trade_date, 3) == (
        trade_date + timedelta(days=3), 3,
    )


def test_horizon_eval_date_skips_nan_closes_and_pushes_the_count_out():
    d = date
    trade_date = d(2026, 7, 1)
    prices = {trade_date: 100.0}
    benchmark = {trade_date: 3000.0}
    for i in range(1, 4):
        day = trade_date + timedelta(days=i)
        prices[day] = 100.0 + i
        benchmark[day] = 3000.0 + i
    # Day 2 has a NaN ticker close — must not count as one of the N days;
    # the 2nd VALID trading day is pushed out to day 3.
    prices[trade_date + timedelta(days=2)] = float("nan")
    result = auto_resolve._horizon_eval_date("AAPL", "SPY", prices, benchmark, trade_date, 2)
    assert result == (trade_date + timedelta(days=3), 2)


def test_horizon_eval_date_returns_none_when_entry_close_itself_is_unusable():
    d = date
    trade_date = d(2026, 7, 1)
    prices = {trade_date: float("nan"), trade_date + timedelta(days=1): 101.0}
    benchmark = {trade_date: 3000.0, trade_date + timedelta(days=1): 3010.0}
    assert auto_resolve._horizon_eval_date("AAPL", "SPY", prices, benchmark, trade_date, 1) is None


async def test_resolve_ticker_still_resolves_variable_window_when_horizon_not_yet_reached(monkeypatch):
    """Regression guard for the stall this project already hit once
    (`_common_eval_date`'s own docstring) — the fixed trading-day horizon
    (10 days) being unreached must NEVER prevent the existing variable-
    window resolution from completing. Only 2 trading days exist in this
    fixture, far short of the fixed horizon, but resolve_outcome (the
    variable-window path) must still fire."""
    trade_date = date(2026, 7, 1)
    today = local_today("overseas")
    resolved = {}

    async def fake_get_pool():
        return FakePool([_pending_row(trade_date)])

    async def fake_get_price_series(ticker, start, end):
        if ticker == "AAPL":
            return {trade_date: 150.0, today: 165.0}
        return {trade_date: 400.0, today: 404.0}

    async def fake_resolve_outcome(
        decision_id, ticker, trade_date_, raw_return, alpha_return, holding_days, reflection_text,
        benchmark_ticker=None, *, resolved_on=None,
    ):
        resolved["called"] = True

    horizon_calls = []

    async def spy_set_horizon_outcome(*args, **kwargs):
        horizon_calls.append(args)

    monkeypatch.setattr(auto_resolve, "get_pool", fake_get_pool)
    monkeypatch.setattr(auto_resolve, "get_price_series", fake_get_price_series)
    monkeypatch.setattr(auto_resolve, "resolve_outcome", fake_resolve_outcome)
    monkeypatch.setattr(auto_resolve, "set_horizon_outcome", spy_set_horizon_outcome)

    count = await auto_resolve.resolve_ticker("AAPL", benchmark_ticker="SPY")
    assert count == 1
    assert resolved.get("called") is True
    # The fixed 10-trading-day horizon isn't reached by this 2-date fixture,
    # so the opportunistic fill must not have fired — but that must not have
    # blocked the variable-window resolution above.
    assert horizon_calls == []


# ---------------------------------------------------------------------------
# backfill_horizon_returns (Stage 1, F6, 2026-08-13)
# ---------------------------------------------------------------------------


class FakeHorizonConn:
    """Returns configured pending rows on the first `fetch`; every UPDATE
    executed via `set_horizon_outcome` goes through the module's own
    get_pool (monkeypatched to this same fake), recorded via `execute`."""

    def __init__(self, rows):
        self._rows = rows
        self.executed: list[tuple] = []

    async def fetch(self, query, *args):
        return self._rows

    async def execute(self, query, *args):
        self.executed.append(args)


def _horizon_row(row_id, ticker, trade_date):
    return {"id": row_id, "ticker": ticker, "trade_date": trade_date}


async def test_backfill_horizon_returns_only_targets_null_rows(monkeypatch):
    """The SQL WHERE clause itself is what enforces this in production; here
    we confirm the function queries with that filter present."""
    conn = FakeHorizonConn([])
    captured = {}

    class SpyPool:
        def acquire(self):
            return _AcquireCM(conn)

    async def fake_get_pool_spy():
        return SpyPool()

    orig_fetch = conn.fetch

    async def spy_fetch(query, *args):
        captured["query"] = query
        captured["args"] = args
        return await orig_fetch(query, *args)

    conn.fetch = spy_fetch

    monkeypatch.setattr(auto_resolve, "get_pool", fake_get_pool_spy)

    total = await auto_resolve.backfill_horizon_returns()
    assert total == 0
    assert "horizon_alpha_return IS NULL" in captured["query"]
    assert "status = 'resolved'" in captured["query"]


async def test_backfill_horizon_returns_fetches_ticker_series_once(monkeypatch):
    trade_date_1 = date(2026, 7, 1)
    trade_date_2 = date(2026, 7, 2)
    rows = [_horizon_row(1, "AAPL", trade_date_1), _horizon_row(2, "AAPL", trade_date_2)]
    conn = FakeHorizonConn(rows)

    class SpyPool:
        def acquire(self):
            return _AcquireCM(conn)

    async def fake_get_pool():
        return SpyPool()

    fetch_calls = []

    async def fake_get_price_series(ticker, start, end):
        fetch_calls.append(ticker)
        today = local_today("overseas")
        series = {}
        d = trade_date_1
        while d <= today:
            series[d] = 100.0 + (d - trade_date_1).days
            d = d + timedelta(days=1)
        return series

    async def noop_set_horizon_outcome(*args, **kwargs):
        pass

    monkeypatch.setattr(auto_resolve, "get_pool", fake_get_pool)
    monkeypatch.setattr(auto_resolve, "get_price_series", fake_get_price_series)
    # Not this test's concern (covered by its own dedicated tests) — stubbed
    # out so it can't attempt a real DB connection via app.memory.store's
    # OWN get_pool (a separate module reference, unaffected by patching
    # auto_resolve.get_pool above).
    monkeypatch.setattr(auto_resolve, "set_horizon_outcome", noop_set_horizon_outcome)

    await auto_resolve.backfill_horizon_returns()
    # AAPL's own series fetched once, benchmark (SPY) fetched once — not once
    # per pending row (2 rows must not mean 4 calls).
    assert fetch_calls.count("AAPL") == 1
    assert fetch_calls.count("SPY") == 1


async def test_backfill_horizon_returns_one_ticker_failure_does_not_block_others(monkeypatch):
    rows = [_horizon_row(1, "AAPL", date(2026, 7, 1)), _horizon_row(2, "MSFT", date(2026, 7, 1))]
    conn = FakeHorizonConn(rows)

    class SpyPool:
        def acquire(self):
            return _AcquireCM(conn)

    async def fake_get_pool():
        return SpyPool()

    async def fake_get_price_series(ticker, start, end):
        if ticker == "AAPL":
            raise RuntimeError("vendor outage")
        today = local_today("overseas")
        series = {}
        d = start
        while d <= today:
            series[d] = 100.0
            d = d + timedelta(days=1)
        return series

    horizon_calls = []

    async def fake_set_horizon_outcome(decision_id, *args, **kwargs):
        horizon_calls.append(decision_id)

    monkeypatch.setattr(auto_resolve, "get_pool", fake_get_pool)
    monkeypatch.setattr(auto_resolve, "get_price_series", fake_get_price_series)
    monkeypatch.setattr(auto_resolve, "set_horizon_outcome", fake_set_horizon_outcome)

    total = await auto_resolve.backfill_horizon_returns()  # must not raise
    # MSFT's row still gets processed despite AAPL's failure (flat 100.0
    # series never produces a NaN/zero close, so its horizon does resolve).
    assert 2 in horizon_calls
    assert 1 not in horizon_calls
    assert total == len(horizon_calls)


async def test_backfill_horizon_returns_never_touches_status(monkeypatch):
    rows = [_horizon_row(1, "AAPL", date(2026, 7, 1))]
    conn = FakeHorizonConn(rows)

    class SpyPool:
        def acquire(self):
            return _AcquireCM(conn)

    async def fake_get_pool():
        return SpyPool()

    async def fake_get_price_series(ticker, start, end):
        today = local_today("overseas")
        series = {}
        d = start
        while d <= today:
            series[d] = 100.0 + (d - start).days
            d = d + timedelta(days=1)
        return series

    monkeypatch.setattr(auto_resolve, "get_pool", fake_get_pool)
    # set_horizon_outcome (app.memory.store) reads its OWN module's get_pool,
    # not auto_resolve's — must be patched separately so the real function
    # (not a spy) actually writes to our fake conn, which is the whole point
    # of this test (confirming the real UPDATE never touches `status`).
    monkeypatch.setattr(store, "get_pool", fake_get_pool)
    monkeypatch.setattr(auto_resolve, "get_price_series", fake_get_price_series)
    # Override the `_stub_set_horizon_outcome` autouse no-op — this test's whole point is
    # exercising the REAL write path.
    monkeypatch.setattr(auto_resolve, "set_horizon_outcome", _REAL_SET_HORIZON_OUTCOME)

    await auto_resolve.backfill_horizon_returns()
    assert conn.executed  # sanity: the real set_horizon_outcome did write something
    for args in conn.executed:
        # set_horizon_outcome's own UPDATE never includes a `status` set —
        # asserted indirectly here by confirming no executed statement's
        # bound args carry a status-like string value in the wrong place.
        assert len(args) == 5  # id, raw, alpha, trading_days, eval_date only


# ---------------------------------------------------------------------------
# KOSDAQ ('.KS'-stored, Yahoo-'.KQ') price-lookup fallback (2026-09-28)
# settings.memory_krx_fallback_since: None = OFF (default); date = rows with
# trade_date >= it may be priced via the other suffix. See handoff/CURRENT.md.
# ---------------------------------------------------------------------------

KRX = "123456.KS"
KRX_ALT = "123456.KQ"
BENCH = "^KS11"
TODAY_KST = date(2026, 7, 16)  # FakeClock date; local_today("domestic")


def _krx_series(entry: date, entry_price=100.0, last_price=110.0) -> dict:
    return {entry: entry_price, TODAY_KST: last_price}


def _bench_series(entry: date) -> dict:
    return {entry: 1000.0, TODAY_KST: 1010.0}


def _krx_setup(monkeypatch, rows, table, *, since=None):
    """Wires resolve_ticker to fakes. `table` maps price symbol -> series, or to an
    Exception instance to raise. Returns (fetch_calls, resolved)."""
    fetch_calls: list[tuple] = []
    resolved: list[dict] = []

    async def fake_get_pool():
        return FakePool(rows)

    async def fake_get_price_series(symbol, start, end):
        fetch_calls.append((symbol, start))
        entry = table.get(symbol)
        if entry is None:
            raise ValueError(f"no price history returned for {symbol!r}")
        if isinstance(entry, Exception):
            raise entry
        return entry

    async def fake_resolve_outcome(decision_id, ticker, trade_date, raw, alpha, holding_days, reflection,
                                   benchmark, *, resolved_on):
        resolved.append({"id": decision_id, "ticker": ticker, "trade_date": trade_date, "alpha": alpha,
                         "holding_days": holding_days, "benchmark": benchmark, "resolved_on": resolved_on})

    async def noop_set_horizon_outcome(*args, **kwargs):
        pass

    monkeypatch.setattr(auto_resolve, "get_pool", fake_get_pool)
    monkeypatch.setattr(auto_resolve, "get_price_series", fake_get_price_series)
    monkeypatch.setattr(auto_resolve, "resolve_outcome", fake_resolve_outcome)
    monkeypatch.setattr(auto_resolve, "set_horizon_outcome", noop_set_horizon_outcome)
    monkeypatch.setattr(auto_resolve.settings, "memory_krx_fallback_since", since)
    return fetch_calls, resolved


def test_alternate_krx_symbol_only_swaps_numeric_krx_codes():
    assert auto_resolve._alternate_krx_symbol("123456.KS") == "123456.KQ"
    assert auto_resolve._alternate_krx_symbol("123456.KQ") == "123456.KS"
    for not_krx in ("AAPL", "^KS11", "ABC.KS", "SPY", ""):
        assert auto_resolve._alternate_krx_symbol(not_krx) is None


async def test_krx_fallback_resolves_row_priced_under_alternate_suffix(monkeypatch):
    entry = date(2026, 7, 1)
    fetch_calls, resolved = _krx_setup(
        monkeypatch, [_pending_row(entry, decision_id=7)],
        {KRX_ALT: _krx_series(entry), BENCH: _bench_series(entry)}, since=date(2026, 7, 1),
    )
    assert await auto_resolve.resolve_ticker(KRX) == 1
    assert [c[0] for c in fetch_calls] == [KRX, KRX_ALT, BENCH]
    assert len(resolved) == 1
    assert resolved[0]["ticker"] == KRX  # STORED ticker, never the alternate symbol
    assert resolved[0]["benchmark"] == BENCH
    assert resolved[0]["alpha"] == pytest.approx(0.10 - 0.01)


async def test_krx_fallback_is_off_by_default(monkeypatch):
    entry = date(2026, 7, 1)
    fetch_calls, resolved = _krx_setup(
        monkeypatch, [_pending_row(entry)], {KRX_ALT: _krx_series(entry), BENCH: _bench_series(entry)}, since=None,
    )
    with pytest.raises(ValueError, match=KRX):
        await auto_resolve.resolve_ticker(KRX)
    assert [c[0] for c in fetch_calls] == [KRX]  # the alternate suffix is never even tried
    assert resolved == []


async def test_krx_fallback_both_symbols_fail_raises_original_error(monkeypatch):
    entry = date(2026, 7, 1)
    fetch_calls, resolved = _krx_setup(
        monkeypatch, [_pending_row(entry)], {BENCH: _bench_series(entry)}, since=date(2026, 7, 1),
    )
    with pytest.raises(ValueError) as excinfo:
        await auto_resolve.resolve_ticker(KRX)
    assert KRX in str(excinfo.value) and KRX_ALT not in str(excinfo.value)  # the ORIGINAL error
    assert isinstance(excinfo.value.__cause__, ValueError)  # chained from the alternate failure
    assert [c[0] for c in fetch_calls] == [KRX, KRX_ALT]
    assert resolved == []


async def test_krx_fallback_does_not_treat_a_rate_limit_as_missing_history(monkeypatch):
    entry = date(2026, 7, 1)
    fetch_calls, resolved = _krx_setup(
        monkeypatch, [_pending_row(entry)],
        {KRX: RuntimeError("429 Too Many Requests"), KRX_ALT: _krx_series(entry), BENCH: _bench_series(entry)},
        since=date(2026, 7, 1),
    )
    with pytest.raises(RuntimeError):
        await auto_resolve.resolve_ticker(KRX)
    assert [c[0] for c in fetch_calls] == [KRX]  # non-ValueError: no alternate attempt
    assert resolved == []


async def test_krx_fallback_alternate_non_valueerror_propagates(monkeypatch):
    entry = date(2026, 7, 1)
    fetch_calls, resolved = _krx_setup(
        monkeypatch, [_pending_row(entry)],
        {KRX_ALT: RuntimeError("network down"), BENCH: _bench_series(entry)}, since=date(2026, 7, 1),
    )
    with pytest.raises(RuntimeError, match="network down"):
        await auto_resolve.resolve_ticker(KRX)
    assert resolved == []


async def test_krx_fallback_session_guard_and_eval_get_the_stored_ticker(monkeypatch):
    # session_in_progress() treats any non-'.KS' symbol as overseas, which would accept a
    # still-forming domestic bar as settled -- it must NEVER see the '.KQ' fetch symbol.
    entry = date(2026, 7, 1)
    seen: list[str] = []

    def spy(ticker, row_date):
        seen.append(ticker)
        return False

    _krx_setup(
        monkeypatch, [_pending_row(entry)], {KRX_ALT: _krx_series(entry), BENCH: _bench_series(entry)},
        since=date(2026, 7, 1),
    )
    monkeypatch.setattr(auto_resolve, "session_in_progress", spy)
    assert await auto_resolve.resolve_ticker(KRX) == 1
    assert set(seen) == {KRX, BENCH}
    assert KRX_ALT not in seen


async def test_krx_fallback_since_gate_leaves_older_rows_pending(monkeypatch, caplog):
    old, new = date(2026, 6, 20), date(2026, 7, 2)
    fetch_calls, resolved = _krx_setup(
        monkeypatch, [_pending_row(old, decision_id=1), _pending_row(new, decision_id=2)],
        {KRX_ALT: {new: 100.0, TODAY_KST: 110.0}, BENCH: {new: 1000.0, TODAY_KST: 1010.0}},
        since=date(2026, 7, 1),
    )
    with caplog.at_level("INFO", logger=auto_resolve.logger.name):
        assert await auto_resolve.resolve_ticker(KRX) == 1
    assert [r["id"] for r in resolved] == [2]  # only the row on/after `since`
    assert (KRX_ALT, new) in fetch_calls  # alternate series starts at the ELIGIBLE row's date
    assert any("1 older row(s) left untouched" in r.getMessage() for r in caplog.records)
    assert not any("skipping" in r.getMessage() for r in caplog.records)  # no per-old-row warning noise


async def test_krx_fallback_no_eligible_rows_reraises_without_trying_alternate(monkeypatch):
    entry = date(2026, 6, 20)
    fetch_calls, resolved = _krx_setup(
        monkeypatch, [_pending_row(entry)], {KRX_ALT: _krx_series(entry), BENCH: _bench_series(entry)},
        since=date(2026, 7, 1),
    )
    with pytest.raises(ValueError):
        await auto_resolve.resolve_ticker(KRX)
    assert [c[0] for c in fetch_calls] == [KRX]
    assert resolved == []


async def test_krx_fallback_never_applies_to_overseas_tickers(monkeypatch):
    entry = date(2026, 7, 1)
    fetch_calls, resolved = _krx_setup(
        monkeypatch, [_pending_row(entry)], {"SPY": _bench_series(entry)}, since=date(2026, 7, 1),
    )
    with pytest.raises(ValueError):
        await auto_resolve.resolve_ticker("AAPL")
    assert [c[0] for c in fetch_calls] == ["AAPL"]


async def test_krx_stored_symbol_success_is_unchanged_even_for_rows_older_than_since(monkeypatch):
    old = date(2026, 6, 20)
    fetch_calls, resolved = _krx_setup(
        monkeypatch, [_pending_row(old, decision_id=1)],
        {KRX: {old: 100.0, TODAY_KST: 110.0}, BENCH: {old: 1000.0, TODAY_KST: 1010.0}}, since=date(2026, 7, 1),
    )
    assert await auto_resolve.resolve_ticker(KRX) == 1  # the gate only governs the FALLBACK
    assert [c[0] for c in fetch_calls] == [KRX, BENCH]
    assert [r["id"] for r in resolved] == [1]


async def _run_horizon_backfill(monkeypatch, *, since, table):
    entry = date(2026, 7, 1)
    conn = FakeHorizonConn([_horizon_row(1, KRX, entry)])

    class SpyPool:
        def acquire(self):
            return _AcquireCM(conn)

    async def fake_get_pool():
        return SpyPool()

    fetch_calls: list[str] = []
    horizon_calls: list[tuple] = []

    async def fake_get_price_series(symbol, start, end):
        fetch_calls.append(symbol)
        if symbol not in table:
            raise ValueError(f"no price history returned for {symbol!r}")
        return table[symbol]

    async def spy_set_horizon_outcome(*args, **kwargs):
        horizon_calls.append(args)

    monkeypatch.setattr(auto_resolve, "get_pool", fake_get_pool)
    monkeypatch.setattr(auto_resolve, "get_price_series", fake_get_price_series)
    monkeypatch.setattr(auto_resolve, "set_horizon_outcome", spy_set_horizon_outcome)
    monkeypatch.setattr(auto_resolve.settings, "memory_krx_fallback_since", since)
    total = await auto_resolve.backfill_horizon_returns()
    return total, fetch_calls, horizon_calls


def _daily(start: date, base: float) -> dict:
    return {start + timedelta(days=i): base + i for i in range((TODAY_KST - start).days + 1)}


async def test_backfill_horizon_returns_uses_krx_fallback_when_enabled(monkeypatch):
    entry = date(2026, 7, 1)
    table = {KRX_ALT: _daily(entry, 100.0), BENCH: _daily(entry, 1000.0)}
    total, fetch_calls, horizon_calls = await _run_horizon_backfill(
        monkeypatch, since=date(2026, 7, 1), table=table,
    )
    assert total == 1 and len(horizon_calls) == 1
    assert fetch_calls[:2] == [KRX, KRX_ALT]


async def test_backfill_horizon_returns_fallback_is_off_by_default(monkeypatch):
    entry = date(2026, 7, 1)
    table = {KRX_ALT: _daily(entry, 100.0), BENCH: _daily(entry, 1000.0)}
    total, fetch_calls, horizon_calls = await _run_horizon_backfill(monkeypatch, since=None, table=table)
    assert total == 0 and horizon_calls == []
    assert fetch_calls == [KRX]  # failed ticker is logged and skipped exactly as before
