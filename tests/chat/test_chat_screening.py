"""app.chat.screening — pct_change/rank (pure), screen() integration with
get_price_series faked, no network (F-stage)."""

from __future__ import annotations

import asyncio
from datetime import date

import pytest

from app.chat import screening
from app.config import settings


def test_pct_change_computes_first_to_last():
    series = {date(2026, 1, 1): 100.0, date(2026, 1, 5): 110.0, date(2026, 1, 10): 120.0}
    assert screening.pct_change(series) == pytest.approx(0.20)


def test_pct_change_none_for_fewer_than_two_points():
    assert screening.pct_change({date(2026, 1, 1): 100.0}) is None
    assert screening.pct_change({}) is None


def test_pct_change_none_when_first_price_is_zero():
    series = {date(2026, 1, 1): 0.0, date(2026, 1, 2): 10.0}
    assert screening.pct_change(series) is None


def test_rank_sorts_descending_and_drops_none_scores():
    scores = {"AAPL": 0.05, "MSFT": None, "TSLA": 0.12, "GOOG": -0.02}
    result = screening.rank(scores, top_n=5)
    assert result == [("TSLA", 0.12), ("AAPL", 0.05), ("GOOG", -0.02)]


def test_rank_respects_top_n():
    scores = {"A": 0.1, "B": 0.2, "C": 0.3}
    assert screening.rank(scores, top_n=2) == [("C", 0.3), ("B", 0.2)]


def test_risk_adjusted_score_divides_by_atr():
    assert screening.risk_adjusted_score(0.10, 0.05) == pytest.approx(2.0)


def test_risk_adjusted_score_falls_back_to_raw_when_atr_missing():
    assert screening.risk_adjusted_score(0.10, None) == pytest.approx(0.10)


def test_risk_adjusted_score_falls_back_to_raw_when_atr_non_positive():
    assert screening.risk_adjusted_score(0.10, 0.0) == pytest.approx(0.10)
    assert screening.risk_adjusted_score(0.10, -0.01) == pytest.approx(0.10)


async def test_momentum_score_is_never_nan_when_atr_is_unavailable(monkeypatch):
    """2026-08-21 fix (ATR TA-Lib PoC review): this only stays correct
    because `get_atr_pct` returns `None`, never a bare NaN, for "not
    enough history yet" — `risk_adjusted_score`'s `atr_pct <= 0` guard is
    silently always False for a NaN, which used to let a NaN ATR reach
    `raw_pct / atr_pct` and hand `rank()` an unorderable NaN score."""
    async def fake_price_series(ticker, start, end):
        return {date(2026, 7, 1): 100.0, date(2026, 7, 10): 110.0}

    async def fake_atr_pct(ticker, as_of=None):
        return None  # the contract get_atr_pct/_atr must honor

    monkeypatch.setattr(screening, "get_price_series", fake_price_series)
    monkeypatch.setattr(screening, "get_atr_pct", fake_atr_pct)
    result = await screening._momentum("AAPL", lookback_days=10)
    assert result.atr_pct is None
    assert result.score == pytest.approx(result.raw_pct)  # fell back to raw_pct, not nan


def test_dedupe_correlated_keeps_only_highest_scored_per_cluster():
    ranked = [("A", 0.30), ("B", 0.20), ("C", 0.10)]
    clusters = {"A": 0, "B": 0, "C": 1}  # A and B correlated; C its own
    assert screening.dedupe_correlated(ranked, clusters) == [("A", 0.30), ("C", 0.10)]


def test_dedupe_correlated_keeps_tickers_missing_from_clusters():
    # A data gap (its own price history fetch failed) must not drop the
    # ticker from the ranked pool.
    ranked = [("A", 0.30), ("B", 0.20)]
    assert screening.dedupe_correlated(ranked, {"A": 0}) == [("A", 0.30), ("B", 0.20)]


def test_dedupe_correlated_no_op_when_every_ticker_its_own_cluster():
    ranked = [("A", 0.30), ("B", 0.20), ("C", 0.10)]
    clusters = {"A": 0, "B": 1, "C": 2}
    assert screening.dedupe_correlated(ranked, clusters) == ranked


@pytest.fixture(autouse=True)
def _no_atr_by_default(monkeypatch):
    # Every test below only cares about the momentum/ranking mechanics, not
    # volatility adjustment specifically — degrade to the raw-pct behavior
    # (risk_adjusted_score's None-ATR fallback) unless a test overrides this.
    async def fake_atr_pct(ticker, as_of=None):
        raise RuntimeError("no ATR data in this test")

    monkeypatch.setattr(screening, "get_atr_pct", fake_atr_pct)


@pytest.fixture(autouse=True)
def _no_correlation_clustering_by_default(monkeypatch):
    # Every ticker its own singleton cluster (i.e. no dedup) unless a test
    # overrides this — real compute_correlation_clusters fetches a full
    # price history per ticker via yfinance, which every test below not
    # explicitly testing correlation dedup shouldn't pay for (or depend on
    # for a deterministic result).
    async def fake_clusters(tickers, as_of=None, lookback_days=None):
        return {ticker: i for i, ticker in enumerate(tickers)}

    monkeypatch.setattr(screening, "compute_correlation_clusters", fake_clusters)


async def test_momentum_returns_none_on_non_value_error_from_price_series(monkeypatch):
    """FINDING A-1 regression: _momentum used to only catch ValueError around
    get_price_series — a network error / rate limit / yfinance-internal
    exception from the unguarded yfinance fallback leg used to propagate and
    kill the whole scan instead of excluding just that one ticker."""

    async def fake_price_series(ticker, start, end):
        raise RuntimeError("simulated network failure, not the empty-data ValueError")

    monkeypatch.setattr(screening, "get_price_series", fake_price_series)
    assert await screening._momentum("AAPL", 10) is None


async def test_scored_candidates_excludes_one_failing_ticker_not_the_whole_scan(monkeypatch):
    """FINDING A-1/A-2 regression: one ticker's data failure must not take
    down the other tickers' scores."""

    async def fake_price_series(ticker, start, end):
        if ticker == "BAD":
            raise RuntimeError("simulated failure")
        return {date(2026, 7, 1): 100.0, date(2026, 7, 10): 110.0}

    monkeypatch.setattr(screening, "get_price_series", fake_price_series)
    scored = await screening._scored_candidates(["AAPL", "BAD", "MSFT"], 10)
    assert set(scored) == {"AAPL", "MSFT"}


async def test_scored_candidates_excludes_ticker_whose_momentum_call_itself_raises(monkeypatch):
    """FINDING A-2 regression: return_exceptions=True must isolate a failure
    from _momentum itself (not just from get_price_series inside it) — a
    layer A-1 alone can't cover."""

    async def fake_price_series(ticker, start, end):
        return {date(2026, 7, 1): 100.0, date(2026, 7, 10): 110.0}

    monkeypatch.setattr(screening, "get_price_series", fake_price_series)

    real_momentum = screening._momentum

    async def flaky_momentum(ticker, lookback_days, as_of=None):
        if ticker == "BOOM":
            raise TypeError("simulated bug inside _momentum, not a data-source failure")
        return await real_momentum(ticker, lookback_days, as_of=as_of)

    monkeypatch.setattr(screening, "_momentum", flaky_momentum)
    scored = await screening._scored_candidates(["AAPL", "BOOM"], 10)
    assert set(scored) == {"AAPL"}


async def test_scored_candidates_propagates_real_cancellation(monkeypatch):
    """FINDING A-2 regression: return_exceptions=True must not swallow a
    genuine asyncio.CancelledError (e.g. server shutdown mid-scan) as if it
    were just another failed ticker — it must still propagate."""

    async def cancelling_momentum(ticker, lookback_days, as_of=None):
        raise asyncio.CancelledError()

    monkeypatch.setattr(screening, "_momentum", cancelling_momentum)
    with pytest.raises(asyncio.CancelledError):
        await screening._scored_candidates(["AAPL"], 10)


async def test_screen_returns_no_data_message_when_all_scores_none(monkeypatch):
    async def fake_price_series(ticker, start, end):
        raise ValueError("no price history")

    monkeypatch.setattr(screening, "get_price_series", fake_price_series)
    result = await screening.screen(["AAPL", "MSFT"])
    assert result == "No momentum data available for the current trade allowlist."


async def test_screen_ranks_and_formats_momentum(monkeypatch):
    # Overseas volatility-adjustment/dedup default to OFF in production
    # (settings.screening_volatility_adjustment_overseas — see screen()'s
    # docstring on why domestic/overseas now use their own live settings
    # rather than a hardcoded always-on). This test wants the
    # risk-adjusted-score behavior specifically, so it opts overseas back in.
    monkeypatch.setattr(settings, "screening_volatility_adjustment_overseas", True)

    async def fake_price_series(ticker, start, end):
        if ticker == "AAPL":
            return {date(2026, 7, 1): 100.0, date(2026, 7, 10): 110.0}
        return {date(2026, 7, 1): 200.0, date(2026, 7, 10): 190.0}

    monkeypatch.setattr(screening, "get_price_series", fake_price_series)
    result = await screening.screen(["AAPL", "MSFT"], top_n=5, lookback_days=10)
    assert "Momentum-ranked candidates" in result
    assert "AAPL: +10.00% over the last 10 trading days (ATR unavailable, risk-adjusted score +0.10)" in result
    assert result.index("AAPL") < result.index("MSFT")  # AAPL ranked first (higher momentum)


async def test_screen_uses_atr_when_available(monkeypatch):
    monkeypatch.setattr(settings, "screening_volatility_adjustment_overseas", True)

    async def fake_price_series(ticker, start, end):
        return {date(2026, 7, 1): 100.0, date(2026, 7, 10): 110.0}

    async def fake_atr_pct(ticker, as_of=None):
        return 0.05

    monkeypatch.setattr(screening, "get_price_series", fake_price_series)
    monkeypatch.setattr(screening, "get_atr_pct", fake_atr_pct)
    result = await screening.screen(["AAPL"], top_n=5, lookback_days=10)
    assert "5.0% ATR, risk-adjusted score +2.00" in result


async def test_screen_uses_per_market_lookback_when_not_overridden(monkeypatch):
    """FINDING G regression: lookback_days=None (the new default) must read
    each market's own settings.momentum_lookback_days_domestic/_overseas,
    not one hardcoded value shared by both markets."""
    monkeypatch.setattr(settings, "momentum_lookback_days_domestic", 15)
    monkeypatch.setattr(settings, "momentum_lookback_days_overseas", 10)
    seen_lookback: dict[str, int] = {}

    async def fake_price_series(ticker, start, end):
        seen_lookback[ticker] = (end - start).days
        return {date(2026, 7, 1): 100.0, date(2026, 7, 10): 110.0}

    monkeypatch.setattr(screening, "get_price_series", fake_price_series)
    await screening.screen(["005930.KS", "AAPL"], top_n=5)
    assert seen_lookback["005930.KS"] == 15
    assert seen_lookback["AAPL"] == 10


async def test_screen_uses_per_market_volatility_adjustment_flag(monkeypatch):
    """FINDING G regression: domestic (adjustment ON by default) and
    overseas (OFF by default) must be scored/labeled independently in the
    same screen() call — not one shared flag for the whole universe."""
    monkeypatch.setattr(settings, "screening_volatility_adjustment_domestic", True)
    monkeypatch.setattr(settings, "screening_volatility_adjustment_overseas", False)

    async def fake_price_series(ticker, start, end):
        return {date(2026, 7, 1): 100.0, date(2026, 7, 10): 110.0}

    async def fake_atr_pct(ticker, as_of=None):
        return 0.05

    monkeypatch.setattr(screening, "get_price_series", fake_price_series)
    monkeypatch.setattr(screening, "get_atr_pct", fake_atr_pct)
    result = await screening.screen(["005930.KS", "AAPL"], top_n=5)
    assert "005930.KS: +10.00% over the last" in result
    assert "risk-adjusted score" in result  # domestic section
    assert "AAPL: +10.00% over the last" in result
    assert "momentum score" in result  # overseas section, not "risk-adjusted"


async def test_screen_never_merges_domestic_and_overseas_into_one_ranking(monkeypatch):
    """FINDING G regression: domestic and overseas must render as two
    separate sections, each re-starting its own ranking — never one
    combined leaderboard, even when one market's scores are all higher."""

    async def fake_price_series(ticker, start, end):
        # Domestic (005930.KS) scores much lower than overseas (AAPL) — a
        # combined ranking would put AAPL above 005930.KS. Two separate
        # sections must each still show their own ticker regardless.
        if ticker == "005930.KS":
            return {date(2026, 7, 1): 100.0, date(2026, 7, 10): 101.0}
        return {date(2026, 7, 1): 100.0, date(2026, 7, 10): 150.0}

    monkeypatch.setattr(screening, "get_price_series", fake_price_series)
    result = await screening.screen(["005930.KS", "AAPL"], top_n=5)
    assert "Domestic (KRX)" in result
    assert "Overseas" in result
    assert "005930.KS" in result
    assert "AAPL" in result


async def test_screen_correlation_clustering_never_mixes_markets(monkeypatch):
    """FINDING G regression: compute_correlation_clusters must be called
    once per market, never with both markets' tickers in one call — a
    cross-market cluster call could let a KRX name and a US name dedupe
    each other out, which is meaningless (they aren't actually correlated
    the way the same-market clustering model assumes)."""
    # pool_size = min(universe_size, top_n * multiplier) must exceed top_n
    # per market, or _ranked_and_deduped skips clustering as "nothing to
    # dedupe" — 2 tickers/market with top_n=1, multiplier=2 clears that.
    monkeypatch.setattr(settings, "screening_correlation_pool_multiplier", 2)
    # Force both markets' dedup on so both actually call clustering — the
    # overseas default is OFF, which would make this test trivially pass
    # with only the domestic call ever happening.
    monkeypatch.setattr(settings, "screening_correlation_dedup_domestic", True)
    monkeypatch.setattr(settings, "screening_correlation_dedup_overseas", True)
    calls: list[list[str]] = []

    async def fake_price_series(ticker, start, end):
        return {date(2026, 7, 1): 100.0, date(2026, 7, 10): 110.0}

    async def fake_clusters(tickers, as_of=None, lookback_days=None):
        calls.append(list(tickers))
        return {ticker: i for i, ticker in enumerate(tickers)}

    monkeypatch.setattr(screening, "get_price_series", fake_price_series)
    monkeypatch.setattr(screening, "compute_correlation_clusters", fake_clusters)
    await screening.screen(["005930.KS", "402340.KS", "AAPL", "MSFT"], top_n=1)
    assert len(calls) == 2  # one per market — both actually ran clustering
    for call in calls:
        markets = {"domestic" if t.endswith(".KS") else "overseas" for t in call}
        assert len(markets) <= 1


async def test_top_candidates_returns_tickers_ranked_by_momentum(monkeypatch):
    async def fake_price_series(ticker, start, end):
        if ticker == "AAPL":
            return {date(2026, 7, 1): 100.0, date(2026, 7, 10): 110.0}
        return {date(2026, 7, 1): 200.0, date(2026, 7, 10): 190.0}

    monkeypatch.setattr(screening, "get_price_series", fake_price_series)
    result = await screening.top_candidates(["AAPL", "MSFT"], top_n=5, lookback_days=10)
    assert result == ["AAPL", "MSFT"]  # AAPL (+10%) ranked above MSFT (-5%)


async def test_screening_pass_ranked_matches_top_candidates_exactly(monkeypatch):
    """§4-1 (2026-09-08): top_candidates now delegates to screening_pass —
    pins that the delegation changed nothing about the ranking itself.
    Also checks screening_pass's wider contract: `scored` carries every
    universe ticker (not just the top_n cut) and `score_key` names the
    field the ranking actually sorted by."""
    async def fake_price_series(ticker, start, end):
        if ticker == "AAPL":
            return {date(2026, 7, 1): 100.0, date(2026, 7, 10): 110.0}
        return {date(2026, 7, 1): 200.0, date(2026, 7, 10): 190.0}

    monkeypatch.setattr(screening, "get_price_series", fake_price_series)

    top_n = await screening.top_candidates(["AAPL", "MSFT"], top_n=5, lookback_days=10)
    result = await screening.screening_pass(["AAPL", "MSFT"], top_n=5, lookback_days=10)

    assert [t for t, _ in result.ranked] == top_n == ["AAPL", "MSFT"]
    assert set(result.scored) == {"AAPL", "MSFT"}
    assert result.score_key == "score"


async def test_top_candidates_ranks_by_volatility_adjusted_score_not_raw_pct(monkeypatch):
    # AAPL moves less (+8%) but on much lower volatility than TSLA (+10%,
    # very volatile) — risk-adjusted, AAPL should outrank TSLA even though
    # its raw move is smaller.
    async def fake_price_series(ticker, start, end):
        pct = 0.08 if ticker == "AAPL" else 0.10
        return {date(2026, 7, 1): 100.0, date(2026, 7, 10): 100.0 * (1 + pct)}

    async def fake_atr_pct(ticker, as_of=None):
        return 0.02 if ticker == "AAPL" else 0.20

    monkeypatch.setattr(screening, "get_price_series", fake_price_series)
    monkeypatch.setattr(screening, "get_atr_pct", fake_atr_pct)
    result = await screening.top_candidates(["AAPL", "TSLA"], top_n=5, lookback_days=10)
    assert result == ["AAPL", "TSLA"]


async def test_top_candidates_respects_top_n(monkeypatch):
    async def fake_price_series(ticker, start, end):
        return {date(2026, 7, 1): 100.0, date(2026, 7, 10): 100.0 + int(ticker)}

    monkeypatch.setattr(screening, "get_price_series", fake_price_series)
    universe = [str(i) for i in range(10)]
    result = await screening.top_candidates(universe, top_n=3, lookback_days=10)
    assert result == ["9", "8", "7"]


async def test_top_candidates_drops_correlated_near_duplicate(monkeypatch):
    # Two names in the same correlation cluster -> only the higher-scored
    # one should survive, letting a third (different-cluster) candidate
    # into the final top_n instead of being crowded out.
    async def fake_price_series(ticker, start, end):
        pct = {"A": 0.30, "B": 0.25, "C": 0.10}[ticker]
        return {date(2026, 7, 1): 100.0, date(2026, 7, 10): 100.0 * (1 + pct)}

    async def fake_clusters(tickers, as_of=None, lookback_days=None):
        return {"A": 0, "B": 0, "C": 1}  # A and B correlated

    monkeypatch.setattr(screening, "get_price_series", fake_price_series)
    monkeypatch.setattr(screening, "compute_correlation_clusters", fake_clusters)
    result = await screening.top_candidates(["A", "B", "C"], top_n=2, lookback_days=10)
    assert result == ["A", "C"]  # B dropped (same cluster as higher-scored A)


async def test_top_candidates_skips_clustering_when_pool_not_wider_than_top_n(monkeypatch):
    async def fake_price_series(ticker, start, end):
        return {date(2026, 7, 1): 100.0, date(2026, 7, 10): 110.0}

    async def fail_if_called(tickers, as_of=None, lookback_days=None):
        raise AssertionError("must not cluster when the ranked pool is already <= top_n")

    monkeypatch.setattr(screening, "get_price_series", fake_price_series)
    monkeypatch.setattr(screening, "compute_correlation_clusters", fail_if_called)
    result = await screening.top_candidates(["A", "B"], top_n=5, lookback_days=10)
    assert set(result) == {"A", "B"}


async def test_top_candidates_falls_back_to_plain_ranking_when_clustering_fails(monkeypatch):
    async def fake_price_series(ticker, start, end):
        pct = {"A": 0.30, "B": 0.20, "C": 0.10}[ticker]
        return {date(2026, 7, 1): 100.0, date(2026, 7, 10): 100.0 * (1 + pct)}

    async def raising_clusters(tickers, as_of=None, lookback_days=None):
        raise RuntimeError("yfinance history fetch failed")

    monkeypatch.setattr(screening, "get_price_series", fake_price_series)
    monkeypatch.setattr(screening, "compute_correlation_clusters", raising_clusters)
    result = await screening.top_candidates(["A", "B", "C"], top_n=2, lookback_days=10)
    assert result == ["A", "B"]  # degrades to the plain ranked pool, no dedup


async def test_top_candidates_use_volatility_adjustment_false_ranks_by_raw_pct(monkeypatch):
    # AAPL moves less (+8%) but on much lower volatility than TSLA (+10%) —
    # with the flag off, ranking must ignore ATR and go by raw move alone,
    # putting TSLA first (opposite of the risk-adjusted-on test above).
    async def fake_price_series(ticker, start, end):
        pct = 0.08 if ticker == "AAPL" else 0.10
        return {date(2026, 7, 1): 100.0, date(2026, 7, 10): 100.0 * (1 + pct)}

    async def fake_atr_pct(ticker, as_of=None):
        return 0.02 if ticker == "AAPL" else 0.20

    monkeypatch.setattr(screening, "get_price_series", fake_price_series)
    monkeypatch.setattr(screening, "get_atr_pct", fake_atr_pct)
    result = await screening.top_candidates(
        ["AAPL", "TSLA"], top_n=5, lookback_days=10, use_volatility_adjustment=False
    )
    assert result == ["TSLA", "AAPL"]


async def test_top_candidates_use_correlation_dedup_false_keeps_correlated_duplicate(monkeypatch):
    # Same correlated-pair setup as test_top_candidates_drops_correlated_near_duplicate,
    # but with dedup off B must survive even though it's in the same cluster as A.
    async def fake_price_series(ticker, start, end):
        pct = {"A": 0.30, "B": 0.25, "C": 0.10}[ticker]
        return {date(2026, 7, 1): 100.0, date(2026, 7, 10): 100.0 * (1 + pct)}

    async def fail_if_called(tickers, as_of=None, lookback_days=None):
        raise AssertionError("must not cluster when correlation dedup is disabled")

    monkeypatch.setattr(screening, "get_price_series", fake_price_series)
    monkeypatch.setattr(screening, "compute_correlation_clusters", fail_if_called)
    result = await screening.top_candidates(
        ["A", "B", "C"], top_n=2, lookback_days=10, use_correlation_dedup=False
    )
    assert result == ["A", "B"]  # B (2nd by raw score) survives, no clustering call made


async def test_top_candidates_returns_empty_list_when_no_data(monkeypatch):
    async def fake_price_series(ticker, start, end):
        raise ValueError("no price history")

    monkeypatch.setattr(screening, "get_price_series", fake_price_series)
    result = await screening.top_candidates(["AAPL", "MSFT"])
    assert result == []


async def test_screening_pass_observe_only_never_reaches_ranking(monkeypatch):
    """2026-09-09 §0단계 계측 core invariant: a ticker passed via
    observe_only must get a score in `.observed` but must NEVER appear in
    `.ranked` — even when it would clearly outrank everything in
    `universe` if it had been included there instead."""
    async def fake_price_series(ticker, start, end):
        # LOCKED would trivially win the ranking if it were in `universe`.
        pct = {"AAPL": 0.05, "MSFT": 0.03, "LOCKED": 0.90}[ticker]
        return {date(2026, 7, 1): 100.0, date(2026, 7, 10): 100.0 * (1 + pct)}

    monkeypatch.setattr(screening, "get_price_series", fake_price_series)
    result = await screening.screening_pass(
        ["AAPL", "MSFT"], top_n=5, lookback_days=10, observe_only=["LOCKED"],
    )

    assert [t for t, _ in result.ranked] == ["AAPL", "MSFT"]
    assert set(result.scored) == {"AAPL", "MSFT"}
    assert set(result.observed) == {"LOCKED"}
    assert result.observed["LOCKED"].raw_pct == pytest.approx(0.90)


async def test_screening_pass_observe_only_dedupes_against_universe(monkeypatch):
    """A ticker already in `universe` must not be re-scored via observe_only
    (a caller mistake, e.g. passing the pre-filter universe instead of just
    the dropped subset) — it must not be double-counted or double-fetched."""
    call_count: dict[str, int] = {}

    async def fake_price_series(ticker, start, end):
        call_count[ticker] = call_count.get(ticker, 0) + 1
        return {date(2026, 7, 1): 100.0, date(2026, 7, 10): 110.0}

    monkeypatch.setattr(screening, "get_price_series", fake_price_series)
    result = await screening.screening_pass(
        ["AAPL"], top_n=5, lookback_days=10, observe_only=["AAPL", "MSFT"],
    )

    assert call_count["AAPL"] == 1  # not fetched twice
    assert set(result.scored) == {"AAPL"}
    assert set(result.observed) == {"MSFT"}


async def test_screening_pass_defaults_to_empty_observed(monkeypatch):
    """No `observe_only` passed (every pre-2026-09-09 caller) must behave
    byte-identically to before this feature existed."""
    async def fake_price_series(ticker, start, end):
        return {date(2026, 7, 1): 100.0, date(2026, 7, 10): 110.0}

    monkeypatch.setattr(screening, "get_price_series", fake_price_series)
    result = await screening.screening_pass(["AAPL"], top_n=5, lookback_days=10)
    assert result.observed == {}
