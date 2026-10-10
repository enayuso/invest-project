"""app.backtest.portfolio_harness's checkpoint/resume wiring — every
boundary run_portfolio_backtest touches (price series, correlation
clusters, portfolio/position snapshots, the per-(ticker,date) pipeline
step, and app.backtest.checkpoint itself) is faked, so these exercise only
the orchestration: does a resumed run skip already-completed dates, skip
clear_backtest_positions, and still checkpoint/clear correctly. The
per-date replay mechanics themselves (gap stop-loss, reflection, trailing
state) are already covered by test_backtest_harness.py against
_process_ticker_date directly — not re-tested here."""

from __future__ import annotations

from datetime import date

import pytest

from app.backtest import checkpoint
from app.backtest import portfolio_harness as ph
from app.backtest.metrics import BacktestRecord
from app.execution.ledger import Position
from app.execution.portfolio_risk import PortfolioSnapshot
from app.state import PortfolioDecision, PortfolioRating

TICKERS = ["AAPL", "MSFT"]
DATES = [date(2026, 1, 1), date(2026, 1, 8), date(2026, 1, 15)]
PRICES = {t: {d: 100.0 + i for i, d in enumerate(DATES)} for t in TICKERS + ["SPY"]}


def _decision() -> PortfolioDecision:
    return PortfolioDecision(rating=PortfolioRating.HOLD, executive_summary="s", investment_thesis="t")


def _record(ticker: str, d: date) -> BacktestRecord:
    return BacktestRecord(trade_date=d, decision=_decision(), close=PRICES[ticker][d])


@pytest.fixture(autouse=True)
def _stub_boundaries(monkeypatch):
    async def fake_get_price_series(ticker, start, end):
        return PRICES[ticker]

    async def fake_compute_correlation_clusters(tickers, as_of):
        return {}

    async def fake_portfolio_snapshot(tickers, as_of, price_overrides=None):
        return PortfolioSnapshot(per_ticker_value={}, total_value=1000.0)

    async def fake_get_position(ticker):
        return Position(cash=0.0, shares=0)

    async def fake_clear_backtest_positions(tickers):
        pass

    monkeypatch.setattr(ph, "get_price_series", fake_get_price_series)
    monkeypatch.setattr(ph, "compute_correlation_clusters", fake_compute_correlation_clusters)
    monkeypatch.setattr(ph, "portfolio_snapshot", fake_portfolio_snapshot)
    monkeypatch.setattr(ph, "get_position", fake_get_position)
    monkeypatch.setattr(ph, "clear_backtest_positions", fake_clear_backtest_positions)


async def test_fresh_run_calls_clear_backtest_positions_and_processes_every_date(monkeypatch):
    processed: list[date] = []

    async def fake_process_ticker_date(ticker, trade_date, prev, *a, **kw):
        processed.append(trade_date)
        return _record(ticker, trade_date)

    clear_calls = []

    async def tracking_clear(tickers):
        clear_calls.append(tickers)

    saved = []

    async def fake_save(*a, **kw):
        saved.append(kw.get("state") or a[-1])

    monkeypatch.setattr(ph, "_process_ticker_date", fake_process_ticker_date)
    monkeypatch.setattr(ph, "clear_backtest_positions", tracking_clear)
    monkeypatch.setattr(ph, "load_checkpoint", lambda *a, **kw: _none())
    monkeypatch.setattr(ph, "save_checkpoint", fake_save)
    monkeypatch.setattr(ph, "clear_checkpoint", lambda *a, **kw: _noop())

    result = await ph.run_portfolio_backtest(
        tickers=TICKERS, start=DATES[0], end=DATES[-1], cadence_days=1, label="my-run",
    )

    assert len(clear_calls) == 1  # fresh run — must wipe stale backtest positions
    # 2 tickers x 3 dates = 6 _process_ticker_date calls, all 3 dates present
    assert sorted(set(processed)) == DATES
    # One ticker-granular save per ticker (2 tickers x 3 dates = 6) plus one
    # end-of-date save per date (3) — see checkpoint.py's mid-date resume.
    assert len(saved) == 9
    assert saved[-1].completed_date_index == 2
    assert saved[-1].in_flight_date_index == -1  # end-of-date save clears the in-flight marker
    assert result.trade_dates == DATES


async def test_resumed_run_skips_completed_dates_and_never_clears_positions(monkeypatch):
    processed: list[date] = []

    async def fake_process_ticker_date(ticker, trade_date, prev, *a, **kw):
        processed.append(trade_date)
        return _record(ticker, trade_date)

    clear_calls = []

    async def tracking_clear(tickers):
        clear_calls.append(tickers)

    prior_record_aapl = _record("AAPL", DATES[0])
    prior_record_msft = _record("MSFT", DATES[0])
    existing_checkpoint = checkpoint.CheckpointState(
        completed_date_index=0,  # DATES[0] already done — resume should start at DATES[1]
        prev_records={"AAPL": prior_record_aapl, "MSFT": prior_record_msft},
        per_ticker_records={"AAPL": [prior_record_aapl], "MSFT": [prior_record_msft]},
        per_ticker_equity_curves={"AAPL": [1000.0], "MSFT": [1000.0]},
        portfolio_equity_curve=[1000.0],
        running_high_water_mark=1000.0,
        ever_selected={"AAPL", "MSFT"},
        cap_trigger_counts={},
    )

    async def fake_load(*a, **kw):
        return existing_checkpoint

    saved = []

    async def fake_save(label, tickers, start, end, cadence_days, benchmark_ticker, state):
        saved.append(state)

    monkeypatch.setattr(ph, "_process_ticker_date", fake_process_ticker_date)
    monkeypatch.setattr(ph, "clear_backtest_positions", tracking_clear)
    monkeypatch.setattr(ph, "load_checkpoint", fake_load)
    monkeypatch.setattr(ph, "save_checkpoint", fake_save)
    monkeypatch.setattr(ph, "clear_checkpoint", lambda *a, **kw: _noop())

    result = await ph.run_portfolio_backtest(
        tickers=TICKERS, start=DATES[0], end=DATES[-1], cadence_days=1, label="my-run",
    )

    assert clear_calls == []  # resuming — must NOT wipe the killed run's positions
    # Only DATES[1] and DATES[2] should have been (re)processed, never DATES[0]
    assert DATES[0] not in processed
    assert set(processed) == {DATES[1], DATES[2]}
    # 2 tickers x 2 dates ticker-granular saves + 2 end-of-date saves
    assert len(saved) == 6
    # Prior date's record carried forward into the final per-ticker results
    assert len(result.per_ticker_results["AAPL"].records) == 3  # 1 restored + 2 new
    assert result.per_ticker_results["AAPL"].records[0].trade_date == DATES[0]


async def test_resume_mid_date_skips_already_completed_ticker(monkeypatch):
    # A crash between AAPL's fill and MSFT's on DATES[0] — resume must
    # replay DATES[0]'s SAME rotation order, skip AAPL (already committed:
    # rebalance_to_target's 1% cap bounds only the shares a trade ADDS, so
    # reprocessing AAPL would grant a SECOND cap-bounded increment on top of
    # the one already committed), and still process MSFT and every later date.
    processed: list[tuple[str, date]] = []

    async def fake_process_ticker_date(ticker, trade_date, prev, *a, **kw):
        processed.append((ticker, trade_date))
        return _record(ticker, trade_date)

    prior_record_aapl = _record("AAPL", DATES[0])
    existing_checkpoint = checkpoint.CheckpointState(
        completed_date_index=-1,  # no date fully completed yet
        prev_records={"AAPL": prior_record_aapl},
        per_ticker_records={"AAPL": [prior_record_aapl]},
        per_ticker_equity_curves={"AAPL": [1000.0]},
        portfolio_equity_curve=[],
        running_high_water_mark=1000.0,
        ever_selected={"AAPL", "MSFT"},
        cap_trigger_counts={},
        in_flight_date_index=0,
        in_flight_rotated_tickers=["AAPL", "MSFT"],
        completed_tickers_this_date={"AAPL"},
    )

    async def fake_load(*a, **kw):
        return existing_checkpoint

    saved = []

    async def fake_save(label, tickers, start, end, cadence_days, benchmark_ticker, state):
        saved.append(state)

    clear_calls = []

    async def tracking_clear(tickers):
        clear_calls.append(tickers)

    monkeypatch.setattr(ph, "_process_ticker_date", fake_process_ticker_date)
    monkeypatch.setattr(ph, "clear_backtest_positions", tracking_clear)
    monkeypatch.setattr(ph, "load_checkpoint", fake_load)
    monkeypatch.setattr(ph, "save_checkpoint", fake_save)
    monkeypatch.setattr(ph, "clear_checkpoint", lambda *a, **kw: _noop())

    result = await ph.run_portfolio_backtest(
        tickers=TICKERS, start=DATES[0], end=DATES[-1], cadence_days=1, label="my-run",
    )

    assert clear_calls == []  # resuming — must NOT wipe the killed run's positions
    assert ("AAPL", DATES[0]) not in processed  # already committed pre-crash, not reprocessed
    assert ("MSFT", DATES[0]) in processed
    # DATES[1] and DATES[2] both get their normal full rotation
    assert {t for t, d in processed if d == DATES[1]} == {"AAPL", "MSFT"}
    assert {t for t, d in processed if d == DATES[2]} == {"AAPL", "MSFT"}
    # No duplicate DATES[0] record for AAPL (the restored one, not a second)
    assert len(result.per_ticker_results["AAPL"].records) == 3
    assert result.per_ticker_results["AAPL"].records[0].trade_date == DATES[0]
    # First save (mid DATES[0], after MSFT) must clear the in-flight marker
    # only once that date is FULLY done — check the very first save is
    # still in-flight or already cleared, both are valid transitional
    # states, but the LAST save overall must show date 2 fully completed.
    assert saved[-1].completed_date_index == 2
    assert saved[-1].in_flight_date_index == -1


async def test_stale_in_flight_index_is_ignored(monkeypatch):
    # in_flight_date_index pointing at a date that ISN'T resume_from_date_index
    # (completed_date_index + 1) must be ignored entirely — fail safe to the
    # pre-existing "re-walk the whole date" behavior rather than trust it.
    processed: list[tuple[str, date]] = []

    async def fake_process_ticker_date(ticker, trade_date, prev, *a, **kw):
        processed.append((ticker, trade_date))
        return _record(ticker, trade_date)

    stale_checkpoint = checkpoint.CheckpointState(
        completed_date_index=0,  # resume_from_date_index == 1
        prev_records={"AAPL": _record("AAPL", DATES[0]), "MSFT": _record("MSFT", DATES[0])},
        per_ticker_records={"AAPL": [_record("AAPL", DATES[0])], "MSFT": [_record("MSFT", DATES[0])]},
        per_ticker_equity_curves={"AAPL": [1000.0], "MSFT": [1000.0]},
        portfolio_equity_curve=[1000.0],
        running_high_water_mark=1000.0,
        ever_selected={"AAPL", "MSFT"},
        cap_trigger_counts={},
        in_flight_date_index=0,  # stale -- points at the date already fully completed, not date 1
        in_flight_rotated_tickers=["AAPL"],
        completed_tickers_this_date={"AAPL"},
    )

    async def fake_load(*a, **kw):
        return stale_checkpoint

    monkeypatch.setattr(ph, "_process_ticker_date", fake_process_ticker_date)
    monkeypatch.setattr(ph, "clear_backtest_positions", lambda *a, **kw: _noop())
    monkeypatch.setattr(ph, "load_checkpoint", fake_load)
    monkeypatch.setattr(ph, "save_checkpoint", lambda *a, **kw: _noop())
    monkeypatch.setattr(ph, "clear_checkpoint", lambda *a, **kw: _noop())

    await ph.run_portfolio_backtest(
        tickers=TICKERS, start=DATES[0], end=DATES[-1], cadence_days=1, label="my-run",
    )

    # Both tickers processed on DATES[1] and DATES[2] -- the stale in-flight
    # marker granted no skip on either date.
    assert {t for t, d in processed if d == DATES[1]} == {"AAPL", "MSFT"}
    assert {t for t, d in processed if d == DATES[2]} == {"AAPL", "MSFT"}


# ---------------------------------------------------------------------------
# starting_total_equity_usd — 2026-09-11 fix: this used to be a blind
# settings.backtest_starting_cash * len(tickers) multiply, which assumes
# every ticker's ledger.starting_cash_for() succeeded. On an FX-lookup
# failure that function silently falls back to a raw, unconverted (and much
# smaller, for a KRW ticker) value -- the blind multiply couldn't see that
# degradation, so portfolio_return opened artificially negative. It's now a
# real portfolio_snapshot taken while every ticker is provably flat.
# ---------------------------------------------------------------------------


async def test_starting_total_equity_usd_uses_a_real_snapshot_not_blind_multiply(monkeypatch):
    # The fake portfolio_snapshot returns a value that does NOT equal
    # settings.backtest_starting_cash * len(TICKERS) -- simulating an FX
    # lookup degrading one ticker's real starting cash.
    async def fake_process_ticker_date(ticker, trade_date, prev, *a, **kw):
        return _record(ticker, trade_date)

    async def fake_snapshot(tickers, as_of, price_overrides=None):
        return PortfolioSnapshot(per_ticker_value={}, total_value=100_073.0)

    monkeypatch.setattr(ph, "_process_ticker_date", fake_process_ticker_date)
    monkeypatch.setattr(ph, "portfolio_snapshot", fake_snapshot)
    monkeypatch.setattr(ph, "load_checkpoint", lambda *a, **kw: _none())
    monkeypatch.setattr(ph, "save_checkpoint", lambda *a, **kw: _noop())
    monkeypatch.setattr(ph, "clear_checkpoint", lambda *a, **kw: _noop())

    result = await ph.run_portfolio_backtest(
        tickers=TICKERS, start=DATES[0], end=DATES[-1], cadence_days=1, label="my-run",
    )

    from app.config import settings

    blind_multiply = settings.backtest_starting_cash * len(TICKERS)
    assert result.starting_total_equity_usd == 100_073.0
    assert result.starting_total_equity_usd != blind_multiply


async def test_resumed_run_reuses_the_checkpointed_starting_total_equity_usd(monkeypatch):
    # A resume must NOT re-snapshot (positions aren't flat any more) --
    # it reuses whatever the original run captured while genuinely flat.
    snapshot_calls = []

    async def fake_process_ticker_date(ticker, trade_date, prev, *a, **kw):
        return _record(ticker, trade_date)

    async def fake_snapshot(tickers, as_of, price_overrides=None):
        snapshot_calls.append(as_of)
        return PortfolioSnapshot(per_ticker_value={}, total_value=999_999.0)  # would be WRONG if used

    existing_checkpoint = checkpoint.CheckpointState(
        completed_date_index=0,
        prev_records={"AAPL": _record("AAPL", DATES[0]), "MSFT": _record("MSFT", DATES[0])},
        per_ticker_records={"AAPL": [_record("AAPL", DATES[0])], "MSFT": [_record("MSFT", DATES[0])]},
        per_ticker_equity_curves={"AAPL": [1000.0], "MSFT": [1000.0]},
        portfolio_equity_curve=[1000.0],
        running_high_water_mark=1000.0,
        ever_selected={"AAPL", "MSFT"},
        cap_trigger_counts={},
        starting_total_equity_usd=100_073.0,
    )

    async def fake_load(*a, **kw):
        return existing_checkpoint

    monkeypatch.setattr(ph, "_process_ticker_date", fake_process_ticker_date)
    monkeypatch.setattr(ph, "portfolio_snapshot", fake_snapshot)
    monkeypatch.setattr(ph, "load_checkpoint", fake_load)
    monkeypatch.setattr(ph, "save_checkpoint", lambda *a, **kw: _noop())
    monkeypatch.setattr(ph, "clear_checkpoint", lambda *a, **kw: _noop())

    result = await ph.run_portfolio_backtest(
        tickers=TICKERS, start=DATES[0], end=DATES[-1], cadence_days=1, label="my-run",
    )

    assert result.starting_total_equity_usd == 100_073.0  # reused, not re-snapshotted at 999999.0


# ---------------------------------------------------------------------------
# git_commit — 2026-09-11 fix: CheckpointState.git_commit used to be
# recomputed fresh (via _params()'s own internal current_git_commit() call)
# on EVERY save_checkpoint call, not just load_checkpoint's resume gate. A
# mid-run deploy into the same checkout would silently re-stamp a later
# save's stored git_commit to the NEW commit, so a resume after a crash
# would find the (already-updated) stored value matching a freshly
# re-evaluated current one even though the run actually spans two commits'
# worth of decisions. It's now frozen once per run (see CheckpointState.
# git_commit's own docstring and portfolio_harness's resolution mirroring
# starting_total_equity_usd's own pattern above).
# ---------------------------------------------------------------------------


async def test_fresh_run_captures_current_git_commit_once(monkeypatch):
    # A fresh run must call current_git_commit() to seed run_git_commit, and
    # every save_checkpoint call for the life of the run must carry that
    # SAME frozen value -- not a fresh call each time.
    commit_calls = []

    def fake_current_git_commit():
        commit_calls.append(1)
        return "commit-at-run-start"

    async def fake_process_ticker_date(ticker, trade_date, prev, *a, **kw):
        return _record(ticker, trade_date)

    saved = []

    async def fake_save(label, tickers, start, end, cadence_days, benchmark_ticker, state):
        saved.append(state)

    monkeypatch.setattr(ph, "current_git_commit", fake_current_git_commit)
    monkeypatch.setattr(ph, "_process_ticker_date", fake_process_ticker_date)
    monkeypatch.setattr(ph, "load_checkpoint", lambda *a, **kw: _none())
    monkeypatch.setattr(ph, "save_checkpoint", fake_save)
    monkeypatch.setattr(ph, "clear_checkpoint", lambda *a, **kw: _noop())

    await ph.run_portfolio_backtest(
        tickers=TICKERS, start=DATES[0], end=DATES[-1], cadence_days=1, label="my-run",
    )

    # Called exactly once to seed run_git_commit, regardless of how many
    # save_checkpoint calls follow (9, per test_fresh_run_calls_clear_
    # backtest_positions_and_processes_every_date above).
    assert len(commit_calls) == 1
    assert saved  # sanity: saves did happen
    assert all(state.git_commit == "commit-at-run-start" for state in saved)


async def test_resumed_run_reuses_the_checkpointed_git_commit_not_a_fresh_one(monkeypatch):
    # A resumed run must reuse the checkpoint's OWN stored git_commit
    # verbatim -- never re-call current_git_commit() for it. Simulate a
    # mid-run deploy: the checkpoint was frozen under "old-commit", but
    # current_git_commit() would now return "new-commit" if (incorrectly)
    # called again for this purpose.
    commit_calls = []

    def fake_current_git_commit():
        commit_calls.append(1)
        return "new-commit"  # would be WRONG if used to seed run_git_commit

    async def fake_process_ticker_date(ticker, trade_date, prev, *a, **kw):
        return _record(ticker, trade_date)

    existing_checkpoint = checkpoint.CheckpointState(
        completed_date_index=0,
        prev_records={"AAPL": _record("AAPL", DATES[0]), "MSFT": _record("MSFT", DATES[0])},
        per_ticker_records={"AAPL": [_record("AAPL", DATES[0])], "MSFT": [_record("MSFT", DATES[0])]},
        per_ticker_equity_curves={"AAPL": [1000.0], "MSFT": [1000.0]},
        portfolio_equity_curve=[1000.0],
        running_high_water_mark=1000.0,
        ever_selected={"AAPL", "MSFT"},
        cap_trigger_counts={},
        git_commit="old-commit",
    )

    async def fake_load(*a, **kw):
        return existing_checkpoint

    saved = []

    async def fake_save(label, tickers, start, end, cadence_days, benchmark_ticker, state):
        saved.append(state)

    monkeypatch.setattr(ph, "current_git_commit", fake_current_git_commit)
    monkeypatch.setattr(ph, "_process_ticker_date", fake_process_ticker_date)
    monkeypatch.setattr(ph, "load_checkpoint", fake_load)
    monkeypatch.setattr(ph, "save_checkpoint", fake_save)
    monkeypatch.setattr(ph, "clear_checkpoint", lambda *a, **kw: _noop())

    await ph.run_portfolio_backtest(
        tickers=TICKERS, start=DATES[0], end=DATES[-1], cadence_days=1, label="my-run",
    )

    # Never called: run_git_commit came from the checkpoint's own stored
    # value, not a fresh evaluation.
    assert commit_calls == []
    assert saved
    assert all(state.git_commit == "old-commit" for state in saved)
    assert all(state.git_commit != "new-commit" for state in saved)


async def _none():
    return None


async def _noop():
    return None
