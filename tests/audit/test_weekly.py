"""app.audit.weekly — asyncpg pool/connection faked, no real Postgres, same
style as tests/audit/test_audit_stats.py. Also fakes app.audit.stats's own
compute_* functions where convenient, since weekly.py composes them rather
than re-implementing their SQL."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from app.audit import stats, weekly
from app.audit.stats import ConfidenceCalibration, RatingStats
from app.clock import FakeClock, LiveClock, set_clock


class FakeConn:
    def __init__(self, fetch_result=None):
        self.fetch_result = fetch_result if fetch_result is not None else []
        self.queries: list[tuple] = []

    async def fetch(self, query, *args):
        self.queries.append((query, args))
        return self.fetch_result


class _AcquireCM:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


class FakePool:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        return _AcquireCM(self._conn)


async def _async_return(value):
    return value


# ---------------------------------------------------------------------------
# Week boundary math (Monday..Sunday, KST)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "anchor,expected_monday,expected_sunday",
    [
        (date(2026, 8, 13), date(2026, 8, 10), date(2026, 8, 16)),  # Thursday
        (date(2026, 8, 10), date(2026, 8, 10), date(2026, 8, 16)),  # Monday itself
        (date(2026, 8, 16), date(2026, 8, 10), date(2026, 8, 16)),  # Sunday itself
        (date(2026, 1, 1), date(2025, 12, 29), date(2026, 1, 4)),   # year boundary
    ],
)
def test_week_bounds_monday_to_sunday(anchor, expected_monday, expected_sunday):
    monday, sunday = weekly._week_bounds(anchor)
    assert monday == expected_monday
    assert sunday == expected_sunday
    assert monday.weekday() == 0  # Monday
    assert sunday.weekday() == 6  # Sunday


def test_week_label_and_parse_round_trip():
    d = date(2026, 8, 13)
    label = weekly.week_label(d)
    assert label == "2026-W33"
    monday = weekly.parse_week_label(label)
    assert monday.weekday() == 0
    assert monday <= d <= monday + timedelta(days=6)


def test_parse_week_label_rejects_malformed_input():
    with pytest.raises(ValueError):
        weekly.parse_week_label("not-a-week")


# ---------------------------------------------------------------------------
# compute_weekly_report — window wiring
# ---------------------------------------------------------------------------


def _rating_stats(rating="Buy", n=10, win_rate=0.6):
    return RatingStats(
        rating=rating, n=n, win_rate=win_rate, mean_alpha=0.01, mean_raw_return=0.02,
        win_condition="alpha > 0", unscorable_n=0,
    )


async def test_compute_weekly_report_uses_three_distinct_windows(monkeypatch):
    calls: list[dict] = []

    async def fake_hit_rate(*, market, scope, account_mode, trigger, basis, since, until):
        calls.append({"since": since, "until": until})
        return []

    async def fake_confidence(*, market, scope, account_mode, trigger, basis):
        return ConfidenceCalibration(buckets=[], n_with_confidence=0, n_without_confidence=0, signed_alpha_corr=None)

    async def fake_f7(*, scope, account_mode, trigger, basis):
        return weekly.F7Panel(cells=[], total_overridden=0)

    monkeypatch.setattr(weekly, "compute_hit_rate_by_rating", fake_hit_rate)
    monkeypatch.setattr(weekly, "compute_confidence_calibration", fake_confidence)
    monkeypatch.setattr(weekly, "_compute_f7_panel", fake_f7)

    report = await weekly.compute_weekly_report(week=date(2026, 8, 13))

    assert len(calls) == 3
    this_week, last_8_weeks, cumulative = calls
    assert this_week["since"] == date(2026, 8, 10)
    assert this_week["until"] == date(2026, 8, 16)
    # 8 KST weeks total, inclusive of the target week.
    assert last_8_weeks["since"] == date(2026, 8, 10) - timedelta(weeks=7)
    assert last_8_weeks["until"] == date(2026, 8, 16)
    assert cumulative["since"] is None
    assert cumulative["until"] is None
    assert report.week_start == date(2026, 8, 10)
    assert report.week_end == date(2026, 8, 16)


async def test_compute_weekly_report_defaults_to_live_paper(monkeypatch):
    seen = {}

    async def fake_hit_rate(*, market, scope, account_mode, trigger, basis, since, until):
        seen["scope"] = scope
        seen["account_mode"] = account_mode
        return []

    async def fake_confidence(*, market, scope, account_mode, trigger, basis):
        return ConfidenceCalibration(buckets=[], n_with_confidence=0, n_without_confidence=0, signed_alpha_corr=None)

    async def fake_f7(*, scope, account_mode, trigger, basis):
        return weekly.F7Panel(cells=[], total_overridden=0)

    monkeypatch.setattr(weekly, "compute_hit_rate_by_rating", fake_hit_rate)
    monkeypatch.setattr(weekly, "compute_confidence_calibration", fake_confidence)
    monkeypatch.setattr(weekly, "_compute_f7_panel", fake_f7)

    await weekly.compute_weekly_report(week=date(2026, 8, 13))
    assert seen["scope"] == "live"
    assert seen["account_mode"] == "paper"


async def test_compute_weekly_report_defaults_week_to_today_kst(monkeypatch):
    fixed = datetime(2026, 8, 13, 3, 0, 0, tzinfo=ZoneInfo("UTC"))  # 03:00 UTC == 12:00 KST, same day
    clock = FakeClock(fixed)
    set_clock(clock)
    try:
        seen = {}

        async def fake_hit_rate(*, market, scope, account_mode, trigger, basis, since, until):
            seen.setdefault("windows", []).append((since, until))
            return []

        async def fake_confidence(*, market, scope, account_mode, trigger, basis):
            return ConfidenceCalibration(buckets=[], n_with_confidence=0, n_without_confidence=0, signed_alpha_corr=None)

        async def fake_f7(*, scope, account_mode, trigger, basis):
            return weekly.F7Panel(cells=[], total_overridden=0)

        monkeypatch.setattr(weekly, "compute_hit_rate_by_rating", fake_hit_rate)
        monkeypatch.setattr(weekly, "compute_confidence_calibration", fake_confidence)
        monkeypatch.setattr(weekly, "_compute_f7_panel", fake_f7)

        report = await weekly.compute_weekly_report()
        # 2026-08-13 03:00 UTC == 2026-08-13 12:00 KST -> still Thursday 8/13 in KST.
        assert report.week_start == date(2026, 8, 10)
        assert report.week_end == date(2026, 8, 16)
    finally:
        set_clock(LiveClock())


async def test_compute_weekly_report_rejects_invalid_scope():
    with pytest.raises(ValueError):
        await weekly.compute_weekly_report(scope="mars")


async def test_compute_weekly_report_rejects_invalid_account_mode():
    with pytest.raises(ValueError):
        await weekly.compute_weekly_report(account_mode="mars")


def _stub_stats_and_f7(monkeypatch):
    async def fake_hit_rate(*, market, scope, account_mode, trigger, basis, since, until):
        return []

    async def fake_confidence(*, market, scope, account_mode, trigger, basis):
        return ConfidenceCalibration(buckets=[], n_with_confidence=0, n_without_confidence=0, signed_alpha_corr=None)

    async def fake_f7(*, scope, account_mode, trigger, basis):
        return weekly.F7Panel(cells=[], total_overridden=0)

    monkeypatch.setattr(weekly, "compute_hit_rate_by_rating", fake_hit_rate)
    monkeypatch.setattr(weekly, "compute_confidence_calibration", fake_confidence)
    monkeypatch.setattr(weekly, "_compute_f7_panel", fake_f7)


# ---------------------------------------------------------------------------
# Slippage section (2026-09-08 fee/tax/slippage audit) — the only piece of
# this otherwise DB-only report that does real network I/O (a KIS execution-
# history fetch via app.audit.slippage_validation.compute_slippage). Must
# degrade to a "not available" note rather than fail the whole report.
# ---------------------------------------------------------------------------


async def test_compute_weekly_report_passes_the_week_window_and_account_mode_to_slippage(monkeypatch):
    _stub_stats_and_f7(monkeypatch)
    seen = {}

    async def fake_compute_slippage(*, start, end, account_mode):
        seen["start"] = start
        seen["end"] = end
        seen["account_mode"] = account_mode
        return weekly.SlippageResult()

    monkeypatch.setattr(weekly, "compute_slippage", fake_compute_slippage)

    report = await weekly.compute_weekly_report(week=date(2026, 8, 13), account_mode="real")

    assert seen["start"] == date(2026, 8, 10)
    assert seen["end"] == date(2026, 8, 16)
    assert seen["account_mode"] == "real"
    assert report.slippage is not None
    assert report.slippage_error is None


async def test_compute_weekly_report_degrades_when_slippage_raises(monkeypatch):
    _stub_stats_and_f7(monkeypatch)

    async def raising_compute_slippage(*, start, end, account_mode):
        raise RuntimeError("KIS broker not configured — cannot fetch execution history")

    monkeypatch.setattr(weekly, "compute_slippage", raising_compute_slippage)

    report = await weekly.compute_weekly_report(week=date(2026, 8, 13))

    assert report.slippage is None
    assert "KIS broker not configured" in report.slippage_error


def _slippage_row(ticker="005930.KS", slippage_pct=0.001, side="Buy"):
    return weekly.SlippageResult(matched=[
        _row_for(ticker, side, slippage_pct),
    ])


def _row_for(ticker, side, slippage_pct):
    from app.audit.slippage_validation import SlippageRow

    return SlippageRow(
        ticker=ticker, trade_date=date(2026, 8, 11), side=side,
        ledger_price=100.0, ledger_shares=10, kis_price=100.0 * (1 + slippage_pct), kis_shares=10.0,
        slippage_pct=slippage_pct, fill_delay_days=0,
    )


def test_render_weekly_report_shows_slippage_summary_when_available(monkeypatch):
    report = weekly.WeeklyReport(
        week_start=date(2026, 8, 10), week_end=date(2026, 8, 16), scope="live", account_mode="paper",
        trigger="auto", basis="variable", generated_at=datetime(2026, 8, 17, tzinfo=ZoneInfo("UTC")),
        this_week_n=0, this_week_unscorable_n=0, last_8_weeks_n=0, cumulative_n=0,
        ratings=[], confidence=ConfidenceCalibration(buckets=[], n_with_confidence=0, n_without_confidence=0, signed_alpha_corr=None),
        f7=weekly.F7Panel(cells=[], total_overridden=0),
        slippage=_slippage_row(slippage_pct=0.002),
    )
    text = weekly.render_weekly_report(report)
    assert "## Slippage" in text
    assert "1 matched trade(s)" in text
    assert "mean +0.200%" in text


def test_render_weekly_report_shows_unavailable_note_when_slippage_failed():
    report = weekly.WeeklyReport(
        week_start=date(2026, 8, 10), week_end=date(2026, 8, 16), scope="live", account_mode="paper",
        trigger="auto", basis="variable", generated_at=datetime(2026, 8, 17, tzinfo=ZoneInfo("UTC")),
        this_week_n=0, this_week_unscorable_n=0, last_8_weeks_n=0, cumulative_n=0,
        ratings=[], confidence=ConfidenceCalibration(buckets=[], n_with_confidence=0, n_without_confidence=0, signed_alpha_corr=None),
        f7=weekly.F7Panel(cells=[], total_overridden=0),
        slippage=None, slippage_error="KIS broker not configured",
    )
    text = weekly.render_weekly_report(report)
    assert "## Slippage" in text
    assert "Not available this run (KIS broker not configured)" in text


def test_render_weekly_report_shows_no_matches_note_when_slippage_empty():
    report = weekly.WeeklyReport(
        week_start=date(2026, 8, 10), week_end=date(2026, 8, 16), scope="live", account_mode="paper",
        trigger="auto", basis="variable", generated_at=datetime(2026, 8, 17, tzinfo=ZoneInfo("UTC")),
        this_week_n=0, this_week_unscorable_n=0, last_8_weeks_n=0, cumulative_n=0,
        ratings=[], confidence=ConfidenceCalibration(buckets=[], n_with_confidence=0, n_without_confidence=0, signed_alpha_corr=None),
        f7=weekly.F7Panel(cells=[], total_overridden=0),
        slippage=weekly.SlippageResult(),
    )
    text = weekly.render_weekly_report(report)
    assert "No same-day matched trades this week" in text


# ---------------------------------------------------------------------------
# F7 panel — n<30 gate
# ---------------------------------------------------------------------------


def _f7_row(bucket, n, wins, inferred_n=0):
    return {"bucket": bucket, "n": n, "wins": wins, "inferred_n": inferred_n}


async def test_f7_panel_shows_verdict_when_n_at_or_above_30(monkeypatch):
    conn = FakeConn(fetch_result=[_f7_row("Hold (held)", 30, 21)])
    monkeypatch.setattr(weekly, "get_pool", lambda: _async_return(FakePool(conn)))
    panel = await weekly._compute_f7_panel(scope="live", account_mode="paper", basis="variable")
    held = next(c for c in panel.cells if c.row_label == "Hold (held)")
    assert held.n == 30
    assert held.win_rate == pytest.approx(0.7)


async def test_f7_panel_suppresses_verdict_when_n_below_30(monkeypatch):
    conn = FakeConn(fetch_result=[_f7_row("Hold (held)", 29, 20)])
    monkeypatch.setattr(weekly, "get_pool", lambda: _async_return(FakePool(conn)))
    panel = await weekly._compute_f7_panel(scope="live", account_mode="paper", basis="variable")
    held = next(c for c in panel.cells if c.row_label == "Hold (held)")
    assert held.n == 29
    assert held.win_rate is None  # N shown, no verdict


async def test_f7_panel_unknown_bucket_never_gets_a_verdict(monkeypatch):
    conn = FakeConn(fetch_result=[_f7_row("Hold (held/flat unknown)", 500, 0)])
    monkeypatch.setattr(weekly, "get_pool", lambda: _async_return(FakePool(conn)))
    panel = await weekly._compute_f7_panel(scope="live", account_mode="paper", basis="variable")
    unknown = next(c for c in panel.cells if c.row_label == "Hold (held/flat unknown)")
    assert unknown.n == 500
    assert unknown.win_rate is None


async def test_f7_panel_query_filters_trader_wanted_trade_final_was_hold(monkeypatch):
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(weekly, "get_pool", lambda: _async_return(FakePool(conn)))
    await weekly._compute_f7_panel(scope="live", account_mode="paper", basis="variable")
    query, args = conn.queries[0]
    assert "trader_action <> 'Hold'" in query
    assert "rating = 'Hold'" in query
    assert args[0] is False  # scope="live" -> is_backtest = false
    assert args[1] == "paper"


async def test_f7_panel_rejects_invalid_scope():
    with pytest.raises(ValueError):
        await weekly._compute_f7_panel(scope="mars", account_mode="paper", basis="variable")


async def test_f7_panel_query_coalesces_inferred_column(monkeypatch):
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(weekly, "get_pool", lambda: _async_return(FakePool(conn)))
    await weekly._compute_f7_panel(scope="live", account_mode="paper", basis="variable")
    query, _ = conn.queries[0]
    assert "COALESCE(held_at_decision, held_at_decision_inferred)" in query


async def test_f7_panel_query_scores_held_true_on_raw_return_not_alpha(monkeypatch):
    # Fix (2026-08-27): mirrors compute_hit_rate_by_rating's Hold(held) fix —
    # this system is long-only, so the real counterfactual to "keep holding"
    # is cash (raw_return > 0), not the benchmark (alpha_return > 0).
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(weekly, "get_pool", lambda: _async_return(FakePool(conn)))
    await weekly._compute_f7_panel(scope="live", account_mode="paper", basis="variable")
    query, _ = conn.queries[0]
    assert "WHEN held_effective IS TRUE THEN raw_return > 0" in query
    assert "WHEN held_effective IS FALSE THEN alpha_return < 0" in query
    # The `ok` filter must require both alpha_return and raw_return to be
    # present/non-NaN now that raw_return feeds the wins CASE too.
    assert "raw_return IS NOT NULL AND raw_return <> 'NaN'::float8" in query


async def test_f7_panel_reports_inferred_n(monkeypatch):
    conn = FakeConn(fetch_result=[_f7_row("Hold (held)", 53, 14, inferred_n=51)])
    monkeypatch.setattr(weekly, "get_pool", lambda: _async_return(FakePool(conn)))
    panel = await weekly._compute_f7_panel(scope="live", account_mode="paper", basis="variable")
    held = next(c for c in panel.cells if c.row_label == "Hold (held)")
    assert held.inferred_n == 51


def test_render_weekly_report_f7_hides_verdict_text_below_threshold():
    panel = weekly.F7Panel(
        cells=[
            weekly.F7Cell(row_label="Hold (held)", n=29, wins=20, win_rate=None),
            weekly.F7Cell(row_label="Hold (flat)", n=0, wins=0, win_rate=None),
            weekly.F7Cell(row_label="Hold (held/flat unknown)", n=0, wins=0, win_rate=None),
        ],
        total_overridden=29,
    )
    report = _minimal_report(f7=panel)
    rendered = weekly.render_weekly_report(report)
    assert "no verdict" in rendered
    assert "20/29" not in rendered
    assert "68.9%" not in rendered  # no percentage should leak out for a suppressed cell


def test_render_weekly_report_f7_shows_verdict_at_or_above_threshold():
    panel = weekly.F7Panel(
        cells=[
            weekly.F7Cell(row_label="Hold (held)", n=30, wins=21, win_rate=0.7),
            weekly.F7Cell(row_label="Hold (flat)", n=0, wins=0, win_rate=None),
            weekly.F7Cell(row_label="Hold (held/flat unknown)", n=0, wins=0, win_rate=None),
        ],
        total_overridden=30,
    )
    report = _minimal_report(f7=panel)
    rendered = weekly.render_weekly_report(report)
    assert "70.0%" in rendered


# ---------------------------------------------------------------------------
# Small-sample guard on ratio metrics (n < 10 -> "—(n=X)")
# ---------------------------------------------------------------------------


def test_fmt_ratio_below_min_n_shows_dash_with_count():
    assert weekly._fmt_ratio(0.5, 5) == "—(n=5)"


def test_fmt_ratio_at_or_above_min_n_shows_percentage():
    assert weekly._fmt_ratio(0.5, 10) == "50.0%"


def test_fmt_ratio_none_win_rate_shows_dash():
    assert weekly._fmt_ratio(None, 3) == "—(n=3)"
    assert weekly._fmt_ratio(None, 0) == "—"


def test_render_weekly_report_small_sample_rating_row_shows_dash():
    report = _minimal_report(
        ratings=[
            weekly.RatingRow(
                rating="Buy",
                this_week=_rating_stats(n=3, win_rate=1.0),
                last_8_weeks=None,
                cumulative=None,
            ),
        ],
    )
    rendered = weekly.render_weekly_report(report)
    assert "—(n=3)" in rendered
    assert "100.0%" not in rendered


def _minimal_report(**overrides):
    defaults = dict(
        week_start=date(2026, 8, 10),
        week_end=date(2026, 8, 16),
        scope="live",
        account_mode="paper",
        trigger="auto",
        basis="variable",
        generated_at=datetime(2026, 8, 13, 12, 0, 0),
        this_week_n=0,
        this_week_unscorable_n=0,
        last_8_weeks_n=0,
        cumulative_n=0,
        ratings=[],
        confidence=ConfidenceCalibration(buckets=[], n_with_confidence=0, n_without_confidence=0, signed_alpha_corr=None),
        f7=weekly.F7Panel(cells=[], total_overridden=0),
    )
    defaults.update(overrides)
    return weekly.WeeklyReport(**defaults)


def test_render_weekly_report_shows_n_before_ratio_in_header_line():
    report = _minimal_report(this_week_n=12, this_week_unscorable_n=1, last_8_weeks_n=40, cumulative_n=200)
    rendered = weekly.render_weekly_report(report)
    header_line = next(l for l in rendered.splitlines() if l.startswith("This week:"))
    assert "N=12" in header_line
    assert "unscorable=1" in header_line
    assert "N=40" in header_line
    assert "N=200" in header_line


def test_render_weekly_report_empty_ratings_says_nothing_to_compare():
    report = _minimal_report()
    rendered = weekly.render_weekly_report(report)
    assert "nothing to compare" in rendered
