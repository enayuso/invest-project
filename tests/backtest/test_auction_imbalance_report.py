"""app.backtest.auction_imbalance_report — the decile-sort viability report
for the closing-auction imbalance signal. Builds RoundTrip/AuctionTrade
objects directly (no cache/DB, no harness run) since this module's own
statistics are what's under test, not the backtest loop that produces its
inputs elsewhere."""

from __future__ import annotations

from datetime import date, timedelta

from app.backtest.auction_imbalance_report import (
    AuctionTrade,
    bootstrap_ci_date_clustered,
    build_decile_report,
    spearman_decile,
    within_date_permutation_test,
)
from app.backtest.scalping_viability import RoundTrip

_BASE_DATE = date(2026, 6, 1)


def _rt(ticker: str, day_offset: int, net_bp: float, gap_inflated: bool = False) -> RoundTrip:
    return RoundTrip(
        ticker=ticker, trade_date=_BASE_DATE + timedelta(days=day_offset),
        entry_price=100.0, exit_price=100.0 * (1 + net_bp / 10_000),
        net_bp=net_bp, gap_inflated=gap_inflated, holding_minutes=14.0,
        entry_filled_time="151500",
    )


def _trade(ticker: str, day_offset: int, net_bp: float, imb_resid: float, gap_inflated: bool = False) -> AuctionTrade:
    return AuctionTrade(round_trip=_rt(ticker, day_offset, net_bp, gap_inflated), imb_resid=imb_resid, drift_resid=0.0)


# --- decile sort + spearman -------------------------------------------------


def test_spearman_is_near_one_for_a_clean_monotone_relationship():
    # 10 tickers on the SAME date, imb_resid and net_bp both increasing
    # together in lockstep — a textbook monotone relationship.
    trades = [_trade(f"T{i}.KS", 0, net_bp=float(i), imb_resid=float(i)) for i in range(30)]
    from app.backtest.auction_imbalance_report import decile_table
    deciles = decile_table(trades, n_bootstrap=200)
    assert spearman_decile(deciles) > 0.95


def test_spearman_is_near_zero_for_no_relationship():
    # imb_resid increases steadily but net_bp alternates independent of it
    # — deciles should show no consistent trend.
    trades = [
        _trade(f"T{i}.KS", i % 10, net_bp=(10.0 if i % 2 == 0 else -10.0), imb_resid=float(i))
        for i in range(40)
    ]
    from app.backtest.auction_imbalance_report import decile_table
    deciles = decile_table(trades, n_bootstrap=200)
    assert abs(spearman_decile(deciles)) < 0.5


# --- date-clustered bootstrap CI --------------------------------------------


def test_date_clustered_ci_is_driven_by_between_date_variation_not_trade_count():
    # 5 dates, 20 tickers each, but every ticker on a given date shares
    # that date's OWN net_bp exactly (a stand-in for "all 15 names move
    # together intraday") — the true uncertainty is over 5 dates, not 100
    # trades. A CI that ignored clustering would be implausibly tight.
    date_means = [-40.0, -10.0, 20.0, 50.0, 80.0]
    trades = [
        _trade(f"T{i}.KS", day_idx, net_bp=date_means[day_idx], imb_resid=0.0)
        for day_idx in range(5) for i in range(20)
    ]
    lo, hi = bootstrap_ci_date_clustered(trades, n_bootstrap=2000, seed=42)
    # The CI must be wide enough to plausibly cover the actual spread of
    # per-date means (-40..80) rather than collapsing to a near-point
    # estimate the way a per-trade (unclustered) CI over 100 nearly-
    # identical-within-date values would.
    assert hi - lo > 40


def test_date_clustered_ci_returns_zero_tuple_for_empty_input():
    assert bootstrap_ci_date_clustered([]) == (0.0, 0.0)


# --- within-date permutation test -------------------------------------------


def test_permutation_p_is_low_for_a_real_within_date_separation():
    # Within EVERY date, the top-imbalance ticker genuinely earns more
    # than the bottom-imbalance ticker — a real cross-sectional effect
    # that should survive permuting ranks within each date.
    trades = []
    for day_idx in range(20):
        for rank in range(10):
            trades.append(_trade(f"T{rank}.KS", day_idx, net_bp=float(rank * 10), imb_resid=float(rank)))
    p = within_date_permutation_test(trades, n_permutations=500, seed=1)
    assert p < 0.05


def test_permutation_p_is_high_when_imbalance_is_uninformative():
    # imb_resid is assigned independent of net_bp within each date — no
    # real relationship, permutation should not reject the null.
    import random
    rng = random.Random(7)
    trades = []
    for day_idx in range(20):
        resids = list(range(10))
        rng.shuffle(resids)
        for rank, resid in enumerate(resids):
            trades.append(_trade(f"T{rank}.KS", day_idx, net_bp=float(rank), imb_resid=float(resid)))
    p = within_date_permutation_test(trades, n_permutations=500, seed=2)
    assert p > 0.05


# --- build_decile_report verdicts -------------------------------------------


def _many_dates_monotone_trades(n_dates: int = 60, tickers_per_date: int = 15) -> list[AuctionTrade]:
    trades = []
    for day_idx in range(n_dates):
        for rank in range(tickers_per_date):
            # Top-ranked ticker each day earns a strong, clearly positive
            # return; bottom-ranked earns a clear loss — clean monotone
            # gradient across the whole cross-section.
            net_bp = -30.0 + rank * 6.0  # rank 0 -> -30bp .. rank 14 -> +54bp
            trades.append(_trade(f"T{rank}.KS", day_idx, net_bp=net_bp, imb_resid=float(rank)))
    return trades


def test_build_decile_report_passes_on_a_clean_strong_monotone_sample():
    trades = _many_dates_monotone_trades()
    report = build_decile_report(trades, n_bootstrap=500, n_permutations=500)
    assert report.data_quality_ok is True
    assert report.verdict == "PASS"


def test_build_decile_report_kills_on_a_flat_no_relationship_sample():
    # Every ticker earns the same slightly-negative return regardless of
    # imb_resid — no monotonicity, top decile mean well below the bar.
    trades = [
        _trade(f"T{rank}.KS", day_idx, net_bp=-5.0, imb_resid=float(rank))
        for day_idx in range(60) for rank in range(15)
    ]
    report = build_decile_report(trades, n_bootstrap=500, n_permutations=500)
    assert report.verdict == "KILL"


def test_build_decile_report_flags_data_quality_inconclusive_on_heavy_gap_exclusion():
    trades = [_trade(f"T{i}.KS", i, net_bp=10.0, imb_resid=float(i), gap_inflated=True) for i in range(20)]
    trades += [_trade(f"T{i}.KS", i, net_bp=10.0, imb_resid=float(i)) for i in range(2)]  # <10% would-be-included
    report = build_decile_report(trades, n_bootstrap=200, n_permutations=200)
    assert report.data_quality_ok is False
    assert report.verdict == "DATA_QUALITY_INCONCLUSIVE"


def test_build_decile_report_does_not_kill_on_a_tiny_noisy_underpowered_top_decile():
    # 2026-09-11 fix: the KILL branch used to check ONLY top.mean_net_bp /
    # top.ci_date_clustered[0] / spearman, with no minimum-sample guard —
    # unlike PASS, which already requires top.n_trades/n_dates to clear
    # AUCTION_MIN_DECILE_TRADES/AUCTION_MIN_DISTINCT_DATES first. This is
    # the flat-no-relationship KILL sample from
    # test_build_decile_report_kills_on_a_flat_no_relationship_sample above,
    # but with only 5 distinct dates instead of 60 — far below both
    # minimums, so the top decile's bad-looking mean/spearman here is
    # exactly the kind of noise a too-small sample produces, not a real,
    # trustworthy result. Before the fix this fired a confident "KILL"
    # ("indistinguishable from noise") off a top decile with only a
    # handful of trades/dates; now it must fall through to
    # DIRECTIONAL_INSUFFICIENT instead — "not enough data to call it either
    # way" is the honest verdict for a sample this thin.
    trades = [
        _trade(f"T{rank}.KS", day_idx, net_bp=-5.0, imb_resid=float(rank))
        for day_idx in range(5) for rank in range(15)
    ]
    report = build_decile_report(trades, n_bootstrap=500, n_permutations=500)
    assert report.verdict != "KILL"
    assert report.verdict == "DIRECTIONAL_INSUFFICIENT"


def test_build_decile_report_still_kills_a_well_powered_flat_sample():
    # Regression companion to the test above: a REAL, well-powered kill
    # (plenty of trades and distinct dates — same as
    # test_build_decile_report_kills_on_a_flat_no_relationship_sample) must
    # still fire KILL after the fix. The new minimum-sample gate must not
    # accidentally swallow a legitimate kill just because it's now checking
    # sample size at all.
    trades = [
        _trade(f"T{rank}.KS", day_idx, net_bp=-5.0, imb_resid=float(rank))
        for day_idx in range(60) for rank in range(15)
    ]
    report = build_decile_report(trades, n_bootstrap=500, n_permutations=500)
    assert report.verdict == "KILL"


def test_build_decile_report_directional_insufficient_on_a_small_promising_but_underpowered_sample():
    # A real-looking monotone gradient but far too few distinct dates to
    # clear AUCTION_MIN_DISTINCT_DATES (40) — should not KILL a
    # potentially-real pattern just for being underpowered, and should
    # not PASS on a sample this thin either.
    trades = [
        _trade(f"T{rank}.KS", day_idx, net_bp=-30.0 + rank * 6.0, imb_resid=float(rank))
        for day_idx in range(5) for rank in range(15)
    ]
    report = build_decile_report(trades, n_bootstrap=500, n_permutations=500)
    assert report.verdict == "DIRECTIONAL_INSUFFICIENT"
