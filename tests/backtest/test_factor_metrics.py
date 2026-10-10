"""app.backtest.factor_metrics — pure, no network, no DB. The kernel copies
consolidated out of app.audit.signal_ic / app.backtest.auction_imbalance_report
plus the new factor statistics for app.backtest.screening_sweep."""

from __future__ import annotations

import pytest

from app.backtest import factor_metrics as fm


# --- rank_avg ----------------------------------------------------------

def test_rank_avg_simple_ascending():
    assert fm.rank_avg([10.0, 20.0, 30.0]) == [1.0, 2.0, 3.0]


def test_rank_avg_unsorted_input():
    assert fm.rank_avg([30.0, 10.0, 20.0]) == [3.0, 1.0, 2.0]


def test_rank_avg_ties_share_mean_position():
    assert fm.rank_avg([5.0, 5.0, 9.0]) == [1.5, 1.5, 3.0]


def test_rank_avg_all_tied():
    assert fm.rank_avg([7.0, 7.0, 7.0, 7.0]) == [2.5, 2.5, 2.5, 2.5]


def test_rank_avg_empty():
    assert fm.rank_avg([]) == []


# --- spearman --------------------------------------------------------

def test_spearman_perfect_positive():
    assert fm.spearman([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]) == pytest.approx(1.0)


def test_spearman_perfect_negative():
    assert fm.spearman([1.0, 2.0, 3.0, 4.0], [40.0, 30.0, 20.0, 10.0]) == pytest.approx(-1.0)


def test_spearman_below_min_pairs_is_none():
    assert fm.spearman([1.0, 2.0], [1.0, 2.0], min_pairs=3) is None


def test_spearman_zero_variance_x_is_none():
    assert fm.spearman([5.0, 5.0, 5.0, 5.0], [1.0, 2.0, 3.0, 4.0]) is None


def test_spearman_zero_variance_y_is_none():
    assert fm.spearman([1.0, 2.0, 3.0, 4.0], [2.0, 2.0, 2.0, 2.0]) is None


def test_spearman_mismatched_lengths_is_none():
    assert fm.spearman([1.0, 2.0, 3.0], [1.0, 2.0]) is None


# case 9 — monotone-nonlinear: rank not Pearson
def test_spearman_on_y_equals_x_cubed_is_exactly_one():
    xs = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    ys = [x ** 3 for x in xs]
    assert fm.spearman(xs, ys) == pytest.approx(1.0)


def test_spearman_handles_ties():
    # two tied x, monotone y -> still strongly positive, defined
    val = fm.spearman([1.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0])
    assert val is not None and 0 < val <= 1.0


# --- percentile -----------------------------------------------------

def test_percentile_endpoints():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert fm.percentile(vals, 0) == 1.0
    assert fm.percentile(vals, 100) == 5.0


def test_percentile_median():
    assert fm.percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50) == 3.0


def test_percentile_empty_is_zero():
    assert fm.percentile([], 50) == 0.0


# --- rank_ic --------------------------------------------------------

def test_rank_ic_uses_whole_universe_not_top_n():
    scores = {t: float(i) for i, t in enumerate("ABCDEFGH")}
    forward = {t: float(i) for i, t in enumerate("ABCDEFGH")}
    assert fm.rank_ic(scores, forward) == pytest.approx(1.0)


def test_rank_ic_pairwise_drops_none_on_either_side():
    scores = {"A": 1.0, "B": 2.0, "C": None, "D": 4.0, "E": 5.0, "F": 6.0}
    forward = {"A": 1.0, "B": 2.0, "C": 3.0, "D": None, "E": 5.0, "F": 6.0}
    # usable pairs: A,B,E,F -> 4 < default min_cross_section 5 -> None
    assert fm.rank_ic(scores, forward) is None


def test_rank_ic_negative_means_reversal():
    scores = {t: float(i) for i, t in enumerate("ABCDEF")}
    forward = {t: float(-i) for i, t in enumerate("ABCDEF")}
    assert fm.rank_ic(scores, forward) == pytest.approx(-1.0)


def test_rank_ic_below_min_cross_section_is_none():
    scores = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}
    forward = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}
    assert fm.rank_ic(scores, forward, min_cross_section=5) is None


def test_rank_ic_missing_ticker_on_forward_side_dropped():
    scores = {t: float(i) for i, t in enumerate("ABCDEFG")}
    forward = {t: float(i) for i, t in enumerate("ABCDE")}  # F, G absent
    assert fm.rank_ic(scores, forward) == pytest.approx(1.0)


# --- icir ----------------------------------------------------------

def test_icir_mean_over_stdev():
    import statistics

    series = [0.4, 0.2, 0.3, 0.5]
    assert fm.icir(series) == pytest.approx(statistics.mean(series) / statistics.stdev(series))


def test_icir_ignores_none_entries():
    import statistics

    series = [0.4, None, 0.2, 0.3, None]
    assert fm.icir(series) == pytest.approx(statistics.mean([0.4, 0.2, 0.3]) / statistics.stdev([0.4, 0.2, 0.3]))


def test_icir_fewer_than_three_valid_is_none():
    assert fm.icir([0.3, 0.4, None]) is None


def test_icir_zero_stdev_is_none():
    assert fm.icir([0.3, 0.3, 0.3, 0.3]) is None


# --- excess_return -----------------------------------------------

def test_excess_return_basic():
    assert fm.excess_return([0.10, 0.20], [0.05, 0.05, 0.05, 0.05]) == pytest.approx(0.10)


def test_excess_return_empty_picks_is_none():
    assert fm.excess_return([], [0.01, 0.02]) is None


def test_excess_return_empty_universe_is_none():
    assert fm.excess_return([0.01], []) is None


# --- quantile_spread -------------------------------------------

def test_quantile_spread_positive_when_high_scores_win():
    scores = {t: float(i) for i, t in enumerate("ABCDEFGHIJ")}
    forward = {t: float(i) for i, t in enumerate("ABCDEFGHIJ")}
    spread = fm.quantile_spread(scores, forward, fraction=0.2)
    assert spread is not None and spread > 0


def test_quantile_spread_below_five_pairs_is_none():
    scores = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}
    forward = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}
    assert fm.quantile_spread(scores, forward) is None


def test_quantile_spread_negative_for_reversal():
    scores = {t: float(i) for i, t in enumerate("ABCDEFGHIJ")}
    forward = {t: float(-i) for i, t in enumerate("ABCDEFGHIJ")}
    spread = fm.quantile_spread(scores, forward, fraction=0.3)
    assert spread is not None and spread < 0


# --- effective_sample_size --------------------------------

def test_effective_sample_size_no_overlap():
    # holding 5 <= cadence 5 -> overlap 1 -> n_eff == n_dates
    assert fm.effective_sample_size(40, 5, 5) == pytest.approx(40.0)


def test_effective_sample_size_overlap_deflates():
    # holding 20, cadence 5 -> overlap 4 -> 40 / 4 == 10
    assert fm.effective_sample_size(40, 5, 20) == pytest.approx(10.0)


def test_effective_sample_size_partial_overlap_ceils():
    # holding 12, cadence 5 -> ceil(2.4) == 3
    assert fm.effective_sample_size(30, 5, 12) == pytest.approx(10.0)


def test_effective_sample_size_zero_dates_is_none():
    assert fm.effective_sample_size(0, 5, 10) is None


def test_effective_sample_size_zero_cadence_is_none():
    assert fm.effective_sample_size(10, 0, 10) is None


# --- block_bootstrap_ci ---------------------------------

def test_block_bootstrap_ci_fewer_than_three_blocks_is_none():
    assert fm.block_bootstrap_ci([[0.1], [0.2]]) is None


def test_block_bootstrap_ci_empty_is_none():
    assert fm.block_bootstrap_ci([]) is None


def test_block_bootstrap_ci_all_empty_blocks_is_none():
    assert fm.block_bootstrap_ci([[], [], []]) is None


def test_block_bootstrap_ci_lo_le_hi_and_brackets_mean():
    blocks = [[0.01, 0.02], [0.03], [-0.01, 0.00], [0.02, 0.04]]
    ci = fm.block_bootstrap_ci(blocks, n_bootstrap=2000, seed=0)
    assert ci is not None
    lo, hi = ci
    assert lo <= hi


# case 28/29 — seed determinism
def test_block_bootstrap_ci_same_seed_same_result():
    blocks = [[0.01, 0.02], [0.03, -0.02], [-0.01, 0.05], [0.02, 0.04]]
    a = fm.block_bootstrap_ci(blocks, n_bootstrap=3000, seed=7)
    b = fm.block_bootstrap_ci(blocks, n_bootstrap=3000, seed=7)
    assert a == b


def test_block_bootstrap_ci_different_seed_differs():
    blocks = [[0.01, 0.02], [0.03, -0.02], [-0.01, 0.05], [0.02, 0.04], [0.0, 0.01]]
    a = fm.block_bootstrap_ci(blocks, n_bootstrap=3000, seed=1)
    b = fm.block_bootstrap_ci(blocks, n_bootstrap=3000, seed=2)
    assert a != b


# case 33 — clustering widens the CI: this pins the whole thesis
def test_block_bootstrap_ci_grouped_blocks_wider_than_per_pick():
    # Same 9 numbers. Version (a): 9 singleton blocks. Version (b): 3
    # date-grouped blocks, where each date's picks moved together.
    per_date = [[0.10, 0.10, 0.10], [0.00, 0.00, 0.00], [-0.10, -0.10, -0.10]]
    per_pick = [[v] for block in per_date for v in block]

    ci_pick = fm.block_bootstrap_ci(per_pick, n_bootstrap=5000, seed=0)
    ci_date = fm.block_bootstrap_ci(per_date, n_bootstrap=5000, seed=0)
    assert ci_pick is not None and ci_date is not None
    assert (ci_date[1] - ci_date[0]) > (ci_pick[1] - ci_pick[0])


# --- turnover -------------------------------------------

def test_turnover_all_new():
    assert fm.turnover(["A", "B"], ["C", "D"]) == pytest.approx(1.0)


def test_turnover_none_new():
    assert fm.turnover(["A", "B", "C"], ["C", "B", "A"]) == pytest.approx(0.0)


def test_turnover_half_new():
    assert fm.turnover(["A", "B"], ["A", "C"]) == pytest.approx(0.5)


def test_turnover_empty_picks_is_none():
    assert fm.turnover(["A"], []) is None


def test_turnover_order_independent():
    assert fm.turnover(["A", "B"], ["B", "A"]) == fm.turnover(["B", "A"], ["A", "B"])


# --- worth_analyzing_rate / upside_hit_rate ------------

def test_worth_analyzing_rate_counts_both_directions():
    # -0.05 and +0.04 clear the 0.03 bar; +0.01 and -0.02 do not
    assert fm.worth_analyzing_rate([-0.05, 0.04, 0.01, -0.02]) == pytest.approx(0.5)


def test_worth_analyzing_rate_empty_is_none():
    assert fm.worth_analyzing_rate([]) is None


def test_worth_analyzing_rate_custom_bar():
    assert fm.worth_analyzing_rate([0.05, 0.05, 0.01], bar=0.10) == pytest.approx(0.0)


def test_upside_hit_rate_is_directional():
    # only the +0.04 counts; the -0.05 does not (unlike worth_analyzing_rate)
    assert fm.upside_hit_rate([-0.05, 0.04, 0.01, -0.02]) == pytest.approx(0.25)


def test_upside_hit_rate_empty_is_none():
    assert fm.upside_hit_rate([]) is None


def test_interpretation_guide_mentions_the_hard_rules():
    guide = fm.INTERPRETATION_GUIDE
    assert "RankIC" in guide and "ICIR" in guide
    assert "n_eff" in guide
    assert "0.02" in guide and "0.05" in guide and "0.10" in guide
