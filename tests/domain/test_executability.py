"""app.domain.executability — pure functions, no I/O.

2026-09-01 audit root cause #6: a ₩200,000 real account burned a full
multi-agent LLM pipeline on candidates that couldn't be sized to even 1
share. These are the SAME kernels app.execution.ledger uses to size a real
trade — see test_ledger.py's parity tests for the cross-module guarantee
that this module's notion of "affordable" cannot silently drift from the
ledger's actual binding math.
"""

from __future__ import annotations

import pytest

from app.domain.executability import (
    ExecutabilityInputs,
    SizingLimits,
    affordable_price_ceiling,
    allocation_target_shares,
    atr_unavailable,
    best_case_buyable_shares,
    blocks_new_entry_without_atr,
    cash_affordable_shares,
    is_unexecutable,
    min_stop_distance,
    risk_capped_shares,
)


# --- allocation_target_shares ------------------------------------------------


def test_allocation_target_shares_floors():
    assert allocation_target_shares(200_000, 0.4, 61_000) == 1


def test_allocation_target_shares_non_positive_inputs_return_zero():
    assert allocation_target_shares(0, 0.4, 61_000) == 0
    assert allocation_target_shares(-100, 0.4, 61_000) == 0
    assert allocation_target_shares(200_000, 0, 61_000) == 0
    assert allocation_target_shares(200_000, -0.1, 61_000) == 0
    assert allocation_target_shares(200_000, 0.4, 0) == 0
    assert allocation_target_shares(200_000, 0.4, -1) == 0


# --- min_stop_distance -------------------------------------------------------


def test_min_stop_distance_exact():
    # price * atr * multiple
    assert min_stop_distance(10_000, 0.05, 2.5) == 10_000 * 0.05 * 2.5


def test_min_stop_distance_clamped_at_extreme_atr():
    # 50% ATR * 2.5 = 125% of price -- must clamp to 95% of price.
    assert min_stop_distance(10_000, 0.50, 2.5) == 10_000 * 0.95


def test_min_stop_distance_none_when_atr_missing_or_non_positive():
    assert min_stop_distance(10_000, None, 2.5) is None
    assert min_stop_distance(10_000, 0, 2.5) is None
    assert min_stop_distance(10_000, -0.01, 2.5) is None


# --- risk_capped_shares -------------------------------------------------------


def test_risk_capped_shares_floors_and_never_negative():
    # 1% of 200,000 = 2,000 risk budget / 500 risk-per-share = 4 shares
    assert risk_capped_shares(200_000, 500, 0.01) == 4


def test_risk_capped_shares_none_when_uncapped():
    assert risk_capped_shares(200_000, 0, 0.01) is None
    assert risk_capped_shares(200_000, -10, 0.01) is None


def test_risk_capped_shares_zero_equity_floors_to_zero_not_none():
    # Deliberately asymmetric with the other kernels: matches
    # app.execution.ledger's exact behavior (max_dollar_risk=0 -> 0 shares,
    # a restrictive result), not "uncapped."
    assert risk_capped_shares(0, 500, 0.01) == 0


# --- cash_affordable_shares ---------------------------------------------------


def test_cash_affordable_shares_floors():
    assert cash_affordable_shares(200_000, 61_000) == 3


def test_cash_affordable_shares_zero_when_cash_below_price():
    assert cash_affordable_shares(1_000, 61_000) == 0


def test_cash_affordable_shares_non_positive_inputs_return_zero():
    assert cash_affordable_shares(0, 61_000) == 0
    assert cash_affordable_shares(200_000, 0) == 0


# --- best_case_buyable_shares / is_unexecutable ------------------------------


def test_audit_sample_ticker_is_unexecutable_risk_bound():
    # The 2026-09-01 audit's own arithmetic: E=200,000, cash=200,000,
    # P=61,000, ATR=5% -> risk cap is 0 shares, well before allocation or
    # cash bind. (max_allocation_fraction=0.4 mirrors
    # vol_adjusted_allocation_for(BUY, atr=0.05).)
    inputs = ExecutabilityInputs(price=61_000, atr_pct=0.05, equity=200_000, cash=200_000)
    limits = SizingLimits(max_allocation_fraction=0.4, max_risk_pct_of_equity=0.01, min_stop_atr_multiple=2.5)
    verdict = best_case_buyable_shares(inputs, limits)
    assert verdict.max_buyable_shares == 0
    assert verdict.binding_constraint == "risk"
    assert is_unexecutable(inputs, limits) is True


def test_low_atr_crossover_allocation_binds_instead_of_risk():
    # Below the ~0.4% ATR crossover (see audit design §2.1), the allocation
    # cap becomes tighter than the risk cap -- pins that crossover point so
    # a future change to either formula can't silently invert it.
    inputs = ExecutabilityInputs(price=10_000, atr_pct=0.003, equity=200_000, cash=10_000_000)
    limits = SizingLimits(max_allocation_fraction=1.0, max_risk_pct_of_equity=0.01, min_stop_atr_multiple=2.5)
    verdict = best_case_buyable_shares(inputs, limits)
    assert verdict.binding_constraint == "allocation"


def test_cash_binds_when_account_is_small_and_price_moderate():
    inputs = ExecutabilityInputs(price=4_000, atr_pct=0.02, equity=10_000_000, cash=5_000)
    limits = SizingLimits(max_allocation_fraction=1.0, max_risk_pct_of_equity=0.01, min_stop_atr_multiple=2.5)
    verdict = best_case_buyable_shares(inputs, limits)
    assert verdict.binding_constraint == "cash"
    assert verdict.max_buyable_shares == 1


def test_missing_atr_leaves_risk_cap_uncapped():
    inputs = ExecutabilityInputs(price=10_000, atr_pct=None, equity=200_000, cash=200_000)
    limits = SizingLimits(max_allocation_fraction=1.0, max_risk_pct_of_equity=0.01, min_stop_atr_multiple=2.5)
    verdict = best_case_buyable_shares(inputs, limits)
    assert verdict.risk_cap_shares is None
    assert verdict.binding_constraint in ("allocation", "cash")


def test_affordable_ticker_is_not_unexecutable():
    inputs = ExecutabilityInputs(price=12_000, atr_pct=0.02, equity=200_000, cash=200_000)
    limits = SizingLimits(max_allocation_fraction=0.7, max_risk_pct_of_equity=0.01, min_stop_atr_multiple=2.5)
    verdict = best_case_buyable_shares(inputs, limits)
    assert verdict.max_buyable_shares >= 1
    assert is_unexecutable(inputs, limits) is False


# --- atr_unavailable / blocks_new_entry_without_atr (2026-09-02, id=1169) ---


def test_atr_unavailable_true_for_none_zero_and_negative():
    assert atr_unavailable(None) is True
    assert atr_unavailable(0.0) is True
    assert atr_unavailable(-0.01) is True


def test_atr_unavailable_false_for_small_positive():
    assert atr_unavailable(0.001) is False
    assert atr_unavailable(0.05) is False


def test_blocks_new_entry_without_atr_flat_and_no_atr_blocks():
    assert blocks_new_entry_without_atr(None, current_shares=0) is True
    assert blocks_new_entry_without_atr(0.0, current_shares=0) is True
    assert blocks_new_entry_without_atr(-0.01, current_shares=-5) is True  # never negative in practice, still covered


def test_blocks_new_entry_without_atr_held_position_is_exempt():
    assert blocks_new_entry_without_atr(None, current_shares=10) is False
    assert blocks_new_entry_without_atr(0.0, current_shares=1) is False


def test_blocks_new_entry_without_atr_false_when_atr_present():
    assert blocks_new_entry_without_atr(0.02, current_shares=0) is False


def test_reason_string_names_all_three_caps_and_the_binding_one():
    inputs = ExecutabilityInputs(price=61_000, atr_pct=0.05, equity=200_000, cash=200_000)
    limits = SizingLimits(max_allocation_fraction=0.4, max_risk_pct_of_equity=0.01, min_stop_atr_multiple=2.5)
    verdict = best_case_buyable_shares(inputs, limits)
    assert "배분캡" in verdict.reason
    assert "리스크캡" in verdict.reason
    assert "현금캡" in verdict.reason
    assert verdict.binding_constraint in verdict.reason


# --- affordable_price_ceiling (2026-09-02, screening-improvement-research
# #3 follow-up) — the inverse of best_case_buyable_shares: not "how many
# shares of this ticker", but "what's the highest price a ticker could have
# and still clear every cap at >=1 share". Built for
# app.live.auto_trade_loop._domestic_price_ceiling, which needs this BEFORE
# any candidate exists to look a real ATR up for. -----------------------


def test_affordable_price_ceiling_matches_the_brief_arithmetic():
    # The corporate-account audit's own numbers: E=2,000,000, cash matches
    # equity (no existing positions), shipped risk defaults
    # (max_risk_pct_of_equity=0.01, min_stop_atr_multiple=2.5).
    limits = SizingLimits(max_allocation_fraction=1.0, max_risk_pct_of_equity=0.01, min_stop_atr_multiple=2.5)
    assert affordable_price_ceiling(2_000_000, 2_000_000, 0.02, limits, slack_multiple=1.0) == pytest.approx(400_000.0)
    assert affordable_price_ceiling(
        2_000_000, 2_000_000, 0.03, limits, slack_multiple=1.0
    ) == pytest.approx(2_000_000 * 0.01 / 0.075)
    assert affordable_price_ceiling(2_000_000, 2_000_000, 0.04, limits, slack_multiple=1.0) == pytest.approx(200_000.0)


def test_ceiling_is_the_exact_boundary_of_best_case_buyable_shares():
    # Load-bearing: proves the inverse can never silently drift from the
    # kernel it inverts. slack_multiple=1.0 throughout so the ceiling is
    # the EXACT (not generously-widened) boundary.
    cases = [
        (2_000_000.0, 2_000_000.0, 0.02, SizingLimits(1.0, 0.01, 2.5)),
        (500_000.0, 500_000.0, 0.03, SizingLimits(1.0, 0.01, 2.5)),
        (10_000_000.0, 10_000_000.0, 0.05, SizingLimits(0.5, 0.02, 2.0)),
        (1_000_000.0, 200_000.0, 0.02, SizingLimits(1.0, 0.01, 2.5)),  # cash binds
    ]
    for equity, cash, atr_pct, limits in cases:
        ceiling = affordable_price_ceiling(equity, cash, atr_pct, limits, slack_multiple=1.0)
        assert ceiling is not None
        at_ceiling = ExecutabilityInputs(price=ceiling, atr_pct=atr_pct, equity=equity, cash=cash)
        just_above = ExecutabilityInputs(price=ceiling * 1.0001, atr_pct=atr_pct, equity=equity, cash=cash)
        assert best_case_buyable_shares(at_ceiling, limits).max_buyable_shares >= 1
        assert best_case_buyable_shares(just_above, limits).max_buyable_shares == 0


def test_slack_multiple_widens_only_the_risk_term():
    limits = SizingLimits(max_allocation_fraction=1.0, max_risk_pct_of_equity=0.01, min_stop_atr_multiple=2.5)

    # Risk-bound case (cash/allocation both far above the risk ceiling):
    # the ceiling must scale by slack_multiple exactly.
    risk_bound_1x = affordable_price_ceiling(2_000_000, 2_000_000, 0.02, limits, slack_multiple=1.0)
    risk_bound_1_5x = affordable_price_ceiling(2_000_000, 2_000_000, 0.02, limits, slack_multiple=1.5)
    assert risk_bound_1_5x == pytest.approx(risk_bound_1x * 1.5)

    # Cash-bound case: slack must NOT touch the cash term at all.
    cash_bound_1x = affordable_price_ceiling(2_000_000, 300_000, 0.02, limits, slack_multiple=1.0)
    cash_bound_1_5x = affordable_price_ceiling(2_000_000, 300_000, 0.02, limits, slack_multiple=1.5)
    assert cash_bound_1x == pytest.approx(300_000.0)
    assert cash_bound_1_5x == pytest.approx(300_000.0)


def test_cash_binds_when_lower_than_the_risk_ceiling():
    limits = SizingLimits(max_allocation_fraction=1.0, max_risk_pct_of_equity=0.01, min_stop_atr_multiple=2.5)
    ceiling = affordable_price_ceiling(2_000_000, 300_000, 0.02, limits, slack_multiple=1.0)
    assert ceiling == pytest.approx(300_000.0)


def test_allocation_fraction_binds_when_risk_pct_is_large():
    # max_risk_pct_of_equity=0.5 pushes the risk term far above
    # equity*max_allocation_fraction, so the allocation term is the
    # binding (smallest) one.
    limits = SizingLimits(max_allocation_fraction=0.4, max_risk_pct_of_equity=0.5, min_stop_atr_multiple=2.5)
    ceiling = affordable_price_ceiling(1_000_000, 10_000_000, 0.02, limits, slack_multiple=1.0)
    assert ceiling == pytest.approx(400_000.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(equity=0),
        dict(equity=-1),
        dict(cash=0),
        dict(cash=-1),
        dict(assumed_atr_pct=0),
        dict(assumed_atr_pct=-0.01),
        dict(slack_multiple=0),
        dict(slack_multiple=-1),
    ],
)
def test_returns_none_on_unusable_inputs(kwargs):
    base = dict(equity=2_000_000, cash=2_000_000, assumed_atr_pct=0.02, slack_multiple=1.0)
    base.update(kwargs)
    limits = SizingLimits(max_allocation_fraction=1.0, max_risk_pct_of_equity=0.01, min_stop_atr_multiple=2.5)
    assert affordable_price_ceiling(
        base["equity"], base["cash"], base["assumed_atr_pct"], limits, slack_multiple=base["slack_multiple"]
    ) is None


@pytest.mark.parametrize(
    "limits",
    [
        SizingLimits(max_allocation_fraction=1.0, max_risk_pct_of_equity=0.0, min_stop_atr_multiple=2.5),
        SizingLimits(max_allocation_fraction=1.0, max_risk_pct_of_equity=-0.01, min_stop_atr_multiple=2.5),
        SizingLimits(max_allocation_fraction=1.0, max_risk_pct_of_equity=0.01, min_stop_atr_multiple=0.0),
        SizingLimits(max_allocation_fraction=1.0, max_risk_pct_of_equity=0.01, min_stop_atr_multiple=-2.5),
        SizingLimits(max_allocation_fraction=0.0, max_risk_pct_of_equity=0.01, min_stop_atr_multiple=2.5),
        SizingLimits(max_allocation_fraction=-1.0, max_risk_pct_of_equity=0.01, min_stop_atr_multiple=2.5),
    ],
)
def test_returns_none_on_unusable_limits(limits):
    assert affordable_price_ceiling(2_000_000, 2_000_000, 0.02, limits, slack_multiple=1.0) is None


def test_extreme_atr_mirrors_min_stop_distances_95pct_clamp():
    # atr_pct * min_stop_atr_multiple = 0.5 * 2.5 = 1.25 > 0.95 -- must
    # clamp to the same 0.95 min_stop_distance itself clamps to.
    equity, cash, atr_pct = 2_000_000.0, 10_000_000.0, 0.5
    limits = SizingLimits(max_allocation_fraction=1.0, max_risk_pct_of_equity=0.01, min_stop_atr_multiple=2.5)
    ceiling = affordable_price_ceiling(equity, cash, atr_pct, limits, slack_multiple=1.0)
    assert ceiling == pytest.approx(equity * 0.01 / 0.95)

    at_ceiling = ExecutabilityInputs(price=ceiling, atr_pct=atr_pct, equity=equity, cash=cash)
    just_above = ExecutabilityInputs(price=ceiling * 1.0001, atr_pct=atr_pct, equity=equity, cash=cash)
    assert best_case_buyable_shares(at_ceiling, limits).max_buyable_shares >= 1
    assert best_case_buyable_shares(just_above, limits).max_buyable_shares == 0
