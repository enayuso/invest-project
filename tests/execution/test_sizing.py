"""app.execution.sizing — rating -> target allocation mapping, including the
volatility (ATR) and model-confidence down-only adjustments."""

from __future__ import annotations

import pytest

from app.config import settings
from app.execution import sizing
from app.state import PortfolioRating


@pytest.mark.parametrize(
    "confidence,expected_modifier",
    [
        (None, 0.0),
        (0.0, 0.0),
        (0.49, 0.0),
        (0.5, 0.5),
        (0.59, 0.5),
        (0.6, 0.7),
        (0.74, 0.7),
        (0.75, 0.9),
        (0.89, 0.9),
        (0.9, 1.0),
        (1.0, 1.0),
    ],
)
def test_confidence_size_modifier_bands(confidence, expected_modifier):
    assert sizing.confidence_size_modifier(confidence) == expected_modifier


def test_missing_confidence_is_scored_as_the_lowest_band_not_as_no_adjustment():
    """An omitted confidence must NOT buy a full-size position — otherwise the
    model is rewarded for staying silent (2026-08-19: GOOGL unset -> 49% full
    allocation vs GOOG 0.65 -> 0.7x). Skip (None), not liquidate (0.0)."""
    assert sizing.target_allocation_for(PortfolioRating.BUY, confidence=None) is None
    assert sizing.target_allocation_for(PortfolioRating.OVERWEIGHT, confidence=None) is None
    assert sizing.target_allocation_for(PortfolioRating.UNDERWEIGHT, confidence=None) is None


def test_missing_confidence_still_lets_a_sell_liquidate():
    """Sell's own base allocation is 0.0 (flat), so a missing confidence must
    not turn a full exit into a skip — the bearish path is never blocked."""
    assert sizing.target_allocation_for(PortfolioRating.SELL, confidence=None) == 0.0
    assert sizing.target_allocation_for(PortfolioRating.HOLD, confidence=None) is None


def test_confidence_is_required_and_keyword_only():
    """No positional/defaulted path back into the old silent full-size bug."""
    with pytest.raises(TypeError):
        sizing.target_allocation_for(PortfolioRating.BUY)
    with pytest.raises(TypeError):
        sizing.target_allocation_for(PortfolioRating.BUY, None, 0.9)


def test_vol_adjusted_allocation_is_the_confidence_free_preview(monkeypatch):
    """What _atr_context shows the PM before it has stated any confidence."""
    assert sizing.vol_adjusted_allocation_for(PortfolioRating.BUY) == 1.0
    assert sizing.vol_adjusted_allocation_for(PortfolioRating.OVERWEIGHT) == 0.75
    assert sizing.vol_adjusted_allocation_for(PortfolioRating.UNDERWEIGHT) == 0.25
    assert sizing.vol_adjusted_allocation_for(PortfolioRating.SELL) == 0.0
    assert sizing.vol_adjusted_allocation_for(PortfolioRating.HOLD) is None
    monkeypatch.setattr(settings, "sizing_target_atr_pct", 0.02)
    monkeypatch.setattr(settings, "sizing_min_vol_multiplier", 0.5)
    assert sizing.vol_adjusted_allocation_for(
        PortfolioRating.BUY, current_atr_pct=0.04,
    ) == pytest.approx(0.5)


def test_hold_ignores_confidence():
    assert sizing.target_allocation_for(PortfolioRating.HOLD, confidence=0.1) is None


def test_low_confidence_skips_a_buy_rather_than_liquidating():
    """0.0x modifier must read as `None` (skip, like Hold) not `0.0` (Sell's
    own base allocation, meaning liquidate to flat) - see target_allocation_for's
    own docstring for the real incident this distinguishes."""
    assert sizing.target_allocation_for(PortfolioRating.BUY, confidence=0.3) is None
    assert sizing.target_allocation_for(PortfolioRating.OVERWEIGHT, confidence=0.3) is None
    assert sizing.target_allocation_for(PortfolioRating.UNDERWEIGHT, confidence=0.3) is None


def test_low_confidence_leaves_sell_at_zero():
    """Sell's own base allocation is already 0.0 (liquidate), so a low
    confidence modifier must not turn it into `None` (which would mean skip
    the sell and keep holding)."""
    assert sizing.target_allocation_for(PortfolioRating.SELL, confidence=0.3) == 0.0


def test_full_confidence_never_exceeds_the_base_allocation():
    assert sizing.target_allocation_for(PortfolioRating.BUY, confidence=1.0) == 1.0


def test_confidence_and_atr_adjustments_compose(monkeypatch):
    monkeypatch.setattr(settings, "sizing_target_atr_pct", 0.02)
    monkeypatch.setattr(settings, "sizing_min_vol_multiplier", 0.5)
    current_atr_pct = 0.04  # double the target -> vol_multiplier = target/atr = 0.5
    expected = 1.0 * 0.5 * 0.7  # BUY base x ATR adjustment x confidence(0.6-0.75 band)

    result = sizing.target_allocation_for(
        PortfolioRating.BUY, current_atr_pct=current_atr_pct, confidence=0.65,
    )
    assert result == pytest.approx(expected)
