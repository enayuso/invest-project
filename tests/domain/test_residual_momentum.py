"""app.domain.residual_momentum — pure functions, no I/O."""

from __future__ import annotations

import pytest

from app.domain.residual_momentum import residual_pct


def test_identical_benchmark_move_collapses_residual_to_zero():
    assert residual_pct(0.05, 0.05) == pytest.approx(0.0)
    assert residual_pct(0.10, 0.10) == pytest.approx(0.0)


def test_outperformance_is_positive_residual():
    assert residual_pct(0.08, 0.03) == pytest.approx(0.05)


def test_underperformance_is_negative_residual():
    assert residual_pct(0.01, 0.03) == pytest.approx(-0.02)


def test_stock_up_while_market_down_amplifies_residual():
    assert residual_pct(0.02, -0.04) == pytest.approx(0.06)


def test_benchmark_none_returns_raw_unchanged():
    assert residual_pct(0.05, None) == 0.05


def test_beta_scales_the_market_component():
    assert residual_pct(0.10, 0.05, beta=1.5) == pytest.approx(0.025)


def test_zero_benchmark_move_is_identity():
    assert residual_pct(0.05, 0.0) == pytest.approx(0.05)
