"""app.backtest.atr_sweep.parse_atr_multipliers — 2026-09-11 fix: a
malformed --atr-multipliers entry used to only fail deep inside every
spawned child process (settings validation), reported as an ordinary
per-group failure rather than a clear upfront error."""

from __future__ import annotations

import pytest

from app.backtest.atr_sweep import parse_atr_multipliers


def test_parses_and_strips_whitespace():
    assert parse_atr_multipliers(" 1.0 , 1.5 , 2.25 ") == ["1.0", "1.5", "2.25"]


def test_preserves_original_textual_form():
    # Labels are built as f"{prefix}-atr{token}" and atr_sweep_report
    # matches on that exact text -- "1.50" must not become "1.5".
    assert parse_atr_multipliers("1.50") == ["1.50"]


def test_rejects_non_numeric_entry():
    with pytest.raises(ValueError, match="not a number"):
        parse_atr_multipliers("1.0,abc")


@pytest.mark.parametrize("bad", ["0", "-1", "nan", "inf", "-inf"])
def test_rejects_non_positive_or_non_finite_entry(bad):
    with pytest.raises(ValueError, match="positive finite"):
        parse_atr_multipliers(bad)


def test_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one value"):
        parse_atr_multipliers("")
    with pytest.raises(ValueError, match="at least one value"):
        parse_atr_multipliers(",,")
