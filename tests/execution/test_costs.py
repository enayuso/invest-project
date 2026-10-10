"""app.execution.costs — the shared commission/tax formula both
app.execution.ledger and app.backtest.scalping_harness depend on. Pure
functions, no fixtures/mocking needed beyond monkeypatching the two
settings knobs the formula reads."""

from __future__ import annotations

import pytest

from app.config import settings
from app.execution import costs


def test_domestic_sell_charges_commission_plus_tax(monkeypatch):
    monkeypatch.setattr(settings, "transaction_cost_bps", 5.0)
    monkeypatch.setattr(settings, "sell_tax_bps_domestic", 20.0)

    assert costs.transaction_cost_bps_for("005930.KS", "sell") == pytest.approx(25.0)
    assert costs.transaction_cost("005930.KS", "sell", 1000.0) == pytest.approx(2.5)


def test_domestic_buy_is_commission_only(monkeypatch):
    monkeypatch.setattr(settings, "transaction_cost_bps", 5.0)
    monkeypatch.setattr(settings, "sell_tax_bps_domestic", 20.0)

    assert costs.transaction_cost_bps_for("005930.KS", "buy") == pytest.approx(5.0)
    assert costs.transaction_cost("005930.KS", "buy", 1000.0) == pytest.approx(0.5)


def test_overseas_sell_is_not_charged_the_korean_tax(monkeypatch):
    monkeypatch.setattr(settings, "transaction_cost_bps", 5.0)
    monkeypatch.setattr(settings, "transaction_cost_bps_overseas", 25.0)
    monkeypatch.setattr(settings, "sell_tax_bps_domestic", 20.0)

    assert costs.transaction_cost_bps_for("AAPL", "sell") == pytest.approx(25.0)
    assert costs.transaction_cost("AAPL", "sell", 1000.0) == pytest.approx(2.5)


def test_overseas_buy_is_commission_only(monkeypatch):
    monkeypatch.setattr(settings, "transaction_cost_bps", 5.0)
    monkeypatch.setattr(settings, "transaction_cost_bps_overseas", 25.0)
    monkeypatch.setattr(settings, "sell_tax_bps_domestic", 20.0)

    assert costs.transaction_cost_bps_for("AAPL", "buy") == pytest.approx(25.0)
    assert costs.transaction_cost("AAPL", "buy", 1000.0) == pytest.approx(2.5)


def test_overseas_uses_its_own_commission_rate_not_the_domestic_one(monkeypatch):
    """2026-09-08 fee/tax audit: before the market split, every overseas
    trade was silently charged the DOMESTIC commission (5bp), understating
    real US round-trip cost roughly 5x. Pins that the two settings are now
    read independently."""
    monkeypatch.setattr(settings, "transaction_cost_bps", 5.0)
    monkeypatch.setattr(settings, "transaction_cost_bps_overseas", 25.0)
    monkeypatch.setattr(settings, "sell_tax_bps_domestic", 20.0)

    assert costs.transaction_cost_bps_for("AAPL", "buy") != costs.transaction_cost_bps_for("005930.KS", "buy")
    assert costs.transaction_cost_bps_for("AAPL", "buy") == pytest.approx(settings.transaction_cost_bps_overseas)


def test_zero_sell_tax_collapses_to_symmetric_commission_only(monkeypatch):
    # Escape hatch / backward-compat pin, same as
    # tests/execution/test_ledger.py's identical case for record_trade: with
    # the tax knob at 0, domestic and overseas sells cost the same as a buy.
    monkeypatch.setattr(settings, "transaction_cost_bps", 5.0)
    monkeypatch.setattr(settings, "sell_tax_bps_domestic", 0.0)

    assert costs.transaction_cost_bps_for("005930.KS", "sell") == pytest.approx(5.0)
    assert costs.transaction_cost_bps_for("005930.KS", "sell") == pytest.approx(
        costs.transaction_cost_bps_for("005930.KS", "buy")
    )
