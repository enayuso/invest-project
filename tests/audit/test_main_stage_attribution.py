"""app.audit.__main__ CLI wiring for stage-attribution, plus the --basis default
plumbing regression (every other command must still default to 'variable')."""

from __future__ import annotations

import sys
from datetime import date

import pytest

from app.audit import __main__ as audit_main

_SA = ["app.audit", "stage-attribution", "--market", "domestic", "--account-mode", "real"]


def _parse(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", argv)
    return audit_main._parse_args()


@pytest.mark.parametrize("argv", [
    ["app.audit", "stage-attribution", "--market", "domestic"],  # no --account-mode
    ["app.audit", "stage-attribution", "--account-mode", "real"],  # no --market
    [*_SA, "--scope", "backtest"],
    [*_SA, "--ticker", "005930.KS"],
    ["app.audit", "report", "--show-underpowered"],
    ["app.audit", "report", "--era-boundary", "2026-09-17T03:10:52Z"],
    [*_SA, "--era-boundary", "not-a-date"],
    ["app.audit", "entry-paired", "--market", "domestic", "--scope", "live", "--account-mode", "real",
     "--recompute-pending", "--start-date", "2026-09-01"],
    ["app.audit", "stage-attribution", "--market", "overseas", "--account-mode", "real", "--recompute-pending"],
])
def test_invalid_invocations_exit(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        audit_main._parse_args()


def test_basis_default_is_horizon_only_for_stage_attribution(monkeypatch):
    assert _parse(monkeypatch, _SA).basis == "horizon"
    assert _parse(monkeypatch, [*_SA, "--basis", "variable"]).basis == "variable"
    assert _parse(monkeypatch, ["app.audit", "report"]).basis == "variable"
    assert _parse(monkeypatch, ["app.audit", "report", "--basis", "horizon"]).basis == "horizon"
    for cmd in ("entry-paired", "entry-version-strat", "guard-alpha", "confidence", "weekly"):
        assert _parse(monkeypatch, ["app.audit", cmd]).basis == "variable", cmd


def test_stage_attribution_scope_defaults_to_live_and_recompute_allows_dates(monkeypatch):
    args = _parse(monkeypatch, [*_SA, "--recompute-pending", "--start-date", "2026-09-18"])
    assert args.scope == "live" and args.recompute_pending and args.start_date == "2026-09-18"


async def test_main_dispatch_plain_and_recompute(monkeypatch, capsys):
    calls = {}

    async def fake_compute(**kwargs):
        calls["compute"] = kwargs
        return "REPORT"

    async def fake_recompute(**kwargs):
        calls["recompute"] = kwargs
        return "RECOMPUTE"

    monkeypatch.setattr(audit_main, "compute_stage_attribution", fake_compute)
    monkeypatch.setattr(audit_main, "compute_stage_attribution_with_recompute", fake_recompute)
    monkeypatch.setattr(
        audit_main, "render_stage_attribution", lambda r, show_underpowered=False: f"{r}|{show_underpowered}",
    )
    monkeypatch.setattr(
        audit_main, "render_stage_attribution_recompute",
        lambda r, show_underpowered=False: f"{r}|{show_underpowered}",
    )

    monkeypatch.setattr(sys, "argv", [*_SA, "--start-date", "2026-09-18", "--show-underpowered"])
    await audit_main._main()
    assert "REPORT|True" in capsys.readouterr().out
    assert calls["compute"] == dict(
        market="domestic", account_mode="real", trigger=None, basis="horizon",
        since=date(2026, 9, 18), until=None, era_boundary=None,
    )

    monkeypatch.setattr(sys, "argv", [*_SA, "--recompute-pending", "--start-date", "2026-09-18"])
    await audit_main._main()
    assert "RECOMPUTE|False" in capsys.readouterr().out
    assert calls["recompute"]["since"] == date(2026, 9, 18)
