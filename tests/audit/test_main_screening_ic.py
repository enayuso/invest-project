"""app.audit.__main__'s screening-ic CLI wiring — 2026-09-10 §0단계 P0 fix:
--start-date/--end-date must be honored instead of only --weeks, so a range
can be pinned exactly (e.g. a single day, or from a known deploy date) rather
than always counting back from today."""

from __future__ import annotations

import sys
from datetime import date

import pytest

from app.audit import __main__ as audit_main


class _FakeReport:
    pass


async def _fake_compute_screening_ic(market, start, end, *, holding_days=None, account_mode=None):
    return {"market": market, "start": start, "end": end, "holding_days": holding_days, "account_mode": account_mode}


def _fake_render_screening_ic(report):
    return str(report)


@pytest.fixture(autouse=True)
def _patch_screening_ic(monkeypatch):
    monkeypatch.setattr(audit_main, "compute_screening_ic", _fake_compute_screening_ic)
    monkeypatch.setattr(audit_main, "render_screening_ic", _fake_render_screening_ic)


async def test_start_and_end_date_both_given_are_used_verbatim(monkeypatch, capsys):
    monkeypatch.setattr(
        sys, "argv",
        ["app.audit", "screening-ic", "--market", "domestic", "--account-mode", "real",
         "--start-date", "2026-09-10", "--end-date", "2026-09-10"],
    )
    await audit_main._main()
    out = capsys.readouterr().out
    assert "'start': datetime.date(2026, 9, 10)" in out
    assert "'end': datetime.date(2026, 9, 10)" in out
    assert "'account_mode': 'real'" in out


async def test_end_date_defaults_to_today_when_omitted(monkeypatch, capsys):
    fixed_today = date(2026, 9, 10)

    class _FixedClock:
        def today(self):
            return fixed_today

    import app.clock as clock_module
    monkeypatch.setattr(clock_module, "get_clock", lambda: _FixedClock())

    monkeypatch.setattr(
        sys, "argv",
        ["app.audit", "screening-ic", "--market", "domestic", "--start-date", "2026-09-01"],
    )
    await audit_main._main()
    out = capsys.readouterr().out
    assert "'start': datetime.date(2026, 9, 1)" in out
    assert "'end': datetime.date(2026, 9, 10)" in out


async def test_weeks_still_works_when_no_start_date_given(monkeypatch, capsys):
    fixed_today = date(2026, 9, 10)

    class _FixedClock:
        def today(self):
            return fixed_today

    import app.clock as clock_module
    monkeypatch.setattr(clock_module, "get_clock", lambda: _FixedClock())

    monkeypatch.setattr(
        sys, "argv",
        ["app.audit", "screening-ic", "--market", "domestic", "--weeks", "1"],
    )
    await audit_main._main()
    out = capsys.readouterr().out
    assert "'start': datetime.date(2026, 9, 3)" in out  # 1 week back from fixed today
    assert "'end': datetime.date(2026, 9, 10)" in out


async def test_start_date_given_ignores_weeks(monkeypatch, capsys):
    monkeypatch.setattr(
        sys, "argv",
        ["app.audit", "screening-ic", "--market", "domestic",
         "--start-date", "2026-09-10", "--end-date", "2026-09-10", "--weeks", "52"],
    )
    await audit_main._main()
    out = capsys.readouterr().out
    assert "'start': datetime.date(2026, 9, 10)" in out  # not 52 weeks back
