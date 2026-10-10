"""app.live.kr_holidays.is_market_open_on — memory/DB/KIS cache layers,
degrade-on-failure. No real DB/network/KIS — pool, connection, and broker
all faked, same pattern as tests/execution/test_market_regime.py."""

from __future__ import annotations

from datetime import date

import pytest

from app.live import kr_holidays
from app.web import state
from conftest import broker_stub


class _AcquireCM:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


class FakeConn:
    def __init__(self, fetchrow_result=None):
        self.fetchrow_result = fetchrow_result
        self.executed: list[tuple] = []

    async def fetchrow(self, query, *args):
        return self.fetchrow_result

    async def execute(self, query, *args):
        if "CREATE TABLE" in query:
            return
        self.executed.append((query, args))


class FakePool:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        return _AcquireCM(self._conn)


class FakeBroker:
    def __init__(self, is_open=True, raise_exc=None):
        self._is_open = is_open
        self._raise = raise_exc
        self.calls: list[str] = []

    async def fetch_is_market_open_on(self, day):
        self.calls.append(day)
        if self._raise is not None:
            raise self._raise
        return self._is_open


async def _async_return(value):
    return value


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setattr(kr_holidays, "_schema_ready", False)
    monkeypatch.setattr(kr_holidays, "_memory_cache", {})
    monkeypatch.setattr(state, "kis_broker", broker_stub(None))


async def test_returns_db_cached_value_without_calling_kis(monkeypatch):
    conn = FakeConn(fetchrow_result={"is_open": False})
    monkeypatch.setattr(kr_holidays, "_ready_pool", lambda: _async_return(FakePool(conn)))
    broker = FakeBroker()
    monkeypatch.setattr(state, "kis_broker", broker_stub(broker))

    result = await kr_holidays.is_market_open_on(date(2026, 1, 1))

    assert result is False
    assert broker.calls == []  # DB cache hit — never reached the KIS call


async def test_calls_kis_and_persists_on_first_lookup(monkeypatch):
    conn = FakeConn(fetchrow_result=None)  # not cached yet
    monkeypatch.setattr(kr_holidays, "_ready_pool", lambda: _async_return(FakePool(conn)))
    broker = FakeBroker(is_open=False)
    monkeypatch.setattr(state, "kis_broker", broker_stub(broker))

    result = await kr_holidays.is_market_open_on(date(2026, 1, 1))

    assert result is False
    assert broker.calls == ["20260101"]
    assert any("INSERT INTO kr_market_holiday_cache" in q for q, _ in conn.executed)


async def test_second_lookup_same_date_uses_memory_cache_not_kis(monkeypatch):
    conn = FakeConn(fetchrow_result=None)
    monkeypatch.setattr(kr_holidays, "_ready_pool", lambda: _async_return(FakePool(conn)))
    broker = FakeBroker(is_open=True)
    monkeypatch.setattr(state, "kis_broker", broker_stub(broker))

    first = await kr_holidays.is_market_open_on(date(2026, 8, 4))
    second = await kr_holidays.is_market_open_on(date(2026, 8, 4))

    assert first is True and second is True
    assert broker.calls == ["20260804"]  # only the first lookup actually called KIS


async def test_degrades_to_open_when_no_broker_configured(monkeypatch):
    conn = FakeConn(fetchrow_result=None)
    monkeypatch.setattr(kr_holidays, "_ready_pool", lambda: _async_return(FakePool(conn)))
    result = await kr_holidays.is_market_open_on(date(2026, 1, 1))
    assert result is True


async def test_degrades_to_open_when_kis_call_fails(monkeypatch):
    conn = FakeConn(fetchrow_result=None)
    monkeypatch.setattr(kr_holidays, "_ready_pool", lambda: _async_return(FakePool(conn)))
    broker = FakeBroker(raise_exc=RuntimeError("KIS unreachable"))
    monkeypatch.setattr(state, "kis_broker", broker_stub(broker))

    result = await kr_holidays.is_market_open_on(date(2026, 1, 1))

    assert result is True


async def test_kis_call_failure_is_cached_in_memory_not_retried(monkeypatch):
    # Confirmed live 2026-08-04: a paper account's holiday-check TR rejects
    # unconditionally (EGW02006), so without caching the failure itself,
    # every poll of latest_domestic_signal_card_reset_at re-hit KIS and re-logged the
    # same traceback for the rest of the process's life (4 identical
    # tracebacks in the first 20 minutes of one run).
    conn = FakeConn(fetchrow_result=None)
    monkeypatch.setattr(kr_holidays, "_ready_pool", lambda: _async_return(FakePool(conn)))
    broker = FakeBroker(raise_exc=RuntimeError("EGW02006: not a paper-account TR"))
    monkeypatch.setattr(state, "kis_broker", broker_stub(broker))

    first = await kr_holidays.is_market_open_on(date(2026, 8, 4))
    second = await kr_holidays.is_market_open_on(date(2026, 8, 4))

    assert first is True and second is True
    assert broker.calls == ["20260804"]  # only the first lookup actually called KIS
    # Deliberately memory-only — kr_market_holiday_cache means "confirmed by
    # KIS", and this was never confirmed, just assumed.
    assert not any("INSERT INTO kr_market_holiday_cache" in q for q, _ in conn.executed)


async def test_db_read_failure_falls_through_to_a_live_kis_check(monkeypatch):
    # A broken DB cache alone must not force "assume open" — it should just
    # fall through to the live KIS check and trust whatever THAT says.
    def _broken_ready_pool():
        raise RuntimeError("DB unreachable")

    monkeypatch.setattr(kr_holidays, "_ready_pool", _broken_ready_pool)
    broker = FakeBroker(is_open=False)
    monkeypatch.setattr(state, "kis_broker", broker_stub(broker))

    result = await kr_holidays.is_market_open_on(date(2026, 1, 1))

    assert result is False  # the live KIS answer, not a blanket "assume open"
    assert broker.calls == ["20260101"]


# ---------------------------------------------------------------------------
# is_kr_market_holiday — the static table, sync, used by
# app.live.market_hours.is_domestic_market_open (the actual trading gate).
# Unlike is_market_open_on above, this never touches the DB/KIS at all.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("day", [
    date(2026, 1, 1),   # 신정
    date(2026, 2, 17),  # 설날
    date(2026, 3, 2),   # 삼일절 대체
    date(2026, 5, 1),   # 근로자의 날
    date(2026, 5, 25),  # 부처님오신날 대체
    date(2026, 8, 17),  # 광복절 대체
    date(2026, 9, 25),  # 추석
    date(2026, 10, 9),  # 한글날
    date(2026, 12, 25),  # 성탄절
    date(2026, 12, 31),  # 연말 휴장
])
def test_known_2026_holidays_are_flagged(day):
    assert kr_holidays.is_kr_market_holiday(day) is True


def test_an_ordinary_2026_weekday_is_not_a_holiday():
    assert kr_holidays.is_kr_market_holiday(date(2026, 7, 16)) is False


def test_an_uncovered_year_degrades_to_not_a_holiday_and_warns_once(caplog, monkeypatch):
    monkeypatch.setattr(kr_holidays, "_warned_years", set())
    with caplog.at_level("WARNING", logger="app.live.kr_holidays"):
        assert kr_holidays.is_kr_market_holiday(date(2099, 1, 1)) is False
        assert kr_holidays.is_kr_market_holiday(date(2099, 6, 1)) is False  # same year, no second warning
    warnings = [r for r in caplog.records if "no static holiday table" in r.message]
    assert len(warnings) == 1
