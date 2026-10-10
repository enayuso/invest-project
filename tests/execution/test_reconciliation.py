"""app.execution.reconciliation — pure matching logic (no DB/broker), plus
asyncpg pool/connection faked (in-memory rows) for the DB-reading half, same
convention as tests/execution/test_pending_orders.py."""

from __future__ import annotations

from datetime import date, datetime, timezone

import asyncpg
import pytest

from app.domain.models import BrokerExecution
from app.execution import reconciliation
from app.execution.reconciliation import LocalTrade, find_unmirrored_trades, match_trades_against_kis


def _execution(code="005930", side="buy", order_date="20260805", filled_qty=10) -> BrokerExecution:
    return BrokerExecution(
        order_date=order_date, order_no="0000000001", code=code, name="삼성전자", side=side,
        order_qty=filled_qty, filled_qty=filled_qty, remaining_qty=0, avg_price=70_000.0,
        filled_amount=filled_qty * 70_000.0,
    )


def _local_trade(trade_id=1, ticker="005930.KS", side="buy", shares=10, trade_date=date(2026, 8, 5),
                  created_at=None, reason="momentum buy") -> LocalTrade:
    return LocalTrade(
        id=trade_id, ticker=ticker, trade_date=trade_date, side=side, shares=shares,
        created_at=created_at or datetime(2026, 8, 5, 10, 0, 0, tzinfo=timezone.utc),
        reason=reason,
    )


def test_no_matching_execution_is_critical():
    trade = _local_trade()
    discrepancies = match_trades_against_kis([trade], domestic_executions=[], overseas_executions=[])
    assert len(discrepancies) == 1
    assert discrepancies[0].severity == "CRITICAL"
    assert discrepancies[0].trade_id == 1


def test_full_match_is_not_a_discrepancy():
    trade = _local_trade(shares=10)
    execution = _execution(filled_qty=10)
    discrepancies = match_trades_against_kis([trade], domestic_executions=[execution], overseas_executions=[])
    assert discrepancies == []


def test_partial_fill_is_warning():
    trade = _local_trade(shares=10)
    execution = _execution(filled_qty=4)
    discrepancies = match_trades_against_kis([trade], domestic_executions=[execution], overseas_executions=[])
    assert len(discrepancies) == 1
    assert discrepancies[0].severity == "WARNING"
    assert "4/10" in discrepancies[0].note


def test_wrong_side_or_date_does_not_match():
    trade = _local_trade(side="buy", trade_date=date(2026, 8, 5))
    wrong_side = _execution(side="sell", filled_qty=10)
    wrong_date = _execution(order_date="20260804", filled_qty=10)
    discrepancies = match_trades_against_kis(
        [trade], domestic_executions=[wrong_side, wrong_date], overseas_executions=[]
    )
    assert len(discrepancies) == 1
    assert discrepancies[0].severity == "CRITICAL"


def test_greedy_matching_pairs_multiple_same_day_trades_independently():
    # Two BUYs of the same ticker/day, one fully filled, one never reached
    # KIS at all (the actual crash scenario) — the fully-filled one must not
    # "consume" the wrong execution and leave the phantom one falsely matched.
    trade_a = _local_trade(trade_id=1, shares=10, created_at=datetime(2026, 8, 5, 9, 0, tzinfo=timezone.utc))
    trade_b = _local_trade(trade_id=2, shares=5, created_at=datetime(2026, 8, 5, 9, 5, tzinfo=timezone.utc))
    execution = _execution(filled_qty=10)  # only one real KIS execution exists
    discrepancies = match_trades_against_kis(
        [trade_b, trade_a],  # out of order on purpose — function must sort by created_at
        domestic_executions=[execution], overseas_executions=[],
    )
    assert len(discrepancies) == 1
    assert discrepancies[0].trade_id == 2
    assert discrepancies[0].severity == "CRITICAL"


# ---------------------------------------------------------------------------
# Delayed fill matching (2026-08-13) — a self-audit found two real trades
# (000810.KS, 006340.KS) whose KIS fills landed 5-8 days after the order
# date, which the old exact-date match flagged as a false CRITICAL. See
# match_trades_against_kis's own module comment (_MAX_FILL_DELAY_DAYS).
# ---------------------------------------------------------------------------


def test_fill_several_days_later_still_matches():
    trade = _local_trade(trade_date=date(2026, 8, 4))
    execution = _execution(order_date="20260812", filled_qty=10)  # 8 days later
    discrepancies = match_trades_against_kis([trade], domestic_executions=[execution], overseas_executions=[])
    assert discrepancies == []


def test_fill_beyond_the_delay_window_does_not_match():
    trade = _local_trade(trade_date=date(2026, 8, 4))
    execution = _execution(order_date="20260901", filled_qty=10)  # far beyond the window
    discrepancies = match_trades_against_kis([trade], domestic_executions=[execution], overseas_executions=[])
    assert len(discrepancies) == 1
    assert discrepancies[0].severity == "CRITICAL"


def test_fill_before_the_order_date_never_matches():
    # A fill can't precede the order that produced it — this must stay a
    # discrepancy even though it's chronologically "close."
    trade = _local_trade(trade_date=date(2026, 8, 4))
    execution = _execution(order_date="20260803", filled_qty=10)
    discrepancies = match_trades_against_kis([trade], domestic_executions=[execution], overseas_executions=[])
    assert len(discrepancies) == 1
    assert discrepancies[0].severity == "CRITICAL"


def test_delayed_partial_fill_note_mentions_the_delay():
    trade = _local_trade(trade_date=date(2026, 8, 4), shares=10)
    execution = _execution(order_date="20260810", filled_qty=4)  # 6 days later, partial
    discrepancies = match_trades_against_kis([trade], domestic_executions=[execution], overseas_executions=[])
    assert len(discrepancies) == 1
    assert "6 day(s)" in discrepancies[0].note


def test_same_day_fill_preferred_over_later_same_quantity_candidate():
    # Two candidates fill the exact same quantity — same-day must win over
    # a coincidentally-identical-quantity fill on a later date, so an
    # active ticker's genuinely same-day trade never gets mismatched.
    trade = _local_trade(trade_date=date(2026, 8, 4), shares=10)
    same_day = _execution(order_date="20260804", filled_qty=10)
    later = _execution(order_date="20260810", filled_qty=10)
    discrepancies = match_trades_against_kis(
        [trade], domestic_executions=[later, same_day], overseas_executions=[]
    )
    assert discrepancies == []  # matched same_day, not later — both fully filled either way


def test_overseas_ticker_matches_against_overseas_executions():
    trade = _local_trade(ticker="AAPL", shares=3)
    execution = _execution(code="AAPL", filled_qty=3)
    discrepancies = match_trades_against_kis([trade], domestic_executions=[], overseas_executions=[execution])
    assert discrepancies == []


def test_stop_loss_no_match_is_critical_with_unprotected_note():
    trade = _local_trade(reason="stop-loss triggered: breached 85200.0")
    discrepancies = match_trades_against_kis([trade], domestic_executions=[], overseas_executions=[])
    assert len(discrepancies) == 1
    assert discrepancies[0].severity == "CRITICAL"
    assert "UNPROTECTED" in discrepancies[0].note


def test_stop_loss_partial_fill_is_critical_not_warning():
    trade = _local_trade(shares=10, reason="stop-loss triggered: breached 85200.0")
    execution = _execution(filled_qty=4)
    discrepancies = match_trades_against_kis([trade], domestic_executions=[execution], overseas_executions=[])
    assert len(discrepancies) == 1
    assert discrepancies[0].severity == "CRITICAL"
    assert "4/10" in discrepancies[0].note
    assert "UNPROTECTED" in discrepancies[0].note


def test_non_stop_loss_partial_fill_stays_warning():
    trade = _local_trade(shares=10, reason="momentum buy")
    execution = _execution(filled_qty=4)
    discrepancies = match_trades_against_kis([trade], domestic_executions=[execution], overseas_executions=[])
    assert len(discrepancies) == 1
    assert discrepancies[0].severity == "WARNING"


class _AcquireCM:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


class FakePool:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        return _AcquireCM(self._conn)


class FakeTradesConn:
    def __init__(self, rows):
        self._rows = rows
        self.fetch_args = None

    async def fetch(self, query, *args):
        assert "reversal:%" in query  # reversal rows always excluded via SQL, never in Python
        self.fetch_args = args
        return self._rows


async def _async_return(value):
    return value


@pytest.fixture
def fake_trades_rows(monkeypatch):
    def _install(rows):
        conn = FakeTradesConn(rows)
        monkeypatch.setattr(reconciliation, "get_pool", lambda: _async_return(FakePool(conn)))
        return conn
    return _install


async def test_fetch_recent_local_trades_maps_rows(fake_trades_rows):
    fake_trades_rows([{
        "id": 7, "ticker": "005930.KS", "trade_date": date(2026, 8, 5), "side": "buy",
        "shares": 10, "created_at": datetime(2026, 8, 5, 10, 0, tzinfo=timezone.utc),
        "reason": "momentum buy",
    }])
    trades = await reconciliation.fetch_recent_local_trades(
        reconciliation.timedelta(hours=48), reconciliation.timedelta(minutes=2)
    )
    assert len(trades) == 1
    assert trades[0].id == 7 and trades[0].ticker == "005930.KS"


async def test_fetch_recent_local_trades_passes_raw_timedeltas_not_strings(fake_trades_rows):
    # Confirmed live 2026-08-06: passing f"{td.total_seconds()} seconds"
    # (a plain str) here failed with "invalid input for query argument $2:
    # '172800.0 seconds' ('str' object has no attribute 'days')" — the
    # query's explicit $2::interval/$3::interval casts make asyncpg prepare
    # those placeholders as interval type and encode them with its own
    # timedelta codec, which requires an actual datetime.timedelta (it calls
    # .days on the value), not a string for Postgres to parse server-side.
    conn = fake_trades_rows([])
    lookback, grace_period = reconciliation.timedelta(hours=48), reconciliation.timedelta(minutes=2)
    await reconciliation.fetch_recent_local_trades(lookback, grace_period)
    _account_mode, _user_id, arg_lookback, arg_grace = conn.fetch_args
    assert arg_lookback is lookback
    assert arg_grace is grace_period


async def test_fetch_recent_local_trades_degrades_on_interface_error(monkeypatch):
    async def broken_get_pool():
        raise asyncpg.InterfaceError("pool is closing")

    monkeypatch.setattr(reconciliation, "get_pool", broken_get_pool)
    trades = await reconciliation.fetch_recent_local_trades(
        reconciliation.timedelta(hours=48), reconciliation.timedelta(minutes=2)
    )
    assert trades == []


async def test_find_unmirrored_trades_short_circuits_when_no_local_trades(fake_trades_rows):
    fake_trades_rows([])
    discrepancies = await find_unmirrored_trades(domestic_executions=[_execution()], overseas_executions=[])
    assert discrepancies == []
