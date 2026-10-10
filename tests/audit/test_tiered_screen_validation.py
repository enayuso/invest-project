"""app.audit.tiered_screen_validation — asyncpg pool/connection faked, no real
Postgres (same FakePool/FakeConn pattern as tests/audit/test_audit_stats.py).

Focus: the 2026-08-27 `_fetch_position_seed` fix. `_held_state_before` walks
only the 30-day-widened `_fetch_trades` window and used to default to
`shares=0, stop=None` when a ticker had no row at all in that window — which
silently misreported a long-held, quiet position (opened more than 30 days
before `start_date`, never traded since) as "not currently held". Since the
2026-08-13 rework, "not currently held" is the ONLY remaining Tier-1
(skip-eligible) gate for a stable ticker, so that misreport could let a real
Sell on such a position pass through as Tier-1-eligible and silently
undercount this script's headline safety number ("would any sell-side action
have been missed")."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from app.audit import tiered_screen_validation as tsv


class FakeConn:
    def __init__(self, fetch_results=None):
        # Either a flat list (every .fetch() call gets the same result) or a
        # list-of-lists consumed in call order (for tests that stub multiple
        # distinct queries against the same fake connection).
        self._fetch_results = fetch_results if fetch_results is not None else []
        self._sequenced = isinstance(self._fetch_results, list) and self._fetch_results and isinstance(
            self._fetch_results[0], list
        )
        self._call_n = 0
        self.queries: list[tuple] = []

    async def fetch(self, query, *args):
        self.queries.append((query, args))
        if self._sequenced:
            result = self._fetch_results[self._call_n] if self._call_n < len(self._fetch_results) else []
            self._call_n += 1
            return result
        return self._fetch_results


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


async def _async_return(value):
    return value


def _decision(ticker="005930.KS", trade_date=date(2026, 8, 10), rating="Hold", action_label="Hold",
              trader_action="Hold"):
    return {
        "ticker": ticker, "trade_date": trade_date, "rating": rating,
        "action_label": action_label, "trader_action": trader_action,
    }


def _trade(ticker="005930.KS", trade_date=date(2026, 8, 1), shares_after=10, stop_loss=None,
           created_at=None):
    return {
        "ticker": ticker, "trade_date": trade_date, "shares_after": shares_after,
        "stop_loss": stop_loss, "created_at": created_at,
    }


# ---------------------------------------------------------------------------
# Direct unit tests: _held_state_before / _fetch_position_seed
# ---------------------------------------------------------------------------


def test_held_state_before_empty_history_returns_seed_unchanged():
    shares, stop = tsv._held_state_before([], date(2026, 8, 10), seed=(10, 95.0))
    assert shares == 10
    assert stop == 95.0


def test_held_state_before_default_seed_is_flat():
    shares, stop = tsv._held_state_before([], date(2026, 8, 10))
    assert shares == 0
    assert stop is None


def test_held_state_before_history_overrides_seed():
    trades = [_trade(trade_date=date(2026, 8, 5), shares_after=0, stop_loss=None)]
    shares, stop = tsv._held_state_before(trades, date(2026, 8, 10), seed=(10, 95.0))
    assert shares == 0  # the actual sell in the window wins over the stale seed


async def test_fetch_position_seed_returns_last_row_before_the_widened_window(monkeypatch):
    conn = FakeConn(fetch_results=[
        {"ticker": "005930.KS", "shares_after": 10, "stop_loss": 90.0},
    ])
    monkeypatch.setattr(tsv, "get_pool", lambda: _async_return(FakePool(conn)))

    seed = await tsv._fetch_position_seed(date(2026, 8, 10), "paper", "u1")

    assert seed == {"005930.KS": {"shares_after": 10, "stop_loss": 90.0}}
    query, args = conn.queries[0]
    assert "trade_date < $3" in query
    assert args == ("paper", "u1", date(2026, 8, 10) - timedelta(days=30))


async def test_fetch_position_seed_empty_when_no_prior_rows(monkeypatch):
    conn = FakeConn(fetch_results=[])
    monkeypatch.setattr(tsv, "get_pool", lambda: _async_return(FakePool(conn)))

    seed = await tsv._fetch_position_seed(date(2026, 8, 10), "paper", "u1")

    assert seed == {}


# ---------------------------------------------------------------------------
# End-to-end: simulate_tiered_screen threads the seed through both
# _held_state_before call sites (near-stop Tier-0 gate, and the Tier-1
# "must not currently be held" gate).
# ---------------------------------------------------------------------------


def _patch_simulate(monkeypatch, decisions, trades_rows, seed_rows):
    decisions_conn = FakeConn(fetch_results=decisions)
    trades_conn = FakeConn(fetch_results=trades_rows)
    seed_conn = FakeConn(fetch_results=seed_rows)

    async def fake_fetch_decisions(start_date, account_mode, user_id):
        return [dict(r) for r in decisions_conn._fetch_results]

    async def fake_fetch_trades(start_date, account_mode, user_id):
        by_ticker: dict[str, list[dict]] = {}
        for r in trades_rows:
            by_ticker.setdefault(r["ticker"], []).append(r)
        return by_ticker

    async def fake_fetch_seed(start_date, account_mode, user_id):
        return {r["ticker"]: {"shares_after": r["shares_after"], "stop_loss": r["stop_loss"]} for r in seed_rows}

    monkeypatch.setattr(tsv, "_fetch_decisions", fake_fetch_decisions)
    monkeypatch.setattr(tsv, "_fetch_trades", fake_fetch_trades)
    monkeypatch.setattr(tsv, "_fetch_position_seed", fake_fetch_seed)

    async def fake_get_last_close(ticker, as_of=None):
        return 100.0

    async def fake_get_atr_pct(ticker, as_of=None):
        return 0.02  # small, so price_move_beyond_atr never fires in these tests

    monkeypatch.setattr(tsv, "get_last_close", fake_get_last_close)
    monkeypatch.setattr(tsv, "get_atr_pct", fake_get_atr_pct)


async def test_long_held_quiet_position_is_not_treated_as_flat(monkeypatch):
    # Position opened 45 days before start_date (well past the 30-day
    # widened _fetch_trades window) and never traded since. Before the fix,
    # _held_state_before's empty-history default (shares=0) would make this
    # ticker Tier-1-eligible on the "not currently held" check; after the
    # fix, the seed correctly reports shares=10 (held).
    start_date = date(2026, 8, 1)
    ticker = "005930.KS"
    opened = start_date - timedelta(days=45)

    decisions = [
        # Two prior Holds (to satisfy MIN_CONSECUTIVE_HOLDS) then a Sell,
        # each spaced beyond STALENESS_LIMIT_DAYS apart is irrelevant here —
        # staleness/prior_not_hold etc. would force Tier 0 anyway and mask
        # the bug, so keep decisions close together and rating alternating
        # only on the final row.
        _decision(ticker, start_date, "Hold", "Hold", "Hold"),
        _decision(ticker, start_date + timedelta(days=1), "Hold", "Hold", "Hold"),
        _decision(ticker, start_date + timedelta(days=2), "Sell", "매도", "Sell"),
    ]
    seed_rows = [{"ticker": ticker, "shares_after": 10, "stop_loss": None}]
    trades_rows = []  # nothing in the 30-day-widened window — the bug's exact trigger

    _patch_simulate(monkeypatch, decisions, trades_rows, seed_rows)

    result = await tsv.simulate_tiered_screen(start_date=start_date, account_mode="paper", user_id="u1")

    # The Sell must NOT have been screened out as a "missed action" via a
    # false Tier-1 pass — it should show up as tier0 (forced full analysis)
    # because the position is genuinely held, so it can never even reach the
    # missed_actions bucket.
    assert result.missed_actions == []


async def test_without_seed_the_same_scenario_would_undercount(monkeypatch):
    # Sanity check that the scenario above actually exercises the bug when
    # the seed is empty (regression guard for the test itself, and a direct
    # demonstration of the pre-fix behavior via the real code path with an
    # empty seed).
    start_date = date(2026, 8, 1)
    ticker = "005930.KS"

    decisions = [
        _decision(ticker, start_date, "Hold", "Hold", "Hold"),
        _decision(ticker, start_date + timedelta(days=1), "Hold", "Hold", "Hold"),
        _decision(ticker, start_date + timedelta(days=2), "Sell", "매도", "Sell"),
    ]
    seed_rows = []  # no seed at all -> reproduces the old always-(0, None) default
    trades_rows = []

    _patch_simulate(monkeypatch, decisions, trades_rows, seed_rows)

    result = await tsv.simulate_tiered_screen(start_date=start_date, account_mode="paper", user_id="u1")

    # With no seed data available (position genuinely unknown/flat from this
    # script's point of view), the Sell is free to be screened as Tier 1 and
    # counted as a missed action -- demonstrating why the seed matters.
    assert len(result.missed_actions) == 1
    assert result.missed_actions[0].direction == "sell"


async def test_flat_ticker_with_no_seed_and_no_trades_stays_eligible_for_tier1(monkeypatch):
    # A ticker that's genuinely never been traded (no seed row, no trades
    # row) must still be able to reach Tier 1 on a Hold decision -- the fix
    # must not force everything to Tier 0 by accident.
    start_date = date(2026, 8, 1)
    ticker = "005930.KS"

    decisions = [
        _decision(ticker, start_date, "Hold", "Hold", "Hold"),
        _decision(ticker, start_date + timedelta(days=1), "Hold", "Hold", "Hold"),
        _decision(ticker, start_date + timedelta(days=2), "Hold", "Hold", "Hold"),
    ]
    _patch_simulate(monkeypatch, decisions, trades_rows=[], seed_rows=[])

    result = await tsv.simulate_tiered_screen(start_date=start_date, account_mode="paper", user_id="u1")

    assert result.tier1_count >= 1


async def test_seeded_stop_loss_still_triggers_near_stop_tier0(monkeypatch):
    # The seed also feeds the near-stop-loss Tier-0 gate (the other
    # _held_state_before call site), not just the Tier-1 "not held" check.
    start_date = date(2026, 8, 1)
    ticker = "005930.KS"

    decisions = [
        _decision(ticker, start_date, "Hold", "Hold", "Hold"),
        # price_now from fake_get_last_close is 100.0; stop_loss=97 is within
        # STOP_LOSS_TIER0_PCT (0.05) of price, so this must force Tier 0.
        _decision(ticker, start_date + timedelta(days=1), "Hold", "Hold", "Hold"),
    ]
    seed_rows = [{"ticker": ticker, "shares_after": 10, "stop_loss": 97.0}]
    _patch_simulate(monkeypatch, decisions, trades_rows=[], seed_rows=seed_rows)

    result = await tsv.simulate_tiered_screen(start_date=start_date, account_mode="paper", user_id="u1")

    assert result.tier0_reason_counts.get("near_stop_loss", 0) >= 1


def test_fetch_position_seed_call_site_present_in_simulate():
    import inspect

    src = inspect.getsource(tsv.simulate_tiered_screen)
    assert "_fetch_position_seed" in src
