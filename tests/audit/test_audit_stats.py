"""app.audit.stats — asyncpg pool/connection faked, no real Postgres (F-stage)."""

from __future__ import annotations

from datetime import date, datetime

import pytest

from app.audit import stats
from app.user_context import get_current_user_id, user_scope

# Every stats query is scoped to the caller (Phase 4), so each one carries
# the current identity as its final bind parameter.
_WHO = get_current_user_id()


class FakeConn:
    def __init__(self, fetch_result=None):
        self.fetch_result = fetch_result if fetch_result is not None else []
        self.queries: list[tuple] = []

    async def fetch(self, query, *args):
        self.queries.append((query, args))
        return self.fetch_result


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


def _row(rating="Buy", n=4, wins=3, mean_alpha=0.02, mean_raw=0.03, unscorable=0, inferred_n=0):
    return {
        "rating": rating, "n": n, "wins": wins, "mean_alpha": mean_alpha, "mean_raw": mean_raw,
        # NULL/NaN rows excluded from every figure above (2026-08-13) — the
        # query reports them separately instead of letting them count as wins.
        "unscorable": unscorable,
        # How many of `n` came from held_at_decision_inferred rather than the
        # observed column (2026-08-20 ledger backfill) — see RatingStats.
        "inferred_n": inferred_n,
    }


async def test_compute_hit_rate_by_rating_parses_rows(monkeypatch):
    conn = FakeConn(fetch_result=[_row()])
    monkeypatch.setattr(stats, "get_pool", lambda: _async_return(FakePool(conn)))
    result = await stats.compute_hit_rate_by_rating()
    assert len(result) == 1
    assert result[0].rating == "Buy" and result[0].n == 4
    assert result[0].win_rate == pytest.approx(0.75)
    assert result[0].mean_alpha == pytest.approx(0.02)


async def test_compute_hit_rate_by_rating_query_dedupes_per_ticker_date(monkeypatch):
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(stats, "get_pool", lambda: _async_return(FakePool(conn)))
    await stats.compute_hit_rate_by_rating()
    query, _ = conn.queries[0]
    assert "DISTINCT ON (ticker, trade_date)" in query
    assert "ORDER BY ticker, trade_date, created_at DESC" in query


async def test_compute_hit_rate_by_rating_filters_by_ticker(monkeypatch):
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(stats, "get_pool", lambda: _async_return(FakePool(conn)))
    await stats.compute_hit_rate_by_rating(ticker="AAPL")
    query, args = conn.queries[0]
    assert "ticker = $1" in query
    assert args == ("AAPL", _WHO)


async def test_compute_hit_rate_by_rating_filters_by_market_domestic(monkeypatch):
    # Domestic/overseas separation audit (2026-07-27): a market filter lets a
    # report see one market's hit rate instead of always pooling both.
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(stats, "get_pool", lambda: _async_return(FakePool(conn)))
    await stats.compute_hit_rate_by_rating(market="domestic")
    query, args = conn.queries[0]
    assert "ticker LIKE '%.KS'" in query
    assert args == (_WHO,)


async def test_compute_hit_rate_by_rating_filters_by_market_overseas(monkeypatch):
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(stats, "get_pool", lambda: _async_return(FakePool(conn)))
    await stats.compute_hit_rate_by_rating(market="overseas")
    query, _ = conn.queries[0]
    assert "ticker NOT LIKE '%.KS'" in query


async def test_compute_hit_rate_by_rating_rejects_invalid_market(monkeypatch):
    with pytest.raises(ValueError):
        await stats.compute_hit_rate_by_rating(market="mars")


async def test_compute_hit_rate_by_rating_filters_by_scope_live(monkeypatch):
    # 2026-08-03: once live decisions started auto-resolving, live and
    # backtest rows need to be separable rather than always pooled.
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(stats, "get_pool", lambda: _async_return(FakePool(conn)))
    await stats.compute_hit_rate_by_rating(scope="live")
    query, _ = conn.queries[0]
    assert "is_backtest = false" in query


async def test_compute_hit_rate_by_rating_filters_by_scope_backtest(monkeypatch):
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(stats, "get_pool", lambda: _async_return(FakePool(conn)))
    await stats.compute_hit_rate_by_rating(scope="backtest")
    query, _ = conn.queries[0]
    assert "is_backtest = true" in query


async def test_compute_hit_rate_by_rating_no_scope_pools_both(monkeypatch):
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(stats, "get_pool", lambda: _async_return(FakePool(conn)))
    await stats.compute_hit_rate_by_rating()
    query, _ = conn.queries[0]
    assert "is_backtest" not in query


async def test_compute_hit_rate_by_rating_rejects_invalid_scope(monkeypatch):
    with pytest.raises(ValueError):
        await stats.compute_hit_rate_by_rating(scope="mars")


async def test_compute_hit_rate_by_rating_filters_by_account_mode_paper(monkeypatch):
    # 2026-08-04: once real-mode trading starts, paper and real decisions
    # need to be separable rather than always pooled — see project memory:
    # user-account-mode-db-separation-design.
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(stats, "get_pool", lambda: _async_return(FakePool(conn)))
    await stats.compute_hit_rate_by_rating(account_mode="paper")
    query, args = conn.queries[0]
    assert "account_mode = $1" in query
    assert args == ("paper", _WHO)


async def test_compute_hit_rate_by_rating_filters_by_account_mode_real(monkeypatch):
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(stats, "get_pool", lambda: _async_return(FakePool(conn)))
    await stats.compute_hit_rate_by_rating(account_mode="real")
    query, _ = conn.queries[0]
    assert "account_mode = $1" in query


async def test_compute_hit_rate_by_rating_no_account_mode_pools_both(monkeypatch):
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(stats, "get_pool", lambda: _async_return(FakePool(conn)))
    await stats.compute_hit_rate_by_rating()
    query, _ = conn.queries[0]
    assert "account_mode" not in query


async def test_compute_hit_rate_by_rating_rejects_invalid_account_mode(monkeypatch):
    with pytest.raises(ValueError):
        await stats.compute_hit_rate_by_rating(account_mode="mars")


async def test_compute_hit_rate_by_rating_combines_ticker_and_account_mode_params(monkeypatch):
    # ticker's param index is dynamic (added first) — account_mode must
    # slot in after it, not collide on $1.
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(stats, "get_pool", lambda: _async_return(FakePool(conn)))
    await stats.compute_hit_rate_by_rating(ticker="AAPL", account_mode="real")
    query, args = conn.queries[0]
    assert "ticker = $1" in query
    assert "account_mode = $2" in query
    assert args == ("AAPL", "real", _WHO)


async def test_compute_hit_rate_by_rating_excludes_degraded_analyst_panel(monkeypatch):
    # analysts_failed (app.memory.db) added specifically so a degraded
    # (1-3-analyst) decision doesn't pool with a full 4-analyst one — NULL
    # (predates the column) stays included, only explicit >0 is excluded.
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(stats, "get_pool", lambda: _async_return(FakePool(conn)))
    await stats.compute_hit_rate_by_rating()
    query, _ = conn.queries[0]
    assert "(analysts_failed IS NULL OR analysts_failed = 0)" in query


async def test_compute_hit_rate_by_rating_empty_returns_empty_list(monkeypatch):
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(stats, "get_pool", lambda: _async_return(FakePool(conn)))
    result = await stats.compute_hit_rate_by_rating()
    assert result == []


async def test_compute_hit_rate_by_rating_no_trigger_pools_both(monkeypatch):
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(stats, "get_pool", lambda: _async_return(FakePool(conn)))
    await stats.compute_hit_rate_by_rating()
    query, args = conn.queries[0]
    assert "decision_trigger" not in query
    assert args == (_WHO,)


async def test_compute_hit_rate_by_rating_filters_by_trigger_auto(monkeypatch):
    # 'auto' must match NULL too — rows predating the decision_trigger
    # column (2026-08-20) can't be retroactively told apart from a genuine
    # auto-triggered row, and excluding them would drop nearly this table's
    # entire pre-tagging history from an "auto-only" view.
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(stats, "get_pool", lambda: _async_return(FakePool(conn)))
    await stats.compute_hit_rate_by_rating(trigger="auto")
    query, args = conn.queries[0]
    assert "decision_trigger IS DISTINCT FROM 'manual'" in query
    # Literal-only fragment, no bind param — doesn't shift the user_id index.
    assert args == (_WHO,)


async def test_compute_hit_rate_by_rating_filters_by_trigger_manual(monkeypatch):
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(stats, "get_pool", lambda: _async_return(FakePool(conn)))
    await stats.compute_hit_rate_by_rating(trigger="manual")
    query, args = conn.queries[0]
    assert "decision_trigger = 'manual'" in query
    assert args == (_WHO,)


async def test_compute_hit_rate_by_rating_rejects_invalid_trigger(monkeypatch):
    with pytest.raises(ValueError):
        await stats.compute_hit_rate_by_rating(trigger="mars")


# ---------------------------------------------------------------------------
# compute_debate_impact
# ---------------------------------------------------------------------------


async def test_compute_debate_impact_counts_changed_and_unchanged(monkeypatch):
    rows = [
        {"trader_action": "Buy", "rating": "Buy"},        # unchanged
        {"trader_action": "Buy", "rating": "Overweight"},  # unchanged (Overweight buckets to Buy)
        {"trader_action": "Buy", "rating": "Sell"},        # changed
        {"trader_action": "Hold", "rating": "Underweight"},  # changed (Underweight buckets to Sell)
    ]
    conn = FakeConn(fetch_result=rows)
    monkeypatch.setattr(stats, "get_pool", lambda: _async_return(FakePool(conn)))
    result = await stats.compute_debate_impact()
    assert result.n == 4
    assert result.changed == 2
    assert result.changed_pct == pytest.approx(0.5)
    assert result.breakdown == {"Buy->Buy": 2, "Buy->Sell": 1, "Hold->Sell": 1}


async def test_compute_debate_impact_query_excludes_null_trader_action(monkeypatch):
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(stats, "get_pool", lambda: _async_return(FakePool(conn)))
    await stats.compute_debate_impact()
    query, _ = conn.queries[0]
    assert "trader_action IS NOT NULL" in query
    assert "DISTINCT ON (ticker, trade_date)" in query


async def test_compute_debate_impact_filters_by_ticker(monkeypatch):
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(stats, "get_pool", lambda: _async_return(FakePool(conn)))
    await stats.compute_debate_impact(ticker="AAPL")
    query, args = conn.queries[0]
    assert "ticker = $1" in query
    assert args == ("AAPL", _WHO)


async def test_compute_debate_impact_filters_by_market_domestic(monkeypatch):
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(stats, "get_pool", lambda: _async_return(FakePool(conn)))
    await stats.compute_debate_impact(market="domestic")
    query, _ = conn.queries[0]
    assert "ticker LIKE '%.KS'" in query


async def test_compute_debate_impact_rejects_invalid_market(monkeypatch):
    with pytest.raises(ValueError):
        await stats.compute_debate_impact(market="mars")


async def test_compute_debate_impact_filters_by_scope_live(monkeypatch):
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(stats, "get_pool", lambda: _async_return(FakePool(conn)))
    await stats.compute_debate_impact(scope="live")
    query, _ = conn.queries[0]
    assert "is_backtest = false" in query


async def test_compute_debate_impact_rejects_invalid_scope(monkeypatch):
    with pytest.raises(ValueError):
        await stats.compute_debate_impact(scope="mars")


async def test_compute_debate_impact_filters_by_account_mode(monkeypatch):
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(stats, "get_pool", lambda: _async_return(FakePool(conn)))
    await stats.compute_debate_impact(account_mode="paper")
    query, args = conn.queries[0]
    assert "account_mode = $1" in query
    assert args == ("paper", _WHO)


async def test_compute_debate_impact_rejects_invalid_account_mode(monkeypatch):
    with pytest.raises(ValueError):
        await stats.compute_debate_impact(account_mode="mars")


async def test_compute_debate_impact_filters_by_trigger_auto(monkeypatch):
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(stats, "get_pool", lambda: _async_return(FakePool(conn)))
    await stats.compute_debate_impact(trigger="auto")
    query, args = conn.queries[0]
    assert "decision_trigger IS DISTINCT FROM 'manual'" in query
    assert args == (_WHO,)


async def test_compute_debate_impact_filters_by_trigger_manual(monkeypatch):
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(stats, "get_pool", lambda: _async_return(FakePool(conn)))
    await stats.compute_debate_impact(trigger="manual")
    query, args = conn.queries[0]
    assert "decision_trigger = 'manual'" in query
    assert args == (_WHO,)


async def test_compute_debate_impact_rejects_invalid_trigger(monkeypatch):
    with pytest.raises(ValueError):
        await stats.compute_debate_impact(trigger="mars")


async def test_compute_debate_impact_empty_reports_zero_n(monkeypatch):
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(stats, "get_pool", lambda: _async_return(FakePool(conn)))
    result = await stats.compute_debate_impact()
    assert result.n == 0
    assert result.changed_pct == 0.0
    assert result.breakdown == {}


def test_render_debate_impact_shows_breakdown_and_pct():
    result = stats.render_debate_impact(
        stats.DebateImpactStats(n=4, changed=2, changed_pct=0.5, breakdown={"Buy->Buy": 2, "Buy->Sell": 1, "Hold->Sell": 1})
    )
    assert "4" in result
    assert "50.0%" in result
    assert "Buy->Sell" in result


def test_render_debate_impact_empty():
    result = stats.render_debate_impact(stats.DebateImpactStats(n=0, changed=0, changed_pct=0.0, breakdown={}))
    assert "No decisions with a recorded trader_action" in result


async def test_get_decision_history_returns_all_attempts_in_order(monkeypatch):
    rows = [
        {
            "created_at": datetime(2026, 7, 23, 10, 0, 0),
            "rating": "Hold",
            "status": "resolved",
            "alpha_return": -0.01,
            "raw_return": 0.01,
            "reflection": "before fix",
            "is_backtest": True,
            "account_mode": "paper",
        },
        {
            "created_at": datetime(2026, 7, 23, 18, 0, 0),
            "rating": "Buy",
            "status": "pending",
            "alpha_return": None,
            "raw_return": None,
            "reflection": None,
            "is_backtest": False,
            "account_mode": "real",
        },
    ]
    conn = FakeConn(fetch_result=rows)
    monkeypatch.setattr(stats, "get_pool", lambda: _async_return(FakePool(conn)))
    result = await stats.get_decision_history("AAPL", date(2026, 4, 1))
    assert len(result) == 2
    assert result[0].rating == "Hold" and result[0].reflection == "before fix"
    assert result[0].is_backtest is True and result[0].account_mode == "paper"
    assert result[1].rating == "Buy" and result[1].alpha_return is None
    assert result[1].is_backtest is False and result[1].account_mode == "real"
    query, args = conn.queries[0]
    assert args == ("AAPL", date(2026, 4, 1), _WHO)


def test_render_labels_market_scope_when_no_ticker_given():
    result = stats.render([stats.RatingStats(rating="Buy", n=1, win_rate=1.0, mean_alpha=0.01, mean_raw_return=0.01)], market="domestic")
    assert "all domestic tickers" in result


def test_render_warns_when_scope_not_given():
    # 2026-08-03: pooling live+backtest is still the default (opt-in split,
    # matching `market`'s own convention) but must be visibly flagged now
    # that live decisions actually resolve.
    result = stats.render([stats.RatingStats(rating="Buy", n=1, win_rate=1.0, mean_alpha=0.01, mean_raw_return=0.01)])
    assert "pooling live and backtest" in result


def test_render_labels_run_scope_when_given():
    result = stats.render(
        [stats.RatingStats(rating="Buy", n=1, win_rate=1.0, mean_alpha=0.01, mean_raw_return=0.01)], run_scope="live"
    )
    assert "live" in result
    assert "pooling live and backtest" not in result


def test_render_warns_when_account_mode_not_given():
    result = stats.render([stats.RatingStats(rating="Buy", n=1, win_rate=1.0, mean_alpha=0.01, mean_raw_return=0.01)])
    assert "pooling paper and real" in result


def test_render_labels_account_mode_when_given():
    result = stats.render(
        [stats.RatingStats(rating="Buy", n=1, win_rate=1.0, mean_alpha=0.01, mean_raw_return=0.01)],
        account_mode="real",
    )
    assert "real" in result
    assert "pooling paper and real" not in result


def test_render_warns_when_trigger_not_given():
    result = stats.render([stats.RatingStats(rating="Buy", n=1, win_rate=1.0, mean_alpha=0.01, mean_raw_return=0.01)])
    assert "pooling auto-triggered and manually-triggered" in result


def test_render_labels_trigger_when_given():
    result = stats.render(
        [stats.RatingStats(rating="Buy", n=1, win_rate=1.0, mean_alpha=0.01, mean_raw_return=0.01)],
        run_trigger="auto",
    )
    assert "auto" in result
    assert "pooling auto-triggered and manually-triggered" not in result


def test_render_history_shows_all_attempts_and_footer():
    attempts = [
        stats.DecisionAttempt(
            created_at=datetime(2026, 7, 23, 10, 0, 0), rating="Hold", status="resolved",
            alpha_return=-0.01, raw_return=0.01, reflection="before fix",
            is_backtest=True, account_mode="paper",
        ),
        stats.DecisionAttempt(
            created_at=datetime(2026, 7, 23, 18, 0, 0), rating="Buy", status="pending",
            alpha_return=None, raw_return=None, reflection=None,
            is_backtest=False, account_mode="real",
        ),
    ]
    result = stats.render_history(attempts, "AAPL", date(2026, 4, 1))
    assert "AAPL" in result and "2026-04-01" in result
    assert "before fix" in result
    assert "2 attempt(s) total" in result
    assert "env=backtest/paper" in result
    assert "env=live/real" in result


def test_render_history_empty():
    result = stats.render_history([], "AAPL", date(2026, 4, 1))
    assert "No attempts on record" in result


# ---------------------------------------------------------------------------
# Per-user scoping (multi-user migration Phase 4, plan §2.2): "Win-rate /
# alpha reporting blends all users into one number."
#
# Unlike the other Tier B gaps this one is silently WRONG rather than unsafe —
# a blended figure looks perfectly plausible, so nothing prompts anyone to
# question it, and it is the number a person uses to decide how much money to
# trust the system with.
# ---------------------------------------------------------------------------


async def test_hit_rate_query_is_scoped_to_the_caller(monkeypatch):
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(stats, "get_pool", lambda: _async_return(FakePool(conn)))
    with user_scope("u-alice"):
        await stats.compute_hit_rate_by_rating()
    query, args = conn.queries[0]
    assert "user_id = $1" in query
    assert args == ("u-alice",)


async def test_debate_impact_query_is_scoped_to_the_caller(monkeypatch):
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(stats, "get_pool", lambda: _async_return(FakePool(conn)))
    with user_scope("u-alice"):
        await stats.compute_debate_impact()
    query, args = conn.queries[0]
    assert "user_id = $1" in query
    assert args == ("u-alice",)


async def test_decision_history_is_scoped_to_the_caller(monkeypatch):
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(stats, "get_pool", lambda: _async_return(FakePool(conn)))
    with user_scope("u-alice"):
        await stats.get_decision_history("AAPL", date(2026, 4, 1))
    query, args = conn.queries[0]
    assert "user_id = $3" in query
    assert args == ("AAPL", date(2026, 4, 1), "u-alice")


async def test_user_scoping_composes_with_the_other_filters(monkeypatch):
    """The user predicate must be ANDed on, not replace anything — a scoped
    query that quietly dropped the account_mode filter would blend paper and
    real results instead of users."""
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(stats, "get_pool", lambda: _async_return(FakePool(conn)))
    with user_scope("u-alice"):
        await stats.compute_hit_rate_by_rating(ticker="AAPL", account_mode="real")
    query, args = conn.queries[0]
    assert "ticker = $1" in query
    assert "account_mode = $2" in query
    assert "user_id = $3" in query
    assert args == ("AAPL", "real", "u-alice")


# ---------------------------------------------------------------------------
# Direction-aware scoring (2026-08-13)
#
# Until this landed, every rating was scored on `alpha_return > 0`, which
# measured the two exposure-REDUCING tiers exactly backwards. Live evidence at
# the time: Sell (n=11) reported 72.7% when the correct figure was 27.3%, and
# Underweight (n=5) reported 0.0% when it was actually 100%.
# ---------------------------------------------------------------------------


def test_win_direction_map_covers_only_determinable_ratings():
    # Stage 1 (2026-08-13): Hold (held)/Hold (flat) added — the held_at_
    # decision column now lets these two be scored like any other bucket.
    # 2026-08-20: 'Buy (gated)'/'Sell (gated)' added — a hysteresis-gate
    # downgrade to Hold is scored on the model's original call, not as Hold.
    assert stats._WIN_DIRECTION == {
        "Buy": 1, "Overweight": 1, "Underweight": -1, "Sell": -1,
        "Hold (held)": 1, "Hold (flat)": -1,
        "Buy (gated)": 1, "Sell (gated)": -1,
    }
    # Bare 'Hold' must still NOT be in the map: without held_at_decision
    # recorded, direction is genuinely unknown.
    assert "Hold" not in stats._WIN_DIRECTION


async def test_query_scores_exposure_reducing_ratings_on_negative_alpha(monkeypatch):
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(stats, "get_pool", lambda: _async_return(FakePool(conn)))
    await stats.compute_hit_rate_by_rating()
    query, _ = conn.queries[0]
    assert "WHEN rating IN ('Buy', 'Overweight', 'Buy (gated)') THEN alpha_return > 0" in query
    assert "WHEN rating IN ('Sell', 'Underweight', 'Sell (gated)') THEN alpha_return < 0" in query


async def test_query_excludes_nan_returns_from_every_figure(monkeypatch):
    # Postgres sorts NaN above every real number, so `alpha_return > 0` was
    # TRUE for a NaN row — it counted as a win. It must be filtered out.
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(stats, "get_pool", lambda: _async_return(FakePool(conn)))
    await stats.compute_hit_rate_by_rating()
    query, _ = conn.queries[0]
    assert "alpha_return <> 'NaN'::float8" in query
    assert "raw_return <> 'NaN'::float8" in query
    assert "COUNT(*) FILTER (WHERE NOT ok) AS unscorable" in query


async def test_sell_win_rate_uses_negative_alpha_direction(monkeypatch):
    # 10 Sells, 3 of which had alpha < 0 -> 30%, not the 70% the old test
    # would have produced from the same underlying rows.
    conn = FakeConn(fetch_result=[_row(rating="Sell", n=10, wins=3, mean_alpha=0.022)])
    monkeypatch.setattr(stats, "get_pool", lambda: _async_return(FakePool(conn)))
    result = await stats.compute_hit_rate_by_rating()
    assert result[0].win_rate == pytest.approx(0.3)
    assert result[0].win_condition == "alpha < 0"


async def test_hold_is_reported_but_unscored(monkeypatch):
    conn = FakeConn(fetch_result=[_row(rating="Hold", n=146, wins=0, mean_alpha=-0.0013)])
    monkeypatch.setattr(stats, "get_pool", lambda: _async_return(FakePool(conn)))
    result = await stats.compute_hit_rate_by_rating()
    # Counted and averaged, but win_rate is None — NOT 0%, which would read as
    # "every Hold was wrong".
    assert result[0].n == 146
    assert result[0].win_rate is None
    assert result[0].mean_alpha == pytest.approx(-0.0013)
    assert result[0].win_condition == stats._UNSCORED_CONDITION


async def test_unscorable_rows_are_surfaced_not_hidden(monkeypatch):
    conn = FakeConn(fetch_result=[_row(rating="Sell", n=10, wins=3, unscorable=1)])
    monkeypatch.setattr(stats, "get_pool", lambda: _async_return(FakePool(conn)))
    result = await stats.compute_hit_rate_by_rating()
    assert result[0].n == 10  # scorable only
    assert result[0].unscorable_n == 1


async def test_all_rows_unscorable_gives_no_figure_rather_than_zero(monkeypatch):
    conn = FakeConn(fetch_result=[_row(rating="Buy", n=0, wins=0, mean_alpha=None, mean_raw=None, unscorable=2)])
    monkeypatch.setattr(stats, "get_pool", lambda: _async_return(FakePool(conn)))
    result = await stats.compute_hit_rate_by_rating()
    assert result[0].win_rate is None
    assert result[0].mean_alpha is None
    assert result[0].unscorable_n == 2


def test_directional_totals_matches_render_headline():
    # Lifted out of render() (2026-08-20) so the CLI and the web API can
    # never disagree — this pins the exact figures render()'s own headline
    # test below also asserts against.
    rating_stats = [
        stats.RatingStats(rating="Buy", n=10, win_rate=0.6, wins=6, mean_alpha=0.01, mean_raw_return=0.02,
                          win_condition="alpha > 0"),
        stats.RatingStats(rating="Hold", n=90, win_rate=None, mean_alpha=0.0, mean_raw_return=0.05,
                          win_condition=stats._UNSCORED_CONDITION, unscorable_n=1),
    ]
    totals = stats.directional_totals(rating_stats)
    assert totals.scored_n == 10
    assert totals.scored_wins == 6
    assert totals.win_rate == pytest.approx(0.6)
    assert totals.unscored_hold_n == 90
    assert totals.unscorable_n == 1


def test_directional_totals_uses_exact_wins_not_rounded_win_rate():
    # Fix (2026-08-27): scored_wins must come from the raw SQL-computed
    # `wins` int, not a round(win_rate * n) reconstruction — Python's
    # round-half-to-even means round(0.65 * 10) == round(6.5) == 6, but the
    # true win count here is 7 (win_rate itself carries rounding error from
    # upstream display/storage). The old round-trip formula would silently
    # under-report this bucket by one win.
    rating_stats = [
        stats.RatingStats(rating="Buy", n=10, win_rate=0.65, wins=7, mean_alpha=0.01, mean_raw_return=0.02,
                          win_condition="alpha > 0"),
    ]
    totals = stats.directional_totals(rating_stats)
    assert totals.scored_wins == 7  # NOT round(0.65 * 10) == 6
    assert round(0.65 * 10) == 6  # pin the old formula's wrong answer for contrast


def test_directional_totals_is_none_when_no_directional_rows():
    totals = stats.directional_totals([
        stats.RatingStats(rating="Hold", n=5, win_rate=None, mean_alpha=None, mean_raw_return=None,
                          win_condition=stats._UNSCORED_CONDITION),
    ])
    assert totals.scored_n == 0
    assert totals.win_rate is None
    assert totals.unscored_hold_n == 5


def test_render_headline_excludes_hold_and_states_the_condition():
    rendered = stats.render([
        stats.RatingStats(rating="Buy", n=10, win_rate=0.6, wins=6, mean_alpha=0.01, mean_raw_return=0.02,
                          win_condition="alpha > 0"),
        stats.RatingStats(rating="Hold", n=90, win_rate=None, mean_alpha=0.0, mean_raw_return=0.05,
                          win_condition=stats._UNSCORED_CONDITION),
    ])
    # Headline covers the 10 directional calls only, not the 100 total.
    assert "Directional decisions (Buy/Overweight/Sell/Underweight): 10" in rendered
    assert "Directional win rate: 60.0%" in rendered
    assert "Unscored (Hold — held vs flat not recorded): 90" in rendered
    assert "DIRECTION-AWARE" in rendered


def test_render_shows_dash_not_zero_for_unscored_rows():
    rendered = stats.render([
        stats.RatingStats(rating="Hold", n=5, win_rate=None, mean_alpha=None, mean_raw_return=None,
                          win_condition=stats._UNSCORED_CONDITION),
    ])
    assert "0.0%" not in rendered.split("'Win rate'")[0]
    assert "—" in rendered


def test_render_warns_when_rows_were_excluded_as_nan():
    rendered = stats.render([
        stats.RatingStats(rating="Sell", n=10, win_rate=0.3, mean_alpha=0.02, mean_raw_return=0.03,
                          win_condition="alpha < 0", unscorable_n=2),
    ])
    assert "2 resolved row(s) excluded entirely" in rendered


# ---------------------------------------------------------------------------
# Hold held/flat bucketing (Stage 1, F5, 2026-08-13)
#
# held_at_decision now lets a Hold be scored in the correct direction instead
# of being excluded entirely — 'Hold (held)' wins like Buy (alpha > 0),
# 'Hold (flat)' wins like Sell (alpha < 0), and only rows predating the
# column (or where the position lookup failed) remain unscored.
# ---------------------------------------------------------------------------


def test_win_direction_scores_hold_held_and_hold_flat_oppositely():
    assert stats._WIN_DIRECTION["Hold (held)"] == 1
    assert stats._WIN_DIRECTION["Hold (flat)"] == -1


async def test_query_buckets_hold_by_held_at_decision(monkeypatch):
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(stats, "get_pool", lambda: _async_return(FakePool(conn)))
    await stats.compute_hit_rate_by_rating()
    query, _ = conn.queries[0]
    assert "held_at_decision" in query
    assert "'Hold (held)'" in query
    assert "'Hold (flat)'" in query
    assert "'Hold (held/flat unknown)'" in query
    # 2026-08-20: Hold (held) tests raw_return, not alpha_return — see
    # _RAW_RETURN_RATINGS' own comment (long-only system, real counterfactual
    # to keeping a position is cash, not the benchmark).
    assert "WHEN rating = 'Hold (held)' THEN raw_return > 0" in query
    assert "WHEN rating = 'Hold (flat)' THEN alpha_return < 0" in query
    # Stage 0's direction-aware Buy/Sell branches must survive unmodified
    # (now also covering the gated buckets, 2026-08-20).
    assert "WHEN rating IN ('Buy', 'Overweight', 'Buy (gated)') THEN alpha_return > 0" in query
    assert "WHEN rating IN ('Sell', 'Underweight', 'Sell (gated)') THEN alpha_return < 0" in query


async def test_hold_held_scores_on_raw_return_not_alpha(monkeypatch):
    conn = FakeConn(fetch_result=[_row(rating="Hold (held)", n=10, wins=7, mean_alpha=0.01)])
    monkeypatch.setattr(stats, "get_pool", lambda: _async_return(FakePool(conn)))
    result = await stats.compute_hit_rate_by_rating()
    assert result[0].win_rate == pytest.approx(0.7)
    assert result[0].win_condition == "raw > 0"


async def test_hold_flat_scores_like_a_sell(monkeypatch):
    conn = FakeConn(fetch_result=[_row(rating="Hold (flat)", n=10, wins=4, mean_alpha=-0.005)])
    monkeypatch.setattr(stats, "get_pool", lambda: _async_return(FakePool(conn)))
    result = await stats.compute_hit_rate_by_rating()
    assert result[0].win_rate == pytest.approx(0.4)
    assert result[0].win_condition == "alpha < 0"


async def test_hold_held_flat_unknown_stays_unscored(monkeypatch):
    conn = FakeConn(fetch_result=[_row(rating="Hold (held/flat unknown)", n=50, wins=0, mean_alpha=0.0)])
    monkeypatch.setattr(stats, "get_pool", lambda: _async_return(FakePool(conn)))
    result = await stats.compute_hit_rate_by_rating()
    assert result[0].win_rate is None
    assert result[0].win_condition == stats._UNSCORED_CONDITION


# ---------------------------------------------------------------------------
# held_at_decision_inferred (2026-08-20 ledger backfill) — COALESCE fallback
# and inferred_n transparency
# ---------------------------------------------------------------------------


async def test_query_falls_back_to_held_at_decision_inferred(monkeypatch):
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(stats, "get_pool", lambda: _async_return(FakePool(conn)))
    await stats.compute_hit_rate_by_rating()
    query, _ = conn.queries[0]
    assert "COALESCE(held_at_decision, held_at_decision_inferred)" in query


async def test_hold_bucket_reports_inferred_n(monkeypatch):
    conn = FakeConn(fetch_result=[_row(rating="Hold (held)", n=53, wins=14, mean_alpha=-0.017, inferred_n=51)])
    monkeypatch.setattr(stats, "get_pool", lambda: _async_return(FakePool(conn)))
    result = await stats.compute_hit_rate_by_rating()
    assert result[0].n == 53
    assert result[0].inferred_n == 51
    assert result[0].win_rate == pytest.approx(14 / 53)


# ---------------------------------------------------------------------------
# 2026-08-20 — hysteresis-gate-downgraded rows scored on the ORIGINAL call,
# not as a Hold (real incident: id=639, 316140.KS said "전량 매도" in its own
# prose, was gated to Hold for unset confidence, then scored as a losing
# Hold when the Sell call was in fact correct)
# ---------------------------------------------------------------------------


async def test_query_recovers_pre_gate_rating_from_decision_text(monkeypatch):
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(stats, "get_pool", lambda: _async_return(FakePool(conn)))
    await stats.compute_hit_rate_by_rating()
    query, _ = conn.queries[0]
    assert r"\[Hysteresis gate: downgraded from (\w+) to Hold" in query
    assert "gate_orig_rating" in query
    assert "gate_orig_rating || ' (gated)'" in query
    # Checks BOTH the short-form prefix (decision_text) and the long-form
    # note (full_transcript->>'investment_thesis') — some real rows only
    # carry the long-form one (predate the short-form prefix fix).
    assert "full_transcript->>'investment_thesis'" in query


async def test_gated_buy_scores_like_ungated_buy(monkeypatch):
    conn = FakeConn(fetch_result=[_row(rating="Buy (gated)", n=5, wins=3, mean_alpha=0.02)])
    monkeypatch.setattr(stats, "get_pool", lambda: _async_return(FakePool(conn)))
    result = await stats.compute_hit_rate_by_rating()
    assert result[0].win_rate == pytest.approx(0.6)
    assert result[0].win_condition == "alpha > 0"


async def test_gated_sell_scores_like_ungated_sell(monkeypatch):
    conn = FakeConn(fetch_result=[_row(rating="Sell (gated)", n=3, wins=2, mean_alpha=-0.03)])
    monkeypatch.setattr(stats, "get_pool", lambda: _async_return(FakePool(conn)))
    result = await stats.compute_hit_rate_by_rating()
    assert result[0].win_rate == pytest.approx(2 / 3)
    assert result[0].win_condition == "alpha < 0"


def test_gated_buckets_excluded_from_directional_headline():
    # Directional headline text names exactly four ratings — a '(gated)' row
    # is real signal but under a different label, so it stays visible in the
    # per-rating table without silently inflating a headline that says
    # "Buy/Overweight/Sell/Underweight".
    assert "Buy (gated)" not in stats._DIRECTIONAL_RATINGS
    assert "Sell (gated)" not in stats._DIRECTIONAL_RATINGS


async def test_non_hold_rating_reports_zero_inferred_n_by_default(monkeypatch):
    conn = FakeConn(fetch_result=[_row(rating="Buy", n=10, wins=6)])
    monkeypatch.setattr(stats, "get_pool", lambda: _async_return(FakePool(conn)))
    result = await stats.compute_hit_rate_by_rating()
    assert result[0].inferred_n == 0


def test_render_shows_inferred_note_only_when_nonzero():
    inferred = stats.RatingStats(
        rating="Hold (held)", n=53, win_rate=0.264, mean_alpha=-0.017, mean_raw_return=0.03,
        win_condition="alpha > 0", inferred_n=51,
    )
    observed = stats.RatingStats(
        rating="Buy", n=10, win_rate=0.6, mean_alpha=0.01, mean_raw_return=0.02,
        win_condition="alpha > 0", inferred_n=0,
    )
    rendered = stats.render([inferred, observed])
    assert "(51 inferred)" in rendered
    lines = rendered.splitlines()
    buy_line = next(line for line in lines if line.startswith("Buy"))
    assert "inferred" not in buy_line


# ---------------------------------------------------------------------------
# `basis` parameter (Stage 1, F6, 2026-08-13) — fixed 10-trading-day horizon
# columns as an alternative to the existing variable-window ones.
# ---------------------------------------------------------------------------


async def test_basis_variable_is_the_default_and_uses_existing_columns(monkeypatch):
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(stats, "get_pool", lambda: _async_return(FakePool(conn)))
    await stats.compute_hit_rate_by_rating()
    query, _ = conn.queries[0]
    assert "raw_return AS raw_return,\n                    held_at_decision, held_at_decision_inferred" in query
    assert "horizon_alpha_return" not in query


async def test_basis_horizon_selects_the_fixed_window_columns(monkeypatch):
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(stats, "get_pool", lambda: _async_return(FakePool(conn)))
    await stats.compute_hit_rate_by_rating(basis="horizon")
    query, _ = conn.queries[0]
    assert "horizon_alpha_return AS alpha_return" in query
    assert "horizon_raw_return AS raw_return" in query


async def test_basis_rejects_invalid_value(monkeypatch):
    with pytest.raises(ValueError):
        await stats.compute_hit_rate_by_rating(basis="mars")


# ---------------------------------------------------------------------------
# compute_confidence_calibration (Stage 1, F5, 2026-08-13)
# ---------------------------------------------------------------------------


def _confidence_bucket_row(bucket="[0.90, 1.00]", n=8, wins=6, mean_signed_alpha=0.015, unscorable=0):
    return {
        "bucket": bucket, "n": n, "unscorable": unscorable, "wins": wins,
        "mean_signed_alpha": mean_signed_alpha,
    }


async def test_compute_confidence_calibration_maps_bucket_rows(monkeypatch):
    conn = FakeConn()

    class SequencedConn(FakeConn):
        def __init__(self):
            super().__init__()
            self._n = 0

        async def fetch(self, query, *args):
            self._n += 1
            self.queries.append((query, args))
            if self._n == 1:
                return [_confidence_bucket_row()]
            return [{"total": 10, "with_confidence": 8, "alpha_corr": 0.42}]

    conn = SequencedConn()
    monkeypatch.setattr(stats, "get_pool", lambda: _async_return(FakePool(conn)))
    result = await stats.compute_confidence_calibration()
    assert result.n_with_confidence == 8
    assert result.n_without_confidence == 2
    assert result.signed_alpha_corr == pytest.approx(0.42)
    top_bucket = next(b for b in result.buckets if b.label == "[0.90, 1.00]")
    assert top_bucket.n == 8
    assert top_bucket.win_rate == pytest.approx(0.75)
    assert top_bucket.size_multiplier == pytest.approx(1.0)


async def test_compute_confidence_calibration_bucket_boundaries_match_sizing():
    # Must mirror app.execution.sizing._CONFIDENCE_MODIFIER_BANDS exactly —
    # a mismatch here would describe a different sizing scheme than the one
    # actually executing trades.
    labels_and_multipliers = [(b[0], b[2]) for b in stats._CONFIDENCE_BUCKETS]
    assert labels_and_multipliers == [
        ("[0.90, 1.00]", 1.0),
        ("[0.75, 0.90)", 0.9),
        ("[0.60, 0.75)", 0.7),
        ("[0.50, 0.60)", 0.5),
        ("<0.50 (skipped)", 0.0),
    ]


async def test_compute_confidence_calibration_excludes_null_confidence_separately(monkeypatch):
    class SequencedConn(FakeConn):
        def __init__(self):
            super().__init__()
            self._n = 0

        async def fetch(self, query, *args):
            self._n += 1
            self.queries.append((query, args))
            if self._n == 1:
                return []
            return [{"total": 5, "with_confidence": 0, "alpha_corr": None}]

    conn = SequencedConn()
    monkeypatch.setattr(stats, "get_pool", lambda: _async_return(FakePool(conn)))
    result = await stats.compute_confidence_calibration()
    assert result.n_with_confidence == 0
    assert result.n_without_confidence == 5
    assert result.signed_alpha_corr is None


async def test_compute_confidence_calibration_query_flips_sell_alpha_sign(monkeypatch):
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(stats, "get_pool", lambda: _async_return(FakePool(conn)))
    await stats.compute_confidence_calibration()
    query, _ = conn.queries[0]
    assert "WHEN rating IN ('Sell', 'Underweight') THEN -alpha_return" in query
    assert (
        "WHEN rating = 'Hold' AND COALESCE(held_at_decision, held_at_decision_inferred) IS FALSE"
        in query
    )
    assert "THEN -alpha_return" in query
    # 2026-08-27 fix: Hold(held) now scores raw_return (not alpha_return),
    # mirroring compute_hit_rate_by_rating's Hold(held) fix — see that
    # function's own 2026-08-20 comment on why raw_return is the correct
    # test for a long-only system's held position.
    assert (
        "WHEN rating = 'Hold' AND COALESCE(held_at_decision, held_at_decision_inferred) IS TRUE"
        in query
    )
    assert "THEN raw_return" in query
    # Hysteresis-gate recovery, mirrored from compute_hit_rate_by_rating.
    assert "gate_orig_rating" in query


async def test_compute_confidence_calibration_query_holds_on_inferred_column(monkeypatch):
    # A row with held_at_decision IS NULL but held_at_decision_inferred IS
    # TRUE must be picked up via COALESCE, not dropped to ELSE NULL.
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(stats, "get_pool", lambda: _async_return(FakePool(conn)))
    await stats.compute_confidence_calibration()
    query, _ = conn.queries[0]
    assert "held_at_decision_inferred" in query


async def test_compute_confidence_calibration_excludes_degraded_analyst_panel(monkeypatch):
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(stats, "get_pool", lambda: _async_return(FakePool(conn)))
    await stats.compute_confidence_calibration()
    query, _ = conn.queries[0]
    assert "(analysts_failed IS NULL OR analysts_failed = 0)" in query


async def test_compute_confidence_calibration_rejects_invalid_basis():
    with pytest.raises(ValueError):
        await stats.compute_confidence_calibration(basis="mars")


async def test_compute_confidence_calibration_filters_by_trigger_auto(monkeypatch):
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(stats, "get_pool", lambda: _async_return(FakePool(conn)))
    await stats.compute_confidence_calibration(trigger="auto")
    query, _ = conn.queries[0]
    assert "decision_trigger IS DISTINCT FROM 'manual'" in query


async def test_compute_confidence_calibration_filters_by_trigger_manual(monkeypatch):
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(stats, "get_pool", lambda: _async_return(FakePool(conn)))
    await stats.compute_confidence_calibration(trigger="manual")
    query, _ = conn.queries[0]
    assert "decision_trigger = 'manual'" in query


async def test_compute_confidence_calibration_rejects_invalid_trigger():
    with pytest.raises(ValueError):
        await stats.compute_confidence_calibration(trigger="mars")


def test_render_confidence_calibration_empty_when_no_confidence_recorded():
    result = stats.render_confidence_calibration(
        stats.ConfidenceCalibration(buckets=[], n_with_confidence=0, n_without_confidence=12, signed_alpha_corr=None)
    )
    assert "No decisions with a recorded confidence yet" in result
    assert "12" in result


# ---------------------------------------------------------------------------
# since/until date filters (Stage 2/weekly report, 2026-08-13)
# ---------------------------------------------------------------------------


async def test_hit_rate_since_filters_on_horizon_eval_date(monkeypatch):
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(stats, "get_pool", lambda: _async_return(FakePool(conn)))
    await stats.compute_hit_rate_by_rating(since=date(2026, 8, 1))
    query, args = conn.queries[0]
    assert "horizon_eval_date >= $1" in query
    assert args == (date(2026, 8, 1), _WHO)


async def test_hit_rate_until_filters_on_horizon_eval_date(monkeypatch):
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(stats, "get_pool", lambda: _async_return(FakePool(conn)))
    await stats.compute_hit_rate_by_rating(until=date(2026, 8, 7))
    query, args = conn.queries[0]
    assert "horizon_eval_date <= $1" in query
    assert args == (date(2026, 8, 7), _WHO)


async def test_hit_rate_since_and_until_compose(monkeypatch):
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(stats, "get_pool", lambda: _async_return(FakePool(conn)))
    await stats.compute_hit_rate_by_rating(since=date(2026, 8, 1), until=date(2026, 8, 7))
    query, args = conn.queries[0]
    assert "horizon_eval_date >= $1" in query
    assert "horizon_eval_date <= $2" in query
    assert args == (date(2026, 8, 1), date(2026, 8, 7), _WHO)


async def test_hit_rate_no_since_until_applies_no_date_filter(monkeypatch):
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(stats, "get_pool", lambda: _async_return(FakePool(conn)))
    await stats.compute_hit_rate_by_rating()
    query, _ = conn.queries[0]
    assert "horizon_eval_date" not in query


async def test_hit_rate_since_composes_with_other_filters(monkeypatch):
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(stats, "get_pool", lambda: _async_return(FakePool(conn)))
    await stats.compute_hit_rate_by_rating(ticker="AAPL", account_mode="real", since=date(2026, 8, 1))
    query, args = conn.queries[0]
    assert "ticker = $1" in query
    assert "account_mode = $2" in query
    assert "horizon_eval_date >= $3" in query
    assert args == ("AAPL", "real", date(2026, 8, 1), _WHO)


async def test_confidence_calibration_since_until_filter_on_horizon_eval_date(monkeypatch):
    class SequencedConn(FakeConn):
        def __init__(self):
            super().__init__()
            self._n = 0

        async def fetch(self, query, *args):
            self._n += 1
            self.queries.append((query, args))
            if self._n == 1:
                return []
            return [{"total": 0, "with_confidence": 0, "alpha_corr": None}]

    conn = SequencedConn()
    monkeypatch.setattr(stats, "get_pool", lambda: _async_return(FakePool(conn)))
    await stats.compute_confidence_calibration(since=date(2026, 8, 1), until=date(2026, 8, 7))
    query, args = conn.queries[0]
    assert "horizon_eval_date >= $1" in query
    assert "horizon_eval_date <= $2" in query
    assert args == (date(2026, 8, 1), date(2026, 8, 7), _WHO)


async def test_debate_impact_since_until_fall_back_to_created_at_kst(monkeypatch):
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(stats, "get_pool", lambda: _async_return(FakePool(conn)))
    await stats.compute_debate_impact(since=date(2026, 8, 1), until=date(2026, 8, 7))
    query, args = conn.queries[0]
    assert "(created_at AT TIME ZONE 'Asia/Seoul')::date >= $1" in query
    assert "(created_at AT TIME ZONE 'Asia/Seoul')::date <= $2" in query
    assert "horizon_eval_date" not in query
    assert args == (date(2026, 8, 1), date(2026, 8, 7), _WHO)


async def test_debate_impact_no_since_until_applies_no_date_filter(monkeypatch):
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(stats, "get_pool", lambda: _async_return(FakePool(conn)))
    await stats.compute_debate_impact()
    query, _ = conn.queries[0]
    assert "created_at) ::date" not in query
    assert "AT TIME ZONE" not in query


def test_render_confidence_calibration_shows_buckets_and_correlation():
    calibration = stats.ConfidenceCalibration(
        buckets=[
            stats.ConfidenceBucketStats(
                label="[0.90, 1.00]", lower=0.90, size_multiplier=1.0, n=8,
                win_rate=0.75, mean_signed_alpha=0.015, unscorable_n=0,
            ),
        ],
        n_with_confidence=8, n_without_confidence=3, signed_alpha_corr=0.42,
    )
    result = stats.render_confidence_calibration(calibration)
    assert "[0.90, 1.00]" in result
    assert "75.0%" in result
    assert "+0.420" in result
    assert "3" in result
