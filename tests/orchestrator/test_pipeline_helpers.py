"""app.orchestrator.pipeline's context-builder helpers — each I/O dependency
mocked, no DB/network (D-stage). run_pipeline's full-integration wiring is
covered separately in test_pipeline_integration.py."""

from __future__ import annotations

from datetime import date

import pytest

from app.config import settings
from app.execution.ledger import (
    FORCED_EXIT_THRESHOLD_PRICE_NOTE_PREFIX,
    REVERSAL_REASON_PREFIX,
    STOP_LOSS_REASON_PREFIX,
    EquityBasisUnavailable,
    LastTrade,
    Position,
)
from app.execution.portfolio_risk import PortfolioSnapshot
from app.orchestrator import pipeline
from app.state import (
    FlatPortfolioDecision,
    FlatPortfolioRating,
    FlatResearchPlan,
    FlatResearchRecommendation,
    PortfolioDecision,
    PortfolioRating,
    ResearchPlan,
    TraderAction,
    TraderProposal,
    is_state_flipping_rating,
)
from conftest import broker_stub


# ---------------------------------------------------------------------------
# _position_context
# ---------------------------------------------------------------------------


async def test_position_context_holding_shares_with_trade_history(monkeypatch):
    async def fake_get_position(ticker):
        return Position(cash=0.0, shares=10)

    async def fake_get_last_trade(ticker):
        return LastTrade(date(2026, 7, 1), "buy", 150.0, "**Rating**: Buy\n\n...")

    async def fake_real_shares_held(ticker):
        return None  # simulate broker not connected — falls back to the ledger

    monkeypatch.setattr(pipeline, "get_position", fake_get_position)
    monkeypatch.setattr(pipeline, "get_last_trade", fake_get_last_trade)
    monkeypatch.setattr(pipeline, "real_shares_held", fake_real_shares_held)
    held, result = await pipeline._position_context("AAPL", None)
    assert held is True
    assert "Currently holding 10 shares" in result
    assert "entered via a buy at 150.00 USD on 2026-07-01" in result


async def test_position_context_holding_shares_without_trade_history(monkeypatch):
    async def fake_get_position(ticker):
        return Position(cash=0.0, shares=5)

    async def fake_get_last_trade(ticker):
        return None

    async def fake_real_shares_held(ticker):
        return None  # simulate broker not connected — falls back to the ledger

    monkeypatch.setattr(pipeline, "get_position", fake_get_position)
    monkeypatch.setattr(pipeline, "get_last_trade", fake_get_last_trade)
    monkeypatch.setattr(pipeline, "real_shares_held", fake_real_shares_held)
    held, result = await pipeline._position_context("AAPL", None)
    assert held is True
    assert result == "Currently holding 5 shares (internal paper ledger — real KIS balance unavailable)."


async def test_position_context_flat_with_prior_trade(monkeypatch):
    async def fake_get_position(ticker):
        return Position(cash=1000.0, shares=0)

    async def fake_get_last_trade(ticker):
        return LastTrade(date(2026, 6, 15), "sell", 200.0, "**Rating**: Sell\n\n...")

    async def fake_real_shares_held(ticker):
        return None  # simulate broker not connected — falls back to the ledger

    monkeypatch.setattr(pipeline, "get_position", fake_get_position)
    monkeypatch.setattr(pipeline, "get_last_trade", fake_get_last_trade)
    monkeypatch.setattr(pipeline, "real_shares_held", fake_real_shares_held)
    held, result = await pipeline._position_context("AAPL", None)
    assert held is False
    assert result == (
        "Currently flat (no position) (internal paper ledger — real KIS balance unavailable) — "
        "last sell at 200.00 USD on 2026-06-15."
    )


async def test_position_context_flat_after_stop_loss_price_above_exit(monkeypatch):
    """Real incident this fixes: 005930 stopped out at 198,714 (2026-07-29),
    re-bought 21h later at 214,500 (+7.94%) with the Portfolio Manager
    having no idea the prior exit was a forced safety stop, or that it was
    paying a meaningfully higher price to get back in (memory:
    ai-reentry-context-2026-08-14)."""

    async def fake_get_position(ticker):
        return Position(cash=1000.0, shares=0)

    async def fake_get_last_trade(ticker):
        return LastTrade(
            date(2026, 7, 29), "sell", 198714.0,
            f"{STOP_LOSS_REASON_PREFIX}stop-loss monitor — breach 2026-07-29: low 195000.00 <= stop 198714.00",
        )

    async def fake_real_shares_held(ticker):
        return None

    async def fake_get_last_close(ticker, as_of=None):
        return 214500.0

    monkeypatch.setattr(pipeline, "get_position", fake_get_position)
    monkeypatch.setattr(pipeline, "get_last_trade", fake_get_last_trade)
    monkeypatch.setattr(pipeline, "real_shares_held", fake_real_shares_held)
    monkeypatch.setattr(pipeline, "get_last_close", fake_get_last_close)
    held, result = await pipeline._position_context("005930.KS", None)

    assert held is False
    assert "FORCE-EXITED by the automated stop-loss" in result
    assert "not sold by your own decision" in result
    assert "198714.00 KRW" in result
    assert "214500.00 KRW" in result
    assert "+7.94%" in result
    assert "your judgment call" in result


async def test_position_context_flat_after_stop_loss_price_below_exit(monkeypatch):
    async def fake_get_position(ticker):
        return Position(cash=1000.0, shares=0)

    async def fake_get_last_trade(ticker):
        return LastTrade(date(2026, 7, 29), "sell", 100.0, f"{STOP_LOSS_REASON_PREFIX}stop-loss monitor")

    async def fake_real_shares_held(ticker):
        return None

    async def fake_get_last_close(ticker, as_of=None):
        return 95.0

    monkeypatch.setattr(pipeline, "get_position", fake_get_position)
    monkeypatch.setattr(pipeline, "get_last_trade", fake_get_last_trade)
    monkeypatch.setattr(pipeline, "real_shares_held", fake_real_shares_held)
    monkeypatch.setattr(pipeline, "get_last_close", fake_get_last_close)
    held, result = await pipeline._position_context("AAPL", None)

    assert "-5.00%" in result


async def test_position_context_flat_after_stop_loss_price_fetch_fails(monkeypatch):
    """Narrative-only context — a price-fetch failure must not raise or
    block the decision, just omit the delta clause."""

    async def fake_get_position(ticker):
        return Position(cash=1000.0, shares=0)

    async def fake_get_last_trade(ticker):
        return LastTrade(date(2026, 7, 29), "sell", 100.0, f"{STOP_LOSS_REASON_PREFIX}stop-loss monitor")

    async def fake_real_shares_held(ticker):
        return None

    async def failing_get_last_close(ticker, as_of=None):
        raise RuntimeError("KIS unavailable")

    monkeypatch.setattr(pipeline, "get_position", fake_get_position)
    monkeypatch.setattr(pipeline, "get_last_trade", fake_get_last_trade)
    monkeypatch.setattr(pipeline, "real_shares_held", fake_real_shares_held)
    monkeypatch.setattr(pipeline, "get_last_close", failing_get_last_close)
    held, result = await pipeline._position_context("AAPL", None)

    assert "FORCE-EXITED by the automated stop-loss" in result
    assert "versus that stop-out price" not in result


async def test_position_context_forced_exit_with_threshold_price_says_the_price_is_unknown(monkeypatch):
    """047040.KS id=280 (2026-08-27): the daily-bar backstop recorded the
    breached stop THRESHOLD (19,729.24) as trades.price, not the real
    ~21,000 fill. Narrating that threshold as an exit price — or computing a
    % move against it — would fabricate the same 6.4% error this
    adjustment_note marker exists to prevent from reaching the Portfolio
    Manager/Trader/risk-debate nodes."""

    async def fake_get_position(ticker):
        return Position(cash=1000.0, shares=0)

    async def fake_get_last_trade(ticker):
        return LastTrade(
            date(2026, 8, 27), "sell", 19729.24,
            f"{STOP_LOSS_REASON_PREFIX}stop-loss monitor — breach 2026-08-27: low 18760.00 <= floor 19729.24",
            adjustment_note=(
                f"{FORCED_EXIT_THRESHOLD_PRICE_NOTE_PREFIX} (recorded before the 2026-08-27 fix) — "
                "recorded at the stop threshold 19,729.24, which is a threshold and NOT a traded price; "
                "exclude this row from slippage and return statistics"
            ),
        )

    async def fake_real_shares_held(ticker):
        return None

    async def fake_get_last_close(ticker, as_of=None):
        return 21000.0

    monkeypatch.setattr(pipeline, "get_position", fake_get_position)
    monkeypatch.setattr(pipeline, "get_last_trade", fake_get_last_trade)
    monkeypatch.setattr(pipeline, "real_shares_held", fake_real_shares_held)
    monkeypatch.setattr(pipeline, "get_last_close", fake_get_last_close)
    held, result = await pipeline._position_context("047040.KS", None)

    assert held is False
    assert "NOT a traded price" in result
    assert "UNKNOWN" in result
    assert "versus that stop-out price" not in result
    assert "%" not in result


async def test_position_context_threshold_row_never_calls_get_last_close(monkeypatch):
    """Proves the delta computation is skipped entirely for a threshold row,
    not just omitted from the narrative after a silent failure — the old
    code path (see the "price fetch fails" test above) only reaches its
    except-and-omit behavior by actually calling get_last_close first."""

    async def fake_get_position(ticker):
        return Position(cash=1000.0, shares=0)

    async def fake_get_last_trade(ticker):
        return LastTrade(
            date(2026, 8, 27), "sell", 19729.24,
            f"{STOP_LOSS_REASON_PREFIX}stop-loss monitor",
            adjustment_note=FORCED_EXIT_THRESHOLD_PRICE_NOTE_PREFIX,
        )

    async def fake_real_shares_held(ticker):
        return None

    async def must_not_be_called(ticker, as_of=None):
        raise AssertionError("get_last_close must not be called for a threshold-placeholder exit")

    monkeypatch.setattr(pipeline, "get_position", fake_get_position)
    monkeypatch.setattr(pipeline, "get_last_trade", fake_get_last_trade)
    monkeypatch.setattr(pipeline, "real_shares_held", fake_real_shares_held)
    monkeypatch.setattr(pipeline, "get_last_close", must_not_be_called)
    held, result = await pipeline._position_context("047040.KS", None)  # must not raise

    assert "NOT a traded price" in result


async def test_position_context_unmarked_forced_exit_keeps_the_old_narrative(monkeypatch):
    """No-regression lock: the pre-existing 005930 case (adjustment_note=None,
    i.e. a tick-monitor exit whose price IS real) must keep producing the
    original narrative, delta and all."""

    async def fake_get_position(ticker):
        return Position(cash=1000.0, shares=0)

    async def fake_get_last_trade(ticker):
        return LastTrade(
            date(2026, 7, 29), "sell", 198714.0,
            f"{STOP_LOSS_REASON_PREFIX}stop-loss monitor — breach 2026-07-29: low 195000.00 <= stop 198714.00",
        )

    async def fake_real_shares_held(ticker):
        return None

    async def fake_get_last_close(ticker, as_of=None):
        return 214500.0

    monkeypatch.setattr(pipeline, "get_position", fake_get_position)
    monkeypatch.setattr(pipeline, "get_last_trade", fake_get_last_trade)
    monkeypatch.setattr(pipeline, "real_shares_held", fake_real_shares_held)
    monkeypatch.setattr(pipeline, "get_last_close", fake_get_last_close)
    held, result = await pipeline._position_context("005930.KS", None)

    assert "+7.94%" in result
    assert "NOT a traded price" not in result


async def test_position_context_holding_omits_basis_when_last_trade_price_is_a_threshold(monkeypatch):
    """The shares>0 branch's "entered via a {side} at {price}" clause must
    also be withheld when the most recent row is a threshold placeholder —
    reachable on a mirror-succeeded/stale-balance edge where the ledger
    still points at a forced-exit row."""

    async def fake_get_position(ticker):
        return Position(cash=0.0, shares=5)

    async def fake_get_last_trade(ticker):
        return LastTrade(
            date(2026, 8, 27), "sell", 19729.24,
            f"{STOP_LOSS_REASON_PREFIX}stop-loss monitor",
            adjustment_note=FORCED_EXIT_THRESHOLD_PRICE_NOTE_PREFIX,
        )

    async def fake_real_shares_held(ticker):
        return 5

    monkeypatch.setattr(pipeline, "get_position", fake_get_position)
    monkeypatch.setattr(pipeline, "get_last_trade", fake_get_last_trade)
    monkeypatch.setattr(pipeline, "real_shares_held", fake_real_shares_held)
    held, result = await pipeline._position_context("047040.KS", None)

    assert held is True
    assert "entered via a" not in result
    assert "Currently holding 5 shares" in result


async def test_position_context_flat_after_reversed_trade(monkeypatch):
    """A reversed/never-executed row is not a real market exit — must not
    be narrated as one, and must not reach further back for the
    pre-reversal row (that could surface the very trade that was reversed
    as if it stood)."""

    async def fake_get_position(ticker):
        return Position(cash=1000.0, shares=0)

    async def fake_get_last_trade(ticker):
        return LastTrade(date(2026, 8, 11), "buy", 211000.0, f"{REVERSAL_REASON_PREFIX}BLOCKED: circuit-breaker state unknown")

    async def fake_real_shares_held(ticker):
        return None

    monkeypatch.setattr(pipeline, "get_position", fake_get_position)
    monkeypatch.setattr(pipeline, "get_last_trade", fake_get_last_trade)
    monkeypatch.setattr(pipeline, "real_shares_held", fake_real_shares_held)
    held, result = await pipeline._position_context("035420.KS", None)

    assert held is False
    assert "correction/reversal" in result
    assert "cannot be stated reliably" in result
    assert "FORCE-EXITED" not in result


async def test_position_context_never_traded(monkeypatch):
    async def fake_get_position(ticker):
        return Position(cash=1000.0, shares=0)

    async def fake_get_last_trade(ticker):
        return None

    async def fake_real_shares_held(ticker):
        return None  # simulate broker not connected — falls back to the ledger

    monkeypatch.setattr(pipeline, "get_position", fake_get_position)
    monkeypatch.setattr(pipeline, "get_last_trade", fake_get_last_trade)
    monkeypatch.setattr(pipeline, "real_shares_held", fake_real_shares_held)
    held, result = await pipeline._position_context("AAPL", None)
    assert held is False
    assert result == "No prior position history for this ticker — this would be a new entry."


async def test_position_context_prefers_real_kis_balance_when_available(monkeypatch):
    """Real incident: the internal ledger said 255 shares (accumulated
    paper/backtest trades) while the real KIS account held 1 — the
    Portfolio Manager's whole narrative reasoned off the wrong number.
    Live calls (as_of=None) must prefer the real balance."""

    async def fake_get_position(ticker):
        return Position(cash=0.0, shares=255)

    async def fake_get_last_trade(ticker):
        return None

    async def fake_real_shares_held(ticker):
        return 1

    monkeypatch.setattr(pipeline, "get_position", fake_get_position)
    monkeypatch.setattr(pipeline, "get_last_trade", fake_get_last_trade)
    monkeypatch.setattr(pipeline, "real_shares_held", fake_real_shares_held)
    held, result = await pipeline._position_context("035420.KS", None)
    assert held is True
    assert result == "Currently holding 1 shares."


async def test_position_context_backtest_never_calls_real_balance(monkeypatch):
    """Backtesting (`as_of` set) has no real broker to query for a historical
    replay — must always use the internal ledger, never `real_shares_held`."""

    async def fake_get_position(ticker):
        return Position(cash=0.0, shares=7)

    async def fake_get_last_trade(ticker):
        return None

    async def fail_real_shares_held(ticker):
        raise AssertionError("real_shares_held must not be called during a backtest")

    monkeypatch.setattr(pipeline, "get_position", fake_get_position)
    monkeypatch.setattr(pipeline, "get_last_trade", fake_get_last_trade)
    monkeypatch.setattr(pipeline, "real_shares_held", fail_real_shares_held)
    held, result = await pipeline._position_context("AAPL", date(2026, 1, 1))
    assert held is True
    assert result == "Currently holding 7 shares."


async def test_position_context_includes_open_stop_loss(monkeypatch):
    """A re-analysis must see the stop-loss already on record, not decide
    blind each time (2026-07-28 context gap)."""

    async def fake_get_position(ticker):
        return Position(cash=0.0, shares=10)

    async def fake_get_last_trade(ticker):
        return None

    async def fake_real_shares_held(ticker):
        return None

    async def fake_get_open_stop_loss(ticker):
        return (date(2026, 7, 20), 142.50)

    monkeypatch.setattr(pipeline, "get_position", fake_get_position)
    monkeypatch.setattr(pipeline, "get_last_trade", fake_get_last_trade)
    monkeypatch.setattr(pipeline, "real_shares_held", fake_real_shares_held)
    monkeypatch.setattr(pipeline, "get_open_stop_loss", fake_get_open_stop_loss)
    held, result = await pipeline._position_context("AAPL", None)
    assert held is True
    assert "Current stop-loss on record: 142.50 USD." in result


async def test_position_context_omits_stop_loss_line_when_unset(monkeypatch):
    async def fake_get_position(ticker):
        return Position(cash=0.0, shares=10)

    async def fake_get_last_trade(ticker):
        return None

    async def fake_real_shares_held(ticker):
        return None

    async def fake_get_open_stop_loss(ticker):
        return None

    monkeypatch.setattr(pipeline, "get_position", fake_get_position)
    monkeypatch.setattr(pipeline, "get_last_trade", fake_get_last_trade)
    monkeypatch.setattr(pipeline, "real_shares_held", fake_real_shares_held)
    monkeypatch.setattr(pipeline, "get_open_stop_loss", fake_get_open_stop_loss)
    held, result = await pipeline._position_context("AAPL", None)
    assert "Current stop-loss on record" not in result


async def test_position_context_includes_avg_cost_and_pnl_when_live(monkeypatch):
    """Real gap this fixes: the Portfolio Manager only ever saw the LAST
    trade's price, not the true (possibly multi-buy) weighted-average cost,
    and had no idea whether the position was currently up or down."""

    class _FakePosition:
        avg_price = 130.0
        pnl_amount = 200.0
        pnl_pct = 15.38

    async def fake_get_position(ticker):
        return Position(cash=0.0, shares=10)

    async def fake_get_last_trade(ticker):
        return None

    async def fake_real_shares_held(ticker):
        return 10

    async def fake_get_open_stop_loss(ticker):
        return None

    async def fake_real_position_detail(ticker):
        return _FakePosition()

    monkeypatch.setattr(pipeline, "get_position", fake_get_position)
    monkeypatch.setattr(pipeline, "get_last_trade", fake_get_last_trade)
    monkeypatch.setattr(pipeline, "real_shares_held", fake_real_shares_held)
    monkeypatch.setattr(pipeline, "get_open_stop_loss", fake_get_open_stop_loss)
    monkeypatch.setattr(pipeline, "real_position_detail", fake_real_position_detail)
    held, result = await pipeline._position_context("AAPL", None)
    assert "Avg cost 130.00 USD" in result
    assert "+15.38%" in result
    assert "+200" in result


async def test_position_context_never_calls_real_position_detail_during_backtest(monkeypatch):
    async def fake_get_position(ticker):
        return Position(cash=0.0, shares=10)

    async def fake_get_last_trade(ticker):
        return None

    async def fake_get_open_stop_loss(ticker):
        return None

    async def fail_real_position_detail(ticker):
        raise AssertionError("real_position_detail must not be called during a backtest")

    monkeypatch.setattr(pipeline, "get_position", fake_get_position)
    monkeypatch.setattr(pipeline, "get_last_trade", fake_get_last_trade)
    monkeypatch.setattr(pipeline, "get_open_stop_loss", fake_get_open_stop_loss)
    monkeypatch.setattr(pipeline, "real_position_detail", fail_real_position_detail)
    held, result = await pipeline._position_context("AAPL", date(2026, 1, 1))
    assert "Avg cost" not in result


async def test_position_context_labels_krw_for_a_domestic_ticker(monkeypatch):
    """2026-08-25 fix: `_position_context` had the same unlabeled-KRW defect
    `_risk_cap_context` was fixed for on 2026-08-10 (229200.KS, "약
    1,415,380달러"), five times the blast radius (Trader + all three risk
    debators + Portfolio Manager)."""

    class _FakePosition:
        avg_price = 130.0
        pnl_amount = 200.0
        pnl_pct = 15.38

    async def fake_get_position(ticker):
        return Position(cash=0.0, shares=10)

    async def fake_get_last_trade(ticker):
        return LastTrade(date(2026, 7, 1), "buy", 150.0, "**Rating**: Buy\n\n...")

    async def fake_real_shares_held(ticker):
        return 10

    async def fake_get_open_stop_loss(ticker):
        return (date(2026, 7, 20), 142.50)

    async def fake_real_position_detail(ticker):
        return _FakePosition()

    monkeypatch.setattr(pipeline, "get_position", fake_get_position)
    monkeypatch.setattr(pipeline, "get_last_trade", fake_get_last_trade)
    monkeypatch.setattr(pipeline, "real_shares_held", fake_real_shares_held)
    monkeypatch.setattr(pipeline, "get_open_stop_loss", fake_get_open_stop_loss)
    monkeypatch.setattr(pipeline, "real_position_detail", fake_real_position_detail)
    held, result = await pipeline._position_context("005930.KS", None)
    assert held is True
    assert "150.00 KRW" in result
    assert "130.00 KRW" in result
    assert "142.50 KRW" in result
    assert "200 KRW" in result
    assert "USD" not in result
    assert "$" not in result


async def test_position_context_labels_usd_for_an_overseas_ticker(monkeypatch):
    class _FakePosition:
        avg_price = 130.0
        pnl_amount = 200.0
        pnl_pct = 15.38

    async def fake_get_position(ticker):
        return Position(cash=0.0, shares=10)

    async def fake_get_last_trade(ticker):
        return LastTrade(date(2026, 7, 1), "buy", 150.0, "**Rating**: Buy\n\n...")

    async def fake_real_shares_held(ticker):
        return 10

    async def fake_get_open_stop_loss(ticker):
        return (date(2026, 7, 20), 142.50)

    async def fake_real_position_detail(ticker):
        return _FakePosition()

    monkeypatch.setattr(pipeline, "get_position", fake_get_position)
    monkeypatch.setattr(pipeline, "get_last_trade", fake_get_last_trade)
    monkeypatch.setattr(pipeline, "real_shares_held", fake_real_shares_held)
    monkeypatch.setattr(pipeline, "get_open_stop_loss", fake_get_open_stop_loss)
    monkeypatch.setattr(pipeline, "real_position_detail", fake_real_position_detail)
    held, result = await pipeline._position_context("AAPL", None)
    assert held is True
    assert "USD" in result
    assert "KRW" not in result


async def test_position_context_labels_currency_on_the_forced_stop_exit_narrative(monkeypatch):
    async def fake_get_position(ticker):
        return Position(cash=1000.0, shares=0)

    async def fake_get_last_trade(ticker):
        return LastTrade(
            date(2026, 7, 29), "sell", 198714.0,
            f"{STOP_LOSS_REASON_PREFIX}stop-loss monitor — breach 2026-07-29: low 195000.00 <= stop 198714.00",
        )

    async def fake_real_shares_held(ticker):
        return None

    async def fake_get_last_close(ticker, as_of=None):
        return 214500.0

    monkeypatch.setattr(pipeline, "get_position", fake_get_position)
    monkeypatch.setattr(pipeline, "get_last_trade", fake_get_last_trade)
    monkeypatch.setattr(pipeline, "real_shares_held", fake_real_shares_held)
    monkeypatch.setattr(pipeline, "get_last_close", fake_get_last_close)
    held, result = await pipeline._position_context("005930.KS", None)
    assert "198714.00 KRW" in result
    assert "214500.00 KRW" in result
    assert "USD" not in result
    assert "$" not in result


async def test_position_context_never_emits_a_bare_dollar_sign(monkeypatch):
    async def fake_get_position(ticker):
        return Position(cash=0.0, shares=10)

    async def fake_get_last_trade(ticker):
        return LastTrade(date(2026, 7, 1), "buy", 150.0, "**Rating**: Buy\n\n...")

    async def fake_real_shares_held(ticker):
        return None

    monkeypatch.setattr(pipeline, "get_position", fake_get_position)
    monkeypatch.setattr(pipeline, "get_last_trade", fake_get_last_trade)
    monkeypatch.setattr(pipeline, "real_shares_held", fake_real_shares_held)
    for ticker in ("005930.KS", "AAPL"):
        _, result = await pipeline._position_context(ticker, None)
        assert "$" not in result


# ---------------------------------------------------------------------------
# _execution_status_context
# ---------------------------------------------------------------------------


async def test_execution_status_context_unavailable_when_broker_not_connected(monkeypatch):
    from app.web import state

    monkeypatch.setattr(state, "kis_broker", broker_stub(None))
    result = await pipeline._execution_status_context("AAPL", 100.0)
    assert result == "Execution status unavailable — broker not connected."


async def test_execution_status_context_reports_blockers_when_present(monkeypatch):
    from app.web import state

    monkeypatch.setattr(state, "kis_broker", broker_stub(object()))

    async def fake_execution_blockers(broker, krx_code, ticker, *, checked_out=None):
        return ["BLOCKED: circuit breaker active for 005930 (mkop_cls_code='C1')"]

    async def fake_available_cash_native(broker, krx_code, us_symbol, price):
        return 1_000_000.0, "KRW"

    monkeypatch.setattr(pipeline, "execution_blockers", fake_execution_blockers)
    monkeypatch.setattr(pipeline, "available_cash_native", fake_available_cash_native)
    result = await pipeline._execution_status_context("005930.KS", 70000.0)
    assert "WARNING" in result
    assert "circuit breaker active" in result
    assert "1,000,000.00 KRW" in result


async def test_execution_status_context_reports_clean_domestic_status(monkeypatch):
    # 2026-09-07 (id=1238 halt-status narrative fix): the clean-case wording
    # is now CONFIRMED-strength only when all three underlying gates actually
    # completed a check this cycle (see execution_blockers' own `checked_out`
    # docstring) — the fake below simulates that by populating `checked_out`,
    # same as a real clean run would.
    from app.web import state

    monkeypatch.setattr(state, "kis_broker", broker_stub(object()))

    async def fake_execution_blockers(broker, krx_code, ticker, *, checked_out=None):
        if checked_out is not None:
            checked_out.extend(["circuit_breaker", "stock_halt", "vi"])
        return []

    async def fake_available_cash_native(broker, krx_code, us_symbol, price):
        return 500_000.0, "KRW"

    monkeypatch.setattr(pipeline, "execution_blockers", fake_execution_blockers)
    monkeypatch.setattr(pipeline, "available_cash_native", fake_available_cash_native)
    result = await pipeline._execution_status_context("005930.KS", 70000.0)
    assert "CONFIRMED by the broker just now" in result
    assert "no trading halt, no VI, and no circuit-breaker is currently active" in result
    assert "500,000.00 KRW" in result


async def test_execution_status_context_reports_could_not_verify_when_a_check_is_missing(monkeypatch):
    # 2026-09-07 (id=1238 halt-status narrative fix): an empty blockers list
    # is NOT the same as a confirmed-clear read when one of the three gates
    # didn't actually complete this cycle (an exception, a paper-mode-
    # unsupported skip, or a genuinely unknown circuit-breaker state) — this
    # must get the weaker "could not be fully verified" wording, never the
    # CONFIRMED one.
    from app.web import state

    monkeypatch.setattr(state, "kis_broker", broker_stub(object()))

    async def fake_execution_blockers(broker, krx_code, ticker, *, checked_out=None):
        if checked_out is not None:
            checked_out.extend(["circuit_breaker", "vi"])  # stock_halt missing
        return []

    async def fake_available_cash_native(broker, krx_code, us_symbol, price):
        return 500_000.0, "KRW"

    monkeypatch.setattr(pipeline, "execution_blockers", fake_execution_blockers)
    monkeypatch.setattr(pipeline, "available_cash_native", fake_available_cash_native)
    result = await pipeline._execution_status_context("005930.KS", 70000.0)
    assert "could not be fully verified" in result
    assert "CONFIRMED" not in result
    assert "circuit_breaker" in result and "vi" in result and "stock_halt" not in result


async def test_execution_status_context_skips_domestic_only_blockers_for_overseas(monkeypatch):
    from app.web import state

    monkeypatch.setattr(state, "kis_broker", broker_stub(object()))

    async def fail_execution_blockers(*args, **kwargs):
        raise AssertionError("execution_blockers is domestic-only and must not be called for overseas tickers")

    async def fake_available_cash_native(broker, krx_code, us_symbol, price):
        return 2_000.0, "USD"

    monkeypatch.setattr(pipeline, "execution_blockers", fail_execution_blockers)
    monkeypatch.setattr(pipeline, "available_cash_native", fake_available_cash_native)
    result = await pipeline._execution_status_context("AAPL", 190.0)
    assert "2,000.00 USD" in result


async def test_execution_status_context_flags_unavailable_cash_figure(monkeypatch):
    from app.web import state

    monkeypatch.setattr(state, "kis_broker", broker_stub(object()))

    async def fake_execution_blockers(broker, krx_code, ticker, *, checked_out=None):
        return []

    async def fake_available_cash_native(broker, krx_code, us_symbol, price):
        return None

    monkeypatch.setattr(pipeline, "execution_blockers", fake_execution_blockers)
    monkeypatch.setattr(pipeline, "available_cash_native", fake_available_cash_native)
    result = await pipeline._execution_status_context("005930.KS", 70000.0)
    assert "Actual available cash figure unavailable" in result


# ---------------------------------------------------------------------------
# _atr_context
# ---------------------------------------------------------------------------


async def test_atr_context_returns_unavailable_message_on_exception(monkeypatch):
    async def fake_get_atr_pct(ticker, as_of=None):
        raise RuntimeError("data unavailable")

    monkeypatch.setattr(pipeline, "get_atr_pct", fake_get_atr_pct)
    atr, message = await pipeline._atr_context("AAPL", None, held=True)
    assert atr is None
    # 2026-09-02: message rewritten to be held-aware and to stop reading
    # like an invitation to buy at full unscaled size (see id=1169).
    assert "Volatility data (14-day ATR) is unavailable" in message
    assert "existing position and its recorded stop-loss are unaffected" in message


async def test_atr_context_returns_unavailable_message_on_exception_when_flat(monkeypatch):
    async def fake_get_atr_pct(ticker, as_of=None):
        raise RuntimeError("data unavailable")

    monkeypatch.setattr(pipeline, "get_atr_pct", fake_get_atr_pct)
    atr, message = await pipeline._atr_context("AAPL", None, held=False)
    assert atr is None
    assert "Volatility data (14-day ATR) is unavailable" in message
    assert "REFUSE to open a new position" in message
    assert "Only Hold is actionable here" in message


async def test_atr_context_none_without_exception_does_not_raise(monkeypatch):
    # 2026-09-02 fix: get_atr_pct can return None WITHOUT raising (thin
    # history / a recent listing) — before this fix, the code right after
    # this call unconditionally did `current_atr_pct <= target`, raising
    # TypeError on a bare None and crashing the whole pipeline run for that
    # ticker (see this function's own comment for the full incident).
    async def fake_get_atr_pct(ticker, as_of=None):
        return None

    monkeypatch.setattr(pipeline, "get_atr_pct", fake_get_atr_pct)
    atr, message = await pipeline._atr_context("AAPL", None, held=True)
    assert atr is None
    assert "Volatility data (14-day ATR) is unavailable" in message


async def test_atr_context_zero_atr_is_treated_as_unavailable(monkeypatch):
    # A literal 0.0 (app.dataflows.indicators.atr_pct's degenerate-close
    # case) must be treated the same as None here, matching
    # app.domain.executability.atr_unavailable's predicate.
    async def fake_get_atr_pct(ticker, as_of=None):
        return 0.0

    monkeypatch.setattr(pipeline, "get_atr_pct", fake_get_atr_pct)
    atr, message = await pipeline._atr_context("AAPL", None, held=False)
    assert atr == 0.0  # raw value preserved, distinguishable from None
    assert "Volatility data (14-day ATR) is unavailable" in message


async def test_atr_context_below_target_no_scaling(monkeypatch):
    monkeypatch.setattr(settings, "sizing_target_atr_pct", 0.02)

    async def fake_get_atr_pct(ticker, as_of=None):
        return 0.01

    monkeypatch.setattr(pipeline, "get_atr_pct", fake_get_atr_pct)
    atr, message = await pipeline._atr_context("AAPL", None, held=True)
    assert atr == 0.01
    assert "no volatility-based scaling applies" in message


async def test_atr_context_above_target_scales_allocations(monkeypatch):
    monkeypatch.setattr(settings, "sizing_target_atr_pct", 0.02)

    async def fake_get_atr_pct(ticker, as_of=None):
        return 0.04

    monkeypatch.setattr(pipeline, "get_atr_pct", fake_get_atr_pct)
    atr, message = await pipeline._atr_context("AAPL", None, held=True)
    assert atr == 0.04
    assert "code will automatically scale down" in message
    assert "Buy ->" in message


async def test_atr_context_below_target_no_scaling_flat(monkeypatch):
    monkeypatch.setattr(settings, "sizing_target_atr_pct", 0.02)

    async def fake_get_atr_pct(ticker, as_of=None):
        return 0.01

    monkeypatch.setattr(pipeline, "get_atr_pct", fake_get_atr_pct)
    atr, message = await pipeline._atr_context("AAPL", None, held=False)
    assert atr == 0.01
    assert "no volatility-based scaling applies" in message
    assert "Overweight 75%" not in message
    assert "Underweight 25%" not in message
    assert "100%" in message


async def test_atr_context_above_target_scales_allocations_flat(monkeypatch):
    monkeypatch.setattr(settings, "sizing_target_atr_pct", 0.02)

    async def fake_get_atr_pct(ticker, as_of=None):
        return 0.04

    monkeypatch.setattr(pipeline, "get_atr_pct", fake_get_atr_pct)
    atr, message = await pipeline._atr_context("AAPL", None, held=False)
    assert atr == 0.04
    assert "Buy ->" in message
    assert "Overweight ->" not in message
    assert "Underweight ->" not in message


# ---------------------------------------------------------------------------
# _risk_cap_context
# ---------------------------------------------------------------------------


async def test_risk_cap_context_unavailable_for_nonpositive_price():
    result = await pipeline._risk_cap_context("AAPL", 0.0, as_of=None)
    assert "Risk-cap data unavailable" in result


async def test_risk_cap_context_unavailable_on_equity_lookup_failure(monkeypatch):
    async def fake_resolve_equity_basis(ticker, price, *, as_of, position=None):
        raise RuntimeError("db down")

    monkeypatch.setattr(pipeline, "resolve_equity_basis", fake_resolve_equity_basis)
    result = await pipeline._risk_cap_context("AAPL", 100.0, as_of=None)
    assert "Risk-cap data unavailable" in result


async def test_risk_cap_context_explains_the_real_account_equity_refusal(monkeypatch):
    """A real-account broker-balance outage is NOT the same as generic missing
    data: resolve_equity_basis refuses to fall back to the internal ledger's
    fictional paper balance, and the actual sizing call ~9 LLM calls later
    hits the same refusal and skips the trade outright. The Portfolio Manager
    must be told that specifically, or it writes a narrative promising an
    exposure increase that execution will never perform."""

    async def fake_resolve_equity_basis(ticker, price, *, as_of, position=None):
        raise EquityBasisUnavailable(
            "no equity basis", ticker="AAPL", market="overseas", failure="broker_unavailable"
        )

    monkeypatch.setattr(pipeline, "resolve_equity_basis", fake_resolve_equity_basis)
    result = await pipeline._risk_cap_context("AAPL", 100.0, as_of=None)
    assert "real broker balance could not be read" in result
    assert "skip this trade entirely" in result
    assert "broker_unavailable" in result
    # Must not be mistaken for the generic-failure branch above.
    assert "Risk-cap data unavailable" not in result


async def test_risk_cap_context_computes_share_caps(monkeypatch):
    monkeypatch.setattr(settings, "sizing_max_risk_pct_of_equity", 0.01)

    async def fake_resolve_equity_basis(ticker, price, *, as_of, position=None):
        return (10_000.0, "real_account")

    monkeypatch.setattr(pipeline, "resolve_equity_basis", fake_resolve_equity_basis)
    result = await pipeline._risk_cap_context("AAPL", 100.0, as_of=None)
    # equity = 10,000 (stubbed); max_dollar_risk = 100
    assert "100 USD" in result
    assert "current equity of 10,000 USD" in result


async def test_risk_cap_context_says_zero_shares_at_every_stop_when_the_cap_permits_none(monkeypatch):
    """2026-09-01, id=1157 (009150.KS at 1,431,500 KRW, 200,000 KRW equity):
    the 1% risk budget cannot absorb even one share's 3%-stop risk, so the
    generic 'a tighter stop allows more shares' text is an invitation to a
    lever that does not exist — the risk debate spent ~12 LLM calls acting on
    it. The narrative must say the true thing plainly."""
    monkeypatch.setattr(settings, "sizing_max_risk_pct_of_equity", 0.01)

    async def fake_resolve_equity_basis(ticker, price, *, as_of, position=None):
        return (200_000.0, "real_account")

    monkeypatch.setattr(pipeline, "resolve_equity_basis", fake_resolve_equity_basis)
    result = await pipeline._risk_cap_context("005930.KS", 1_431_500.0, as_of=None)
    assert "ZERO added shares at EVERY stop distance" in result
    assert "A TIGHTER stop allows MORE new shares" not in result


async def test_risk_cap_context_flags_the_real_equity_fallback_when_live(monkeypatch):
    """The equity basis this narrative uses can independently fall back to
    the internal ledger (real broker call unavailable) even though the
    actual sizing call ~9 LLM calls later might succeed or fail on its own —
    the LLM must be told explicitly rather than silently shown a number that
    may not match what execution ends up using (memory:
    live-rebalance-equity-basis-mismatch-2026-08-14)."""
    monkeypatch.setattr(settings, "sizing_max_risk_pct_of_equity", 0.01)

    async def fake_resolve_equity_basis(ticker, price, *, as_of, position=None):
        return (10_000.0, "internal_ledger")

    monkeypatch.setattr(pipeline, "resolve_equity_basis", fake_resolve_equity_basis)
    result = await pipeline._risk_cap_context("AAPL", 100.0, as_of=None)
    assert "real account balance could not be read" in result


async def test_risk_cap_context_does_not_flag_fallback_during_backtest(monkeypatch):
    """Backtest always uses the internal ledger by design (no real account
    exists for a historical replay) — that's not a fallback/failure, so the
    caveat must not appear."""
    monkeypatch.setattr(settings, "sizing_max_risk_pct_of_equity", 0.01)

    async def fake_resolve_equity_basis(ticker, price, *, as_of, position=None):
        return (10_000.0, "internal_ledger")

    monkeypatch.setattr(pipeline, "resolve_equity_basis", fake_resolve_equity_basis)
    result = await pipeline._risk_cap_context("AAPL", 100.0, as_of=date(2026, 7, 16))
    assert "real account balance could not be read" not in result


async def test_risk_cap_context_labels_krw_for_a_domestic_ticker(monkeypatch):
    """2026-08-10 fix: a hardcoded `$` here for a .KS ticker had the Portfolio
    Manager write an actual KRW figure out loud as dollars (229200.KS,
    "약 1,415,380달러") — the label must follow the ticker's own native
    currency, not assume USD."""
    monkeypatch.setattr(settings, "sizing_max_risk_pct_of_equity", 0.01)

    async def fake_resolve_equity_basis(ticker, price, *, as_of, position=None):
        return (10_000.0, "real_account")

    monkeypatch.setattr(pipeline, "resolve_equity_basis", fake_resolve_equity_basis)
    result = await pipeline._risk_cap_context("005930.KS", 100.0, as_of=None)
    assert "100 KRW" in result
    assert "current equity of 10,000 KRW" in result
    assert "USD" not in result
    assert "$" not in result


# ---------------------------------------------------------------------------
# _render_portfolio_risk_context
# ---------------------------------------------------------------------------


def test_render_portfolio_risk_context_no_breach():
    snapshot = PortfolioSnapshot(per_ticker_value={"AAPL": 2500.0, "MSFT": 7500.0}, total_value=10000.0)
    result = pipeline._render_portfolio_risk_context("AAPL", snapshot, clusters={}, drawdown_frac=0.05)
    assert "25.0% of aggregate portfolio equity" in result
    assert "no other correlated names" in result
    assert "no breach" in result


def test_render_portfolio_risk_context_drawdown_breach(monkeypatch):
    monkeypatch.setattr(settings, "portfolio_drawdown_breaker_pct", 0.12)
    snapshot = PortfolioSnapshot(per_ticker_value={"AAPL": 2500.0}, total_value=10000.0)
    result = pipeline._render_portfolio_risk_context("AAPL", snapshot, clusters={}, drawdown_frac=0.15)
    assert "any exposure-increasing trade will be blocked in code" in result


def test_render_portfolio_risk_context_includes_correlation_bucket_mates():
    snapshot = PortfolioSnapshot(
        per_ticker_value={"AAPL": 2000.0, "MSFT": 3000.0, "GOOG": 5000.0}, total_value=10000.0
    )
    clusters = {"AAPL": 1, "MSFT": 1, "GOOG": 2}
    result = pipeline._render_portfolio_risk_context("AAPL", snapshot, clusters=clusters, drawdown_frac=0.0)
    assert "MSFT" in result
    assert "50.0%" in result  # (2000+3000)/10000


# ---------------------------------------------------------------------------
# _company_identity caching
# ---------------------------------------------------------------------------


async def _noop_ensure_loaded():
    return None


def _patch_identity_deps(monkeypatch, *, krx_get=None, overseas_get=None, yf_display_info=None):
    """Common wiring so `_company_identity` tests don't hit the network:
    krx_master/overseas_master's ensure_loaded() would otherwise attempt a
    real HTTPS download, and the yfinance weak-alias lookup would otherwise
    attempt a real yfinance call."""
    monkeypatch.setattr(pipeline.krx_master, "ensure_loaded", _noop_ensure_loaded)
    monkeypatch.setattr(pipeline.krx_master, "get", krx_get or (lambda ticker: None))
    monkeypatch.setattr(pipeline.overseas_master, "ensure_loaded", _noop_ensure_loaded)
    monkeypatch.setattr(pipeline.overseas_master, "get", overseas_get or (lambda ticker: None))

    async def default_yf_display_info(ticker):
        raise RuntimeError("no yfinance data in this test")

    monkeypatch.setattr(pipeline, "yf_get_ticker_display_info", yf_display_info or default_yf_display_info)


async def test_company_identity_caches_successful_lookup(monkeypatch):
    pipeline._company_identity_cache.clear()
    _patch_identity_deps(monkeypatch)
    calls = []

    class FakeInfo:
        name = "Apple Inc."

    async def fake_display_info(ticker):
        calls.append(ticker)
        return FakeInfo()

    monkeypatch.setattr(pipeline, "get_ticker_display_info", fake_display_info)
    first, _aliases1, _weak1, _kis_en1 = await pipeline._company_identity("AAPL")
    second, _aliases2, _weak2, _kis_en2 = await pipeline._company_identity("AAPL")
    assert first == second == "Apple Inc."
    assert calls == ["AAPL"]  # only fetched once


async def test_company_identity_falls_back_to_ticker_on_failure(monkeypatch):
    pipeline._company_identity_cache.clear()
    _patch_identity_deps(monkeypatch)

    async def fake_display_info(ticker):
        raise RuntimeError("no data")

    monkeypatch.setattr(pipeline, "get_ticker_display_info", fake_display_info)
    result, _aliases, _weak, _kis_en = await pipeline._company_identity("XYZ")
    assert result == "XYZ"


async def test_company_identity_alias_set_excludes_the_ticker_and_code(monkeypatch):
    # 2026-08-25: build_alias_set used to be called with the ticker AND its
    # bare KRX code alongside the real names, making app.agents.identity_gate's
    # detectors self-satisfying (the ticker appears in essentially every
    # report by construction). identity_gate.check_identity now subtracts
    # these regardless of what it's given, but the call site should not be
    # handing them over in the first place either.
    pipeline._company_identity_cache.clear()
    _patch_identity_deps(monkeypatch, krx_get=lambda ticker: "우리금융지주")

    async def fake_display_info(ticker):
        raise RuntimeError("not reached — krx_master already resolved a name")

    monkeypatch.setattr(pipeline, "get_ticker_display_info", fake_display_info)
    _display_name, aliases, _weak, _kis_en = await pipeline._company_identity("316140.KS")
    assert "316140" not in aliases
    assert "316140ks" not in aliases


async def test_company_identity_prefers_kis_name_over_yfinance_for_krx_ticker(monkeypatch):
    # Real incident (2026-07-23): yfinance's get_ticker_display_info
    # returned "에코프로비엠" (a real, different company) for 064260.KS
    # (actually 다날) - that wrong name then anchored the Market Analyst's
    # whole report to a wrong company's typical price range. KIS's
    # hts_kor_isnm (app.live.ticker_names, populated by krx_top_n) is
    # authoritative for an exact KRX code and can't suffer that cross-symbol
    # confusion, so it must win even when yfinance already cached something.
    pipeline._company_identity_cache.clear()
    _patch_identity_deps(monkeypatch)
    monkeypatch.setattr(pipeline.ticker_names, "_cache", {"064260.KS": "다날"})

    async def should_not_be_called(ticker):
        raise AssertionError("yfinance should not be consulted when a KIS name is cached")

    monkeypatch.setattr(pipeline, "get_ticker_display_info", should_not_be_called)
    display_name, _aliases, _weak, _kis_en = await pipeline._company_identity("064260.KS")
    assert display_name == "다날"
    assert _kis_en == ""  # no get_kis_english_name stubbed here -> "" (never None) in the tuple


async def test_company_identity_weak_aliases_include_yfinance_short_name(monkeypatch):
    # 2026-08-28: short_name is fed into the weak-alias set alongside name
    # (longName) — a report abbreviating the company differently from
    # longName's own prefix (e.g. yfinance's own shortName form) still needs
    # something to match against. See TickerDisplayInfo.short_name's own
    # docstring.
    pipeline._company_identity_cache.clear()
    _patch_identity_deps(monkeypatch)
    monkeypatch.setattr(pipeline.ticker_names, "_cache", {"298040.KS": "효성중공업"})

    class FakeYfInfo:
        name = "Hyosung Heavy Industries Corporation"
        short_name = "HYOSUNG HEAVY"

    async def fake_yf_display_info(ticker):
        return FakeYfInfo()

    monkeypatch.setattr(pipeline, "yf_get_ticker_display_info", fake_yf_display_info)
    _display_name, _aliases, weak, _kis_en = await pipeline._company_identity("298040.KS")
    assert "hyosung heavy" in weak


async def test_company_identity_degrades_gracefully_when_kis_english_name_raises(monkeypatch):
    # 2026-09-03: get_kis_english_name is real-account-only (a paper broker
    # or missing credentials makes it raise/return None) — _company_identity
    # must build the exact same alias set as before this addition when that
    # happens, not propagate the failure or silently drop the yfinance
    # weak-alias entries alongside it.
    pipeline._company_identity_cache.clear()
    _patch_identity_deps(monkeypatch)
    monkeypatch.setattr(pipeline.ticker_names, "_cache", {"298040.KS": "효성중공업"})

    class FakeYfInfo:
        name = "Hyosung Heavy Industries Corporation"
        short_name = "HYOSUNG HEAVY"

    async def fake_yf_display_info(ticker):
        return FakeYfInfo()

    async def fake_get_kis_english_name(ticker):
        raise RuntimeError("paper broker: EGW02006")

    monkeypatch.setattr(pipeline, "yf_get_ticker_display_info", fake_yf_display_info)
    monkeypatch.setattr(pipeline, "get_kis_english_name", fake_get_kis_english_name)
    _display_name, _aliases, weak, _kis_en = await pipeline._company_identity("298040.KS")
    assert "hyosung heavy" in weak
    assert "hyosung heavy industries corporation" in weak


async def test_company_identity_folds_kis_english_name_into_weak_aliases(monkeypatch):
    # A successful (real-account) lookup feeds the WEAK tier only — never
    # promoted to authoritative in this pass (Stage 2, not implemented here).
    pipeline._company_identity_cache.clear()
    _patch_identity_deps(monkeypatch)
    monkeypatch.setattr(pipeline.ticker_names, "_cache", {"003580.KS": "동양파일"})

    async def fake_get_kis_english_name(ticker):
        return "Tongyang Pile Inc"

    monkeypatch.setattr(pipeline, "get_kis_english_name", fake_get_kis_english_name)
    _display_name, aliases, weak, kis_en = await pipeline._company_identity("003580.KS")
    assert "tongyang pile inc" in weak
    assert "tongyang pile inc" not in aliases
    assert kis_en == "Tongyang Pile Inc"  # 4th tuple element carries the raw KIS English name


async def test_company_identity_falls_back_to_yfinance_when_no_kis_name_cached(monkeypatch):
    pipeline._company_identity_cache.clear()
    _patch_identity_deps(monkeypatch)

    class FakeInfo:
        name = "NAVER Corporation"

    async def fake_display_info(ticker):
        return FakeInfo()

    monkeypatch.setattr(pipeline, "get_ticker_display_info", fake_display_info)
    monkeypatch.setattr(pipeline.ticker_names, "_cache", {})
    display_name, _aliases, _weak, _kis_en = await pipeline._company_identity("035420.KS")
    assert display_name == "NAVER Corporation"


async def test_company_identity_us_ticker_prefers_english_yfinance_name_over_korean_master(monkeypatch):
    # kis_provider.get_ticker_display_info was fixed (2026-08-20) to already
    # prefer English for US tickers, but _company_identity must not
    # re-introduce the Korean transliteration as the display_name itself —
    # only as an (authoritative) alias. Root cause of the 2026-08-20 QQQ/PANW
    # corruption: overseas_master's Korean transliteration ("인베스코 QQQ
    # 트러스트") was handed to an English-output model as if it were the
    # verified label.
    pipeline._company_identity_cache.clear()
    _patch_identity_deps(monkeypatch, overseas_get=lambda ticker: "인베스코 QQQ 트러스트")

    class FakeInfo:
        name = "Invesco QQQ Trust"

    async def fake_display_info(ticker):
        return FakeInfo()

    monkeypatch.setattr(pipeline, "get_ticker_display_info", fake_display_info)
    display_name, aliases, _weak, _kis_en = await pipeline._company_identity("QQQ")
    assert display_name == "Invesco QQQ Trust"
    assert "인베스코 qqq 트러스트" in aliases  # still a legitimate alias for the gate


async def test_company_identity_cache_evicts_oldest_entry_past_the_lru_cap(monkeypatch):
    # Without a cap, a long-running daemon or a many-ticker backtest would
    # grow this cache for the life of the process — one entry per distinct
    # ticker it ever saw (time-invariant, so nothing ever expires it on its
    # own the way the TTL-based _reports_cache expires).
    pipeline._company_identity_cache.clear()
    _patch_identity_deps(monkeypatch)
    monkeypatch.setattr(pipeline, "_COMPANY_NAME_CACHE_MAX", 2)

    class FakeInfo:
        def __init__(self, name):
            self.name = name

    async def fake_display_info(ticker):
        return FakeInfo(f"{ticker} Inc.")

    monkeypatch.setattr(pipeline, "get_ticker_display_info", fake_display_info)
    await pipeline._company_identity("AAA")
    await pipeline._company_identity("BBB")
    await pipeline._company_identity("CCC")  # pushes the cache past the cap of 2

    assert list(pipeline._company_identity_cache.keys()) == ["BBB", "CCC"]
    assert "AAA" not in pipeline._company_identity_cache


async def test_company_identity_cache_touch_on_hit_keeps_recently_used_entries_alive(monkeypatch):
    pipeline._company_identity_cache.clear()
    _patch_identity_deps(monkeypatch)
    monkeypatch.setattr(pipeline, "_COMPANY_NAME_CACHE_MAX", 2)

    class FakeInfo:
        def __init__(self, name):
            self.name = name

    async def fake_display_info(ticker):
        return FakeInfo(f"{ticker} Inc.")

    monkeypatch.setattr(pipeline, "get_ticker_display_info", fake_display_info)
    await pipeline._company_identity("AAA")
    await pipeline._company_identity("BBB")
    await pipeline._company_identity("AAA")  # re-touch AAA — should now outlive BBB
    await pipeline._company_identity("CCC")  # evicts the least-recently-used, which is now BBB

    assert "AAA" in pipeline._company_identity_cache
    assert "BBB" not in pipeline._company_identity_cache


# ---------------------------------------------------------------------------
# _cached_reports TTL
# ---------------------------------------------------------------------------


def test_cached_reports_returns_none_when_absent():
    pipeline._reports_cache.clear()
    assert pipeline._cached_reports("AAPL", "2026-07-16", None, "ko") is None


def test_cached_reports_returns_fresh_entry():
    pipeline._reports_cache.clear()
    import time as time_mod

    key = ("AAPL", "2026-07-16", None, "ko")
    pipeline._reports_cache[key] = (time_mod.monotonic(), {"Market Analyst": "bullish"})
    result = pipeline._cached_reports("AAPL", "2026-07-16", None, "ko")
    assert result == {"Market Analyst": "bullish"}


def test_cached_reports_expires_after_ttl():
    pipeline._reports_cache.clear()
    import time as time_mod

    key = ("AAPL", "2026-07-16", None, "ko")
    stale_timestamp = time_mod.monotonic() - pipeline._REPORTS_CACHE_TTL_SECONDS - 1
    pipeline._reports_cache[key] = (stale_timestamp, {"Market Analyst": "bullish"})
    assert pipeline._cached_reports("AAPL", "2026-07-16", None, "ko") is None


def test_cached_reports_is_scoped_per_language():
    pipeline._reports_cache.clear()
    import time as time_mod

    pipeline._reports_cache[("AAPL", "2026-07-16", None, "ko")] = (
        time_mod.monotonic(), {"Market Analyst": "bullish (Korean)"}
    )
    # A different language for the same ticker/date/as_of must not hit the
    # other language's cached reports — see app.llm.language's rationale.
    assert pipeline._cached_reports("AAPL", "2026-07-16", None, "en") is None


# ---------------------------------------------------------------------------
# _store_cached_reports eviction
# ---------------------------------------------------------------------------


def test_store_cached_reports_evicts_expired_entries_on_write():
    # Regression: a TTL-only cache with no eviction grows for the life of
    # the process — every distinct key ever written (e.g. one per backtest
    # date) stayed forever, expired or not.
    pipeline._reports_cache.clear()
    import time as time_mod

    stale_key = ("AAPL", "2026-01-01", None, "ko")
    stale_timestamp = time_mod.monotonic() - pipeline._REPORTS_CACHE_TTL_SECONDS - 1
    pipeline._reports_cache[stale_key] = (stale_timestamp, {"Market Analyst": "old"})

    pipeline._store_cached_reports("MSFT", "2026-07-16", None, "ko", {"Market Analyst": "new"})

    assert stale_key not in pipeline._reports_cache
    assert pipeline._cached_reports("MSFT", "2026-07-16", None, "ko") == {"Market Analyst": "new"}


def test_store_cached_reports_keeps_unexpired_entries():
    pipeline._reports_cache.clear()
    import time as time_mod

    fresh_key = ("AAPL", "2026-07-16", None, "ko")
    pipeline._reports_cache[fresh_key] = (time_mod.monotonic(), {"Market Analyst": "still fresh"})

    pipeline._store_cached_reports("MSFT", "2026-07-16", None, "ko", {"Market Analyst": "new"})

    assert pipeline._cached_reports("AAPL", "2026-07-16", None, "ko") == {"Market Analyst": "still fresh"}


# ---------------------------------------------------------------------------
# is_state_flipping_rating (app.state) — shared predicate used by both
# _apply_hysteresis_gate below and app.agents.portfolio_manager's retry.
# ---------------------------------------------------------------------------


def test_is_state_flipping_rating_true_for_held_sell():
    assert is_state_flipping_rating(PortfolioRating.SELL, held=True) is True


def test_is_state_flipping_rating_true_for_flat_buy():
    assert is_state_flipping_rating(FlatPortfolioRating.BUY, held=False) is True


@pytest.mark.parametrize(
    ("rating", "held"),
    [
        (PortfolioRating.HOLD, True),
        (PortfolioRating.OVERWEIGHT, True),
        (PortfolioRating.UNDERWEIGHT, True),
        (FlatPortfolioRating.HOLD, False),
    ],
)
def test_is_state_flipping_rating_false_for_non_flip_transitions(rating, held):
    assert is_state_flipping_rating(rating, held=held) is False


# ---------------------------------------------------------------------------
# _apply_hysteresis_gate (ai-engine-24h-operation-analysis P0-1, 2026-08-11)
# ---------------------------------------------------------------------------


def _flat_decision(rating, confidence=None):
    return FlatPortfolioDecision(
        rating=rating, executive_summary="s", investment_thesis="thesis text", confidence=confidence
    )


def _held_decision(rating, confidence=None):
    return PortfolioDecision(
        rating=rating, executive_summary="s", investment_thesis="thesis text", confidence=confidence
    )


def test_hysteresis_gate_downgrades_flat_buy_below_threshold(monkeypatch):
    monkeypatch.setattr(settings, "hysteresis_min_confidence_to_flip", 0.6)
    decision = _flat_decision(FlatPortfolioRating.BUY, confidence=0.4)

    result = pipeline._apply_hysteresis_gate(decision, held=False)

    assert result.rating == FlatPortfolioRating.HOLD
    assert "Hysteresis gate" in result.investment_thesis
    assert "Buy" in result.investment_thesis  # names the original rating it downgraded from


def test_hysteresis_gate_prefixes_executive_summary_with_a_number_free_note(monkeypatch):
    """2026-08-14 fix: render_portfolio_decision (the short default view, and
    what's stored as trades.reason) deliberately omits investment_thesis, so
    the gate's note there was invisible — the visible text used to show
    "Rating: Hold" next to an executive_summary still arguing the original
    Buy/Sell case with zero explanation. The prefix must be a PREFIX (the
    field is clamped to 6 lines downstream) and must not repeat the
    confidence/threshold percentages already in investment_thesis (would
    double-count in language_ab_report's numeric-mismatch scan)."""
    monkeypatch.setattr(settings, "hysteresis_min_confidence_to_flip", 0.6)
    decision = _flat_decision(FlatPortfolioRating.BUY, confidence=0.4)
    original_summary = decision.executive_summary

    result = pipeline._apply_hysteresis_gate(decision, held=False)

    assert result.executive_summary.startswith("[Hysteresis gate:")
    assert result.executive_summary.endswith(original_summary)
    assert "Buy" in result.executive_summary
    assert "%" not in result.executive_summary  # no confidence/threshold numbers here


def test_hysteresis_gate_downgrades_flat_buy_with_no_confidence_stated(monkeypatch):
    # Missing confidence is treated as below the bar, not exempt from it.
    monkeypatch.setattr(settings, "hysteresis_min_confidence_to_flip", 0.6)
    decision = _flat_decision(FlatPortfolioRating.BUY, confidence=None)

    result = pipeline._apply_hysteresis_gate(decision, held=False)

    assert result.rating == FlatPortfolioRating.HOLD


def test_hysteresis_gate_allows_flat_buy_at_or_above_threshold(monkeypatch):
    monkeypatch.setattr(settings, "hysteresis_min_confidence_to_flip", 0.6)
    decision = _flat_decision(FlatPortfolioRating.BUY, confidence=0.6)

    result = pipeline._apply_hysteresis_gate(decision, held=False)

    assert result.rating == FlatPortfolioRating.BUY
    assert result is decision  # untouched, not even a needless copy


def test_hysteresis_gate_leaves_flat_hold_untouched(monkeypatch):
    monkeypatch.setattr(settings, "hysteresis_min_confidence_to_flip", 0.6)
    decision = _flat_decision(FlatPortfolioRating.HOLD, confidence=None)

    result = pipeline._apply_hysteresis_gate(decision, held=False)

    assert result is decision


def test_hysteresis_gate_model_copy_preserves_card_reason_fields(monkeypatch):
    # card_reason_ko/card_reason_en aren't in _apply_hysteresis_gate's
    # model_copy `update={...}` dict, so they must pass through unchanged —
    # pipeline.py relies on this to store the pre-gate card reason alongside
    # the post-gate rating without re-listing every field here.
    monkeypatch.setattr(settings, "hysteresis_min_confidence_to_flip", 0.6)
    decision = FlatPortfolioDecision(
        rating=FlatPortfolioRating.BUY,
        executive_summary="s",
        investment_thesis="thesis text",
        confidence=0.4,
        card_reason_ko="한글 이유",
        card_reason_en="english reason",
    )

    result = pipeline._apply_hysteresis_gate(decision, held=False)

    assert result.rating == FlatPortfolioRating.HOLD  # gate did fire
    assert result.card_reason_ko == "한글 이유"
    assert result.card_reason_en == "english reason"


# ---------------------------------------------------------------------------
# _degradation_note / _apply_degradation_notice (2026-09-03 decision id=1179)
# — a failed analyst report degraded silently; make it visible (not
# consequential — no rating/sizing/confidence change).
# ---------------------------------------------------------------------------


def test_degradation_note_is_empty_when_nothing_degraded():
    assert pipeline._degradation_note([]) == ""


def test_degradation_note_names_the_failed_analysts():
    note = pipeline._degradation_note(["Fundamentals Analyst"])
    assert "1 of 4" in note
    assert "Fundamentals Analyst" in note
    assert "executive_summary" in note


def test_degradation_notice_is_noop_when_no_analyst_failed():
    decision = _held_decision(PortfolioRating.HOLD)
    result = pipeline._apply_degradation_notice(decision, [])
    assert result is decision


def test_degradation_notice_prefixes_executive_summary_and_appends_thesis():
    decision = PortfolioDecision(
        rating=PortfolioRating.HOLD,
        executive_summary="original summary",
        investment_thesis="original thesis",
        confidence=0.5,
    )
    result = pipeline._apply_degradation_notice(decision, ["Fundamentals Analyst"])

    assert result.executive_summary.startswith("[Degraded analysis: 1/4")
    assert result.executive_summary.endswith("original summary")
    assert "Fundamentals Analyst" in result.executive_summary
    assert result.investment_thesis.startswith("original thesis")
    assert result.investment_thesis.rstrip().endswith("]")
    assert "Fundamentals Analyst" in result.investment_thesis
    assert "remaining 3" in result.investment_thesis


def test_degradation_notice_prefixes_both_card_reasons():
    decision = PortfolioDecision(
        rating=PortfolioRating.HOLD,
        executive_summary="s",
        investment_thesis="t",
        confidence=0.5,
        card_reason_ko="한글 이유",
        card_reason_en="english reason",
    )
    result = pipeline._apply_degradation_notice(decision, ["Market Analyst", "Sentiment Analyst"])

    assert result.card_reason_en.startswith("[Degraded 2/4] ")
    assert result.card_reason_ko.startswith("[분석 축소 2/4] ")


@pytest.mark.parametrize("blank", [None, "   "])
def test_degradation_notice_skips_blank_card_reason(blank):
    decision = PortfolioDecision(
        rating=PortfolioRating.HOLD,
        executive_summary="s",
        investment_thesis="t",
        confidence=0.5,
        card_reason_ko=blank,
        card_reason_en=blank,
    )
    result = pipeline._apply_degradation_notice(decision, ["Market Analyst"])

    assert result.card_reason_ko == blank
    assert result.card_reason_en == blank


def test_degradation_notice_composes_with_hysteresis_gate(monkeypatch):
    monkeypatch.setattr(settings, "hysteresis_min_confidence_to_flip", 0.6)
    decision = _flat_decision(FlatPortfolioRating.BUY, confidence=0.3)

    gated = pipeline._apply_hysteresis_gate(decision, held=False)
    result = pipeline._apply_degradation_notice(gated, ["Fundamentals Analyst"])

    assert result.rating == FlatPortfolioRating.HOLD
    assert result.executive_summary.startswith("[Degraded analysis: 1/4")
    assert "[Hysteresis gate:" in result.executive_summary


def test_hysteresis_gate_downgrades_held_sell_below_threshold(monkeypatch):
    monkeypatch.setattr(settings, "hysteresis_min_confidence_to_flip", 0.6)
    decision = _held_decision(PortfolioRating.SELL, confidence=0.3)

    result = pipeline._apply_hysteresis_gate(decision, held=True)

    assert result.rating == PortfolioRating.HOLD
    assert "Sell" in result.investment_thesis


def test_hysteresis_gate_allows_held_sell_at_or_above_threshold(monkeypatch):
    monkeypatch.setattr(settings, "hysteresis_min_confidence_to_flip", 0.6)
    decision = _held_decision(PortfolioRating.SELL, confidence=0.75)

    result = pipeline._apply_hysteresis_gate(decision, held=True)

    assert result.rating == PortfolioRating.SELL


@pytest.mark.parametrize("rating", [PortfolioRating.OVERWEIGHT, PortfolioRating.UNDERWEIGHT, PortfolioRating.HOLD])
def test_hysteresis_gate_never_touches_non_flip_ratings_when_held(monkeypatch, rating):
    # Adjusting an already-open position (Overweight/Underweight) or holding
    # it isn't the round-trip-loss scenario this gate exists for — only
    # entering (flat->Buy) and fully exiting (held->Sell) are gated.
    monkeypatch.setattr(settings, "hysteresis_min_confidence_to_flip", 0.6)
    decision = _held_decision(rating, confidence=0.0)

    result = pipeline._apply_hysteresis_gate(decision, held=True)

    assert result is decision


def test_hysteresis_gate_respects_a_custom_threshold(monkeypatch):
    monkeypatch.setattr(settings, "hysteresis_min_confidence_to_flip", 0.9)
    decision = _flat_decision(FlatPortfolioRating.BUY, confidence=0.8)  # would pass the default 0.6 bar

    result = pipeline._apply_hysteresis_gate(decision, held=False)

    assert result.rating == FlatPortfolioRating.HOLD


# ---------------------------------------------------------------------------
# _desk_consensus_note / _consensus_override_tier / _apply_consensus_override_gate
# (2026-09-04 PM consensus-override incident, id=1212/1217)
# ---------------------------------------------------------------------------


def _flat_plan(rec=FlatResearchRecommendation.HOLD):
    return FlatResearchPlan(recommendation=rec, rationale="r", strategic_actions="a")


def _held_plan(rec=PortfolioRating.HOLD):
    return ResearchPlan(recommendation=rec, rationale="r", strategic_actions="a")


def _tp(action=TraderAction.HOLD):
    return TraderProposal(action=action, reasoning="reasoning")


def test_consensus_gate_downgrades_flat_buy_below_neutral_bar():  # id=1212
    plan = _flat_plan(FlatResearchRecommendation.HOLD)
    proposal = _tp(TraderAction.HOLD)
    decision = _flat_decision(FlatPortfolioRating.BUY, confidence=0.65)

    record: dict = {}
    result = pipeline._apply_consensus_override_gate(decision, plan, proposal, record_out=record)

    assert result.rating == FlatPortfolioRating.HOLD
    assert "[Consensus override gate:" in result.investment_thesis
    assert result.executive_summary.startswith("[Consensus override gate:")
    assert "%" not in result.executive_summary  # number-free prefix, original summary "s"
    assert record["gated"] is True
    assert record["tier"] == "neutral"
    assert record["threshold"] == 0.75


def test_consensus_gate_bearish_tier_downgrades_overweight_to_hold_not_sell():  # id=1217
    plan = _held_plan(PortfolioRating.UNDERWEIGHT)
    proposal = _tp(TraderAction.SELL)
    decision = _held_decision(PortfolioRating.OVERWEIGHT, confidence=0.75)

    record: dict = {}
    result = pipeline._apply_consensus_override_gate(decision, plan, proposal, record_out=record)

    assert result.rating == PortfolioRating.HOLD  # Hold, never Sell
    assert record["tier"] == "bearish"
    assert record["threshold"] == 0.85
    assert record["gated"] is True


def test_consensus_gate_records_a_passed_override_without_downgrading():
    plan = _held_plan(PortfolioRating.UNDERWEIGHT)
    proposal = _tp(TraderAction.SELL)
    decision = _held_decision(PortfolioRating.OVERWEIGHT, confidence=0.85)

    record: dict = {}
    result = pipeline._apply_consensus_override_gate(decision, plan, proposal, record_out=record)

    assert result is decision  # not downgraded
    assert record["gated"] is False
    assert record["tier"] == "bearish"
    assert record["pm_rating"] == "Overweight"


def test_consensus_gate_neutral_bar_is_inclusive():  # id=1204 unaffected
    plan = _flat_plan(FlatResearchRecommendation.HOLD)
    proposal = _tp(TraderAction.HOLD)
    decision = _flat_decision(FlatPortfolioRating.BUY, confidence=0.75)

    record: dict = {}
    result = pipeline._apply_consensus_override_gate(decision, plan, proposal, record_out=record)

    assert result.rating == FlatPortfolioRating.BUY
    assert record["gated"] is False


def test_consensus_gate_treats_missing_confidence_as_below_the_bar():
    plan = _flat_plan(FlatResearchRecommendation.HOLD)
    proposal = _tp(TraderAction.HOLD)
    decision = _flat_decision(FlatPortfolioRating.BUY, confidence=None)

    record: dict = {}
    result = pipeline._apply_consensus_override_gate(decision, plan, proposal, record_out=record)

    assert result.rating == FlatPortfolioRating.HOLD
    assert "(unset)" in result.investment_thesis


@pytest.mark.parametrize("rating", [PortfolioRating.SELL, PortfolioRating.HOLD, PortfolioRating.UNDERWEIGHT])
def test_consensus_gate_never_touches_a_reduce_or_hold(rating):
    plan = _held_plan(PortfolioRating.SELL)
    proposal = _tp(TraderAction.SELL)
    decision = _held_decision(rating, confidence=0.1)

    assert pipeline._consensus_override_tier(decision, plan, proposal) is None
    record: dict = {}
    result = pipeline._apply_consensus_override_gate(decision, plan, proposal, record_out=record)
    assert result is decision
    assert record == {}


def test_consensus_gate_inapplicable_when_research_manager_supports_exposure():
    plan = _held_plan(PortfolioRating.BUY)
    proposal = _tp(TraderAction.HOLD)
    decision = _held_decision(PortfolioRating.BUY, confidence=0.1)

    assert pipeline._consensus_override_tier(decision, plan, proposal) is None
    record: dict = {}
    assert pipeline._apply_consensus_override_gate(decision, plan, proposal, record_out=record) is decision


def test_consensus_gate_inapplicable_when_trader_proposed_buy():
    plan = _held_plan(PortfolioRating.HOLD)
    proposal = _tp(TraderAction.BUY)
    decision = _held_decision(PortfolioRating.BUY, confidence=0.1)

    assert pipeline._consensus_override_tier(decision, plan, proposal) is None


def test_consensus_gate_and_note_no_op_when_research_plan_missing():
    proposal = _tp(TraderAction.HOLD)
    decision = _held_decision(PortfolioRating.BUY, confidence=0.1)

    assert pipeline._desk_consensus_note(None, proposal) == ""
    assert pipeline._consensus_override_tier(decision, None, proposal) is None
    record: dict = {}
    assert pipeline._apply_consensus_override_gate(decision, None, proposal, record_out=record) is decision


def test_consensus_gate_runs_before_hysteresis_so_only_one_prefix_lands(monkeypatch):
    monkeypatch.setattr(settings, "hysteresis_min_confidence_to_flip", 0.6)
    plan = _flat_plan(FlatResearchRecommendation.HOLD)
    proposal = _tp(TraderAction.HOLD)
    decision = _flat_decision(FlatPortfolioRating.BUY, confidence=0.5)  # below BOTH bars

    record: dict = {}
    result = pipeline._apply_consensus_override_gate(decision, plan, proposal, record_out=record)
    result = pipeline._apply_hysteresis_gate(result, held=False)  # same order as the pipeline

    assert result.rating == FlatPortfolioRating.HOLD
    assert result.executive_summary.count("gate: downgraded from") == 1
    assert result.executive_summary.startswith("[Consensus override gate:")
    assert "[Hysteresis gate:" not in result.investment_thesis


def test_desk_consensus_note_adds_the_no_exposure_clause_only_when_both_decline():
    both_decline = pipeline._desk_consensus_note(_held_plan(PortfolioRating.HOLD), _tp(TraderAction.HOLD))
    assert "Neither of them supports increasing exposure" in both_decline

    rm_bullish = pipeline._desk_consensus_note(_held_plan(PortfolioRating.BUY), _tp(TraderAction.HOLD))
    assert "Neither of them supports increasing exposure" not in rm_bullish
    assert "the Research Manager concluded Buy" in rm_bullish


# ---------------------------------------------------------------------------
# _desk_consensus_note's kr_risk_extra / debate_sizing_context branches
# (2026-09-15 correction — the note used to claim the risk debate "saw
# nothing the Trader did not" even when it was fed KRX flow signals and/or
# sizing mechanics the Trader never received; see the function's own
# docstring for the id=1217 rationale)
# ---------------------------------------------------------------------------

_SECOND_PARAGRAPH = (
    " Neither of them supports increasing exposure. A Buy or Overweight from you "
    "here overrides the whole desk on the risk debate alone — name the specific new "
    "fact the debate introduced in your executive_summary, or choose Hold."
)


def test_desk_consensus_note_default_args_match_neither_supplied_branch():
    # No kr_risk_extra/debate_sizing_context passed at all (existing callers/
    # tests) must be byte-identical to explicitly passing both as "".
    plan = _held_plan(PortfolioRating.HOLD)
    proposal = _tp(TraderAction.HOLD)
    assert pipeline._desk_consensus_note(plan, proposal) == pipeline._desk_consensus_note(plan, proposal, "", "")


def test_desk_consensus_note_neither_supplied_keeps_original_sentence_byte_identical():
    plan = _held_plan(PortfolioRating.BUY)  # avoid the second-paragraph branch
    proposal = _tp(TraderAction.HOLD)
    note = pipeline._desk_consensus_note(plan, proposal, "", "")
    assert (
        "reports and the bull/bear debate; the risk debate below started from the "
        "Trader's proposal and saw nothing the Trader did not. Weigh the risk debate "
        "against these two, not in place of them."
    ) in note


def test_desk_consensus_note_both_supplied_drops_saw_nothing_claim():
    plan = _held_plan(PortfolioRating.BUY)
    proposal = _tp(TraderAction.HOLD)
    note = pipeline._desk_consensus_note(plan, proposal, "krx signals here", "sizing context here")
    assert "saw nothing the Trader did not" not in note
    assert "KRX short-selling and program-flow signals for this ticker and the position-sizing" in note
    assert "neither of which the Trader received" in note
    assert "can qualify as a new fact" in note
    assert "never qualify" in note


def test_desk_consensus_note_krx_only_branch():
    plan = _held_plan(PortfolioRating.BUY)
    proposal = _tp(TraderAction.HOLD)
    note = pipeline._desk_consensus_note(plan, proposal, "krx signals here", "")
    assert "saw nothing the Trader did not" not in note
    assert "KRX short-selling and program-flow signals for this ticker, which the Trader did not receive" in note
    assert "can qualify as a new fact" in note
    assert "position-sizing mechanics" not in note


def test_desk_consensus_note_sizing_only_branch_never_qualifies():
    plan = _held_plan(PortfolioRating.BUY)
    proposal = _tp(TraderAction.HOLD)
    note = pipeline._desk_consensus_note(plan, proposal, "", "sizing context here")
    assert "saw nothing the Trader did not" not in note
    assert "position-sizing mechanics (ATR volatility and the 1% risk-cap arithmetic), which the Trader did not receive" in note
    assert "never qualify as a new fact" in note
    assert "KRX short-selling" not in note


@pytest.mark.parametrize(
    "kr_risk_extra,debate_sizing_context",
    [("", ""), ("krx signals here", ""), ("", "sizing context here"), ("krx signals here", "sizing context here")],
)
def test_desk_consensus_note_second_paragraph_unchanged_across_all_branches(kr_risk_extra, debate_sizing_context):
    plan = _held_plan(PortfolioRating.HOLD)
    proposal = _tp(TraderAction.HOLD)
    note = pipeline._desk_consensus_note(plan, proposal, kr_risk_extra, debate_sizing_context)
    assert note.endswith(_SECOND_PARAGRAPH)


# ---------------------------------------------------------------------------
# _apply_unjustified_exit_gate  (2026-09-09 P0-2/3 incident, id=1358)
# ---------------------------------------------------------------------------

_MISUSE_SPAN = ["portfolio_manager: sizing cap cited as exit rationale — 'the risk cap forces the exit'"]


def _uxg(decision, plan, proposal, *, held=True, atr_pct=0.08, current_weight=0.05, misuse_spans=None,
         drawdown_frac=0.0, execution_blocked=False, forced_exit_on_book=False, record=None):
    return pipeline._apply_unjustified_exit_gate(
        decision, plan, proposal, held=held, atr_pct=atr_pct, current_weight=current_weight,
        misuse_spans=misuse_spans if misuse_spans is not None else [], drawdown_frac=drawdown_frac,
        execution_blocked=execution_blocked, forced_exit_on_book=forced_exit_on_book,
        record_out=record if record is not None else {},
    )


def test_unjustified_exit_gate_can_be_disabled(monkeypatch):
    monkeypatch.setattr(settings, "unjustified_exit_gate_enabled", False)
    record: dict = {}
    decision = _held_decision(PortfolioRating.SELL, confidence=0.3)
    result = _uxg(decision, _held_plan(PortfolioRating.UNDERWEIGHT), _tp(TraderAction.HOLD),
                  misuse_spans=_MISUSE_SPAN, record=record)
    assert result is decision
    assert record == {}


def test_unjustified_exit_gate_downgrades_held_sell_with_both_signatures():
    # Enabled by default as of the step-6 calibration.
    record: dict = {}
    decision = _held_decision(PortfolioRating.SELL, confidence=0.3)
    result = _uxg(decision, _held_plan(PortfolioRating.UNDERWEIGHT), _tp(TraderAction.HOLD),
                  misuse_spans=_MISUSE_SPAN, record=record)

    assert result.rating == PortfolioRating.HOLD
    assert record["tier"] == "both"
    assert record["gated"] is True
    assert result.executive_summary.startswith("[Unjustified exit gate:")
    assert result.executive_summary.count("gate: downgraded from") == 1
    assert "%" not in result.executive_summary  # number-free prefix
    assert "Unjustified exit gate" in result.investment_thesis


def test_unjustified_exit_gate_preserves_confidence_through_model_copy(monkeypatch):
    monkeypatch.setattr(settings, "unjustified_exit_gate_enabled", True)
    decision = _held_decision(PortfolioRating.SELL, confidence=0.42)
    result = _uxg(decision, _held_plan(PortfolioRating.UNDERWEIGHT), _tp(TraderAction.HOLD),
                  misuse_spans=_MISUSE_SPAN)
    assert result.confidence == 0.42


def test_unjustified_exit_gate_records_narrative_only_without_downgrading(monkeypatch):
    monkeypatch.setattr(settings, "unjustified_exit_gate_enabled", True)
    record: dict = {}
    decision = _held_decision(PortfolioRating.SELL, confidence=0.3)
    # mechanics fails: position weight far above the ATR-scaled Underweight target
    result = _uxg(decision, _held_plan(PortfolioRating.UNDERWEIGHT), _tp(TraderAction.HOLD),
                  current_weight=0.9, misuse_spans=_MISUSE_SPAN, record=record)
    assert result is decision
    assert record["tier"] == "narrative"
    assert record["gated"] is False


def test_unjustified_exit_gate_records_mechanics_only_without_downgrading():
    # RM Underweight (a trim) + position already at/below the ATR-scaled
    # Underweight target, but the PM text never cited the cap. The Trader is
    # at Sell — deliberately not an exemption (both real incidents were).
    record: dict = {}
    decision = _held_decision(PortfolioRating.SELL, confidence=0.3)
    result = _uxg(decision, _held_plan(PortfolioRating.UNDERWEIGHT), _tp(TraderAction.SELL),
                  current_weight=0.05, misuse_spans=[], record=record)
    assert result is decision
    assert record["tier"] == "mechanics"
    assert record["gated"] is False


def test_unjustified_exit_gate_mechanics_needs_research_manager_underweight():
    # RM Hold is not the "trim expanded to exit" pattern.
    record: dict = {}
    decision = _held_decision(PortfolioRating.SELL, confidence=0.3)
    result = _uxg(decision, _held_plan(PortfolioRating.HOLD), _tp(TraderAction.SELL),
                  current_weight=0.05, misuse_spans=[], record=record)
    assert result is decision
    assert record == {}


@pytest.mark.parametrize("rating", [PortfolioRating.HOLD, PortfolioRating.UNDERWEIGHT, PortfolioRating.OVERWEIGHT])
def test_unjustified_exit_gate_never_touches_a_non_sell(monkeypatch, rating):
    monkeypatch.setattr(settings, "unjustified_exit_gate_enabled", True)
    record: dict = {}
    decision = _held_decision(rating, confidence=0.3)
    result = _uxg(decision, _held_plan(PortfolioRating.UNDERWEIGHT), _tp(TraderAction.HOLD),
                  misuse_spans=_MISUSE_SPAN, record=record)
    assert result is decision
    assert record == {}


def test_unjustified_exit_gate_noop_when_flat(monkeypatch):
    monkeypatch.setattr(settings, "unjustified_exit_gate_enabled", True)
    record: dict = {}
    decision = _held_decision(PortfolioRating.SELL, confidence=0.3)
    result = _uxg(decision, _held_plan(PortfolioRating.UNDERWEIGHT), _tp(TraderAction.HOLD),
                  held=False, misuse_spans=_MISUSE_SPAN, record=record)
    assert result is decision
    assert record == {}


@pytest.mark.parametrize("kwargs", [
    {"drawdown_frac": 0.5},
    {"execution_blocked": True},
    {"forced_exit_on_book": True},
])
def test_unjustified_exit_gate_hard_exceptions_never_gate(monkeypatch, kwargs):
    monkeypatch.setattr(settings, "unjustified_exit_gate_enabled", True)
    record: dict = {}
    decision = _held_decision(PortfolioRating.SELL, confidence=0.3)
    result = _uxg(decision, _held_plan(PortfolioRating.UNDERWEIGHT), _tp(TraderAction.HOLD),
                  misuse_spans=_MISUSE_SPAN, record=record, **kwargs)
    assert result is decision
    assert record == {}


def test_unjustified_exit_gate_mutually_exclusive_with_the_other_two(monkeypatch):
    # A held Sell runs consensus (no-op, not Buy/Overweight) -> this gate ->
    # hysteresis (no-op, Hold is not state-flipping): exactly one prefix.
    monkeypatch.setattr(settings, "unjustified_exit_gate_enabled", True)
    monkeypatch.setattr(settings, "hysteresis_min_confidence_to_flip", 0.6)
    plan, proposal = _held_plan(PortfolioRating.UNDERWEIGHT), _tp(TraderAction.HOLD)
    decision = _held_decision(PortfolioRating.SELL, confidence=0.3)

    result = pipeline._apply_consensus_override_gate(decision, plan, proposal, record_out={})
    result = _uxg(result, plan, proposal, misuse_spans=_MISUSE_SPAN)
    result = pipeline._apply_hysteresis_gate(result, held=True)

    assert result.rating == PortfolioRating.HOLD
    assert result.executive_summary.count("gate: downgraded from") == 1
    assert result.executive_summary.startswith("[Unjustified exit gate:")
