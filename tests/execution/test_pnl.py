"""app.execution.pnl — pure weighted-average realized-P&L matching (no I/O,
no broker/DB). Same style as tests/execution/test_reconciliation.py: hand-built
BrokerExecution fixtures, no FakeBroker needed since compute_realized_pnl takes
already-fetched execution lists."""

from __future__ import annotations

from datetime import date

from app.domain.models import BrokerExecution
from app.execution.pnl import compute_realized_pnl, earliest_fetchable_date


def _execution(code="005930", side="buy", order_date="20260805", filled_qty=10, avg_price=70_000.0,
               order_no="1", name="삼성전자", fee_tax=0.0) -> BrokerExecution:
    return BrokerExecution(
        order_date=order_date, order_no=order_no, code=code, name=name, side=side,
        order_qty=filled_qty, filled_qty=filled_qty, remaining_qty=0, avg_price=avg_price,
        filled_amount=filled_qty * avg_price, fee_tax=fee_tax,
    )


def test_earliest_fetchable_date_is_ninety_days_back():
    assert earliest_fetchable_date(date(2026, 8, 5)) == date(2026, 5, 7)


async def test_single_buy_then_sell_computes_realized_pnl():
    buy = _execution(side="buy", order_date="20260701", filled_qty=10, avg_price=70_000.0, order_no="1")
    sell = _execution(side="sell", order_date="20260710", filled_qty=10, avg_price=75_000.0, order_no="2")
    trades, summary = await compute_realized_pnl(
        date(2026, 7, 1), date(2026, 7, 31), domestic_executions=[buy, sell], overseas_executions=[],
    )
    assert len(trades) == 1
    t = trades[0]
    assert t.ticker == "005930.KS"
    assert t.buy_avg_price == 70_000.0
    assert t.sell_price == 75_000.0
    assert t.shares == 10
    assert t.realized_pnl == 50_000.0  # (75000-70000)*10
    assert t.realized_pnl_pct == 50_000.0 / 700_000.0
    assert summary.total_pnl_domestic == 50_000.0
    assert summary.win_count == 1
    assert summary.loss_count == 0
    assert summary.win_rate == 1.0


async def test_partial_sell_leaves_remaining_shares_for_a_later_sell():
    buy = _execution(side="buy", order_date="20260701", filled_qty=10, avg_price=70_000.0, order_no="1")
    sell1 = _execution(side="sell", order_date="20260710", filled_qty=4, avg_price=75_000.0, order_no="2")
    sell2 = _execution(side="sell", order_date="20260715", filled_qty=6, avg_price=80_000.0, order_no="3")
    trades, summary = await compute_realized_pnl(
        date(2026, 7, 1), date(2026, 7, 31), domestic_executions=[buy, sell1, sell2], overseas_executions=[],
    )
    assert len(trades) == 2
    assert trades[0].shares == 4 and trades[0].realized_pnl == (75_000.0 - 70_000.0) * 4
    assert trades[1].shares == 6 and trades[1].realized_pnl == (80_000.0 - 70_000.0) * 6
    assert summary.win_count == 2


async def test_multiple_buy_lots_use_weighted_average_cost():
    buy1 = _execution(side="buy", order_date="20260701", filled_qty=10, avg_price=60_000.0, order_no="1")
    buy2 = _execution(side="buy", order_date="20260705", filled_qty=10, avg_price=80_000.0, order_no="2")
    # weighted avg = (10*60000 + 10*80000) / 20 = 70000
    sell = _execution(side="sell", order_date="20260710", filled_qty=20, avg_price=90_000.0, order_no="3")
    trades, _ = await compute_realized_pnl(
        date(2026, 7, 1), date(2026, 7, 31), domestic_executions=[buy1, buy2, sell], overseas_executions=[],
    )
    assert len(trades) == 1
    assert trades[0].buy_avg_price == 70_000.0
    assert trades[0].realized_pnl == (90_000.0 - 70_000.0) * 20


async def test_buy_before_query_window_still_sets_cost_basis():
    # The buy happened before `start` (simulating a fetch that started
    # earlier, at earliest_fetchable_date, than the caller's own window) —
    # it must still be used to compute the average cost of a sell that DOES
    # land inside [start, end].
    buy = _execution(side="buy", order_date="20260601", filled_qty=10, avg_price=50_000.0, order_no="1")
    sell = _execution(side="sell", order_date="20260710", filled_qty=10, avg_price=60_000.0, order_no="2")
    trades, _ = await compute_realized_pnl(
        date(2026, 7, 1), date(2026, 7, 31), domestic_executions=[buy, sell], overseas_executions=[],
    )
    assert len(trades) == 1
    assert trades[0].buy_avg_price == 50_000.0
    assert trades[0].realized_pnl == (60_000.0 - 50_000.0) * 10


async def test_sell_outside_date_range_is_excluded():
    buy = _execution(side="buy", order_date="20260601", filled_qty=10, avg_price=50_000.0, order_no="1")
    sell_before_window = _execution(side="sell", order_date="20260615", filled_qty=10, avg_price=55_000.0, order_no="2")
    trades, summary = await compute_realized_pnl(
        date(2026, 7, 1), date(2026, 7, 31), domestic_executions=[buy, sell_before_window], overseas_executions=[],
    )
    assert trades == []
    assert summary.total_pnl_domestic == 0.0
    assert summary.win_rate is None


async def test_unfilled_order_is_ignored():
    buy = _execution(side="buy", order_date="20260701", filled_qty=10, avg_price=70_000.0, order_no="1")
    cancelled = _execution(side="sell", order_date="20260705", filled_qty=0, avg_price=0.0, order_no="2")
    real_sell = _execution(side="sell", order_date="20260710", filled_qty=10, avg_price=75_000.0, order_no="3")
    trades, _ = await compute_realized_pnl(
        date(2026, 7, 1), date(2026, 7, 31), domestic_executions=[buy, cancelled, real_sell], overseas_executions=[],
    )
    assert len(trades) == 1
    assert trades[0].shares == 10


async def test_a_loss_decrements_loss_count_not_win_count():
    buy = _execution(side="buy", order_date="20260701", filled_qty=10, avg_price=70_000.0, order_no="1")
    sell = _execution(side="sell", order_date="20260710", filled_qty=10, avg_price=65_000.0, order_no="2")
    trades, summary = await compute_realized_pnl(
        date(2026, 7, 1), date(2026, 7, 31), domestic_executions=[buy, sell], overseas_executions=[],
    )
    assert trades[0].realized_pnl == -50_000.0
    assert summary.win_count == 0
    assert summary.loss_count == 1
    assert summary.win_rate == 0.0


async def test_overseas_ticker_uses_bare_code_and_usd_currency():
    buy = _execution(code="AAPL", side="buy", order_date="20260701", filled_qty=5, avg_price=150.0, order_no="1")
    sell = _execution(code="AAPL", side="sell", order_date="20260710", filled_qty=5, avg_price=160.0, order_no="2")
    trades, summary = await compute_realized_pnl(
        date(2026, 7, 1), date(2026, 7, 31), domestic_executions=[], overseas_executions=[buy, sell],
    )
    assert len(trades) == 1
    assert trades[0].ticker == "AAPL"
    assert trades[0].currency == "USD"
    assert trades[0].market == "overseas"
    assert summary.total_pnl_overseas == 50.0
    assert summary.total_pnl_domestic == 0.0


async def test_fee_and_tax_reduce_net_pnl_but_not_gross_realized_pnl():
    buy = _execution(side="buy", order_date="20260701", filled_qty=10, avg_price=70_000.0, order_no="1",
                      fee_tax=350.0)  # buy-side commission only, no securities transaction tax
    sell = _execution(side="sell", order_date="20260710", filled_qty=10, avg_price=75_000.0, order_no="2",
                       fee_tax=1_500.0)  # sell-side commission + 증권거래세
    trades, summary = await compute_realized_pnl(
        date(2026, 7, 1), date(2026, 7, 31), domestic_executions=[buy, sell], overseas_executions=[],
    )
    assert len(trades) == 1
    t = trades[0]
    assert t.realized_pnl == 50_000.0  # gross, unaffected by fee/tax
    assert t.fee_tax == 350.0 + 1_500.0
    assert t.net_pnl == 50_000.0 - 1_850.0
    assert t.net_pnl_pct == t.net_pnl / 700_000.0
    assert summary.total_fee_tax_domestic == 1_850.0
    assert summary.total_net_pnl_domestic == 50_000.0 - 1_850.0


async def test_partial_sell_attributes_proportional_fee_and_buy_fee_per_share():
    # buy fee spread evenly per share; sell fee attributed to matched_qty's
    # share of the sell fill when the fill is only partially matched.
    buy = _execution(side="buy", order_date="20260701", filled_qty=10, avg_price=70_000.0, order_no="1",
                      fee_tax=100.0)  # 10/share
    sell = _execution(side="sell", order_date="20260710", filled_qty=4, avg_price=75_000.0, order_no="2",
                       fee_tax=40.0)  # full fee for this 4-share sell fill
    trades, _ = await compute_realized_pnl(
        date(2026, 7, 1), date(2026, 7, 31), domestic_executions=[buy, sell], overseas_executions=[],
    )
    assert len(trades) == 1
    t = trades[0]
    assert t.fee_tax == 10.0 * 4 + 40.0  # buy_fee_per_share * matched_qty + full sell fee (fully matched)
    assert t.net_pnl == t.realized_pnl - t.fee_tax


# No unscoped total_pnl/total_fee_tax/total_net_pnl (removed 2026-08-06) —
# domestic is KRW, overseas is USD, and summing them across both markets
# produced a mixed-currency figure with no real unit. Confirms the
# dataclass has no such field left to accidentally reintroduce, and that
# both markets' totals (pnl, fee/tax, and net) stay independently correct
# when both are present in the same call.
def test_pnl_summary_has_no_unscoped_currency_mixed_fields():
    import dataclasses

    from app.execution.pnl import PnlSummary

    field_names = {f.name for f in dataclasses.fields(PnlSummary)}
    assert "total_pnl" not in field_names
    assert "total_fee_tax" not in field_names
    assert "total_net_pnl" not in field_names


async def test_domestic_and_overseas_totals_stay_independent_in_one_call():
    domestic_buy = _execution(code="005930", side="buy", order_date="20260701", filled_qty=10,
                               avg_price=70_000.0, order_no="1", fee_tax=100.0)
    domestic_sell = _execution(code="005930", side="sell", order_date="20260710", filled_qty=10,
                                avg_price=75_000.0, order_no="2", fee_tax=150.0)
    overseas_buy = _execution(code="AAPL", side="buy", order_date="20260701", filled_qty=5,
                               avg_price=150.0, order_no="1", fee_tax=1.0)
    overseas_sell = _execution(code="AAPL", side="sell", order_date="20260710", filled_qty=5,
                                avg_price=160.0, order_no="2", fee_tax=1.0)
    _, summary = await compute_realized_pnl(
        date(2026, 7, 1), date(2026, 7, 31),
        domestic_executions=[domestic_buy, domestic_sell], overseas_executions=[overseas_buy, overseas_sell],
    )
    assert summary.total_pnl_domestic == 50_000.0
    assert summary.total_pnl_overseas == 50.0
    assert summary.total_fee_tax_domestic == 250.0
    assert summary.total_fee_tax_overseas == 2.0
    assert summary.total_net_pnl_domestic == 50_000.0 - 250.0
    assert summary.total_net_pnl_overseas == 50.0 - 2.0


# _order_no_sort_key (2026-08-06): int(e.order_no or 0) used to raise
# ValueError on any non-numeric order_no, crashing the whole /api/pnl
# response for every user over a single bad execution record.
def test_order_no_sort_key_falls_back_to_zero_on_non_numeric_value():
    from app.execution.pnl import _order_no_sort_key

    assert _order_no_sort_key("42") == 42
    assert _order_no_sort_key(None) == 0
    assert _order_no_sort_key("") == 0
    assert _order_no_sort_key("not-a-number") == 0


async def test_match_realized_trades_tolerates_a_non_numeric_order_no():
    buy = _execution(side="buy", order_date="20260701", filled_qty=10, avg_price=70_000.0, order_no="abc123")
    sell = _execution(side="sell", order_date="20260710", filled_qty=10, avg_price=75_000.0, order_no="2")
    trades, _ = await compute_realized_pnl(
        date(2026, 7, 1), date(2026, 7, 31), domestic_executions=[buy, sell], overseas_executions=[],
    )
    assert len(trades) == 1
    assert trades[0].realized_pnl == 50_000.0


# fee_measured / fee_unavailable_overseas (2026-09-08 fee/tax audit): the
# overseas execution TR never carries a fee/tax figure at all (BrokerExecution.
# fee_tax's own docstring), so fee_tax=0.0/net_pnl==realized_pnl for an
# overseas row means "never measured", not "this trade was free" — these
# flags let a caller (app.web.routers.portfolio) tell the two apart instead
# of silently presenting a gross number as if it were net.
async def test_domestic_trade_is_marked_fee_measured():
    buy = _execution(side="buy", order_date="20260701", filled_qty=10, avg_price=70_000.0, order_no="1")
    sell = _execution(side="sell", order_date="20260710", filled_qty=10, avg_price=75_000.0, order_no="2")
    trades, summary = await compute_realized_pnl(
        date(2026, 7, 1), date(2026, 7, 31), domestic_executions=[buy, sell], overseas_executions=[],
    )
    assert trades[0].fee_measured is True
    assert summary.fee_unavailable_overseas is False


async def test_overseas_trade_is_marked_fee_unmeasured():
    buy = _execution(code="AAPL", side="buy", order_date="20260701", filled_qty=5, avg_price=150.0, order_no="1")
    sell = _execution(code="AAPL", side="sell", order_date="20260710", filled_qty=5, avg_price=160.0, order_no="2")
    trades, summary = await compute_realized_pnl(
        date(2026, 7, 1), date(2026, 7, 31), domestic_executions=[], overseas_executions=[buy, sell],
    )
    assert trades[0].fee_measured is False
    assert summary.fee_unavailable_overseas is True


async def test_fee_unavailable_overseas_stays_false_when_only_domestic_present():
    buy = _execution(side="buy", order_date="20260701", filled_qty=10, avg_price=70_000.0, order_no="1")
    sell = _execution(side="sell", order_date="20260710", filled_qty=10, avg_price=75_000.0, order_no="2")
    _, summary = await compute_realized_pnl(
        date(2026, 7, 1), date(2026, 7, 31), domestic_executions=[buy, sell], overseas_executions=[],
    )
    assert summary.fee_unavailable_overseas is False


async def test_sell_with_no_prior_position_on_record_is_skipped():
    # A sell for a ticker whose buy fell outside the fetched execution
    # window entirely (even earlier than the TR's own cap) — nothing to
    # match it against, so it must not fabricate a phantom realized number.
    sell_only = _execution(side="sell", order_date="20260710", filled_qty=10, avg_price=75_000.0, order_no="1")
    trades, summary = await compute_realized_pnl(
        date(2026, 7, 1), date(2026, 7, 31), domestic_executions=[sell_only], overseas_executions=[],
    )
    assert trades == []
    assert summary.total_pnl_domestic == 0.0
