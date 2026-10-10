"""app.live.universe.krx_top_n / overseas_top_n — broker faked, no network."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.live import ticker_names, universe


@dataclass
class _FakeRankedStock:
    rank: int
    code: str
    name: str = ""
    price: float = 0.0
    change_pct: float = 0.0
    volume: int = 0


class _FakeBroker:
    def __init__(self, rows):
        self._rows = rows
        self.trading_value_rank_calls = []
        self.market_cap_rank_calls = []
        self.fluctuation_rank_calls = []
        self.disparity_rank_calls = []

    async def fetch_trading_value_rank(self, count, *, max_price=None):
        self.trading_value_rank_calls.append({"count": count, "max_price": max_price})
        return self._rows[:count]

    async def fetch_market_cap_rank(self, count, *, max_price=None):
        self.market_cap_rank_calls.append({"count": count, "max_price": max_price})
        return self._rows[:count]

    async def fetch_overseas_trading_value_rank(self, exchange="NAS", count=30):
        return self._rows[:count]

    async def fetch_overseas_market_cap_rank(self, exchange="NAS", count=30):
        return self._rows[:count]

    async def fetch_fluctuation_rank(self, rising=True, count=30, *, min_volume=0, max_price=None):
        self.fluctuation_rank_calls.append(
            {"rising": rising, "count": count, "min_volume": min_volume, "max_price": max_price}
        )
        return self._rows[:count]

    async def fetch_disparity_rank(self, *, low=True, window_days=20, count=30, min_volume=0, max_price=None):
        self.disparity_rank_calls.append(
            {"low": low, "window_days": window_days, "count": count, "min_volume": min_volume, "max_price": max_price}
        )
        return self._rows[:count]


async def test_krx_top_n_returns_ks_suffixed_tickers():
    broker = _FakeBroker([_FakeRankedStock(rank=1, code="005930"), _FakeRankedStock(rank=2, code="000660")])
    result = await universe.krx_top_n(broker, count=2)
    assert result == ["005930.KS", "000660.KS"]


async def test_krx_top_n_respects_count():
    broker = _FakeBroker([_FakeRankedStock(rank=i, code=str(i).zfill(6)) for i in range(1, 6)])
    result = await universe.krx_top_n(broker, count=3)
    assert len(result) == 3


async def test_krx_top_n_returns_empty_when_broker_none():
    assert await universe.krx_top_n(None, count=10) == []


async def test_krx_top_n_returns_empty_and_logs_on_failure():
    class _BrokenBroker:
        async def fetch_trading_value_rank(self, count):
            raise RuntimeError("KIS API down")

    result = await universe.krx_top_n(_BrokenBroker(), count=10)
    assert result == []


async def test_krx_top_n_skips_rows_with_empty_code():
    broker = _FakeBroker([_FakeRankedStock(rank=1, code=""), _FakeRankedStock(rank=2, code="005930")])
    result = await universe.krx_top_n(broker, count=2)
    assert result == ["005930.KS"]


async def test_krx_top_n_skips_non_standard_codes():
    # ELW/warrant-style codes (letters mixed in, or not 6 digits) pass the
    # name-based leverage filter fine since their names look like ordinary
    # products — this is the code-shape check that catches what the name
    # check can't.
    broker = _FakeBroker([
        _FakeRankedStock(rank=1, code="0156T0", name="Some ELW"),
        _FakeRankedStock(rank=2, code="12345", name="Too short"),
        _FakeRankedStock(rank=3, code="005930", name="삼성전자"),
    ])
    result = await universe.krx_top_n(broker, count=3)
    assert result == ["005930.KS"]


async def test_krx_top_n_remembers_korean_name_for_display(monkeypatch):
    monkeypatch.setattr(ticker_names, "_cache", {})
    broker = _FakeBroker([_FakeRankedStock(rank=1, code="005930", name="삼성전자")])
    await universe.krx_top_n(broker, count=1)
    assert ticker_names.get("005930.KS") == "삼성전자"


async def test_krx_market_cap_top_n_returns_ks_suffixed_tickers():
    broker = _FakeBroker([_FakeRankedStock(rank=1, code="005930"), _FakeRankedStock(rank=2, code="000660")])
    result = await universe.krx_market_cap_top_n(broker, count=2)
    assert result == ["005930.KS", "000660.KS"]


async def test_krx_market_cap_top_n_returns_empty_when_broker_none():
    assert await universe.krx_market_cap_top_n(None, count=10) == []


async def test_krx_market_cap_top_n_returns_empty_and_logs_on_failure():
    class _BrokenBroker:
        async def fetch_market_cap_rank(self, count):
            raise RuntimeError("KIS API down")

    result = await universe.krx_market_cap_top_n(_BrokenBroker(), count=10)
    assert result == []


async def test_overseas_top_n_returns_bare_symbols_no_suffix():
    broker = _FakeBroker([_FakeRankedStock(rank=1, code="AAPL"), _FakeRankedStock(rank=2, code="MSFT")])
    result = await universe.overseas_top_n(broker, count=2)
    assert result == ["AAPL", "MSFT"]  # no .KS-style suffix appended, unlike krx_top_n


async def test_overseas_market_cap_top_n_returns_bare_symbols_no_suffix():
    broker = _FakeBroker([_FakeRankedStock(rank=1, code="AAPL"), _FakeRankedStock(rank=2, code="MSFT")])
    result = await universe.overseas_market_cap_top_n(broker, count=2)
    assert result == ["AAPL", "MSFT"]


async def test_overseas_market_cap_top_n_returns_empty_when_broker_none():
    assert await universe.overseas_market_cap_top_n(None, count=10) == []


async def test_overseas_market_cap_top_n_returns_empty_and_logs_on_failure():
    class _BrokenBroker:
        async def fetch_overseas_market_cap_rank(self, exchange="NAS", count=30):
            raise RuntimeError("KIS API down")

    result = await universe.overseas_market_cap_top_n(_BrokenBroker(), count=10)
    assert result == []


async def test_overseas_market_cap_top_n_filters_leveraged_names():
    broker = _FakeBroker([
        _FakeRankedStock(rank=1, code="AAPL", name="Apple Inc."),
        _FakeRankedStock(rank=2, code="TQQQ", name="ProShares UltraPro QQQ"),
    ])
    result = await universe.overseas_market_cap_top_n(broker, count=2)
    assert result == ["AAPL"]


async def test_overseas_top_n_returns_empty_when_broker_none():
    assert await universe.overseas_top_n(None, count=10) == []


async def test_overseas_top_n_returns_empty_and_logs_on_failure():
    class _BrokenBroker:
        async def fetch_overseas_trading_value_rank(self, exchange="NAS", count=30):
            raise RuntimeError("KIS API down")

    result = await universe.overseas_top_n(_BrokenBroker(), count=10)
    assert result == []


async def test_overseas_top_n_skips_rows_with_empty_code():
    broker = _FakeBroker([_FakeRankedStock(rank=1, code=""), _FakeRankedStock(rank=2, code="AAPL")])
    result = await universe.overseas_top_n(broker, count=2)
    assert result == ["AAPL"]


# ---------------------------------------------------------------------------
# Cash-like / bond fund exclusion (2026-09-01) — app.domain.product_filter.
# These rank high in KRX 거래대금 (institutions park cash in them) so the
# discovery feed would surface them constantly; 481050.KS reached the
# analysts this way and every agent misread a CD-rate ETF as a credit
# product.
# ---------------------------------------------------------------------------

async def test_krx_top_n_drops_cash_like_etf():
    broker = _FakeBroker([
        _FakeRankedStock(rank=1, code="481050", name="KODEX CD1년금리플러스액티브(합성)"),
        _FakeRankedStock(rank=2, code="005930", name="삼성전자"),
    ])
    result = await universe.krx_top_n(broker, count=2)
    assert result == ["005930.KS"]


async def test_krx_market_cap_top_n_drops_bond_etf():
    broker = _FakeBroker([
        _FakeRankedStock(rank=1, code="439870", name="KODEX 국고채30년액티브"),
        _FakeRankedStock(rank=2, code="000660", name="SK하이닉스"),
    ])
    result = await universe.krx_market_cap_top_n(broker, count=2)
    assert result == ["000660.KS"]


async def test_krx_top_n_keeps_ordinary_equity_etf():
    broker = _FakeBroker([_FakeRankedStock(rank=1, code="069500", name="KODEX 200")])
    result = await universe.krx_top_n(broker, count=1)
    assert result == ["069500.KS"]


async def test_overseas_top_n_drops_bond_etf():
    broker = _FakeBroker([
        _FakeRankedStock(rank=1, code="TLT", name="ISHARES 20+Y TREASURY BOND"),
        _FakeRankedStock(rank=2, code="AAPL", name="Apple Inc."),
    ])
    result = await universe.overseas_top_n(broker, count=2)
    assert result == ["AAPL"]


async def test_krx_top_n_drops_covered_call_etf():
    # 2026-09-02 (audit id=1170): 498400.KS "KODEX 200타겟위클리커버드콜"
    # passed all 3 gates and burned a full ~17-LLM-call pass. Pure-equity
    # covered-call ETF — no bond keyword — now dropped via the new
    # is_derivative_income_fund predicate.
    broker = _FakeBroker([
        _FakeRankedStock(rank=1, code="498400", name="KODEX 200타겟위클리커버드콜"),
        _FakeRankedStock(rank=2, code="005930", name="삼성전자"),
    ])
    result = await universe.krx_top_n(broker, count=2)
    assert result == ["005930.KS"]


async def test_krx_top_n_covered_call_drop_does_not_over_block_plain_index_etf():
    broker = _FakeBroker([_FakeRankedStock(rank=1, code="069500", name="KODEX 200")])
    result = await universe.krx_top_n(broker, count=1)
    assert result == ["069500.KS"]


async def test_overseas_market_cap_top_n_drops_bond_etf():
    broker = _FakeBroker([
        _FakeRankedStock(rank=1, code="AGG", name="ISHARES CORE US AGGREGATE BOND"),
        _FakeRankedStock(rank=2, code="MSFT", name="Microsoft Corp."),
    ])
    result = await universe.overseas_market_cap_top_n(broker, count=2)
    assert result == ["MSFT"]


# ---------------------------------------------------------------------------
# krx_fluctuation_losers_n / krx_disparity_low_n (2026-09-02,
# screening-improvement-research #3) — two additional entry axes, both
# built on _krx_discovery_tickers (the same admission filter as krx_top_n/
# krx_market_cap_top_n), so the filtering-behavior tests below mirror
# krx_top_n's own test coverage rather than re-deriving it.
# ---------------------------------------------------------------------------


async def test_krx_fluctuation_losers_n_returns_ks_suffixed_tickers():
    broker = _FakeBroker([_FakeRankedStock(rank=1, code="005930"), _FakeRankedStock(rank=2, code="000660")])
    result = await universe.krx_fluctuation_losers_n(broker, count=2)
    assert result == ["005930.KS", "000660.KS"]


async def test_krx_fluctuation_losers_n_requests_the_losing_side():
    # Pins the axis semantics against an accidental future flip to gainers.
    broker = _FakeBroker([_FakeRankedStock(rank=1, code="005930")])
    await universe.krx_fluctuation_losers_n(broker, count=1)
    assert broker.fluctuation_rank_calls[0]["rising"] is False


async def test_krx_fluctuation_losers_n_passes_min_volume_floor_through():
    broker = _FakeBroker([_FakeRankedStock(rank=1, code="005930")])
    await universe.krx_fluctuation_losers_n(broker, count=1, min_volume=100_000)
    assert broker.fluctuation_rank_calls[0]["min_volume"] == 100_000


async def test_krx_fluctuation_losers_n_returns_empty_when_broker_none():
    assert await universe.krx_fluctuation_losers_n(None, count=10) == []


async def test_krx_fluctuation_losers_n_returns_empty_and_logs_on_failure():
    class _BrokenBroker:
        async def fetch_fluctuation_rank(self, rising=True, count=30, *, min_volume=0):
            raise RuntimeError("KIS API down")

    result = await universe.krx_fluctuation_losers_n(_BrokenBroker(), count=10)
    assert result == []


async def test_krx_fluctuation_losers_n_drops_leveraged_and_out_of_scope():
    broker = _FakeBroker([
        _FakeRankedStock(rank=1, code="122630", name="KODEX 레버리지"),
        _FakeRankedStock(rank=2, code="481050", name="KODEX CD1년금리플러스액티브(합성)"),
        _FakeRankedStock(rank=3, code="005930", name="삼성전자"),
    ])
    result = await universe.krx_fluctuation_losers_n(broker, count=3)
    assert result == ["005930.KS"]


async def test_krx_fluctuation_losers_n_skips_non_standard_codes():
    broker = _FakeBroker([
        _FakeRankedStock(rank=1, code="0156T0", name="Some ELW"),
        _FakeRankedStock(rank=2, code="005930", name="삼성전자"),
    ])
    result = await universe.krx_fluctuation_losers_n(broker, count=2)
    assert result == ["005930.KS"]


async def test_krx_disparity_low_n_returns_ks_suffixed_tickers():
    broker = _FakeBroker([_FakeRankedStock(rank=1, code="005930"), _FakeRankedStock(rank=2, code="000660")])
    result = await universe.krx_disparity_low_n(broker, count=2)
    assert result == ["005930.KS", "000660.KS"]


async def test_krx_disparity_low_n_requests_the_low_side():
    broker = _FakeBroker([_FakeRankedStock(rank=1, code="005930")])
    await universe.krx_disparity_low_n(broker, count=1)
    assert broker.disparity_rank_calls[0]["low"] is True


async def test_krx_disparity_low_n_passes_window_days_through():
    broker = _FakeBroker([_FakeRankedStock(rank=1, code="005930")])
    await universe.krx_disparity_low_n(broker, count=1, window_days=60)
    assert broker.disparity_rank_calls[0]["window_days"] == 60


async def test_krx_disparity_low_n_passes_min_volume_floor_through():
    broker = _FakeBroker([_FakeRankedStock(rank=1, code="005930")])
    await universe.krx_disparity_low_n(broker, count=1, min_volume=100_000)
    assert broker.disparity_rank_calls[0]["min_volume"] == 100_000


async def test_krx_disparity_low_n_returns_empty_when_broker_none():
    assert await universe.krx_disparity_low_n(None, count=10) == []


async def test_krx_disparity_low_n_returns_empty_and_logs_on_failure():
    class _BrokenBroker:
        async def fetch_disparity_rank(self, *, low=True, window_days=20, count=30, min_volume=0):
            raise RuntimeError("KIS API down")

    result = await universe.krx_disparity_low_n(_BrokenBroker(), count=10)
    assert result == []


async def test_krx_disparity_low_n_drops_leveraged_and_out_of_scope():
    broker = _FakeBroker([
        _FakeRankedStock(rank=1, code="122630", name="KODEX 레버리지"),
        _FakeRankedStock(rank=2, code="498400", name="KODEX 200타겟위클리커버드콜"),
        _FakeRankedStock(rank=3, code="005930", name="삼성전자"),
    ])
    result = await universe.krx_disparity_low_n(broker, count=3)
    assert result == ["005930.KS"]


async def test_krx_disparity_low_n_skips_non_standard_codes():
    broker = _FakeBroker([
        _FakeRankedStock(rank=1, code="0156T0", name="Some ELW"),
        _FakeRankedStock(rank=2, code="005930", name="삼성전자"),
    ])
    result = await universe.krx_disparity_low_n(broker, count=2)
    assert result == ["005930.KS"]


async def test_krx_disparity_low_n_remembers_korean_name_for_display(monkeypatch):
    monkeypatch.setattr(ticker_names, "_cache", {})
    broker = _FakeBroker([_FakeRankedStock(rank=1, code="005930", name="삼성전자")])
    await universe.krx_disparity_low_n(broker, count=1)
    assert ticker_names.get("005930.KS") == "삼성전자"


# ---------------------------------------------------------------------------
# max_price (2026-09-02, screening-improvement-research #3 follow-up) — an
# account-derived KIS-side price ceiling threaded through all four domestic
# discovery axes with zero interpretation in this module. See krx_top_n's
# own docstring for the full account-affordability story.
# ---------------------------------------------------------------------------


async def test_krx_top_n_threads_max_price_to_the_broker():
    broker = _FakeBroker([_FakeRankedStock(rank=1, code="005930")])
    await universe.krx_top_n(broker, count=1, max_price=400_000)
    assert broker.trading_value_rank_calls[0]["max_price"] == 400_000


async def test_krx_top_n_default_max_price_is_none():
    broker = _FakeBroker([_FakeRankedStock(rank=1, code="005930")])
    await universe.krx_top_n(broker, count=1)
    assert broker.trading_value_rank_calls[0]["max_price"] is None


async def test_krx_market_cap_top_n_threads_max_price_to_the_broker():
    broker = _FakeBroker([_FakeRankedStock(rank=1, code="005930")])
    await universe.krx_market_cap_top_n(broker, count=1, max_price=400_000)
    assert broker.market_cap_rank_calls[0]["max_price"] == 400_000


async def test_krx_market_cap_top_n_default_max_price_is_none():
    broker = _FakeBroker([_FakeRankedStock(rank=1, code="005930")])
    await universe.krx_market_cap_top_n(broker, count=1)
    assert broker.market_cap_rank_calls[0]["max_price"] is None


async def test_krx_fluctuation_losers_n_threads_max_price_to_the_broker():
    broker = _FakeBroker([_FakeRankedStock(rank=1, code="005930")])
    await universe.krx_fluctuation_losers_n(broker, count=1, max_price=400_000)
    assert broker.fluctuation_rank_calls[0]["max_price"] == 400_000


async def test_krx_fluctuation_losers_n_default_max_price_is_none():
    broker = _FakeBroker([_FakeRankedStock(rank=1, code="005930")])
    await universe.krx_fluctuation_losers_n(broker, count=1)
    assert broker.fluctuation_rank_calls[0]["max_price"] is None


async def test_krx_disparity_low_n_threads_max_price_to_the_broker():
    broker = _FakeBroker([_FakeRankedStock(rank=1, code="005930")])
    await universe.krx_disparity_low_n(broker, count=1, max_price=400_000)
    assert broker.disparity_rank_calls[0]["max_price"] == 400_000


async def test_krx_disparity_low_n_default_max_price_is_none():
    broker = _FakeBroker([_FakeRankedStock(rank=1, code="005930")])
    await universe.krx_disparity_low_n(broker, count=1)
    assert broker.disparity_rank_calls[0]["max_price"] is None
