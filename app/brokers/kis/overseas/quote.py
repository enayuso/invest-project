import logging

from app.brokers.kis.const import KISBrokerError
from app.brokers.kis.parser import _as_rows, _kis_datetime_to_iso, _num
from app.domain.models import BidAsk, Candle, OverseasValuation

logger = logging.getLogger(__name__)

_OVERSEAS_PRICE_TR = "HHDFS00000300"  # 해외주식 현재체결가 (실전/모의 공통)
_OVERSEAS_PRICE_PATH = "/uapi/overseas-price/v1/quotations/price"
_OVERSEAS_ASKING_PRICE_TR = "HHDFS76200100"  # 해외주식 현재가 1호가 (실전/모의 공통)
_OVERSEAS_ASKING_PRICE_PATH = "/uapi/overseas-price/v1/quotations/inquire-asking-price"
_OVERSEAS_CANDLES_TR = "HHDFS76240000"  # 해외주식 기간별시세 (실전/모의 공통)
_OVERSEAS_CANDLES_PATH = "/uapi/overseas-price/v1/quotations/dailyprice"
_OVERSEAS_CANDLE_GUBN = {"D": "0", "W": "1", "M": "2"}  # no 년봉 option overseas
_OVERSEAS_NEWS_TR = "HHPSTH60100C1"  # 해외뉴스종합(제목) [해외주식-053]
_OVERSEAS_NEWS_PATH = "/uapi/overseas-price/v1/quotations/news-title"
_OVERSEAS_PRICE_DETAIL_TR = "HHDFS76200200"  # 해외주식 현재가상세 [해외주식-029]
_OVERSEAS_PRICE_DETAIL_PATH = "/uapi/overseas-price/v1/quotations/price-detail"


class OverseasQuoteMixin:
    async def fetch_overseas_price(self, symbol: str, exchange: str = "NAS") -> float:
        """해외주식 현재체결가 — note the quote TR's exchange codes (NAS/NYS/
        AMS) differ from the trading TRs' (NASD/NYSE/AMEX)."""
        resp = await self._request(
            "GET", _OVERSEAS_PRICE_PATH, _OVERSEAS_PRICE_TR,
            params={"AUTH": "", "EXCD": exchange, "SYMB": symbol},
        )
        body = self._checked_body(resp, f"overseas price inquiry for {symbol}")
        price = _num(body.get("output", {}).get("last"))
        if price <= 0:
            raise KISBrokerError(f"overseas price inquiry for {symbol} returned no price")
        return price

    async def fetch_overseas_asking_price(self, symbol: str, exchange: str = "NAS") -> BidAsk:
        """해외주식 현재가 1호가 — best bid/ask for a liquidity/spread
        pre-filter, same "not account-mode gated" property as the domestic
        sibling (fetch_asking_price)."""
        resp = await self._request(
            "GET", _OVERSEAS_ASKING_PRICE_PATH, _OVERSEAS_ASKING_PRICE_TR,
            params={"AUTH": "", "EXCD": exchange, "SYMB": symbol},
        )
        body = self._checked_body(resp, f"overseas asking price inquiry for {symbol}")
        # KIS's official COLUMN_MAPPING for this TR (HHDFS76200100) splits fields
        # across output1 (last/open/high/low/volume), output2 (pbid1/pask1/rclose),
        # and output3 (HK CAS/VCM only). The docs list them under one flat mapping
        # without saying which block, and the old code read pbid1/pask1 from
        # output1 — where they don't exist — so bid/ask came back 0 and
        # spread_pct() returned None, silently disabling the overseas spread
        # filter. Merge all three blocks (shallow, later wins) so the read works
        # regardless of which block KIS actually populates.
        out: dict = {}
        for key in ("output1", "output2", "output3"):
            block = body.get(key)
            if isinstance(block, dict):
                out.update(block)
        bid = _num(out.get("pbid1"))
        ask = _num(out.get("pask1"))
        if bid <= 0 or ask <= 0:
            logger.warning(
                "overseas asking price for %s has no bid/ask fields "
                "(bid=%s ask=%s) — spread filter passes silently for this name",
                symbol, bid, ask,
            )
        return BidAsk(
            code=symbol,
            bid=bid,
            ask=ask,
            last=_num(out.get("last")),
        )

    async def fetch_overseas_candles(
        self, symbol: str, end: str = "", exchange: str = "NAS", period: str = "D"
    ) -> list[Candle]:
        """해외주식 기간별시세 — daily/weekly/monthly OHLCV ending at `end`
        (YYYYMMDD, "" = most recent), oldest first. No 년봉 (yearly) option,
        unlike the domestic TR — only D/W/M. OHLCV lives in `output2`;
        `output1` here is just a header (rsym/zdiv/nrec), not the quote — an
        easy-to-miss KIS quirk. Single call, same "caller widens the range /
        pages via tr_cont for more history" convention as fetch_daily_candles."""
        if period not in _OVERSEAS_CANDLE_GUBN:
            raise ValueError(f"period must be D/W/M for overseas, got {period!r}")
        resp = await self._request(
            "GET", _OVERSEAS_CANDLES_PATH, _OVERSEAS_CANDLES_TR,
            params={
                "AUTH": "", "EXCD": exchange, "SYMB": symbol,
                "GUBN": _OVERSEAS_CANDLE_GUBN[period], "BYMD": end, "MODP": "1",
            },
        )
        body = self._checked_body(resp, f"overseas daily candles for {symbol}")
        candles = [
            Candle(
                day=row.get("xymd", ""),
                open=_num(row.get("open")),
                high=_num(row.get("high")),
                low=_num(row.get("low")),
                close=_num(row.get("clos")),
                volume=int(_num(row.get("tvol"))),
            )
            for row in body.get("output2", [])
            if row.get("xymd")
        ]
        candles.sort(key=lambda c: c.day)
        return candles

    async def fetch_overseas_valuation(self, symbol: str, exchange: str = "NAS") -> OverseasValuation:
        """해외주식 현재가상세 — PER/PBR/시가총액/52주 고저가 한 번에. 국내
        fetch_valuation과 짝이지만 별도 TR(다른 필드셋)이라 별도 메서드로 둠."""
        resp = await self._request(
            "GET", _OVERSEAS_PRICE_DETAIL_PATH, _OVERSEAS_PRICE_DETAIL_TR,
            params={"AUTH": "", "EXCD": exchange, "SYMB": symbol},
        )
        body = self._checked_body(resp, f"overseas valuation inquiry for {symbol}")
        out = body.get("output", {})
        return OverseasValuation(
            symbol=symbol,
            price=_num(out.get("last")),
            per=_num(out.get("perx")),
            pbr=_num(out.get("pbrx")),
            market_cap=_num(out.get("tomv")),
            week52_high=_num(out.get("h52p")),
            week52_low=_num(out.get("l52p")),
        )

    async def fetch_overseas_news(
        self, symbol: str = "", nation_cd: str = "US", exchange: str = "NAS", limit: int = 10
    ) -> list[dict]:
        """해외뉴스종합(제목) — KIS-only news for the frontend, overseas side
        of DomesticQuoteMixin.fetch_news (see that method's docstring for
        the yfinance-fallback policy this replaces). Title/source/time
        only — this KIS TR has no article link or thumbnail either.

        EXCHANGE_CD must be a real exchange (not "") for the SYMB filter to
        actually narrow results — confirmed by hand: blank EXCHANGE_CD with
        a SYMB set returns an empty outblock1 even when that symbol clearly
        has news (verified against AAPL), same NAS/NYS/AMS codes as
        fetch_overseas_price/candles."""
        resp = await self._request(
            "GET", _OVERSEAS_NEWS_PATH, _OVERSEAS_NEWS_TR,
            params={
                "INFO_GB": "", "CLASS_CD": "", "NATION_CD": nation_cd,
                "EXCHANGE_CD": exchange, "SYMB": symbol,
                "DATA_DT": "", "DATA_TM": "", "CTS": "",
            },
        )
        body = self._checked_body(resp, f"overseas news inquiry for {symbol or '(all)'}")
        rows = _as_rows(body.get("outblock1"))
        items = []
        for row in rows[:limit]:
            title = row.get("title", "")
            if not title:
                continue
            items.append({
                "title": title,
                "publisher": row.get("source", ""),
                "published_at": _kis_datetime_to_iso(row.get("data_dt", ""), row.get("data_tm", "")),
            })
        return items
