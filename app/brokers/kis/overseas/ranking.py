# app/brokers/kis/overseas/ranking.py
# 해외주식 순위분석 — 국내와 대칭 기능(거래량/등락률), EXCD로 거래소 지정
from app.brokers.kis.parser import _ranked_rows
from app.domain.models import RankedStock

_OVERSEAS_VOLUME_RANK_TR = "HHDFS76310010"  # 해외주식 거래량순위
_OVERSEAS_VOLUME_RANK_PATH = "/uapi/overseas-stock/v1/ranking/trade-vol"
_OVERSEAS_VALUE_RANK_TR = "HHDFS76320010"  # 해외주식 거래대금순위 (별개 TR, 국내와 달리 파라미터 분기가 아님)
_OVERSEAS_VALUE_RANK_PATH = "/uapi/overseas-stock/v1/ranking/trade-pbmn"
_OVERSEAS_FLUCT_RANK_TR = "HHDFS76290000"  # 해외주식 상승률/하락률
_OVERSEAS_FLUCT_RANK_PATH = "/uapi/overseas-stock/v1/ranking/updown-rate"
_OVERSEAS_MARKET_CAP_RANK_TR = "HHDFS76350100"  # 해외주식 시가총액순위
_OVERSEAS_MARKET_CAP_RANK_PATH = "/uapi/overseas-stock/v1/ranking/market-cap"


class OverseasRankingMixin:
    async def fetch_overseas_volume_rank(
        self, exchange: str = "NAS", count: int = 30
    ) -> list[RankedStock]:
        """해외주식 거래량순위 — most actively traded names on `exchange`
        right now (NAS/NYS/AMS/HKS/SHS/SZS/HSX/HNX/TSE). Response splits into
        output1 (a header: zdiv/stat/nrec/rsym) + output2 (the actual rows) —
        the rows live in output2, not output1."""
        resp = await self._request(
            "GET", _OVERSEAS_VOLUME_RANK_PATH, _OVERSEAS_VOLUME_RANK_TR,
            params={
                "EXCD": exchange, "NDAY": "0", "VOL_RANG": "0",
                "KEYB": "", "AUTH": "", "PRC1": "", "PRC2": "",
            },
        )
        body = self._checked_body(resp, "overseas volume rank")
        return _ranked_rows(body.get("output2") or [], count, code_keys=("symb",))

    async def fetch_overseas_trading_value_rank(
        self, exchange: str = "NAS", count: int = 30
    ) -> list[RankedStock]:
        """해외주식 거래대금순위 — unlike the domestic pair (same TR, one
        FID switches sort), overseas has a genuinely separate TR for this
        (confirmed live: real-time NAS data, symb/name/ename/last/rate/tvol
        fields match fetch_overseas_volume_rank's shape 1:1, just ranked by
        cumulative trading value instead of share count)."""
        resp = await self._request(
            "GET", _OVERSEAS_VALUE_RANK_PATH, _OVERSEAS_VALUE_RANK_TR,
            params={
                "EXCD": exchange, "NDAY": "0", "VOL_RANG": "0",
                "KEYB": "", "AUTH": "", "PRC1": "", "PRC2": "",
            },
        )
        body = self._checked_body(resp, "overseas trading value rank")
        return _ranked_rows(body.get("output2") or [], count, code_keys=("symb",))

    async def fetch_overseas_fluctuation_rank(
        self, exchange: str = "NAS", rising: bool = True, count: int = 30
    ) -> list[RankedStock]:
        """해외주식 상승률/하락률 — top gainers (rising=True) or losers on
        `exchange` right now. GUBN 1=상승율/0=하락율 (opposite polarity from
        the domestic TR's FID_RANK_SORT_CLS_CODE, verified against this TR's
        own field name)."""
        resp = await self._request(
            "GET", _OVERSEAS_FLUCT_RANK_PATH, _OVERSEAS_FLUCT_RANK_TR,
            params={
                "EXCD": exchange, "NDAY": "0", "GUBN": "1" if rising else "0",
                "VOL_RANG": "0", "AUTH": "", "KEYB": "",
            },
        )
        body = self._checked_body(resp, "overseas fluctuation rank")
        return _ranked_rows(body.get("output2") or [], count, code_keys=("symb",))

    async def fetch_overseas_market_cap_rank(
        self, exchange: str = "NAS", count: int = 30
    ) -> list[RankedStock]:
        """해외주식 시가총액순위 — biggest companies on `exchange` by market
        cap right now, not by today's trading activity — same "stable
        large-cap pool vs. momentum/turnover-driven pool" distinction as the
        domestic sibling (DomesticRankingMixin.fetch_market_cap_rank).
        `tomv` (market cap itself) isn't surfaced on RankedStock — nothing
        else in this codebase's ranking rows carries it either, and rank
        order already reflects it."""
        resp = await self._request(
            "GET", _OVERSEAS_MARKET_CAP_RANK_PATH, _OVERSEAS_MARKET_CAP_RANK_TR,
            params={"EXCD": exchange, "VOL_RANG": "0", "KEYB": "", "AUTH": "", "CURR_GB": ""},
        )
        body = self._checked_body(resp, "overseas market cap rank")
        return _ranked_rows(body.get("output2") or [], count, code_keys=("symb",))
