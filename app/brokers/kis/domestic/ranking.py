from app.brokers.kis.parser import _ranked_rows
from app.domain.models import RankedStock

_VOLUME_RANK_TR_ID = "FHPST01710000"  # 거래량순위 (스크리너)
_VOLUME_RANK_PATH = "/uapi/domestic-stock/v1/quotations/volume-rank"
_FLUCT_RANK_TR_ID = "FHPST01700000"  # 등락률순위 (상승/하락)
_FLUCT_RANK_PATH = "/uapi/domestic-stock/v1/ranking/fluctuation"
_MARKET_CAP_RANK_TR_ID = "FHPST01740000"  # 시가총액 상위
_MARKET_CAP_RANK_PATH = "/uapi/domestic-stock/v1/ranking/market-cap"
_DISPARITY_RANK_TR_ID = "FHPST01780000"  # 이격도 순위
_DISPARITY_RANK_PATH = "/uapi/domestic-stock/v1/ranking/disparity"


def _price_ceiling_param(max_price: float | None, *, blank: str) -> str:
    """`FID_INPUT_PRICE_2` — documented identically across all four ranking
    TRs in this file (KIS's own `volume_rank.py` example source: "입력
    가격2 (~ 가격, 전체 가격 대상 조회 시 공란)"), i.e. an INCLUSIVE upper
    price bound: "~ price", blank when no filter is wanted. Added
    2026-09-02 (screening-improvement-research #3 follow-up) so
    `app.live.auto_trade_loop._domestic_price_ceiling` can ask KIS to rank
    only among tickers a small account could actually afford, confirmed
    live the same day: a probe against `fetch_market_cap_rank` with
    `FID_INPUT_PRICE_2="100000"` returned a full, non-truncated 30 rows —
    KIS ranks WITHIN the price-bounded set, before its own top-N cut.

    `blank` is a required keyword, not a hardcoded `""`, because this
    file's four TRs spell "no filter" two different ways for this exact
    field: `_fetch_volume_rank_tr` (shared by `fetch_volume_rank`/
    `fetch_trading_value_rank`) sends literal `"0"`, while
    `fetch_market_cap_rank`/`fetch_fluctuation_rank`/`fetch_disparity_rank`
    send `""`. Passing each caller's own pre-existing spelling back in is
    what keeps `max_price=None` byte-identical to this file's wire
    behavior from before this parameter existed — the settings-off no-op
    guarantee the whole feature is gated behind
    (`settings.auto_trade_universe_price_ceiling_domestic=False` by
    default).

    Floors rather than rounds (`int()`, not `round()`): a computed ceiling
    of 399,999.7 must not become 400,000 and admit a row the caller's own
    arithmetic (`app.domain.executability.affordable_price_ceiling`)
    already excluded at 399,999.7. KRX prices are whole KRW, so flooring
    loses nothing real — there is no such thing as a 399,999.7 KRW quote
    to round toward."""
    if max_price is None or max_price <= 0:
        return blank
    return str(int(max_price))


class DomesticRankingMixin:
    async def _fetch_volume_rank_tr(
        self, blng_cls_code: str, what: str, count: int, *, max_price: float | None = None
    ) -> list[RankedStock]:
        """Shared body for 거래량순위(FHPST01710000) — `FID_BLNG_CLS_CODE`
        picks the sort within this one TR ("0"=평균거래량, "3"=거래금액순,
        confirmed live; also has "1"=거래증가율/"2"=평균거래회전율/"4"=평균거래
        금액회전율 unused here). Same response shape regardless of sort.

        `max_price` — see `_price_ceiling_param`'s docstring for the shared
        `FID_INPUT_PRICE_2` mechanism; this TR's own "no filter" spelling
        is `"0"` (`blank="0"`), unlike the other three TRs in this file."""
        resp = await self._request(
            "GET", _VOLUME_RANK_PATH, _VOLUME_RANK_TR_ID,
            params={
                "FID_COND_MRKT_DIV_CODE": "J", "FID_COND_SCR_DIV_CODE": "20171",
                "FID_INPUT_ISCD": "0000", "FID_DIV_CLS_CODE": "0", "FID_BLNG_CLS_CODE": blng_cls_code,
                "FID_TRGT_CLS_CODE": "111111111", "FID_TRGT_EXLS_CLS_CODE": "0000000000",
                "FID_INPUT_PRICE_1": "0", "FID_INPUT_PRICE_2": _price_ceiling_param(max_price, blank="0"),
                "FID_VOL_CNT": "0",
                "FID_INPUT_DATE_1": "",
            },
        )
        body = self._checked_body(resp, what)
        rows = body.get("output") or body.get("output1") or []
        return _ranked_rows(
            rows, count, code_keys=("mksc_shrn_iscd",), name_keys=("hts_kor_isnm",),
            rank_key="data_rank", price_key="stck_prpr", change_pct_key="prdy_ctrt", volume_key="acml_vol",
        )

    async def fetch_volume_rank(self, count: int = 30, *, max_price: float | None = None) -> list[RankedStock]:
        """거래량순위 — the most actively traded KRX names right now, with each
        one's price and day change. A discovery/screening feed (영웅문 순위).

        `max_price` — see `fetch_market_cap_rank`'s docstring for the full
        story; same `FID_INPUT_PRICE_2` mechanism via `_price_ceiling_param`,
        same `None`-is-a-no-op guarantee, `blank="0"` (this TR's own
        no-filter spelling, via `_fetch_volume_rank_tr`)."""
        return await self._fetch_volume_rank_tr("0", "volume rank", count, max_price=max_price)

    async def fetch_trading_value_rank(
        self, count: int = 30, *, max_price: float | None = None
    ) -> list[RankedStock]:
        """거래대금순위 — same TR as fetch_volume_rank, sorted by cumulative
        trading value (거래대금, `FID_BLNG_CLS_CODE="3"`) instead of raw share
        volume. A high-volume/low-price penny stock ranks lower here than by
        fetch_volume_rank, and a high-price/lower-share-count name (e.g.
        삼성전자 vs a leveraged ETF) ranks higher — closer to "where the real
        money is trading today" than pure share count.

        `max_price` — see `fetch_market_cap_rank`'s docstring for the full
        story; same `FID_INPUT_PRICE_2` mechanism via `_price_ceiling_param`,
        same `None`-is-a-no-op guarantee, `blank="0"` (this TR's own
        no-filter spelling, via `_fetch_volume_rank_tr`)."""
        return await self._fetch_volume_rank_tr("3", "trading value rank", count, max_price=max_price)

    async def fetch_market_cap_rank(self, count: int = 30, *, max_price: float | None = None) -> list[RankedStock]:
        """시가총액 상위(FHPST01740000) — KRX names ranked by current market
        cap, `FID_INPUT_ISCD="0000"` (전체: KOSPI+KOSDAQ combined, not just
        one market). A different discovery axis from fetch_trading_value_rank/
        fetch_volume_rank: those surface whatever's actively trading *today*,
        this surfaces the market's biggest companies regardless of today's
        activity — for a use case that wants a stable "large-cap" pool rather
        than a momentum/turnover-driven one (e.g. seeding a trade allowlist).

        `max_price` (`FID_INPUT_PRICE_2`, a KIS-side inclusive upper price
        bound applied BEFORE this TR's own top-N cut — confirmed live
        2026-09-02: a probe with `max_price=100_000` returned a full,
        non-truncated 30 rows, 28 of them different from the unfiltered
        top-30, meaning KIS ranks WITHIN the bounded price range rather
        than truncating an already-fixed top-30) added 2026-09-02: with a
        small account, "top 30 by market cap" and "top 30 by market cap
        among shares this account can actually afford" are very different
        30-row budgets, and on KRX they diverge far more than one might
        expect, because per-share price is nearly uncorrelated with market
        cap after decades of stock splits (삼성전자 ~₩250,000 vs SK하이닉스
        ~₩1,613,000 in this same probe, both mega-caps). Filtering
        KIS-side is what makes the full 30 rows useful instead of a short
        list after a client-side drop — the same argument `min_volume`
        already makes for liquidity on the other three TRs in this file.
        Defaults to `None`, sending this TR's own pre-existing `""` via
        `_price_ceiling_param` — byte-identical wire behavior for any
        caller that doesn't opt in. See
        `app.domain.executability.affordable_price_ceiling` for where the
        number comes from and `app.live.universe.krx_top_n`'s docstring
        for the fuller account-affordability story."""
        resp = await self._request(
            "GET", _MARKET_CAP_RANK_PATH, _MARKET_CAP_RANK_TR_ID,
            params={
                "FID_INPUT_PRICE_2": _price_ceiling_param(max_price, blank=""),
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_COND_SCR_DIV_CODE": "20174", "FID_DIV_CLS_CODE": "0",
                "FID_INPUT_ISCD": "0000", "FID_TRGT_CLS_CODE": "0",
                "FID_TRGT_EXLS_CLS_CODE": "0", "FID_INPUT_PRICE_1": "", "FID_VOL_CNT": "",
            },
        )
        body = self._checked_body(resp, "market cap rank")
        rows = body.get("output") or body.get("output1") or []
        return _ranked_rows(
            rows, count, code_keys=("mksc_shrn_iscd",), name_keys=("hts_kor_isnm",),
            rank_key="data_rank", price_key="stck_prpr", change_pct_key="prdy_ctrt", volume_key="acml_vol",
        )

    async def fetch_disparity_rank(
        self,
        *,
        low: bool = True,
        window_days: int = 20,
        count: int = 30,
        min_volume: int = 0,
        max_price: float | None = None,
    ) -> list[RankedStock]:
        """이격도 순위(FHPST01780000) — KRX names ranked by 이격도 (today's
        price as a % of its own `window_days`-day moving average), a THIRD
        discovery axis alongside fetch_trading_value_rank/fetch_market_cap_rank
        and structurally unlike either of them. Both of those sort on a
        size/turnover quantity, and on KRX those two are near-collinear —
        measured 2026-09-02: their union is only ~35 overlapping mega-caps
        every cycle, because a name that's big enough to top one turnover/
        market-cap ranking is usually big enough to top the other too. This
        TR's sort key has no size term at all, so it reaches mid/small caps
        neither of the other two axes can structurally surface.

        `low=True` (`FID_RANK_SORT_CLS_CODE="1"`, 이격도하위순 — names
        trading furthest BELOW their own moving average) is the default and
        the only mode currently wired into a discovery feed
        (`app.live.universe.krx_disparity_low_n`). `low=False` (이격도상위순,
        `"0"`) is exposed for symmetry with `fetch_fluctuation_rank`'s
        `rising` parameter but is not currently used by anything.

        `window_days` selects which moving-average window the 이격도 is
        measured against, via `FID_HOUR_CLS_CODE` — a misleading name (it
        reads like a clock-hour selector) for what KIS documents as a
        day-count: this TR accepts ONLY 5/10/20/60/120, nothing else. The
        default of 20 pairs with the ranking layer's existing 14-day ATR
        normalization without reaching into 60/120-day territory, where
        "trading below its mean" stops meaning "pullback" and starts
        meaning "broken" — a much weaker prior for a name that's about to
        turn back up.

        `min_volume` (`FID_VOL_CNT`) is a KIS-side MINIMUM volume floor,
        applied by KIS before ranking, not filtered client-side — so every
        one of the `count` rows returned is already usable, rather than
        handing the caller a shorter list after a client-side drop. This
        matters more here than on fetch_trading_value_rank/
        fetch_market_cap_rank, which are self-limiting on liquidity by
        construction (an illiquid name can't top a 거래대금 ranking in the
        first place). A 이격도 하위 ranking has no such built-in floor and
        will readily surface 관리종목/정리매매-adjacent names trading on
        minimal volume — exactly the names furthest below their own average
        for the worst possible reason. `min_volume=0` sends `""` (KIS's own
        "no floor" spelling, same convention as fetch_market_cap_rank's
        FID_VOL_CNT).

        `FID_TRGT_EXLS_CLS_CODE="0"` (전체, no exclusion): this TR's own
        spec documents ONLY "0: 전체" for this field, unlike 등락률순위
        (fetch_fluctuation_rank) whose spec documents a 10-character
        exclusion mask — so that longer mask form is deliberately not
        attempted here; there is nothing in the spec for this TR to apply
        it to.

        `max_price` — see `fetch_market_cap_rank`'s docstring for the full
        story; same `FID_INPUT_PRICE_2` mechanism, same `None`-is-a-no-op
        guarantee, `blank=""` (this TR's own no-filter spelling)."""
        resp = await self._request(
            "GET", _DISPARITY_RANK_PATH, _DISPARITY_RANK_TR_ID,
            params={
                "FID_INPUT_PRICE_2": _price_ceiling_param(max_price, blank=""), "FID_COND_MRKT_DIV_CODE": "J",
                "FID_COND_SCR_DIV_CODE": "20178", "FID_DIV_CLS_CODE": "0",
                "FID_RANK_SORT_CLS_CODE": "1" if low else "0",
                "FID_HOUR_CLS_CODE": str(window_days),
                "FID_INPUT_ISCD": "0000", "FID_TRGT_CLS_CODE": "0",
                "FID_TRGT_EXLS_CLS_CODE": "0", "FID_INPUT_PRICE_1": "",
                "FID_VOL_CNT": str(min_volume) if min_volume > 0 else "",
            },
        )
        body = self._checked_body(resp, "disparity rank")
        rows = body.get("output") or body.get("output1") or []
        return _ranked_rows(
            rows, count, code_keys=("mksc_shrn_iscd",), name_keys=("hts_kor_isnm",),
            rank_key="data_rank", price_key="stck_prpr", change_pct_key="prdy_ctrt", volume_key="acml_vol",
        )

    async def fetch_fluctuation_rank(
        self, rising: bool = True, count: int = 30, *, min_volume: int = 0, max_price: float | None = None
    ) -> list[RankedStock]:
        """등락률순위 — top gainers (rising=True) or losers (rising=False) on
        KRX right now. Complements the volume-rank screener.

        `min_volume` (`FID_VOL_CNT`) is a KIS-side floor added 2026-09-02
        when this TR was first wired into a live discovery feed
        (`app.live.universe.krx_fluctuation_losers_n`) — see
        `fetch_disparity_rank`'s docstring for why a 하락률/이격도-style
        ranking needs this floor when the turnover/market-cap rankings
        don't. Defaults to 0, which sends `""` — byte-identical wire
        behavior to before this parameter existed, so the pre-existing UI
        caller (app.web.routers.market_data) is unaffected by this change.

        `FID_TRGT_EXLS_CLS_CODE` intentionally stays `"0"` (no exclusion)
        rather than attempting this TR's documented 10-character exclusion
        mask (unlike fetch_disparity_rank, whose spec has no such mask to
        attempt) — the mask's field order is unverified against a live
        account, and a mis-ordered mask would silently exclude the wrong
        class of stock rather than failing loudly. The volume floor above is
        the safer equivalent for now; verifying and switching to the mask is
        a deliberate follow-up, not an oversight.

        `max_price` — see `fetch_market_cap_rank`'s docstring for the full
        story; same `FID_INPUT_PRICE_2` mechanism, same `None`-is-a-no-op
        guarantee, `blank=""` (this TR's own no-filter spelling)."""
        resp = await self._request(
            "GET", _FLUCT_RANK_PATH, _FLUCT_RANK_TR_ID,
            params={
                "FID_COND_MRKT_DIV_CODE": "J", "FID_COND_SCR_DIV_CODE": "20170",
                "FID_INPUT_ISCD": "0000", "FID_RANK_SORT_CLS_CODE": "0" if rising else "1",
                "FID_INPUT_CNT_1": "0", "FID_PRC_CLS_CODE": "0", "FID_INPUT_PRICE_1": "",
                "FID_INPUT_PRICE_2": _price_ceiling_param(max_price, blank=""),
                "FID_VOL_CNT": str(min_volume) if min_volume > 0 else "",
                "FID_TRGT_CLS_CODE": "0",
                "FID_TRGT_EXLS_CLS_CODE": "0", "FID_DIV_CLS_CODE": "0",
                "FID_RSFL_RATE1": "", "FID_RSFL_RATE2": "",
            },
        )
        body = self._checked_body(resp, "fluctuation rank")
        rows = body.get("output") or body.get("output1") or []
        return _ranked_rows(
            rows, count, code_keys=("stck_shrn_iscd", "mksc_shrn_iscd"), name_keys=("hts_kor_isnm",),
            rank_key="data_rank", price_key="stck_prpr", change_pct_key="prdy_ctrt", volume_key="acml_vol",
        )
