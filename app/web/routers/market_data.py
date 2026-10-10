"""KIS/yfinance-backed market-data endpoints — candles, screeners, index and
sector quotes."""

from __future__ import annotations

import asyncio
import logging
from datetime import timedelta

from fastapi import APIRouter

from app.clock import get_clock
from app.dataflows.yfinance import get_price_series
from app.domain.leverage_filter import is_leveraged_or_inverse
from app.domain.product_filter import is_out_of_scope_product
from app.execution.kis_bridge import route_ticker
from app.web import state

logger = logging.getLogger("app.web")

router = APIRouter()

_CANDLE_LOOKBACK_DAYS = {"D": 200, "W": 800, "M": 3300, "Y": 12000}


_OVERSEAS_CANDLE_PERIODS = {"D", "W", "M"}  # KIS overseas has no 년봉 option


@router.get("/api/candles/{ticker}")
async def candles_endpoint(ticker: str, period: str = "D") -> list[dict]:
    """OHLCV candles for a ticker at the given period — D 일 / W 주 / M 월 /
    Y 년봉, oldest first. KIS-only by policy: what the frontend shows must
    come from the same broker as the account/order panels, never yfinance
    (that stays backend-only, for AI-analysis inputs it's actually good at).
    No cross-provider fallback either — KIS overseas candles genuinely have
    no Y (연봉) option for individual stocks (confirmed via the KIS API
    catalog: the one overseas TR that does support Y is index-linked
    (해외지수기간별시세, KIS's own docs say to use the per-stock TR we already
    call instead for arbitrary tickers), so Y for a US ticker always
    returns empty here — "KIS API 없음" for that combination, not a bug."""
    period = period.upper()
    if period not in _CANDLE_LOOKBACK_DAYS:
        return []
    krx_code, us_symbol = route_ticker(ticker.upper())

    def _rows(candles) -> list[dict]:
        return [
            {"day": c.day, "open": c.open, "high": c.high, "low": c.low,
             "close": c.close, "volume": c.volume}
            for c in candles
        ]

    async def _load():
        broker = await state.kis_broker()
        try:
            if krx_code is not None and broker is not None:
                end = get_clock().today()
                start = end - timedelta(days=_CANDLE_LOOKBACK_DAYS[period])
                candles = await broker.fetch_daily_candles(
                    krx_code, start.strftime("%Y%m%d"), end.strftime("%Y%m%d"), period=period
                )
                return _rows(candles)
            if us_symbol is not None and broker is not None and period in _OVERSEAS_CANDLE_PERIODS:
                candles = await broker.fetch_overseas_candles(us_symbol, period=period)
                return _rows(candles)
        except Exception:
            logger.warning("candles failed for %s (%s)", ticker, period, exc_info=True)
        return []

    return await state.cached(f"candles:{ticker.upper()}:{period}", 60, _load)


@router.get("/api/valuation/{ticker}")
async def valuation_endpoint(ticker: str) -> dict:
    """PER/PBR/시가총액(+해외는 52주 고저) — 트레이딩 탭 펀더멘털 요약행.
    market_cap unit differs by scope (domestic: 억원, overseas: raw currency)
    since that's each TR's native unit — frontend already branches on
    domestic/overseas for price formatting, so it formats this the same way.
    배당수익률/외국인지분은 KIS API 자체가 없어 여기 포함 안 함(지어내지 않음)."""
    broker = await state.kis_broker()
    if broker is None:
        return {}
    krx_code, us_symbol = route_ticker(ticker.upper())

    async def _load():
        if krx_code is not None:
            v = await broker.fetch_valuation(krx_code)
            return {"market_cap": v.market_cap, "per": v.per, "pbr": v.pbr,
                    "week52_high": None, "week52_low": None}
        if us_symbol is not None:
            v = await broker.fetch_overseas_valuation(us_symbol)
            return {"market_cap": v.market_cap, "per": v.per, "pbr": v.pbr,
                    "week52_high": v.week52_high, "week52_low": v.week52_low}
        # 라우팅 불가(국내도 해외도 아님) = 확정된 '없음' — 이건 캐시해도 맞다.
        return {}

    try:
        # try/except is OUTSIDE cached() on purpose. Inside the factory, a
        # transient KIS failure returned {} and state.cached stored THAT empty
        # dict for the full 300s TTL — one blip blanked the fundamentals row for
        # five minutes after KIS had already recovered. Letting the exception
        # propagate past cached() means nothing is written to _cache, so the very
        # next request retries for real. Same "the failure is never cached"
        # convention as app.web.account_mode_middleware.bind_account_mode.
        # A genuinely-unroutable ticker still returns {} from inside the factory
        # and IS cached — confirmed-empty and couldn't-fetch must not share a
        # cache entry.
        # 실패는 캐시하지 않음 — 다음 요청이 즉시 다시 시도. 진짜 '없음'은 캐시함.
        return await state.cached(f"valuation:{ticker.upper()}", 300, _load)
    except Exception:
        logger.warning("valuation inquiry failed for %s — not cached, next request retries", ticker, exc_info=True)
        return {}


@router.get("/api/minute/{ticker}")
async def minute_candles_endpoint(ticker: str) -> list[dict]:
    """Today's 1-minute OHLCV for a KRX ticker (intraday chart), oldest first."""
    broker = await state.kis_broker()
    if broker is None:
        return []
    krx_code, _ = route_ticker(ticker.upper())
    if krx_code is None:
        return []
    try:
        candles = await broker.fetch_minute_candles(krx_code)
    except Exception:
        logger.warning("minute candles failed for %s", ticker, exc_info=True)
        return []
    return [
        {"time": c.time, "open": c.open, "high": c.high, "low": c.low,
         "close": c.close, "volume": c.volume}
        for c in candles
    ]


def _rank_rows(rows, overseas: bool) -> list[dict]:
    """Shared shape for every rank screener row — `ticker` gets the same
    .KS-suffix-or-bare convention `route_ticker` expects elsewhere, so a row
    can be click-added to the trade allowlist without extra parsing."""
    return [
        {"rank": r.rank, "ticker": r.code if overseas else f"{r.code}.KS", "name": r.name,
         "price": r.price, "change_pct": r.change_pct, "volume": r.volume}
        for r in rows
    ]


@router.get("/api/ticker-suggestions")
async def ticker_suggestions(scope: str = "domestic", limit: int = 8) -> list[dict]:
    """관심종목 검색창의 빈 입력 상태에서 보여줄 추천 종목 — 현재 거래대금
    순위에서 레버리지/인버스 및 범위 외 상품(현금성/채권형 + 커버드콜/옵션인컴)을
    제외한 목록. 프론트에서
    `_rankCache`를 걸러 쓰지 않고 별도 엔드포인트로 두는 이유: 이 필터들은
    "매 계층이 각자 구현하지 말 것"이 명시된 도메인 로직(app.domain.leverage_
    filter / app.domain.product_filter)이고, 순위 탭의 현재 모드(sector 등)에
    종속되지 않아야 한다.

    검색 결과와 같은 `symbol` 필드명을 쓰므로 프론트 렌더러를 공유할 수 있다.
    브로커 미설정이면 `[]`."""
    broker = await state.kis_broker()
    if broker is None:
        return []
    overseas = scope == "overseas"

    async def _load():
        try:
            rows = (
                await broker.fetch_overseas_trading_value_rank(count=20)
                if overseas
                else await broker.fetch_trading_value_rank(20)
            )
        except Exception:
            logger.warning("ticker suggestions failed (scope=%s)", scope, exc_info=True)
            return []
        ranked = _rank_rows(rows, overseas)
        out = [
            r for r in ranked
            if not is_leveraged_or_inverse(r["name"]) and not is_out_of_scope_product(r["name"])
        ]
        return [
            {"symbol": r["ticker"], "name": r["name"],
             "market": "overseas" if overseas else "domestic",
             "price": r["price"], "change_pct": r["change_pct"]}
            for r in out[:limit]
        ]

    return await state.cached(f"suggest:{scope}:{limit}", 15, _load)


@router.get("/api/volume-rank")
async def volume_rank_endpoint(scope: str = "domestic", count: int = 20) -> list[dict]:
    """거래량 순위 스크리너 — most actively traded names right now, KRX
    (scope=domestic) or NASDAQ (scope=overseas)."""
    broker = await state.kis_broker()
    if broker is None:
        return []
    overseas = scope == "overseas"

    async def _load():
        try:
            rows = (
                await broker.fetch_overseas_volume_rank(count=count)
                if overseas
                else await broker.fetch_volume_rank(count)
            )
        except Exception:
            logger.warning("volume rank inquiry failed (scope=%s)", scope, exc_info=True)
            return []
        return _rank_rows(rows, overseas)

    return await state.cached(f"vrank:{scope}:{count}", 15, _load)


@router.get("/api/trading-value-rank")
async def trading_value_rank_endpoint(scope: str = "domestic", count: int = 20) -> list[dict]:
    """거래대금 순위 — ranked by cumulative trading value rather than raw
    share count (a high-price, lower-share-count name ranks higher here than
    on the volume screener)."""
    broker = await state.kis_broker()
    if broker is None:
        return []
    overseas = scope == "overseas"

    async def _load():
        try:
            rows = (
                await broker.fetch_overseas_trading_value_rank(count=count)
                if overseas
                else await broker.fetch_trading_value_rank(count)
            )
        except Exception:
            logger.warning("trading-value rank inquiry failed (scope=%s)", scope, exc_info=True)
            return []
        return _rank_rows(rows, overseas)

    return await state.cached(f"vrank-value:{scope}:{count}", 15, _load)


@router.get("/api/fluctuation")
async def fluctuation_rank_endpoint(direction: str = "up", scope: str = "domestic", count: int = 20) -> list[dict]:
    """등락률 순위 — top gainers (direction=up) or losers (down), KRX or NASDAQ."""
    broker = await state.kis_broker()
    if broker is None:
        return []
    overseas = scope == "overseas"
    rising = direction != "down"

    async def _load():
        try:
            rows = (
                await broker.fetch_overseas_fluctuation_rank(rising=rising, count=count)
                if overseas
                else await broker.fetch_fluctuation_rank(rising=rising, count=count)
            )
        except Exception:
            logger.warning("fluctuation rank failed (scope=%s)", scope, exc_info=True)
            return []
        return _rank_rows(rows, overseas)

    return await state.cached(f"fluct:{direction}:{scope}:{count}", 15, _load)


@router.get("/api/index")
async def index_quotes_endpoint() -> list[dict]:
    """KOSPI/KOSDAQ current level + day change for the dashboard market bar."""
    broker = await state.kis_broker()
    if broker is None:
        return []

    async def _load():
        out = []
        for code in ("0001", "1001"):
            try:
                q = await broker.fetch_index_price(code)
                out.append({"code": q.code, "name": q.name, "value": q.value,
                            "change": q.change, "change_pct": q.change_pct})
            except Exception:
                logger.warning("index price %s failed", code, exc_info=True)
        return out

    # 35s, not 15s: pollIndex (main.js) polls this every 30s — same
    # shorter-than-poll-interval waste as the "account"/"executions" caches
    # in app/web/state.py, fixed the same way (see those for the reasoning).
    return await state.cached("index", 35, _load)


@router.get("/api/sectors")
async def sectors_endpoint(market: str = "0001") -> list[dict]:
    """업종별 지수 등락 (sector breadth) for KOSPI (0001) or KOSDAQ (1001)."""
    broker = await state.kis_broker()
    if broker is None:
        return []

    async def _load():
        try:
            rows = await broker.fetch_sector_indices(market)
        except Exception:
            logger.warning("sector indices failed", exc_info=True)
            return []
        return [{"code": s.code, "name": s.name, "value": s.value, "change_pct": s.change_pct} for s in rows]

    return await state.cached(f"sectors:{market}", 15, _load)


_MARKET_WIDGET_SPARK_POINTS = 20


def _series_sparkline(series: dict, points: int = _MARKET_WIDGET_SPARK_POINTS) -> list[float]:
    """A yfinance `date -> close` map into a chronological (oldest->newest)
    sparkline of at most `points` values."""
    ordered = [v for _, v in sorted(series.items(), key=lambda kv: kv[0])]
    return ordered[-points:]


def _series_current_change(series: dict) -> tuple[float, float, float] | None:
    """Current/change/change_pct straight off the last two sorted-by-date
    points of a yfinance series — same already-signed, pre-multiplied-percent
    convention as KIS's own change_pct fields (e.g. 1.23 meaning 1.23%, not
    0.0123), so the frontend can treat every entry identically regardless of
    which source produced it."""
    ordered = sorted(series.items(), key=lambda kv: kv[0])
    if len(ordered) < 2:
        return None
    previous, current = ordered[-2][1], ordered[-1][1]
    change = current - previous
    change_pct = (change / previous * 100) if previous else 0.0
    return current, change, change_pct


@router.get("/api/market-widget")
async def market_widget_endpoint() -> list[dict]:
    """5-instrument market-overview widget for the AI 시그널 tab landing
    page (KOSPI/KOSDAQ/USD-KRW/NASDAQ/VIX) — current value, day change, and a
    ~20-point sparkline history for each.

    Deliberately mixes KIS and yfinance in one response, unlike
    `/api/candles/{ticker}`'s KIS-only policy above: that policy exists
    because a per-stock price shown here is something a user could actually
    trade on this exact screen, where a displayed-price/fill-price mismatch
    would matter. Nobody trades "the VIX" or "the NASDAQ" through this app —
    this widget is pure informational macro context, the same category as
    `app.dataflows.kis_provider.get_kr_macro_indicators` (which already
    labels its own output "source: KIS/yfinance"). KOSPI/KOSDAQ current+
    history and NASDAQ's current value/change come from KIS (proven paths
    already used elsewhere — `fetch_index_price`/`fetch_index_daily_chart`/
    `fetch_overseas_index_quote`); NASDAQ's sparkline uses yfinance `^IXIC`
    instead (KIS's overseas index TR has no parsed history method in this
    codebase yet, and adding one purely for a decorative trend line is more
    broker-layer risk than this cosmetic widget is worth); VIX and USD/KRW
    have no KIS equivalent at all, so both fields come entirely from
    yfinance (`^VIX` / `KRW=X`).

    A failed fetch (or a sparkline that ends up with fewer than 2 points)
    just drops that one entry — never 500s, never blocks the other four."""
    today = get_clock().today()
    start = today - timedelta(days=40)
    start_str, end_str = start.strftime("%Y%m%d"), today.strftime("%Y%m%d")
    broker = await state.kis_broker()

    async def _load() -> list[dict]:
        out: list[dict] = []

        if broker is not None:
            for code, key in (("0001", "kospi"), ("1001", "kosdaq")):
                try:
                    q = await broker.fetch_index_price(code)
                    bars = await broker.fetch_index_daily_chart(code, start_str, end_str)
                    spark = [b.close for b in sorted(bars, key=lambda b: b.day)][-_MARKET_WIDGET_SPARK_POINTS:]
                    if len(spark) < 2:
                        continue
                    out.append({"key": key, "value": q.value, "change": q.change,
                                "change_pct": q.change_pct, "sparkline": spark})
                except Exception:
                    logger.warning("market widget: %s failed", key, exc_info=True)

            try:
                nasdaq_quote = await broker.fetch_overseas_index_quote("NDX", "Nasdaq 100", start_str, end_str)
            except Exception:
                logger.warning("market widget: nasdaq quote failed", exc_info=True)
                nasdaq_quote = None
        else:
            nasdaq_quote = None

        async def _load_series(ticker: str) -> dict:
            try:
                return await get_price_series(ticker, start, today)
            except Exception:
                logger.warning("market widget: %s series failed", ticker, exc_info=True)
                return {}

        vix_series, usdkrw_series, nasdaq_series = await asyncio.gather(
            _load_series("^VIX"), _load_series("KRW=X"), _load_series("^IXIC"),
        )

        if nasdaq_quote is not None:
            nasdaq_spark = _series_sparkline(nasdaq_series)
            if len(nasdaq_spark) >= 2:
                out.append({"key": "nasdaq", "value": nasdaq_quote.value, "change": nasdaq_quote.change,
                            "change_pct": nasdaq_quote.change_pct, "sparkline": nasdaq_spark})

        for ticker, key in (("^VIX", "vix"), ("KRW=X", "usdkrw")):
            series = vix_series if key == "vix" else usdkrw_series
            result = _series_current_change(series)
            if result is None:
                continue
            current, change, change_pct = result
            out.append({"key": key, "value": current, "change": change, "change_pct": change_pct,
                        "sparkline": _series_sparkline(series)})

        return out

    return await state.cached("market-widget", 60, _load)
