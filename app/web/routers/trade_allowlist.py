"""Trade-allowlist management endpoints — add/remove tickers and their display info."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.config import settings
from app.domain.leverage_filter import is_leveraged_or_inverse
from app.domain.product_filter import is_out_of_scope_product
from app.live import krx_master, overseas_master
from app.live.market_hours import is_market_open
from app.web import state
from app.web.ticker_validation import validate_ticker

router = APIRouter()

# How many raw matches to pull from each master file before ranking. The
# master `search()` returns in dict-insertion order (= master-file code
# order), so "삼성" would otherwise surface 삼성화재/삼성화재우 ahead of
# 삼성전자 purely by code number. We over-fetch, rank, then trim to `limit`.
_SEARCH_RANK_POOL = 120


def _preferred_or_spac(name: str) -> bool:
    """우선주/스팩 — a real match, but almost never what someone typing a
    company name wants first, so it sinks below the common stock."""
    upper = name.upper()
    return name.endswith("우") or name.endswith("우B") or "스팩" in name or "SPAC" in upper


def _search_rank_key(q: str, code: str, name: str, market: str, scope: str) -> tuple:
    """Sort key for a single search hit — lower sorts first. `code` is the
    bare KRX code / US symbol (no `.KS`), matched against a typed partial
    code; `name` is the display name. Tiers: exact code, exact name, name
    prefix, code prefix, name substring, code substring. Then: tradable
    before leveraged/inverse/cash-bond, common stock before 우선주/스팩, the
    active scope's market before the other, shorter name, code ascending
    (stable against the master file's own order)."""
    ql = q.strip().lower()
    nl = name.lower()
    cl = code.lower()
    if cl == ql:
        tier = 0
    elif nl == ql:
        tier = 1
    elif nl.startswith(ql):
        tier = 2
    elif cl.startswith(ql):
        tier = 3
    elif ql in nl:
        tier = 4
    else:
        tier = 5
    return (
        tier,
        1 if (is_leveraged_or_inverse(name) or is_out_of_scope_product(name)) else 0,
        1 if _preferred_or_spac(name) else 0,
        0 if market == scope else 1,
        len(name),
        code,
    )


@router.get("/api/search-tickers")
async def search_tickers(q: str, limit: int = 8, scope: str = "domestic") -> list[dict]:
    """Name-or-code typeahead for the 관심종목 add box. Backed by the same
    KRX/NASDAQ master-file reverse lookups app.live.krx_master/
    overseas_master already load for display-name resolution — no new data
    source, no network call per search.

    `scope` ("domestic"/"overseas") is a RANKING hint only, not a filter —
    an overseas ticker still shows up in a domestic-scope search, just
    ranked after domestic hits of the same tier (and vice versa). Results
    from both markets are pooled and ranked together (`_search_rank_key`),
    so a small `limit` can't let a full page of domestic hits starve the
    overseas ones.

    `tradable` is `false` for leveraged/inverse/futures products and for
    out-of-scope products — cash-like/bond/bond-mix funds and covered-call/
    option-income funds (same `app.domain.leverage_filter` /
    `app.domain.product_filter` checks the add endpoint rejects on) — the
    client shows these dimmed and non-clickable rather than hiding them, so
    searching "레버리지", "채권" or "커버드콜" doesn't look broken. Product-type checks
    that need a yfinance quoteType (mutual fund / ETN) can't run here and
    surface as a 422 `unsupported_product_type` on add instead.

    Empty `q` returns `[]`. Best-effort: `ensure_loaded()` degrades to an
    empty cache (not an exception) if a master-file download ever failed,
    so this returns fewer/no results rather than erroring."""
    q = q.strip()
    if not q:
        return []
    await krx_master.ensure_loaded()
    await overseas_master.ensure_loaded()
    scope = "overseas" if scope == "overseas" else "domestic"
    # (symbol, name, market, bare_code)
    candidates: list[tuple[str, str, str, str]] = [
        (f"{code}.KS", name, "domestic", code)
        for code, name in krx_master.search(q, _SEARCH_RANK_POOL)
    ]
    candidates += [
        (symbol, name, "overseas", symbol)
        for symbol, name in overseas_master.search(q, _SEARCH_RANK_POOL)
    ]
    candidates.sort(key=lambda c: _search_rank_key(q, c[3], c[1], c[2], scope))
    top = candidates[:limit]

    # Don't let a page full of same-tier scope-market hits completely starve
    # the other market when it has real matches — reserve a few trailing
    # slots for it (finding 테슬라/TSLA from a domestic-scope search still
    # has to work). They land at the end: scope market first, then "also".
    other = "overseas" if scope == "domestic" else "domestic"
    if limit >= 4 and not any(c[2] == other for c in top):
        other_hits = [c for c in candidates if c[2] == other]
        if other_hits:
            reserve = min(len(other_hits), max(1, limit // 4))
            top = top[: limit - reserve] + other_hits[:reserve]

    return [
        {"symbol": sym, "name": name, "market": market,
         "tradable": not (is_leveraged_or_inverse(name) or is_out_of_scope_product(name))}
        for sym, name, market, _ in top
    ]


class TradeAllowlistRequest(BaseModel):
    ticker: str


class TradeAllowlistRemoveRequest(TradeAllowlistRequest):
    # 보유 중(또는 보유 여부 확인 불가)인 종목의 제거를 사람이 명시적으로
    # 승인했음을 뜻함. 기본 False — 클라이언트가 아무것도 안 보내면 안전 쪽.
    force: bool = False


@router.post("/api/trade-allowlist/add")
async def trade_allowlist_add(req: TradeAllowlistRequest) -> dict:
    """Add a ticker to the live trade allowlist after validating it resolves
    to real quote data. Thin HTTP wrapper — the actual add logic (existence
    check, leveraged/inverse filter, display-info fetch, domestic/overseas
    field reassignment, history seeding) lives in
    `app.web.state.add_ticker_to_trade_allowlist`, shared with
    `app.orchestrator.pipeline`'s auto-add-on-BUY-decision hook."""
    ticker = req.ticker.strip().upper()
    if not ticker:
        raise HTTPException(400, "ticker is required")
    ticker = validate_ticker(ticker)
    try:
        status, info = await state.add_ticker_to_trade_allowlist(ticker)
    except state.LeveragedTickerError as exc:
        raise HTTPException(
            422, {"error_code": "leveraged_or_inverse_not_allowed", "ticker": ticker, "detail": str(exc)}
        ) from exc
    except state.MalformedTickerError as exc:
        raise HTTPException(
            422, {"error_code": "invalid_ticker_format", "ticker": ticker, "detail": str(exc)}
        ) from exc
    except state.UnsupportedProductTypeError as exc:
        raise HTTPException(
            422, {"error_code": "unsupported_product_type", "ticker": ticker, "detail": str(exc)}
        ) from exc
    except state.CashOrBondFundError as exc:
        raise HTTPException(
            422, {"error_code": "cash_or_bond_fund", "ticker": ticker, "detail": str(exc)}
        ) from exc
    except state.DerivativeIncomeFundError as exc:
        raise HTTPException(
            422, {"error_code": "derivative_income_fund", "ticker": ticker, "detail": str(exc)}
        ) from exc
    except state.UnresolvedTickerNameError as exc:
        raise HTTPException(
            422, {"error_code": "unresolved_ticker_name", "ticker": ticker, "detail": str(exc)}
        ) from exc
    except Exception as exc:
        # Structured error code (not pre-rendered text) so the client's i18n
        # layer renders this in whichever language is active.
        raise HTTPException(
            422, {"error_code": "ticker_not_found", "ticker": ticker, "detail": str(exc)}
        ) from exc
    # An explicit human add is exactly the "no longer speculative" signal the
    # marker exists to be cleared by (see state.unmark_trade_allowlist_auto_added's
    # docstring). Without this, reconcile_trade_allowlist's prune loop still
    # deletes a ticker a person deliberately put on the list, the moment it has
    # zero shares and no pending order.
    #
    # Unconditional, not `if status == "added"`: the case that matters is
    # status == "exists" — the auto-add already put it on the list, and this
    # request IS the human confirming it.
    #
    # HERE and not in state.add_ticker_to_trade_allowlist, deliberately: that
    # core is shared with app.orchestrator.pipeline's auto-add-on-BUY hook,
    # which marks auto-added right after calling it. "A human did this on
    # purpose" is a router-level fact, not a core-level one. Only reached past
    # the try/except above, so a rejected add never clears the marker.
    # 사람이 직접 추가한 것 = 더 이상 투기적 추가가 아님. 공용 코어가 아니라
    # 라우터에 두는 이유는 파이프라인 자동추가 훅이 같은 코어를 쓰기 때문.
    await state.unmark_trade_allowlist_auto_added(ticker)
    return {"status": status, "ticker": ticker, "name": info.name if info else ticker}


@router.post("/api/trade-allowlist/remove")
async def trade_allowlist_remove(req: TradeAllowlistRemoveRequest) -> dict:
    """trade_allowlist에서 빼는 건 "목록에서 숨기기"가 아니다 —
    app/live/exit_monitor.py와 app/web/state.py의 stop_loss_loop는 둘 다
    보유 포지션이 아니라 settings.trade_allowlist를 순회하므로 실시간 손절/
    트레일링 스탑 감시가 끊기고, app/execution/kis_bridge.py의
    `ticker not in settings.trade_allowlist` 게이트는 side 검사보다 앞에
    있어 **매도 주문까지** BLOCKED된다. 자동 복구(reconcile_trade_allowlist)는
    auto_trade_loop 안에서만 돌고 EXECUTION_MODE=manual이면 그 사이클 자체가
    스킵되므로, 수동 모드에선 되돌아오지도 않는다.

    그래서 보유 중(또는 KIS 장애로 보유 여부를 확인할 수 없을 때)에는 기본
    거부(409)하고, 사람이 확인창에서 승인한 경우에만 `force=true`로 통과시킨다.
    하드 차단이 아닌 이유: 원장이 틀린 유령 포지션(2026-08-13 008290.KS)이
    생기면 그 티커를 UI에서 영원히 지울 수 없게 된다.

    가드는 이 라우터에만 있고 state.remove_ticker_from_trade_allowlist에는
    없다 — reconcile_trade_allowlist의 자동 prune과
    trade_allowlist_admin.reseed_trade_allowlist는 자기 판단으로 부르는
    경로라 사람 확인 개념이 없다."""
    ticker = req.ticker.strip().upper()
    if not req.force:
        check = await state.held_shares_for_removal(ticker)
        if not check.known:
            raise HTTPException(
                409,
                {
                    "error_code": "holdings_unknown",
                    "ticker": ticker,
                    "shares": 0,
                    "source": check.source,
                    "detail": f"cannot confirm whether {ticker} is currently held (KIS lookup unavailable) — removing it would stop stop-loss monitoring and block sell orders",
                },
            )
        if check.shares > 0:
            raise HTTPException(
                409,
                {
                    "error_code": "held_position_exists",
                    "ticker": ticker,
                    "shares": check.shares,
                    "source": check.source,
                    "detail": f"{ticker} still has {check.shares} shares held — removing it would stop stop-loss monitoring and block sell orders",
                },
            )
    await state.remove_ticker_from_trade_allowlist(ticker)
    return {"status": "removed", "ticker": ticker}


@router.get("/api/trade-allowlist-info")
async def trade_allowlist_info() -> list[dict]:
    return [
        {
            "symbol": ticker,
            "name": info.name if (info := state._display_info.get(ticker)) else ticker,
            "previous_close": info.previous_close if info else None,
            "currency": info.currency if info else None,
            "market_open": is_market_open(ticker),
        }
        for ticker in settings.trade_allowlist
    ]


# ---- watchlist (light "관심종목" concept, no trading permission) --------
# No frontend wiring yet (deliberately scoped out of this step — see
# memory: kis-integration-status) — these exist so the backend split can
# be exercised/tested ahead of the UI work.


@router.post("/api/watchlist/add")
async def watchlist_add(req: TradeAllowlistRequest) -> dict:
    """Add a ticker to the watchlist only — no trading permission, no realtime
    feed subscription, no history seeding (see
    `app.web.state.add_ticker_to_watchlist`). Unlike trade_allowlist_add, this
    does not reject leveraged/inverse tickers — the watchlist is just tracking
    interest, not a trading decision."""
    ticker = req.ticker.strip().upper()
    if not ticker:
        raise HTTPException(400, "ticker is required")
    ticker = validate_ticker(ticker)
    try:
        status, info = await state.add_ticker_to_watchlist(ticker)
    except state.MalformedTickerError as exc:
        raise HTTPException(
            422, {"error_code": "invalid_ticker_format", "ticker": ticker, "detail": str(exc)}
        ) from exc
    except Exception as exc:
        raise HTTPException(
            422, {"error_code": "ticker_not_found", "ticker": ticker, "detail": str(exc)}
        ) from exc
    return {"status": status, "ticker": ticker, "name": info.name if info else ticker}


@router.post("/api/watchlist/remove")
async def watchlist_remove(req: TradeAllowlistRequest) -> dict:
    """Removing a ticker still on the trade allowlist is refused — that
    would violate the trade_allowlist ⊆ watchlist invariant (see
    app.config.Settings). Remove it from the trade allowlist first
    (`POST /api/trade-allowlist/remove`)."""
    ticker = req.ticker.strip().upper()
    if ticker in settings.trade_allowlist:
        raise HTTPException(
            409,
            {
                "error_code": "still_on_trade_allowlist",
                "ticker": ticker,
                "detail": f"{ticker} is still on the trade allowlist — remove it from there first",
            },
        )
    await state.remove_ticker_from_watchlist(ticker)
    return {"status": "removed", "ticker": ticker}


@router.get("/api/watchlist-info")
async def watchlist_info() -> list[dict]:
    return [
        {
            "symbol": ticker,
            "name": info.name if (info := state._display_info.get(ticker)) else ticker,
            "previous_close": info.previous_close if info else None,
            "currency": info.currency if info else None,
            "on_trade_allowlist": ticker in settings.trade_allowlist,
        }
        for ticker in settings.watchlist
    ]
