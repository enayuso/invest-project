# app/brokers/kis/client.py
# KIS-specific request plumbing built on the broker-agnostic BaseHttpClient:
# token-backed auth headers, the EGW00201 rate-limit retry, and rt_cd
# response-envelope checking. All domestic/overseas mixins call these
# methods via `self.` — they never talk to httpx directly.
# KIS 전용 요청 처리 (인증 헤더/재시도/응답 검증) — 공통 HTTP 기반 위에 구축

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
import sys
import time
from collections import deque
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TypeVar

import httpx

from app.brokers.base import BaseHttpClient
from app.brokers.kis.auth import KISAuth
from app.brokers.kis.const import KISBrokerError, is_rate_limited, token_cache_file

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

# Per-msg_cd retry backoff. Both EGW00201 (gateway per-second cap) and
# EGW00215 (the LEDGER layer's tighter, separate per-second inquiry cap)
# escalate over two retries: a rate rejection is a self-inflicted momentary
# burst, not an account-state problem, so a short retry is right — fail-closed
# here would both block the trade AND make pipeline._portfolio_risk_context
# inject a fictional 100% drawdown into the LLM.
#
# EGW00201 used to keep a single un-jittered 1.0s retry; on 2026-09-03
# decision id=1179 the ~9 concurrent Fundamentals-Analyst KIS calls that all
# tripped it woke together off that one fixed sleep = thundering herd, and the
# report failed anyway. Widened to two attempts and jittered at the call site
# (below) so the retries de-synchronize. Max added wait ≈ 0.6+1.8 = 2.4s
# before jitter (×1.4 worst case ≈ 3.4s), still inside rebalance_to_target's
# portfolio_risk_lock_wait_seconds=10.0 budget.
# 레이트 거부는 순간 버스트라 짧게 재시도 + 지터로 herd 분산 — fail-closed는 매매도 막고 LLM 서술도 오염.
_RATE_LIMIT_RETRY_SLEEPS = {
    "EGW00201": (0.6, 1.8),
    "EGW00215": (1.0, 2.0),
}

# 2026-10-02 진단 로그(동작 불변): 실계좌에서 매분 :00 즈음 EGW00215 재시도가 전부 실패하는데
# 그 순간 원장을 누가 부르는지 알 수 없었다 — 최근 요청의 호출자를 링버퍼에 남겨 거부 시 1줄로 덤프한다.
# Diagnostic only: ring buffers of recent ledger/general requests (monotonic time, short path,
# caller outside app/brokers) dumped as ONE warning on the first EGW00215/EGW00201 rejection.
_LEDGER_RECENT: deque = deque(maxlen=30)   # (t, path_name, caller) — ledger-lane TRs only
_GENERAL_RECENT: deque = deque(maxlen=60)  # (t, path_name, caller) — every request
_LEDGER_CTX_WINDOW_S = 5.0
_GENERAL_CTX_WINDOW_S = 2.0
_CTX_MAX_ITEMS = 12


def _external_caller() -> str:
    """`module:function` of the first stack frame outside app/brokers, else "?"."""
    try:
        f = sys._getframe(2)
        while f is not None:
            mod = f.f_globals.get("__name__", "?")
            if not mod.startswith("app.brokers"):
                return f"{mod}:{f.f_code.co_name}"
            f = f.f_back
    except Exception:
        pass
    return "?"


def _record_request(path: str, ledger: bool) -> None:
    try:
        entry = (time.monotonic(), path.rsplit("/", 1)[-1] or path, _external_caller())
        _GENERAL_RECENT.append(entry)
        if ledger:
            _LEDGER_RECENT.append(entry)
    except Exception:
        pass


def _format_recent(buf: deque, window_s: float) -> str:
    now = time.monotonic()
    items = [e for e in buf if now - e[0] <= window_s]
    items.sort(key=lambda e: e[0], reverse=True)
    items = items[:_CTX_MAX_ITEMS]
    return "[" + ", ".join(f'"-{now - t:.2f}s {name} <- {caller}"' for t, name, caller in items) + "]"


def _log_rate_limit_context(code: str, path: str) -> None:
    try:
        if code == "EGW00215":
            buf, window, label = _LEDGER_RECENT, _LEDGER_CTX_WINDOW_S, "ledger TRs"
            kind = "ledger"
        else:
            buf, window, label = _GENERAL_RECENT, _GENERAL_CTX_WINDOW_S, "requests"
            kind = "general"
        now = time.monotonic()
        n = sum(1 for e in buf if now - e[0] <= window)
        logger.warning(
            "KIS %s rate-limit context (%s on %s): last %d %s within %gs: %s",
            kind, code, path, min(n, _CTX_MAX_ITEMS), label, window, _format_recent(buf, window),
        )
    except Exception:
        pass


# Every paginated KIS inquiry TR (balance, executions — domestic and
# overseas) caps at this many continuation pages; shared here instead of
# redefined per mixin.
MAX_PAGINATION_PAGES = 10


class KISHttpClient(BaseHttpClient):
    def __init__(
        self,
        *,
        app_key: str,
        app_secret: str,
        base_url: str = "https://openapivts.koreainvestment.com:29443",
        token_cache_path: Path | None = None,
        client: httpx.AsyncClient | None = None,
        min_request_interval: float = 1.0,
        account_min_request_interval: float = 0.0,
        max_requests_per_second: float = 0.0,
        is_real: bool = False,
    ) -> None:
        super().__init__(
            base_url=base_url, client=client, min_request_interval=min_request_interval,
            account_min_request_interval=account_min_request_interval,
            max_requests_per_second=max_requests_per_second,
        )
        self._app_key = app_key
        self._app_secret = app_secret
        self._token_cache_path = token_cache_path or token_cache_file()
        # Real vs paper (모의) — every domestic/overseas mixin method picks
        # its TR ID off this via self._tr(real_id, paper_id) instead of
        # branching individually. Defaults False (paper) so a broker
        # constructed without this kwarg never accidentally targets a real
        # account. See memory: KIS 실전투자 안전 규칙.
        self._is_real = is_real
        # Cache-file separation across accounts (paper vs real, and now
        # per-user) is the CALLER's job — see app.brokers.kis.broker's
        # `token_cache_path=token_cache_path_for(creds.app_key)`, which keys
        # by a hash of the app key itself. That subsumes the narrower
        # is_real-only split this constructor used to compute here: two
        # different app keys already get two different files regardless of
        # paper/real, and a per-user deployment has more than two accounts
        # to keep separate anyway.
        self._auth = KISAuth(
            app_key=app_key,
            app_secret=app_secret,
            base_url=self._base_url,
            token_cache_path=self._token_cache_path,
            client=self._client,
            throttle_func=self._throttle,
        )

    def _tr(self, real_id: _T, paper_id: _T) -> _T:
        """Picks the real- or paper-account TR ID for this broker instance's
        mode — the one place that decision is made, so mixins never branch
        on self._is_real individually. Usually a plain TR ID string; some
        callers pass a {side: tr_id} dict instead (e.g. buy/sell order TRs)
        and index into the result themselves."""
        return real_id if self._is_real else paper_id

    async def _trading_headers(self, tr_id: str) -> dict[str, str]:
        token = await self._auth._get_token()
        return {
            "authorization": f"Bearer {token}",
            "appkey": self._app_key,
            "appsecret": self._app_secret,
            "tr_id": tr_id,
            "custtype": "P",
        }

    async def _request(
        self,
        method: str,
        path: str,
        tr_id: str,
        *,
        params: dict | None = None,
        json_body: dict | None = None,
        tr_cont: str = "",
        priority: str = "low",
        lane: str | None = None,
        _ledger: bool = False,
    ) -> httpx.Response:
        """Throttled request with a per-msg_cd retry on a rate-limit
        rejection (see `_RATE_LIMIT_RETRY_SLEEPS`) — a burst can slip past
        the client-side throttle, and KIS's ledger layer has its own cap
        (EGW00215) that the gateway throttle doesn't bound.

        `priority` ("low" default, "high" for order/account-safety calls —
        see each mixin's own call sites) is forwarded to the shared
        throttle so order submission/cancellation, buyable/sellable,
        balance, executions, and stock/VI status never queue behind routine
        quote/rank/news polling. Every other TR stays "low" by not passing
        this at all — explicit opt-in only, so a call is never accidentally
        high-priority.

        `lane="account"` additionally routes the whole call (retry loop
        included) through `account_lane()` — the ledger-inquiry rate gate.
        Order-submission TRs deliberately never pass this."""
        headers = {**await self._trading_headers(tr_id), "tr_cont": tr_cont}
        lane_cm = self.account_lane() if lane == "account" else contextlib.nullcontext()
        async with lane_cm:
            resp: httpx.Response | None = None
            for attempt in range(1, 4):
                await self._throttle(priority)
                _record_request(path, lane == "account" or _ledger)
                resp = await self._client.request(
                    method, f"{self._base_url}{path}", headers=headers, params=params, json=json_body
                )
                # No status-code gate: KIS returns a rate-limit rejection
                # both as HTTP 500 and as HTTP 200 with rt_cd != "0" (the
                # latter would otherwise fall through to _checked_body and
                # raise instead of retrying).
                code = is_rate_limited(resp.text)
                sleeps = _RATE_LIMIT_RETRY_SLEEPS.get(code, ()) if code else ()
                if code and attempt == 1:
                    _log_rate_limit_context(code, path)  # once per request, first rejection only
                if attempt <= len(sleeps):
                    # WARNING, not info: the client-side throttle already
                    # failed to prevent the rejection — worth noticing in
                    # aggregate. msg_cd verbatim so `grep EGW00215` works.
                    # Jittered so that a burst of concurrent calls all rejected
                    # in the same tick don't wake together and re-collide
                    # (2026-09-03 decision id=1179 thundering herd).
                    base = max(self._min_request_interval, sleeps[attempt - 1])
                    delay = base * random.uniform(0.7, 1.4)
                    logger.warning(
                        "KIS rate limit %s on %s — retry %d/%d after %.2fs",
                        code, path, attempt, len(sleeps), delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                return resp
            return resp  # retries exhausted — _checked_body raises on it

    def _checked_body(self, resp: httpx.Response, what: str) -> dict:
        if resp.status_code != 200:
            raise KISBrokerError(f"{what} failed: HTTP {resp.status_code} {resp.text[:200]}")
        body = resp.json()
        if body.get("rt_cd") != "0":
            raise KISBrokerError(
                f"{what} refused: [{body.get('msg_cd')}] {body.get('msg1')}",
                msg_cd=body.get("msg_cd"), rt_cd=body.get("rt_cd"),
            )
        return body

    async def _paginate(
        self,
        method: str,
        path: str,
        tr_id: str,
        params: dict,
        *,
        what: str,
        ctx_suffix: str = "100",
        priority: str = "low",
        lane: str | None = None,
    ) -> AsyncIterator[dict]:
        """Runs the KIS continuation-page loop shared by every paginated
        inquiry TR (CTX_AREA_FK{suffix}/NK{suffix} request params carried
        forward from each page's ctx_area_fk{suffix}/nk{suffix} response
        fields, tr_cont "M"/"F" response header signaling more pages),
        yielding each page's checked body. `params` is mutated in place with
        the next page's continuation keys — pass a dict the caller doesn't
        need to reuse afterward. Row/summary extraction (output1 vs output,
        position vs execution) stays with the caller since it differs per
        TR; only the pagination mechanics are shared.
        KIS 연속조회 루프 공통화(잔고/체결내역, 국내/해외 4곳 복붙 제거) — 페이지별
        필드 추출은 호출부 담당, 여기선 페이지 넘기기만."""
        fk_key, nk_key = f"CTX_AREA_FK{ctx_suffix}", f"CTX_AREA_NK{ctx_suffix}"
        ctx_fk_field, ctx_nk_field = f"ctx_area_fk{ctx_suffix}", f"ctx_area_nk{ctx_suffix}"
        req_tr_cont = ""
        # Take the ledger-inquiry lane ONCE around the whole continuation
        # loop — not per page. Passing `lane` into each `_request` would
        # deadlock (the lane lock is not re-entrant) and, even without that,
        # would add `account_min_interval` between every one of up to 10
        # pages of a single logical inquiry.
        # 연속조회 루프 전체를 레인 1회로 감쌈 — 페이지마다 잡으면 데드락+과지연.
        lane_cm = self.account_lane() if lane == "account" else contextlib.nullcontext(None)
        async with lane_cm as lane_hold:
            truncated = True
            for page_no in range(MAX_PAGINATION_PAGES):
                # Space individual pages WITHIN the hold, not just holds of
                # the lane against each other — see `_AccountLaneHold`'s own
                # docstring (app/brokers/base/http_client.py) for why the
                # lane's own entry-wait alone let up to MAX_PAGINATION_PAGES
                # ledger TRs fire back-to-back inside one hold (the id=1187
                # EGW00215 fail-closed BUY block, 2026-09-03). Skipped before
                # the first page — nothing to space against yet.
                if page_no and lane_hold is not None:
                    await lane_hold.space()
                resp = await self._request(method, path, tr_id, params=params, tr_cont=req_tr_cont, priority=priority,
                                           _ledger=lane == "account")
                body = self._checked_body(resp, what)
                yield body
                resp_tr_cont = resp.headers.get("tr_cont", "").strip()
                if resp_tr_cont not in ("M", "F"):
                    truncated = False
                    break
                params[fk_key] = body.get(ctx_fk_field, "")
                params[nk_key] = body.get(ctx_nk_field, "")
                req_tr_cont = "N"
            if truncated:
                # Loop exhausted MAX_PAGINATION_PAGES without the server ever
                # signaling "no more pages" (tr_cont not M/F) — remaining rows
                # were silently dropped. Previously unlogged, so truncated
                # 90-day PnL/executions queries looked like complete data.
                logger.warning(
                    "%s: hit MAX_PAGINATION_PAGES=%d with more pages still available — results truncated",
                    what, MAX_PAGINATION_PAGES,
                )
