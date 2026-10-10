"""KIS real-time WebSocket — market-operation (CB) and trade ticks.

The rest of the system is REST-polling; this is the one push channel. For the
domestic (KRX) trade allowlist codes it subscribes to two unencrypted real-time
feeds and keeps the latest per-code state:

- H0STMKO0 장운영정보: trading-halt / VI / market-operation code (carries
  circuit-breaker state — there is no REST "is a CB active?" call).
- H0STCNT0 실시간체결가: live price / change% / volume, for a truly live tick.

H0STASP0 실시간호가 (10-level bid/ask orderbook) was dropped from the default
subscription set 2026-08-05: KIS caps combined real-time registrations at 41
per session (per-account, across every product), and this feed's own `run()`
already had a comment (2026-07-27) documenting that len(codes) * len(trs)
exceeds that on its own for any real-size trade allowlist — orderbook depth is a
dashboard-only display (app.web.routers.market_status.orderbook_endpoint
already degrades gracefully to tick-only data with empty asks/bids), unlike
H0STMKO0 which every circuit-breaker/VI safety gate depends on. Dropping it
buys back one whole TR's worth of registration budget without touching any
gate. See MAX_CODES below for the code-count side of the same cap.

Design notes:
- Approval key (not the REST token) authenticates the socket: POST
  /oauth2/Approval with appkey/secretkey (note `secretkey`, not `appsecret`).
- Paper socket is ws://ops.koreainvestment.com:31000 (no path); verified live
  that a subscribe returns "SUBSCRIBE SUCCESS".
- These TRs are unencrypted (encrypt=N) so no AES is needed. A data frame is
  `0|<tr_id>|<count>|<^-separated fields>`; system messages (subscribe ack,
  PINGPONG) arrive as JSON.
- Fire-and-degrade: any failure just leaves the last-known state and retries;
  the WS must never crash the dashboard it runs inside.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import ClassVar

import httpx
import websockets
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad

from app.brokers.kis.parser import _num

logger = logging.getLogger(__name__)

REAL_WS_URL = "ws://ops.koreainvestment.com:21000"
PAPER_WS_URL = "ws://ops.koreainvestment.com:31000"
_MARKET_STATUS_TR = "H0STMKO0"  # 실전/모의 공통
_TRADE_TR = "H0STCNT0"  # 실전/모의 공통
_ORDERBOOK_TR = "H0STASP0"  # 실전/모의 공통
_FILL_TR_REAL = "H0STCNI0"  # 체결통보 (실전) — tr_key is the HTS ID; frames are AES-encrypted
_FILL_TR_PAPER = "H0STCNI9"  # 체결통보 (모의) — tr_key is the HTS ID; frames are AES-encrypted

# H0STCNI9/H0STCNI0 field order (^-separated, after AES decryption) — identical for both
_CNI_COLUMNS = [
    "cust_id", "acnt_no", "oder_no", "ooder_no", "seln_byov_cls", "rctf_cls",
    "oder_kind", "oder_cond", "code", "cntg_qty", "cntg_unpr", "cntg_hour",
    "rfus_yn", "cntg_yn", "acpt_yn", "brnc_no", "oder_qty", "acnt_name",
    "ord_cond_prc", "ord_exg_gb", "popup_yn", "filler", "crdt_cls",
    "crdt_loan_date", "cntg_isnm40", "oder_prc",
]


def aes_cbc_b64_decrypt(key: str, iv: str, cipher_b64: str) -> str:
    """AES-128/256-CBC + base64, matching KIS's real-time encryption. The
    key/iv are the plaintext strings the subscribe ACK returns for an
    encrypted TR."""
    cipher = AES.new(key.encode("utf-8"), AES.MODE_CBC, iv.encode("utf-8"))
    return unpad(cipher.decrypt(base64.b64decode(cipher_b64)), AES.block_size).decode("utf-8")

# H0STMKO0 field order (^-separated), per the KIS sample.
_MKO_COLUMNS = [
    "mksc_shrn_iscd", "trht_yn", "tr_susp_reas_cntt", "mkop_cls_code",
    "antc_mkop_cls_code", "mrkt_trtm_cls_code", "divi_app_cls_code",
    "iscd_stat_cls_code", "vi_cls_code", "ovtm_vi_cls_code", "EXCH_CLS_CODE",
]

# H0STCNT0 (실시간 체결가) — only the leading fields we use are named; the TR
# emits ~46 fields but the ones we render are all near the front.
_CNT_COLUMNS = [
    "code", "time", "price", "vrss_sign", "vrss", "change_pct", "wghn_avg",
    "open", "high", "low", "ask1", "bid1", "cntg_vol", "acml_vol",
]

# H0STASP0 (실시간 호가) — code, time, hour-cls, then ASKP1..10, BIDP1..10,
# ASKP_RSQN1..10, BIDP_RSQN1..10.
_ASP_COLUMNS = (
    ["code", "time", "hour_cls"]
    + [f"askp{i}" for i in range(1, 11)]
    + [f"bidp{i}" for i in range(1, 11)]
    + [f"askp_rsqn{i}" for i in range(1, 11)]
    + [f"bidp_rsqn{i}" for i in range(1, 11)]
)


@dataclass
class MarketStatus:
    code: str
    trading_halted: bool
    halt_reason: str
    market_op_code: str
    vi_code: str
    stock_state_code: str


@dataclass
class Tick:
    code: str
    time: str
    price: float
    change_pct: float
    volume: int  # 누적거래량 (acml_vol)
    ask1: float
    bid1: float


@dataclass
class OrderBook:
    code: str
    time: str
    asks: list[dict]  # [{price, qty}] best-first, up to 10
    bids: list[dict]


@dataclass
class Fill:
    """A 체결통보 record: an order that was accepted, filled, or rejected."""
    order_no: str  # 주문번호 (oder_no)
    code: str  # 종목코드 (code)
    side: str  # "buy" | "sell"
    order_qty: int  # 주문수량 (oder_qty)
    fill_qty: int  # 체결수량 (cntg_qty)
    fill_price: float  # 체결단가 (cntg_unpr)
    time: str  # 체결시각 (cntg_hour)
    status: str  # "filled" | "accepted" | "rejected"
    name: str  # 종목명 (cntg_isnm40)


def parse_fill_frame(decrypted: str, count: int = 1) -> list[Fill]:
    """Parse an already-AES-decrypted H0STCNI9 body (`f^f^...`) into Fills.
    Pure/testable — decryption is handled separately by the feed.

    `count` is the frame's own record count (the outer `0|tr|count|payload`
    frame's 3rd field — still plaintext, since only `payload` is encrypted),
    same authoritative role `_frame_rows` gives it for the other three TRs:
    each record's true field width is `len(fields) // count`, not a fixed
    `len(_CNI_COLUMNS)`. Hardcoding the named-column count here was exactly
    the bug already found and fixed for market-status/trade/orderbook frames
    (see `_frame_rows`'s docstring) — left unfixed here meant more than one
    fill notification in a single frame would silently mis-chunk or drop a
    trailing partial record."""
    fields = decrypted.split("^")
    if count <= 0:
        return []
    width = len(fields) // count
    if width <= 0:
        return []
    out: list[Fill] = []
    for i in range(0, width * count, width):
        row = dict(zip(_CNI_COLUMNS, fields[i:i + width]))
        if not row.get("oder_no"):
            continue
        if row.get("rfus_yn") == "Y":
            status = "rejected"
        elif row.get("cntg_yn") == "2":  # 2 = 체결, 1 = 주문접수
            status = "filled"
        else:
            status = "accepted"
        out.append(Fill(
            order_no=row.get("oder_no", ""),
            code=row.get("code", ""),
            side="sell" if row.get("seln_byov_cls") == "01" else "buy",
            order_qty=int(_num(row.get("oder_qty"))),
            fill_qty=int(_num(row.get("cntg_qty"))),
            fill_price=_num(row.get("cntg_unpr")),
            time=row.get("cntg_hour", ""),
            status=status,
            name=(row.get("cntg_isnm40") or "").strip(),
        ))
    return out


def _frame_rows(raw: str, tr_id: str, columns: list[str]) -> list[dict]:
    """Split a real-time frame `0|tr|count|f^f^...^f^f^...` into one dict per
    record. `count` (the frame's own 3rd field) is the authoritative number
    of concatenated records — each record's TRUE field width is
    `len(all_fields) // count`, which for TRs like H0STCNT0 (~46 fields/
    record) is much wider than `columns` (the handful we actually name).
    Chunking by `len(columns)` instead of the true width (a previous bug)
    chopped a single record's unused trailing fields into bogus extra
    "records" with garbage codes/prices — harmless-looking until something
    actually reads the polluted dict, which real trading-hours volume (many
    real fields, still count="1") exposed immediately. `zip` naturally
    drops each record's un-named trailing fields once truncated to
    `columns`, so nothing else needs to change.
    실제 프레임의 count 필드로 레코드 폭을 구함(이전엔 이름 붙인 컬럼 수로 억지로
    나눠서, 필드 많은 레코드일수록 남는 필드가 가짜 레코드로 오염됐음)."""
    if not raw or raw[0] not in ("0", "1"):
        return []
    parts = raw.split("|", 3)
    if len(parts) < 4 or parts[1] != tr_id:
        return []
    count = int(parts[2]) if parts[2].isdigit() else 1
    if count <= 0:
        return []
    fields = parts[3].split("^")
    true_width = len(fields) // count
    if true_width <= 0:
        return []
    return [
        dict(zip(columns, fields[i : i + true_width]))
        for i in range(0, true_width * count, true_width)
    ]


def parse_market_status_frame(raw: str) -> list[MarketStatus]:
    """H0STMKO0 -> MarketStatus per record."""
    out = []
    for row in _frame_rows(raw, _MARKET_STATUS_TR, _MKO_COLUMNS):
        if not row.get("mksc_shrn_iscd"):
            continue
        out.append(MarketStatus(
            code=row.get("mksc_shrn_iscd", ""),
            trading_halted=row.get("trht_yn", "N") == "Y",
            halt_reason=(row.get("tr_susp_reas_cntt") or "").strip(),
            market_op_code=(row.get("mkop_cls_code") or "").strip(),
            vi_code=(row.get("vi_cls_code") or "").strip(),
            stock_state_code=(row.get("iscd_stat_cls_code") or "").strip(),
        ))
    return out


def parse_trade_frame(raw: str) -> list[Tick]:
    """H0STCNT0 -> Tick per record."""
    out = []
    for row in _frame_rows(raw, _TRADE_TR, _CNT_COLUMNS):
        if not row.get("code"):
            continue
        out.append(Tick(
            code=row["code"],
            time=row.get("time", ""),
            price=_num(row.get("price")),
            change_pct=_num(row.get("change_pct")),
            volume=int(_num(row.get("acml_vol"))),
            ask1=_num(row.get("ask1")),
            bid1=_num(row.get("bid1")),
        ))
    return out


def parse_orderbook_frame(raw: str) -> list[OrderBook]:
    """H0STASP0 -> OrderBook per record (levels with zero price dropped)."""
    out = []
    for row in _frame_rows(raw, _ORDERBOOK_TR, _ASP_COLUMNS):
        if not row.get("code"):
            continue
        asks = [{"price": _num(row.get(f"askp{i}")), "qty": int(_num(row.get(f"askp_rsqn{i}")))}
                for i in range(1, 11)]
        bids = [{"price": _num(row.get(f"bidp{i}")), "qty": int(_num(row.get(f"bidp_rsqn{i}")))}
                for i in range(1, 11)]
        out.append(OrderBook(
            code=row["code"],
            time=row.get("time", ""),
            asks=[a for a in asks if a["price"] > 0],
            bids=[b for b in bids if b["price"] > 0],
        ))
    return out


# Every other authenticated KIS call goes through BaseHttpClient._throttle
# (min-interval spacing between requests), including REST token issuance —
# but this is a bare module function with no KISHttpClient instance to
# borrow that throttle state from (RealtimeFeed.run() calls it directly on
# every reconnect, outside the REST client entirely). Same min-interval
# gate algorithm as BaseHttpClient._throttle, reimplemented here as module
# state since there's no instance to attach it to: without it, a burst of
# transient WS disconnects (network blip, or KIS's documented "one session
# per appkey" collision) could hit /oauth2/Approval faster than KIS's own
# rate limit, unthrottled by anything on our side.
_approval_key_throttle_lock = asyncio.Lock()
_last_approval_key_request_at = 0.0
_APPROVAL_KEY_MIN_INTERVAL = 1.0  # seconds — matches BaseHttpClient's default


async def _throttle_approval_key() -> None:
    global _last_approval_key_request_at
    async with _approval_key_throttle_lock:
        loop = asyncio.get_running_loop()
        wait = _last_approval_key_request_at + _APPROVAL_KEY_MIN_INTERVAL - loop.time()
        if wait > 0:
            await asyncio.sleep(wait)
        _last_approval_key_request_at = loop.time()


async def fetch_approval_key(base_url: str, app_key: str, app_secret: str) -> str:
    await _throttle_approval_key()
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            f"{base_url.rstrip('/')}/oauth2/Approval",
            json={"grant_type": "client_credentials", "appkey": app_key, "secretkey": app_secret},
        )
    resp.raise_for_status()
    key = resp.json().get("approval_key")
    if not key:
        raise RuntimeError(f"no approval_key in response: {resp.text[:200]}")
    return key


@dataclass
class RealtimeFeed:
    """Subscribes market-operation + trade + orderbook feeds for a set of KRX
    codes and keeps their latest state. Start with
    `asyncio.create_task(feed.run())`; read `feed.statuses/ticks/orderbook`."""

    # Combined real-time registration cap: KIS allows 41 per session across
    # every product (2026-04-20 notice). One slot is reserved for 체결통보
    # (see run()'s own comment on why it's subscribed first), leaving 40
    # registrations to spend across codes at 1-2 TRs/code depending on tier
    # (see cb_only_codes below) — MAX_REGISTRATIONS is the real budget;
    # MAX_CODES is kept only as the all-2TR-codes special case (every
    # existing caller/test that assumes uniform 2TR/code still holds).
    # ClassVars, not dataclass fields: a property of the KIS limit itself,
    # not per-instance state.
    MAX_REGISTRATIONS: ClassVar[int] = 40
    MAX_CODES: ClassVar[int] = 20

    # How long a last-known market-status snapshot stays trustworthy once
    # disconnected, before current_status() starts reporting it as unknown
    # instead. KIS can hold a dropped session ("ALREADY IN USE") for a few
    # minutes on reconnect (see run()'s own docstring) — a brief blip
    # shouldn't halt all domestic trading over a stale-by-a-clock technicality,
    # but a reconnect stuck for minutes must not keep answering "not halted"
    # off a snapshot from before whatever's happening right now.
    _RECONNECT_STALE_AFTER_SECONDS: ClassVar[float] = 60.0

    app_key: str
    app_secret: str
    codes: list[str]
    base_url: str = "https://openapivts.koreainvestment.com:29443"
    ws_url: str = PAPER_WS_URL
    trs: tuple[str, ...] = (_MARKET_STATUS_TR, _TRADE_TR)
    hts_id: str | None = None  # enables the 체결통보 subscription
    # Real vs paper — only the 체결통보 TR differs (H0STCNI0 vs H0STCNI9); the
    # other three TRs and the frame formats are 실전/모의 공통. Caller (see
    # app/web/state.py) is responsible for also passing the matching
    # base_url/ws_url for the same mode.
    is_real: bool = False
    statuses: dict[str, MarketStatus] = field(default_factory=dict)
    ticks: dict[str, Tick] = field(default_factory=dict)
    # Wall-clock receipt time per code's last H0STCNT0 trade tick — added
    # for app.live.exit_monitor (2026-08-10), which needs to know how
    # fresh `ticks[code]` actually is. `Tick.time` (KIS's own wire field)
    # is only an HHMMSS string with no date, so it can't answer that on its
    # own. Wall-clock (not monotonic() like `_status_received_at` below) is
    # deliberate here: this is a short-lived freshness check (tens of
    # seconds), not a long-reconnect state machine where a system clock
    # jump matters, and a plain wall timestamp lets exit_monitor compare it
    # directly against a position's real entry timestamp (also wall-clock,
    # from `trades.created_at`) with one consistent clock.
    tick_received_at: dict[str, datetime] = field(default_factory=dict)
    orderbook: dict[str, OrderBook] = field(default_factory=dict)
    fills: list[Fill] = field(default_factory=list)  # newest last, capped
    _crypto: dict[str, tuple[str, str]] = field(default_factory=dict)  # tr_id -> (key, iv)
    # (tr_id, tr_key) pairs KIS actually ACKed rt_cd="0" for — distinct from
    # `codes`/`trs`, which only record what we ATTEMPTED to register. A
    # rejection ("MAX SUBSCRIBE OVER", "ALREADY IN SUBSCRIBE", ...) never
    # lands here, so a caller can tell "genuinely no event yet on a
    # confirmed subscription" apart from "this registration was silently
    # dropped and will never deliver a frame" — see is_subscribed().
    _subscribed: set[tuple[str, str]] = field(default_factory=set)
    # 2026-10-02: codes add_code() refused because MAX_REGISTRATIONS was spent
    # (never even attempted on the wire) — lets app.live.kis_market_status
    # tell "registration budget exhausted" apart from "ACK not yet received"
    # (both used to be "code_not_subscribed"). A later successful add_code
    # removes the code; a rebuilt feed starts empty.
    capped_codes: set[str] = field(default_factory=set)
    # 2026-10-02: (tr_id, tr_key) pairs KIS answered with rt_cd != "0" — the
    # registration was attempted and explicitly rejected. Cleared on
    # reconnect together with _subscribed (every registration is retried).
    rejected_keys: set[tuple[str, str]] = field(default_factory=set)
    # monotonic() timestamp of the last H0STMKO0 frame per code — lets
    # current_status() distinguish "genuinely quiet all session" (fine, no
    # time bound) from "we've been disconnected/reconnecting long enough
    # that this snapshot predates whatever's happening right now" (unknown).
    # Only checked while NOT connected — see current_status()'s own comment
    # for why a connected-but-quiet code must never go stale by a clock.
    _status_received_at: dict[str, float] = field(default_factory=dict)
    _stop: bool = False
    # True only between a successful subscribe and the next disconnect/retry —
    # lets app.live.kis_market_status tell "connected+subscribed, no event
    # yet" (steady-state on a quiet day) apart from "genuinely not connected."
    connected: bool = False
    # The currently-open connection + its approval key, captured by run()
    # while connected — lets add_code() send an immediate subscribe message
    # on the live socket instead of waiting for the next reconnect. None
    # whenever not connected (cleared in run()'s finally alongside
    # `connected`), so add_code() never sends on a socket that's already
    # closed/closing.
    _ws: object | None = field(default=None, repr=False)
    _approval_key: str | None = field(default=None, repr=False)
    # Codes trading H0STCNT0 (trade-tick) coverage for H0STASP0 (orderbook)
    # instead — 2026-08-24, see app.web.state._prioritized_krx_codes's
    # _ORDERBOOK_ZONE_SIZE comment for why this trade is being made at all
    # (forward-collecting orderbook depth for scalping research within the
    # same 41-registration cap, at the cost of live trade-tick coverage for
    # the trailing, currently-unheld tail of the subscribe list). Every code
    # here MUST also be in `codes` — __post_init__'s truncation and
    # add_code() both keep the two in sync, never adding to one without the
    # other.
    orderbook_codes: frozenset[str] = field(default_factory=frozenset)
    # Codes subscribed to H0STMKO0 (market-status/CB) ONLY — 1 registration
    # instead of 2 — for screening candidates that only need a circuit-
    # breaker/VI answer, not a live trade tick (2026-08-28, domestic AI audit:
    # candidate pre-subscribes were exhausting the fixed 20-code/2TR budget
    # every cycle, leaving 55% of same-day Buy decisions blocked on "circuit-
    # breaker state unknown" — there is no REST fallback for CB, unlike
    # price/ticks, so an unsubscribed code cannot trade at all, not just
    # degrade). A code here still gets accurate price data via the existing
    # 60s REST poller fallback: is_subscribed(_TRADE_TR, code) below already
    # returns False for it, which current_status/latest_tick_received_at and
    # every REST-fallback caller already treat as "not on this feed" — no
    # extra plumbing needed for that half. Mutually exclusive with
    # orderbook_codes (2TR vs 1TR for the same code makes no sense); every
    # member must also be in `codes`, same invariant orderbook_codes keeps.
    cb_only_codes: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        # cb_only and orderbook are mutually exclusive tiers for the same
        # code (1TR vs 2TR) — cb_only wins if both were passed, since it's
        # the narrower ask (a caller wanting cb_only never also wants
        # orderbook depth for the same code).
        overlap = self.cb_only_codes & self.orderbook_codes
        if overlap:
            logger.warning(
                "RealtimeFeed: %s in both cb_only_codes and orderbook_codes — keeping cb_only "
                "(narrower subscription wins)", sorted(overlap),
            )
            self.orderbook_codes = self.orderbook_codes - overlap
        # Enforces MAX_REGISTRATIONS even when `codes` was handed in directly
        # at construction (app.web.state prioritizes held positions first and
        # already truncates before constructing this, but a bare direct
        # construction — a test, a future caller — shouldn't be able to
        # silently exceed the registration cap either). Walks `codes` in
        # order and skips (not stops at) whatever doesn't fit, so a 2TR code
        # that doesn't fit doesn't block a later 1TR (cb_only) code from
        # still claiming the remaining slot — the whole point of the cb_only
        # tier is fitting more codes into the same budget.
        kept: list[str] = []
        used = 0
        dropped: list[str] = []
        for code in self.codes:
            cost = len(self._trs_for(code))
            if used + cost > self.MAX_REGISTRATIONS:
                dropped.append(code)
                continue
            kept.append(code)
            used += cost
        if dropped:
            logger.warning(
                "RealtimeFeed: %d codes need %d registrations, exceeding the %d-registration cap "
                "(KIS's 41 combined real-time registration limit, 1 reserved for 체결통보) — kept "
                "%d, dropping from the initial subscribe: %s",
                len(self.codes), used + sum(len(self._trs_for(c)) for c in dropped),
                self.MAX_REGISTRATIONS, len(kept), dropped,
            )
            self.codes = kept
            # A dropped code must not linger in orderbook_codes/cb_only_codes
            # — nothing re-checks membership there against `codes` later, so
            # a stale entry would silently claim coverage for a code this
            # feed no longer subscribes to at all.
            self.orderbook_codes = self.orderbook_codes & set(self.codes)
            self.cb_only_codes = self.cb_only_codes & set(self.codes)

    def _trs_for(self, code: str) -> tuple[str, ...]:
        """The TR(s) to subscribe `code` to: market-status only for a
        cb_only-tier code (1 registration), the (market-status, orderbook)
        pair for a code in the orderbook zone, or the normal (market-status,
        trade-tick) pair otherwise. Market-status is never dropped for any
        tier: every circuit-breaker/VI safety gate depends on it regardless
        of which (if any) second TR a code gets."""
        if code in self.cb_only_codes:
            return (_MARKET_STATUS_TR,)
        if code in self.orderbook_codes:
            return (_MARKET_STATUS_TR, _ORDERBOOK_TR)
        return self.trs

    def registration_count(self, codes: list[str] | None = None) -> int:
        """Total real-time registrations `codes` (default: self.codes) would
        consume, tier-aware — the actual unit MAX_REGISTRATIONS budgets,
        unlike a plain len(codes) which only holds for the all-2TR case."""
        return sum(len(self._trs_for(c)) for c in (self.codes if codes is None else codes))

    @property
    def _fill_tr(self) -> str:
        return _FILL_TR_REAL if self.is_real else _FILL_TR_PAPER

    def _subscribe_msg(self, approval_key: str, tr_id: str, tr_key: str) -> str:
        return json.dumps({
            "header": {"approval_key": approval_key, "custtype": "P",
                       "tr_type": "1", "content-type": "utf-8"},
            "body": {"input": {"tr_id": tr_id, "tr_key": tr_key}},
        })

    async def _send_subscriptions(self, ws, approval_key: str) -> None:
        """Sends every subscribe message for this session, 0.3s apart (KIS
        throttles subscribe bursts) — run concurrently with the receive loop
        in `run()`, NOT before it. At MAX_CODES=20 codes x 2 TRs + 체결통보
        this takes ~12s; a prior version sent all of it before starting
        `async for raw in ws`, leaving every incoming frame (including this
        same batch's own subscribe ACKs) unread in the socket's buffer for
        that whole window. Right after a reconnect at/near market open is
        the worst possible moment for that: a burst of frames arriving while
        nothing is reading risks the buffer backing up and the connection
        stalling into yet another reconnect. Best-effort: a failure here
        (e.g. the socket closes mid-send) just means some registrations
        never got sent — the outer reconnect loop retries the whole session
        next time, same as any other subscribe failure."""
        try:
            if self.hts_id:  # 체결통보 is account-level: one sub keyed by HTS ID
                await ws.send(self._subscribe_msg(approval_key, self._fill_tr, self.hts_id))
                await asyncio.sleep(0.3)
            for code in self.codes:
                for tr in self._trs_for(code):
                    await ws.send(self._subscribe_msg(approval_key, tr, code))
                    await asyncio.sleep(0.3)  # KIS throttles subscribe bursts
            # "%s x %s" no longer describes every code's TR pair uniformly
            # once orderbook_codes/cb_only_codes is non-empty (those get
            # H0STASP0 or drop H0STCNT0 entirely) — spelling out the split
            # counts here rather than leaving the log to imply a uniform
            # assignment it isn't.
            logger.info(
                "KIS realtime WS subscribed: %s x %s (%d/%d registrations; %d orderbook zone: %s; "
                "%d cb-only zone: %s)%s",
                self.codes, list(self.trs), self.registration_count(), self.MAX_REGISTRATIONS,
                len(self.orderbook_codes), sorted(self.orderbook_codes),
                len(self.cb_only_codes), sorted(self.cb_only_codes),
                " + 체결통보" if self.hts_id else "",
            )
        except Exception:  # noqa: BLE001 - best-effort, see docstring
            logger.warning("KIS realtime WS: sending subscriptions failed partway", exc_info=True)

    async def run(self) -> None:
        """Connect/subscribe/consume, reconnecting on any failure with capped
        backoff — indefinitely (until stop()). Deliberately never gives up:
        KIS allows only ONE WebSocket session per appkey and holds a dropped
        session for a few minutes, so a server restarted before the old
        session times out gets 'ALREADY IN USE' and must keep retrying until
        KIS frees it. Backoff (max 30s) keeps that from busy-looping."""
        backoff = 1
        while not self._stop:
            try:
                approval_key = await fetch_approval_key(self.base_url, self.app_key, self.app_secret)
                async with websockets.connect(self.ws_url, open_timeout=10) as ws:
                    self._ws = ws
                    self._approval_key = approval_key
                    backoff = 1
                    self.connected = True
                    # 체결통보 goes FIRST within _send_subscriptions, before the
                    # per-code market-status/trade/orderbook loop — KIS caps
                    # the number of real-time registrations per session, and
                    # a trade allowlist of any real size (len(codes) *
                    # len(trs)) already exceeds it on its own, so subscribing
                    # per-code first meant 체결통보 (order-fill notifications)
                    # always landed last and was always the one KIS rejected
                    # with "MAX SUBSCRIBE OVER" — confirmed live with a
                    # 24-code trade allowlist (72 per-code registrations sent
                    # before 체결통보 ever got a turn). Fills are account-level
                    # and singular (one registration total) while a per-code
                    # TR is one of many equally-important-but-not-critical
                    # feeds, so giving fills the guaranteed first slot is the
                    # right trade-off even though some per-code registrations
                    # may still be dropped once the batch is large enough to
                    # hit the cap.
                    #
                    # Run as a background task, concurrent with the receive
                    # loop below, not awaited inline before it — see
                    # _send_subscriptions' own docstring for why sending the
                    # whole ~12s batch before reading anything is unsafe.
                    subscribe_task = asyncio.ensure_future(
                        self._send_subscriptions(ws, approval_key)
                    )
                    try:
                        async for raw in ws:
                            if self._stop:
                                break
                            await self._handle(ws, raw)
                    finally:
                        subscribe_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await subscribe_task
            except Exception as exc:  # noqa: BLE001 - never crash the host app
                if self._stop:
                    # stop() closing self._ws (or a real network error racing
                    # shutdown) lands here too — skip the reconnect log/sleep
                    # entirely so shutdown doesn't keep printing "reconnect
                    # in Ns" after the host app has already asked this feed
                    # to stop (observed live 2026-07-27: this sleep was the
                    # visible tail of a slow shutdown).
                    break
                logger.warning("realtime WS error (reconnect in %ds): %s", backoff, exc)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)
            finally:
                self.connected = False
                self._ws = None
                # Every registration must be re-ACKed after a reconnect — an
                # entry surviving from the previous session would let
                # is_subscribed() report "confirmed" before the new
                # subscribe messages have even been sent, let alone ACKed.
                self._subscribed.clear()
                self.rejected_keys.clear()

    async def add_code(self, code: str, *, orderbook: bool = False, cb_only: bool = False) -> bool:
        """Subscribes `code` to this feed's TRs — a no-op (returns True) if
        already watching, whatever tier it's already on: this early-return
        is also what keeps a second `add_code(code, cb_only=True)` call from
        ever demoting an already-2TR code down to cb_only. Appends to
        `self.codes` so a FUTURE reconnect always covers it (run()'s
        subscribe loop reads `self.codes` fresh on every reconnect, never a
        frozen snapshot — see run()'s own loop), and additionally sends the
        registration messages on the CURRENTLY OPEN connection right away
        when one exists, so a ticker newly added to the trade allowlist
        doesn't have to wait for the next reconnect to stop reporting
        circuit-breaker state as unknown (app.live.kis_market_status).
        Best-effort on the live-send: `self.codes` is already updated before
        it's attempted, so a failure here just means the next reconnect
        picks it up instead — never worse than today's permanent-unknown
        state.

        `orderbook=True` puts `code` in the orderbook zone (market-status +
        H0STASP0 instead of + H0STCNT0, still 2 registrations) — see
        `orderbook_codes`'s own comment. `cb_only=True` puts it in the
        market-status-only zone (1 registration) — see `cb_only_codes`'s own
        comment. Passing both is a caller bug (ValueError) — a code can't be
        subscribed to both alternate second-TR tiers at once. Neither is
        meant for promoting an existing zone code to a wider tier: there is
        no unsubscribe protocol on this feed (`_subscribe_msg`'s tr_type is
        always "1"), so that promotion goes through
        app.web.state.reconcile_feed_tier_promotions's full
        stop_market_feed()/ensure_market_feed() rebuild instead.

        Returns False (no-ops, logged) instead of subscribing once
        `MAX_REGISTRATIONS` is already spent by the cost this call would add
        (1 for cb_only, else len(self.trs)) — `code` still trades normally
        via manual orders/the automated path for price (falls back to the
        existing 60s REST poller), but circuit-breaker/VI state has NO REST
        fallback (there is no "is a CB active?" REST call), so an unsubscribed
        code cannot pass the execution gate at all until it fits — callers
        that care (e.g. app.live.auto_trade_loop._ensure_subscribed) should
        surface this to the user rather than let it fail silently at
        execution time minutes later."""
        if code in self.codes:
            self.capped_codes.discard(code)
            return True
        if orderbook and cb_only:
            raise ValueError(f"add_code({code}): orderbook and cb_only are mutually exclusive")
        cost = 1 if cb_only else len(self.trs)
        if self.registration_count() + cost > self.MAX_REGISTRATIONS:
            logger.warning(
                "RealtimeFeed: %d/%d registrations already spent — %s (cost %d) will not get "
                "real-time coverage; circuit-breaker/VI state has no REST fallback, so this code "
                "cannot trade until a slot frees up",
                self.registration_count(), self.MAX_REGISTRATIONS, code, cost,
            )
            self.capped_codes.add(code)
            return False
        self.capped_codes.discard(code)
        self.codes.append(code)
        if orderbook:
            self.orderbook_codes = self.orderbook_codes | {code}
        if cb_only:
            self.cb_only_codes = self.cb_only_codes | {code}
        if self.connected and self._ws is not None and self._approval_key is not None:
            try:
                for tr in self._trs_for(code):
                    await self._ws.send(self._subscribe_msg(self._approval_key, tr, code))
                    await asyncio.sleep(0.3)  # same subscribe-burst throttle as run()'s initial loop
                logger.info("KIS realtime WS: subscribed new code %s (live, no reconnect needed)", code)
            except Exception:  # noqa: BLE001 - best-effort, see docstring
                logger.warning(
                    "KIS realtime WS: live-subscribe for %s failed — will retry on next reconnect",
                    code, exc_info=True,
                )
        return True

    async def _handle(self, ws, raw: str) -> None:
        if raw and raw[0] in ("0", "1"):
            parts = raw.split("|", 3)
            tr_id = parts[1] if len(parts) > 1 else ""
            if tr_id == _MARKET_STATUS_TR:
                for s in parse_market_status_frame(raw):
                    self.statuses[s.code] = s
                    self._status_received_at[s.code] = time.monotonic()
                    if s.trading_halted:
                        logger.warning("TRADING HALT on %s: %s", s.code, s.halt_reason)
            elif tr_id == _TRADE_TR:
                for t in parse_trade_frame(raw):
                    self.ticks[t.code] = t
                    self.tick_received_at[t.code] = datetime.now(timezone.utc)
            elif tr_id == _ORDERBOOK_TR:
                for ob in parse_orderbook_frame(raw):
                    self.orderbook[ob.code] = ob
            elif tr_id == self._fill_tr and len(parts) > 3:  # encrypted 체결통보
                crypto = self._crypto.get(self._fill_tr)
                if not crypto:
                    logger.warning("체결통보 frame arrived but no AES key/iv captured (subscribe ACK gave none)")
                    return
                try:
                    body = aes_cbc_b64_decrypt(crypto[0], crypto[1], parts[3])
                except Exception:  # noqa: BLE001
                    logger.warning("체결통보 decrypt failed", exc_info=True)
                    return
                # parts[2] (record count) is the frame's own plaintext field —
                # only parts[3] (the payload) is AES-encrypted — same
                # authoritative role _frame_rows gives it for the other TRs.
                count = int(parts[2]) if parts[2].isdigit() else 1
                for f in parse_fill_frame(body, count):
                    self.fills.append(f)
                    del self.fills[:-50]  # keep last 50
                    logger.info("체결통보: %s %s %s x%d @ %s", f.status, f.side, f.code, f.fill_qty, f.fill_price)
            return
        try:
            msg = json.loads(raw)
        except (ValueError, TypeError):
            return
        header = msg.get("header", {})
        if header.get("tr_id") == "PINGPONG":
            await ws.pong(raw)
            return
        # subscribe ACK — rt_cd="0" means KIS actually accepted this
        # (tr_id, tr_key) registration; anything else (MAX SUBSCRIBE OVER,
        # ALREADY IN SUBSCRIBE, ...) was previously discarded entirely, with
        # no log and no state change — indistinguishable from a healthy
        # subscription that just hasn't seen an event yet. That's the exact
        # gap is_circuit_breaker_active's "connected + subscribed -> not
        # halted" fallback (kis_market_status.py) relies on being accurate.
        body = msg.get("body", {})
        out = body.get("output", {})
        ack_tr = header.get("tr_id")
        ack_key = header.get("tr_key")
        rt_cd = body.get("rt_cd")
        if ack_tr and rt_cd is not None:
            if rt_cd == "0":
                self._subscribed.add((ack_tr, ack_key))
            else:
                self.rejected_keys.add((ack_tr, ack_key))
                logger.error(
                    "KIS realtime WS subscribe REJECTED: tr_id=%s tr_key=%s rt_cd=%s msg=%s — "
                    "this registration will never deliver a frame",
                    ack_tr, ack_key, rt_cd, body.get("msg1"),
                )
        if ack_tr and out.get("key") and out.get("iv"):
            self._crypto[ack_tr] = (out["key"], out["iv"])

    def is_subscribed(self, tr_id: str, tr_key: str) -> bool:
        """True only if KIS actually ACKed rt_cd="0" for this (tr_id, tr_key)
        — see _handle's subscribe-ACK comment for why this must not be
        inferred from `codes`/`trs` (what we attempted) alone."""
        return (tr_id, tr_key) in self._subscribed

    def current_status(self, code: str) -> MarketStatus | None:
        """Last-known H0STMKO0 status for `code`, or None if never received,
        not currently subscribed, OR stale (see _RECONNECT_STALE_AFTER_
        SECONDS). No time bound at all while `connected` is True — a quiet
        KRX code can legitimately go a full session with zero frames
        (confirmed 2026-07-21), so "old" alone must never mean "unknown"
        during normal operation. The clock only starts once disconnected, so
        a brief reconnect blip doesn't trip this, but a reconnect stuck for
        minutes (KIS's documented "ALREADY IN USE" hold) eventually does.

        The is_subscribed check (2026-08-10 fix) closes a gap the clock
        above can't: run()'s reconnect `finally` clears `_subscribed`
        (every registration must be re-ACKed) but never clears `statuses` —
        so once `connected` flips back to True, this returned `code`'s
        pre-reconnect status forever (the disconnected-only clock above
        never runs again) even if `code`'s own re-subscribe was REJECTED
        (e.g. MAX SUBSCRIBE OVER recurring) and will never receive another
        frame. Requiring is_subscribed here ties trust in a cached status to
        the SAME re-ACK requirement `_subscribed` already enforces, so a
        code stuck un-reconfirmed after a reconnect reports unknown instead
        of a stale pre-reconnect circuit-breaker state."""
        status = self.statuses.get(code)
        if status is None:
            return None
        if not self.is_subscribed(_MARKET_STATUS_TR, code):
            return None
        if not self.connected:
            received_at = self._status_received_at.get(code, 0.0)
            if time.monotonic() - received_at > self._RECONNECT_STALE_AFTER_SECONDS:
                return None
        return status

    def latest_tick_received_at(self, code: str) -> datetime | None:
        """Wall-clock receipt time of `code`'s last H0STCNT0 trade tick
        (`self.ticks[code]`), or `None` if never received (or not
        subscribed to H0STCNT0 — same re-ACK requirement `current_status`
        enforces for H0STMKO0, so a code whose re-subscribe was rejected
        after a reconnect doesn't report a pre-reconnect tick's timestamp
        as if it were current).

        This is what actually gives `app.live.exit_monitor` this feed's
        real per-second resolution for domestic tickers — `app.live.
        store`'s tick cache (what `app.live.poller` fills) only samples
        this feed once per `settings.poll_interval_seconds` (60s default),
        which is too coarse for a 5s-interval exit check to ever see a
        fresh tick. Returning the raw timestamp (not a pre-computed age)
        lets the caller apply the SAME staleness/entry-time comparisons it
        already applies to `app.live.store.Tick.timestamp` for overseas
        tickers, off one consistent wall clock."""
        if code not in self.ticks or not self.is_subscribed(_TRADE_TR, code):
            return None
        return self.tick_received_at.get(code)

    async def stop(self) -> None:
        """Actively closes the live connection instead of only setting the
        stop flag and waiting for run()'s own reconnect/backoff path (or an
        external task.cancel()) to eventually notice — websockets' default
        close_timeout (10s) otherwise makes shutdown wait on a full close
        handshake with KIS's server before run() ever re-checks self._stop.
        Closing here makes run()'s `async for raw in ws` raise immediately,
        landing in the except block above, which now short-circuits on
        self._stop rather than logging/sleeping through a fake reconnect."""
        self._stop = True
        if self._ws is not None:
            await self._ws.close()


# Back-compat alias — the app imported MarketStatusFeed before trade/orderbook
# were added; RealtimeFeed is a strict superset.
MarketStatusFeed = RealtimeFeed
