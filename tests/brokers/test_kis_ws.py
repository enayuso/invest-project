"""Market-status WebSocket frame parser tests — pure, no network."""

from __future__ import annotations

import base64
import json

import httpx
import pytest
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad

from app.brokers.kis.domestic import websocket as ws_module
from app.brokers.kis.domestic.websocket import (
    RealtimeFeed,
    aes_cbc_b64_decrypt,
    fetch_approval_key,
    parse_fill_frame,
    parse_market_status_frame,
    parse_orderbook_frame,
    parse_trade_frame,
)


@pytest.fixture(autouse=True)
def _reset_approval_key_throttle(monkeypatch):
    """The throttle's last-request timestamp is module-global, real
    monotonic time — reset it so one test's call can't make the next
    test's first call think it's throttled."""
    monkeypatch.setattr(ws_module, "_last_approval_key_request_at", 0.0)


def test_aes_cbc_b64_decrypt_roundtrip():
    key, iv = "0123456789abcdef", "abcdef9876543210"  # 16-byte key/iv, as KIS sends
    plain = "005930^1^70000^2^삼성전자"
    ct = base64.b64encode(
        AES.new(key.encode(), AES.MODE_CBC, iv.encode()).encrypt(pad(plain.encode(), AES.block_size))
    ).decode()
    assert aes_cbc_b64_decrypt(key, iv, ct) == plain


def test_parse_fill_frame():
    # 26 fields per _CNI_COLUMNS: oder_no idx2, seln_byov idx4, code idx8,
    # cntg_qty idx9, cntg_unpr idx10, cntg_yn idx13, oder_qty idx16, isnm idx24
    f = ["", "", "0000123", "", "02", "", "", "", "005930", "1", "70000", "093015",
         "N", "2", "Y", "", "1", "", "", "", "", "", "", "", "삼성전자", ""]
    fills = parse_fill_frame("^".join(f))
    assert len(fills) == 1
    assert fills[0].order_no == "0000123" and fills[0].code == "005930"
    assert fills[0].side == "buy" and fills[0].fill_qty == 1 and fills[0].status == "filled"
    assert fills[0].name == "삼성전자"


def test_parse_fill_frame_rejected():
    f = ["", "", "0000124", "", "01", "", "", "", "005930", "0", "0", "093016",
         "Y", "1", "N", "", "1", "", "", "", "", "", "", "", "삼성전자", ""]
    assert parse_fill_frame("^".join(f))[0].status == "rejected"


def test_parse_fill_frame_with_count_chunks_multiple_records():
    # Same regression this fixes as _frame_rows already did for the other
    # three TRs — hardcoding width=len(_CNI_COLUMNS) instead of deriving it
    # from the frame's own count would mis-chunk (or silently drop) a
    # second fill notification packed into the same 체결통보 frame.
    record_a = ["", "", "0000123", "", "02", "", "", "", "005930", "1", "70000", "093015",
                "N", "2", "Y", "", "1", "", "", "", "", "", "", "", "삼성전자", ""]
    record_b = ["", "", "0000456", "", "01", "", "", "", "000660", "2", "150000", "093020",
                "N", "2", "Y", "", "2", "", "", "", "", "", "", "", "SK하이닉스", ""]
    body = "^".join(record_a + record_b)
    fills = parse_fill_frame(body, count=2)
    assert len(fills) == 2
    assert fills[0].order_no == "0000123" and fills[0].code == "005930"
    assert fills[1].order_no == "0000456" and fills[1].code == "000660"
    assert fills[1].side == "sell" and fills[1].fill_qty == 2


def test_parse_fill_frame_defaults_to_a_single_record():
    f = ["", "", "0000123", "", "02", "", "", "", "005930", "1", "70000", "093015",
         "N", "2", "Y", "", "1", "", "", "", "", "", "", "", "삼성전자", ""]
    assert len(parse_fill_frame("^".join(f))) == 1


def test_parse_fill_frame_zero_count_returns_empty():
    assert parse_fill_frame("a^b^c", count=0) == []


def test_parses_normal_open_frame():
    # 0|H0STMKO0|001|<11 ^-separated fields>
    raw = "0|H0STMKO0|001|005930^N^^20^20^0^0^00^0^0^KRX"
    out = parse_market_status_frame(raw)
    assert len(out) == 1
    s = out[0]
    assert s.code == "005930"
    assert s.trading_halted is False
    assert s.market_op_code == "20"


def test_parses_trading_halt_frame():
    raw = "0|H0STMKO0|001|005930^Y^서킷브레이커 발동^40^40^0^0^58^0^0^KRX"
    s = parse_market_status_frame(raw)[0]
    assert s.trading_halted is True
    assert "서킷브레이커" in s.halt_reason
    assert s.market_op_code == "40"
    assert s.stock_state_code == "58"


def test_ignores_system_and_other_frames():
    assert parse_market_status_frame('{"header":{"tr_id":"PINGPONG"}}') == []
    assert parse_market_status_frame("0|H0STCNT0|001|foo^bar") == []
    assert parse_market_status_frame("") == []


def test_parse_trade_frame():
    # H0STCNT0: code, time, price, sign, vrss, change%, wavg, open, high, low, ask1, bid1, cntg_vol, acml_vol
    raw = "0|H0STCNT0|001|005930^101530^254500^5^-3000^-1.16^255000^257000^258000^253000^254600^254400^120^31882652"
    ticks = parse_trade_frame(raw)
    assert len(ticks) == 1
    assert ticks[0].code == "005930" and ticks[0].price == 254500
    assert ticks[0].volume == 31882652 and ticks[0].ask1 == 254600


def test_parse_orderbook_frame():
    asks = "^".join(str(100 + i) for i in range(10))       # askp1..10
    bids = "^".join(str(99 - i) for i in range(10))         # bidp1..10
    askq = "^".join(str((i + 1) * 10) for i in range(10))   # askp_rsqn1..10
    bidq = "^".join(str((i + 1) * 5) for i in range(10))    # bidp_rsqn1..10
    raw = f"0|H0STASP0|001|005930^101530^0^{asks}^{bids}^{askq}^{bidq}"
    obs = parse_orderbook_frame(raw)
    assert len(obs) == 1
    ob = obs[0]
    assert ob.code == "005930"
    assert ob.asks[0] == {"price": 100.0, "qty": 10} and len(ob.asks) == 10
    assert ob.bids[0] == {"price": 99.0, "qty": 5}


def test_trade_and_orderbook_ignore_other_trs():
    assert parse_trade_frame("0|H0STASP0|001|x") == []
    assert parse_orderbook_frame("0|H0STCNT0|001|x") == []


# ---------------------------------------------------------------------------
# RealtimeFeed.latest_tick_received_at (2026-08-10) — the real-time-
# resolution seam app.live.exit_monitor reads for domestic tickers, instead
# of app.live.store's 60s-poll-cadence cache.
# ---------------------------------------------------------------------------

_TRADE_RAW = "0|H0STCNT0|001|005930^101530^254500^5^-3000^-1.16^255000^257000^258000^253000^254600^254400^120^31882652"


async def test_latest_tick_received_at_none_before_any_tick(monkeypatch):
    feed = RealtimeFeed(app_key="k", app_secret="s", codes=["005930"])
    assert feed.latest_tick_received_at("005930") is None


async def test_latest_tick_received_at_set_after_an_acked_trade_tick(monkeypatch):
    feed = RealtimeFeed(app_key="k", app_secret="s", codes=["005930"])
    feed._subscribed.add((ws_module._TRADE_TR, "005930"))
    await feed._handle(None, _TRADE_RAW)
    assert feed.ticks["005930"].price == 254500
    received_at = feed.latest_tick_received_at("005930")
    assert received_at is not None


async def test_latest_tick_received_at_none_when_trade_tr_not_acked():
    # The exact gap this closes: a tick can only ever be trusted as fresh if
    # H0STCNT0 was actually ACKed for this code — not merely attempted (see
    # is_subscribed's own docstring). Directly populating `ticks`/
    # `tick_received_at` without the ACK simulates a rejected subscribe.
    feed = RealtimeFeed(app_key="k", app_secret="s", codes=["005930"])
    for t in parse_trade_frame(_TRADE_RAW):
        feed.ticks[t.code] = t
    import datetime as dt

    feed.tick_received_at["005930"] = dt.datetime.now(dt.timezone.utc)
    assert feed.latest_tick_received_at("005930") is None


def test_parses_multiple_records():
    row = "005930^N^^20^20^0^0^00^0^0^KRX"
    raw = f"0|H0STMKO0|002|{row}^000660^Y^정지^40^40^0^0^58^0^0^KRX"
    out = parse_market_status_frame(raw)
    assert [s.code for s in out] == ["005930", "000660"]
    assert out[1].trading_halted is True


def _mock_client_factory(handler):
    """Returns a drop-in replacement for httpx.AsyncClient whose requests are
    served by `handler` via MockTransport, so fetch_approval_key never hits
    the network."""

    class _MockAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    return _MockAsyncClient


async def test_fetch_approval_key_returns_key_on_success(monkeypatch):
    def handler(request):
        assert request.url.path == "/oauth2/Approval"
        return httpx.Response(200, json={"approval_key": "abc123"})

    monkeypatch.setattr(httpx, "AsyncClient", _mock_client_factory(handler))
    key = await fetch_approval_key("https://openapi.koreainvestment.com:9443", "key", "secret")
    assert key == "abc123"


async def test_fetch_approval_key_raises_when_key_missing(monkeypatch):
    def handler(request):
        return httpx.Response(200, json={})

    monkeypatch.setattr(httpx, "AsyncClient", _mock_client_factory(handler))
    with pytest.raises(RuntimeError, match="no approval_key"):
        await fetch_approval_key("https://openapi.koreainvestment.com:9443", "key", "secret")


async def test_fetch_approval_key_raises_on_http_error(monkeypatch):
    def handler(request):
        return httpx.Response(401, json={"error": "invalid appkey"})

    monkeypatch.setattr(httpx, "AsyncClient", _mock_client_factory(handler))
    with pytest.raises(httpx.HTTPStatusError):
        await fetch_approval_key("https://openapi.koreainvestment.com:9443", "key", "secret")


async def test_fetch_approval_key_throttles_back_to_back_calls(monkeypatch):
    def handler(request):
        return httpx.Response(200, json={"approval_key": "abc123"})

    monkeypatch.setattr(httpx, "AsyncClient", _mock_client_factory(handler))

    sleeps: list[float] = []

    real_sleep = ws_module.asyncio.sleep

    async def fake_sleep(seconds):
        sleeps.append(seconds)
        await real_sleep(0)

    monkeypatch.setattr(ws_module.asyncio, "sleep", fake_sleep)

    await fetch_approval_key("https://openapi.koreainvestment.com:9443", "key", "secret")
    await fetch_approval_key("https://openapi.koreainvestment.com:9443", "key", "secret")

    assert len(sleeps) == 1
    assert 0 < sleeps[0] <= ws_module._APPROVAL_KEY_MIN_INTERVAL


def test_parse_trade_frame_with_unnamed_trailing_fields():
    """Regression test: a real H0STCNT0 record carries ~46 fields, but
    _CNT_COLUMNS only names the first 14 (see its own comment) — chunking
    by len(columns) instead of the frame's own count field used to chop a
    single record's unnamed trailing fields into bogus extra "records"
    with garbage codes. count="001" here with far more than 14 total
    fields must still yield exactly one Tick, not several."""
    named = "005930^101530^254500^5^-3000^-1.16^255000^257000^258000^253000^254600^254400^120^31882652"
    extra_unnamed_fields = "^".join(str(n) for n in range(32))  # pads to ~46 total fields
    raw = f"0|H0STCNT0|001|{named}^{extra_unnamed_fields}"
    ticks = parse_trade_frame(raw)
    assert len(ticks) == 1
    assert ticks[0].code == "005930" and ticks[0].price == 254500
    assert ticks[0].volume == 31882652 and ticks[0].ask1 == 254600


# ---------------------------------------------------------------------------
# RealtimeFeed.add_code — dynamic subscription (circuit-breaker bug 1 fix)
# ---------------------------------------------------------------------------


class _FakeWs:
    def __init__(self, fail: bool = False):
        self.sent: list[str] = []
        self.fail = fail

    async def send(self, msg: str) -> None:
        if self.fail:
            raise RuntimeError("boom")
        self.sent.append(msg)


async def test_add_code_noop_when_already_watching(monkeypatch):
    feed = RealtimeFeed(app_key="k", app_secret="s", codes=["005930"])
    feed.connected = True
    feed._ws = _FakeWs()
    feed._approval_key = "approval"
    monkeypatch.setattr(ws_module.asyncio, "sleep", _fake_sleep)

    await feed.add_code("005930")

    assert feed.codes == ["005930"]  # not duplicated
    assert feed._ws.sent == []  # nothing sent for an already-watched code


async def test_add_code_appends_but_does_not_send_when_not_connected(monkeypatch):
    feed = RealtimeFeed(app_key="k", app_secret="s", codes=["005930"])
    assert feed.connected is False and feed._ws is None

    await feed.add_code("000660")

    assert feed.codes == ["005930", "000660"]  # picked up on the next reconnect


async def test_add_code_sends_immediately_when_connected(monkeypatch):
    feed = RealtimeFeed(app_key="k", app_secret="s", codes=["005930"])
    feed.connected = True
    fake_ws = _FakeWs()
    feed._ws = fake_ws
    feed._approval_key = "approval"

    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(ws_module.asyncio, "sleep", fake_sleep)

    await feed.add_code("000660")

    assert feed.codes == ["005930", "000660"]
    assert len(fake_ws.sent) == len(feed.trs)  # one subscribe message per TR
    for msg in fake_ws.sent:
        assert '"tr_key": "000660"' in msg
    assert len(sleeps) == len(feed.trs)


async def test_add_code_live_send_failure_degrades_quietly(monkeypatch):
    feed = RealtimeFeed(app_key="k", app_secret="s", codes=["005930"])
    feed.connected = True
    feed._ws = _FakeWs(fail=True)
    feed._approval_key = "approval"
    monkeypatch.setattr(ws_module.asyncio, "sleep", _fake_sleep)

    await feed.add_code("000660")  # must not raise

    assert feed.codes == ["005930", "000660"]  # still appended for the next reconnect


# ---------------- MAX_CODES (41 combined real-time registration cap) ----------------


def test_constructing_over_the_cap_truncates_and_keeps_the_first_max_codes():
    codes = [str(i) for i in range(RealtimeFeed.MAX_CODES + 5)]
    feed = RealtimeFeed(app_key="k", app_secret="s", codes=codes)
    assert feed.codes == codes[: RealtimeFeed.MAX_CODES]


def test_constructing_at_or_under_the_cap_keeps_every_code():
    codes = [str(i) for i in range(RealtimeFeed.MAX_CODES)]
    feed = RealtimeFeed(app_key="k", app_secret="s", codes=codes)
    assert feed.codes == codes


async def test_add_code_noop_once_at_the_cap(monkeypatch):
    codes = [str(i) for i in range(RealtimeFeed.MAX_CODES)]
    feed = RealtimeFeed(app_key="k", app_secret="s", codes=list(codes))
    feed.connected = True
    fake_ws = _FakeWs()
    feed._ws = fake_ws
    feed._approval_key = "approval"
    monkeypatch.setattr(ws_module.asyncio, "sleep", _fake_sleep)

    await feed.add_code("one-too-many")

    assert "one-too-many" not in feed.codes
    assert len(feed.codes) == RealtimeFeed.MAX_CODES
    assert fake_ws.sent == []  # never even attempted to subscribe it


async def test_add_code_still_works_just_under_the_cap(monkeypatch):
    codes = [str(i) for i in range(RealtimeFeed.MAX_CODES - 1)]
    feed = RealtimeFeed(app_key="k", app_secret="s", codes=list(codes))
    feed.connected = True
    fake_ws = _FakeWs()
    feed._ws = fake_ws
    feed._approval_key = "approval"
    monkeypatch.setattr(ws_module.asyncio, "sleep", _fake_sleep)

    await feed.add_code("last-slot")

    assert "last-slot" in feed.codes
    assert len(feed.codes) == RealtimeFeed.MAX_CODES
    assert len(fake_ws.sent) == len(feed.trs)


def test_default_trs_no_longer_includes_orderbook():
    # H0STASP0 (orderbook) was dropped 2026-08-05 to buy back registration
    # budget — see websocket.py's module docstring. A dashboard-only display
    # feature, not depended on by any circuit-breaker/VI safety gate.
    feed = RealtimeFeed(app_key="k", app_secret="s", codes=[])
    assert ws_module._ORDERBOOK_TR not in feed.trs
    assert ws_module._MARKET_STATUS_TR in feed.trs
    assert ws_module._TRADE_TR in feed.trs


async def _fake_sleep(*_args) -> None:
    return None


# ---------------------------------------------------------------------------
# orderbook_codes / _trs_for — per-code TR override (2026-08-24), so a
# trailing screening-candidate tail can trade H0STCNT0 (trade-tick) for
# H0STASP0 (orderbook) within the same 41-registration cap. The most
# important regression guard here is that a code NOT in orderbook_codes is
# completely unaffected — held positions and the rest of the priority list
# must keep their existing full (market-status + trade-tick) coverage.
# ---------------------------------------------------------------------------


def test_send_subscriptions_uses_orderbook_trs_for_zone_codes():
    feed = RealtimeFeed(
        app_key="k", app_secret="s", codes=["005930", "000660"],
        orderbook_codes=frozenset({"000660"}),
    )
    assert feed._trs_for("000660") == (ws_module._MARKET_STATUS_TR, ws_module._ORDERBOOK_TR)


def test_send_subscriptions_default_trs_for_non_zone_codes():
    # The regression guard: a code outside orderbook_codes must get the
    # exact same TR pair as before this feature existed.
    feed = RealtimeFeed(
        app_key="k", app_secret="s", codes=["005930", "000660"],
        orderbook_codes=frozenset({"000660"}),
    )
    assert feed._trs_for("005930") == (ws_module._MARKET_STATUS_TR, ws_module._TRADE_TR)
    assert feed._trs_for("005930") == feed.trs


async def test_send_subscriptions_sends_orderbook_tr_only_for_zone_code(monkeypatch):
    feed = RealtimeFeed(
        app_key="k", app_secret="s", codes=["005930", "000660"],
        orderbook_codes=frozenset({"000660"}),
    )
    fake_ws = _FakeWs()
    monkeypatch.setattr(ws_module.asyncio, "sleep", _fake_sleep)

    await feed._send_subscriptions(fake_ws, "approval")

    sent_for = {
        (json.loads(m)["body"]["input"]["tr_id"], json.loads(m)["body"]["input"]["tr_key"])
        for m in fake_ws.sent
    }
    assert (ws_module._TRADE_TR, "005930") in sent_for
    assert (ws_module._ORDERBOOK_TR, "000660") in sent_for
    assert (ws_module._TRADE_TR, "000660") not in sent_for
    assert (ws_module._ORDERBOOK_TR, "005930") not in sent_for


async def test_add_code_orderbook_true_adds_to_orderbook_codes(monkeypatch):
    feed = RealtimeFeed(app_key="k", app_secret="s", codes=["005930"])
    feed.connected = True
    fake_ws = _FakeWs()
    feed._ws = fake_ws
    feed._approval_key = "approval"
    monkeypatch.setattr(ws_module.asyncio, "sleep", _fake_sleep)

    await feed.add_code("000660", orderbook=True)

    assert "000660" in feed.orderbook_codes
    sent_trs = {json.loads(m)["body"]["input"]["tr_id"] for m in fake_ws.sent}
    assert ws_module._ORDERBOOK_TR in sent_trs
    assert ws_module._TRADE_TR not in sent_trs


async def test_add_code_default_does_not_add_to_orderbook_codes(monkeypatch):
    feed = RealtimeFeed(app_key="k", app_secret="s", codes=["005930"])
    feed.connected = True
    feed._ws = _FakeWs()
    feed._approval_key = "approval"
    monkeypatch.setattr(ws_module.asyncio, "sleep", _fake_sleep)

    await feed.add_code("000660")

    assert "000660" not in feed.orderbook_codes


def test_truncation_drops_truncated_codes_from_orderbook_codes_too():
    codes = [str(i) for i in range(RealtimeFeed.MAX_CODES + 5)]
    # The last code (dropped by truncation) is deliberately marked as an
    # orderbook-zone code at construction to prove it doesn't survive there
    # once __post_init__ truncates it out of `codes`.
    dropped_code = codes[-1]
    feed = RealtimeFeed(
        app_key="k", app_secret="s", codes=list(codes),
        orderbook_codes=frozenset({dropped_code, codes[0]}),
    )
    assert dropped_code not in feed.codes
    assert dropped_code not in feed.orderbook_codes


# ---------------------------------------------------------------------------
# cb_only_codes / registration-budget accounting (2026-08-28) — a screening
# candidate only needs the H0STMKO0 (circuit-breaker/VI) answer, not a live
# trade tick, so it costs 1 registration instead of 2. This tier is what lets
# the fixed 40-registration budget cover more codes than MAX_CODES=20 without
# ever weakening the CB gate itself (H0STMKO0 is never dropped for any tier).
# ---------------------------------------------------------------------------


def test_trs_for_cb_only_code_is_market_status_only():
    feed = RealtimeFeed(
        app_key="k", app_secret="s", codes=["005930", "000660"],
        cb_only_codes=frozenset({"000660"}),
    )
    assert feed._trs_for("000660") == (ws_module._MARKET_STATUS_TR,)
    assert feed._trs_for("005930") == feed.trs  # unaffected


def test_registration_count_charges_cb_only_codes_a_single_slot():
    feed = RealtimeFeed(
        app_key="k", app_secret="s", codes=["005930", "000660", "005380"],
        cb_only_codes=frozenset({"000660", "005380"}),
    )
    # 1 code at 2 TRs + 2 codes at 1 TR each = 4.
    assert feed.registration_count() == 4


def test_cb_only_and_orderbook_are_mutually_exclusive_cb_only_wins():
    feed = RealtimeFeed(
        app_key="k", app_secret="s", codes=["000660"],
        cb_only_codes=frozenset({"000660"}), orderbook_codes=frozenset({"000660"}),
    )
    assert "000660" in feed.cb_only_codes
    assert "000660" not in feed.orderbook_codes
    assert feed._trs_for("000660") == (ws_module._MARKET_STATUS_TR,)


def test_construction_fits_more_codes_via_cb_only_tier_than_max_codes_allows():
    # 20 codes at 2TR (the old MAX_CODES ceiling) + 20 more at 1TR (cb_only)
    # = 60 registrations, over budget by 20 — but every 1TR code still fits
    # since truncation skips (not stops at) whatever doesn't fit.
    core = [f"core{i}" for i in range(20)]
    cb_tail = [f"cb{i}" for i in range(20)]
    feed = RealtimeFeed(
        app_key="k", app_secret="s", codes=core + cb_tail,
        cb_only_codes=frozenset(cb_tail),
    )
    # Core (2TR each) fits fully: 20*2 = 40 = MAX_REGISTRATIONS, leaving no
    # room for any cb_only code — proving the tier can't silently exceed the
    # same combined KIS cap the old MAX_CODES-only accounting enforced.
    assert feed.codes == core
    assert feed.registration_count() == RealtimeFeed.MAX_REGISTRATIONS


def test_construction_skips_a_too_big_code_to_fit_a_later_smaller_one():
    # A 2TR code that doesn't fit in the last slot must not block a later
    # 1TR (cb_only) code from claiming that same slot — this is the entire
    # point of walking with `continue`, not `break`, in __post_init__.
    # 19 normal (2TR) codes = 38, "filler" (cb_only, 1TR) = 39, leaving
    # exactly 1: too small for "too-big" (2TR) but enough for "fits" (cb_only).
    core = [f"core{i}" for i in range(19)]
    feed = RealtimeFeed(
        app_key="k", app_secret="s", codes=[*core, "filler", "too-big", "fits"],
        cb_only_codes=frozenset({"filler", "fits"}),
    )
    assert "too-big" not in feed.codes
    assert "filler" in feed.codes
    assert "fits" in feed.codes
    assert feed.registration_count() == RealtimeFeed.MAX_REGISTRATIONS


async def test_add_code_cb_only_costs_one_registration_and_succeeds_past_old_cap(monkeypatch):
    # 19 codes at 2TR = 38 registrations, 2 left — a 2TR add would fail, but
    # a cb_only add (1 registration) must succeed.
    codes = [str(i) for i in range(19)]
    feed = RealtimeFeed(app_key="k", app_secret="s", codes=list(codes))
    feed.connected = True
    feed._ws = _FakeWs()
    feed._approval_key = "approval"
    monkeypatch.setattr(ws_module.asyncio, "sleep", _fake_sleep)

    result = await feed.add_code("candidate", cb_only=True)

    assert result is True
    assert "candidate" in feed.cb_only_codes
    assert feed.registration_count() == 39


async def test_add_code_cb_only_no_ops_when_registration_budget_exhausted(monkeypatch):
    codes = [str(i) for i in range(RealtimeFeed.MAX_CODES)]  # 40/40 registrations spent
    feed = RealtimeFeed(app_key="k", app_secret="s", codes=list(codes))
    feed.connected = True
    fake_ws = _FakeWs()
    feed._ws = fake_ws
    feed._approval_key = "approval"
    monkeypatch.setattr(ws_module.asyncio, "sleep", _fake_sleep)

    result = await feed.add_code("one-too-many", cb_only=True)

    assert result is False
    assert "one-too-many" not in feed.codes
    assert fake_ws.sent == []


async def test_add_code_already_watching_is_a_noop_true_even_for_cb_only(monkeypatch):
    # The early-return on `code in self.codes` is also the demotion guard:
    # a code already on the full 2TR tier must never be downgraded by a
    # later cb_only=True call (e.g. a held position that's also re-selected
    # as a screening candidate the same cycle).
    feed = RealtimeFeed(app_key="k", app_secret="s", codes=["005930"])

    result = await feed.add_code("005930", cb_only=True)

    assert result is True
    assert "005930" not in feed.cb_only_codes  # stayed on the full tier


async def test_add_code_orderbook_and_cb_only_together_is_a_caller_bug():
    feed = RealtimeFeed(app_key="k", app_secret="s", codes=[])
    with pytest.raises(ValueError):
        await feed.add_code("000660", orderbook=True, cb_only=True)


# ---------------------------------------------------------------------------
# 2026-10-02: capped_codes / rejected_keys (cb registration-capacity fix)
# ---------------------------------------------------------------------------


async def test_add_code_records_capped_code_and_clears_it_on_later_success(monkeypatch):
    codes = [str(i) for i in range(RealtimeFeed.MAX_CODES)]  # 40/40
    feed = RealtimeFeed(app_key="k", app_secret="s", codes=list(codes))
    assert await feed.add_code("late", cb_only=True) is False
    assert feed.capped_codes == {"late"}
    # room appears (codes list shrank) -> success removes it
    feed.codes = feed.codes[:-1]
    assert await feed.add_code("late", cb_only=True) is True
    assert feed.capped_codes == set()


async def test_add_code_already_watching_discards_capped_marker():
    feed = RealtimeFeed(app_key="k", app_secret="s", codes=["005930"])
    feed.capped_codes.add("005930")
    assert await feed.add_code("005930", cb_only=True) is True
    assert "005930" not in feed.capped_codes


async def test_subscribe_ack_rejection_recorded_in_rejected_keys():
    import json

    feed = RealtimeFeed(app_key="k", app_secret="s", codes=["005930"])
    msg = {
        "header": {"tr_id": ws_module._MARKET_STATUS_TR, "tr_key": "005930"},
        "body": {"rt_cd": "1", "msg1": "MAX SUBSCRIBE OVER"},
    }
    await feed._handle(None, json.dumps(msg))
    assert (ws_module._MARKET_STATUS_TR, "005930") in feed.rejected_keys
    assert not feed.is_subscribed(ws_module._MARKET_STATUS_TR, "005930")
    ok = {"header": msg["header"], "body": {"rt_cd": "0", "msg1": "SUBSCRIBE SUCCESS"}}
    await feed._handle(None, json.dumps(ok))
    assert feed.is_subscribed(ws_module._MARKET_STATUS_TR, "005930")


def test_new_feed_starts_with_empty_capacity_sets():
    feed = RealtimeFeed(app_key="k", app_secret="s", codes=["005930"])
    assert feed.capped_codes == set() and feed.rejected_keys == set()
