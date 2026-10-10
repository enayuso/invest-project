"""KIS broker client tests — httpx.MockTransport, no network."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import pytest

from app.brokers.kis import BrokerBalance, KISBroker, KISBrokerError
from app.brokers.kis.domestic import order as kis_order
from app.clock import FakeClock, LiveClock, set_clock
from app.domain.models import BrokerExecution, IndexDailyBar

_SEOUL = ZoneInfo("Asia/Seoul")


@pytest.fixture(autouse=True)
def _restore_clock():
    yield
    set_clock(LiveClock())


@pytest.fixture(autouse=True)
def _clear_fee_probe_cache():
    """kis_order._FEE_GROUP_CACHE is module-level (deliberately, per its own
    comment — a closed day's fee never changes) so it must not leak results
    from one test's fake account_8/handler into another test's assertions."""
    kis_order._FEE_GROUP_CACHE.clear()
    yield
    kis_order._FEE_GROUP_CACHE.clear()

TOKEN_BODY = {
    "access_token": "test-token-abc",
    "access_token_token_expired": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"),
    "token_type": "Bearer",
    "expires_in": 86400,
}

BALANCE_BODY = {
    "rt_cd": "0",
    "msg_cd": "20310000",
    "msg1": "모의투자 조회가 완료되었습니다.",
    "ctx_area_fk100": "",
    "ctx_area_nk100": "",
    "output1": [
        {
            "pdno": "005930",
            "prdt_name": "삼성전자",
            "hldg_qty": "60",
            "pchs_avg_pric": "71200.0000",
            "prpr": "74300",
            "evlu_amt": "4458000",
            "evlu_pfls_amt": "186000",
            "evlu_pfls_rt": "4.35",
        },
        {  # sold-out row (quantity 0) must be dropped
            "pdno": "000660",
            "prdt_name": "SK하이닉스",
            "hldg_qty": "0",
            "pchs_avg_pric": "195000.0000",
            "prpr": "201500",
            "evlu_amt": "0",
            "evlu_pfls_amt": "0",
            "evlu_pfls_rt": "0.00",
        },
    ],
    "output2": [
        {
            "dnca_tot_amt": "45542000",
            "tot_evlu_amt": "50000000",
            "evlu_pfls_smtl_amt": "186000",
            "pchs_amt_smtl_amt": "4272000",
        }
    ],
}


def make_broker(tmp_path: Path, handler) -> KISBroker:
    transport = httpx.MockTransport(handler)
    return KISBroker(
        app_key="test-app-key-12345678",
        app_secret="test-secret",
        account_8="12345678",
        base_url="https://paper.test",
        token_cache_path=tmp_path / "token.json",
        client=httpx.AsyncClient(transport=transport),
        min_request_interval=0,  # no throttling against MockTransport
    )


async def test_fetch_balance_parses_summary_and_positions(tmp_path):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        assert request.headers["authorization"] == "Bearer test-token-abc"
        assert request.headers["tr_id"] == "VTTC8434R"
        return httpx.Response(200, json=BALANCE_BODY)

    broker = make_broker(tmp_path, handler)
    balance = await broker.fetch_balance()

    assert isinstance(balance, BrokerBalance)
    assert balance.cash_deposit == 45542000
    # Summed from held positions (4,458,000), not output2's tot_evlu_amt
    # (50,000,000 = cash + securities, a total-assets figure, not the pure
    # holdings evaluation the domestic hero tile shows).
    assert balance.total_eval_amount == 4458000
    assert len(balance.positions) == 1  # zero-qty row dropped
    pos = balance.positions[0]
    assert pos.code == "005930"
    assert pos.name == "삼성전자"
    assert pos.quantity == 60
    assert pos.avg_price == pytest.approx(71200.0)
    assert calls[0] == "/oauth2/tokenP"


async def test_token_cached_to_file_and_reused(tmp_path):
    token_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal token_calls
        if request.url.path == "/oauth2/tokenP":
            token_calls += 1
            return httpx.Response(200, json=TOKEN_BODY)
        return httpx.Response(200, json=BALANCE_BODY)

    broker = make_broker(tmp_path, handler)
    await broker.fetch_balance()
    await broker.fetch_balance()
    assert token_calls == 1  # in-memory reuse

    cached = json.loads((tmp_path / "token.json").read_text(encoding="utf-8"))
    assert cached["token"] == "test-token-abc"

    # A fresh broker (new process) must reuse the file cache, not re-issue.
    broker2 = make_broker(tmp_path, handler)
    await broker2.fetch_balance()
    assert token_calls == 1


async def test_token_cache_file_permissions_restricted_to_owner(tmp_path, monkeypatch):
    # Best-effort defense-in-depth: the token cache is plaintext, so
    # restrict it to owner read/write. os.chmod's effect on Windows is
    # limited (mainly the read-only bit, not real POSIX permissions), so
    # this asserts the call happens with the right mode rather than the
    # resulting bits, which are OS-dependent.
    import stat as stat_module

    chmod_calls = []
    real_chmod = os.chmod

    def spy_chmod(path, mode):
        chmod_calls.append((str(path), mode))
        real_chmod(path, mode)

    monkeypatch.setattr(os, "chmod", spy_chmod)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        return httpx.Response(200, json=BALANCE_BODY)

    broker = make_broker(tmp_path, handler)
    await broker.fetch_balance()

    # The cache is published atomically: chmod happens on a temp file in the
    # same dir, BEFORE os.replace() moves it onto the real path — so the token
    # is never briefly world-readable at its final location.
    assert len(chmod_calls) == 1
    path, mode = chmod_calls[0]
    assert os.path.dirname(path) == str(tmp_path)
    assert os.path.basename(path).startswith("token.json.tmp.")
    assert mode == (stat_module.S_IRUSR | stat_module.S_IWUSR)
    # The temp file is gone (renamed), the real cache file is there.
    assert (tmp_path / "token.json").exists()
    assert not list(tmp_path.glob("token.json.tmp.*"))


async def test_stale_cache_for_other_app_key_ignored(tmp_path):
    (tmp_path / "token.json").write_text(
        json.dumps(
            {
                "token": "other-token",
                "expires_at": (datetime.now() + timedelta(days=1)).isoformat(),
                "app_key_tail": "DIFFERENT",
            }
        ),
        encoding="utf-8",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        assert request.headers["authorization"] == "Bearer test-token-abc"
        return httpx.Response(200, json=BALANCE_BODY)

    broker = make_broker(tmp_path, handler)
    await broker.fetch_balance()


async def test_token_expiry_uses_kst_not_utc(tmp_path):
    """`access_token_token_expired` is a KST wall-clock string. Parsed naive
    it was compared against the container's UTC clock, so every token looked
    valid ~9h longer than it really was. Here the token expires 30 min after
    'now' in KST — inside the 10-min refresh margin almost at once — so a
    fresh process must re-issue, not keep serving a token a UTC skew made
    look 9h fresh."""
    set_clock(FakeClock(datetime(2026, 9, 2, 23, 0, tzinfo=_SEOUL)))
    token_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal token_calls
        if request.url.path == "/oauth2/tokenP":
            token_calls += 1
            return httpx.Response(200, json={**TOKEN_BODY, "access_token_token_expired": "2026-09-02 23:30:00"})
        return httpx.Response(200, json=BALANCE_BODY)

    await make_broker(tmp_path, handler).fetch_balance()
    assert token_calls == 1

    set_clock(FakeClock(datetime(2026, 9, 2, 23, 25, tzinfo=_SEOUL)))
    await make_broker(tmp_path, handler).fetch_balance()  # fresh process, reads disk cache
    assert token_calls == 2  # inside the refresh margin in KST → re-issued


async def test_legacy_naive_cache_file_is_read_as_kst(tmp_path):
    """A cache file written before KST-aware expiries has a naive `expires_at`.
    It must still be readable and compared as KST, not rejected outright."""
    set_clock(FakeClock(datetime(2026, 9, 2, 10, 0, tzinfo=_SEOUL)))
    (tmp_path / "token.json").write_text(
        json.dumps(
            {
                "token": "legacy-token",
                "expires_at": datetime(2026, 9, 2, 18, 0).isoformat(),  # naive, KST-implied
                "app_key_tail": "12345678",
            }
        ),
        encoding="utf-8",
    )
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        assert request.headers["authorization"] == "Bearer legacy-token"
        return httpx.Response(200, json=BALANCE_BODY)

    await make_broker(tmp_path, handler).fetch_balance()
    assert "/oauth2/tokenP" not in calls  # legacy cache reused, no re-issue


async def test_implausible_token_expiry_is_clamped(tmp_path, caplog):
    set_clock(FakeClock(datetime(2026, 9, 2, 10, 0, tzinfo=_SEOUL)))

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(
                200,
                json={**TOKEN_BODY, "access_token_token_expired": "2026-09-30 10:00:00", "expires_in": 86400},
            )
        return httpx.Response(200, json=BALANCE_BODY)

    broker = make_broker(tmp_path, handler)
    with caplog.at_level("WARNING"):
        await broker.fetch_balance()
    assert "implausible token expiry" in caplog.text
    # Clamped to now + expires_in (86400s = 24h), not the ~28-day value.
    assert broker._auth._token_expires_at == datetime(2026, 9, 3, 10, 0, tzinfo=_SEOUL)


async def test_half_written_cache_file_falls_through_cleanly(tmp_path):
    """A concurrent partial write must not raise — just fall through to a
    fresh issuance."""
    (tmp_path / "token.json").write_text('{"token": "abc", "expi', encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        return httpx.Response(200, json=BALANCE_BODY)

    balance = await make_broker(tmp_path, handler).fetch_balance()
    assert isinstance(balance, BrokerBalance)


async def test_inside_margin_disk_token_used_when_issuance_rate_limited(tmp_path):
    """A fresh one-shot CLI process (nothing in memory) that hits EGW00133 can
    still use a disk token that is truly valid, just inside the 10-min refresh
    margin — better than failing every broker call."""
    set_clock(FakeClock(datetime(2026, 9, 2, 10, 0, tzinfo=_SEOUL)))
    (tmp_path / "token.json").write_text(
        json.dumps(
            {
                "token": "disk-token-in-margin",
                "expires_at": datetime(2026, 9, 2, 10, 5, tzinfo=_SEOUL).isoformat(),  # 5 min → inside margin
                "app_key_tail": "12345678",
            }
        ),
        encoding="utf-8",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(403, json={"error_code": "EGW00133", "error_description": "발급 제한"})
        assert request.headers["authorization"] == "Bearer disk-token-in-margin"
        return httpx.Response(200, json=BALANCE_BODY)

    balance = await make_broker(tmp_path, handler).fetch_balance()
    assert balance.cash_deposit == 45542000


async def test_balance_pagination_follows_tr_cont(tmp_path):
    page_calls = 0
    page1 = dict(BALANCE_BODY, ctx_area_fk100="FK1", ctx_area_nk100="NK1")

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal page_calls
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        page_calls += 1
        if page_calls == 1:
            return httpx.Response(200, json=page1, headers={"tr_cont": "M"})
        assert request.url.params["CTX_AREA_FK100"] == "FK1"
        return httpx.Response(200, json=BALANCE_BODY, headers={"tr_cont": "D"})

    broker = make_broker(tmp_path, handler)
    balance = await broker.fetch_balance()
    assert page_calls == 2
    assert len(balance.positions) == 2  # one non-zero position per page


async def test_index_daily_chart_pagination_follows_tr_cont(tmp_path):
    """Real incident (2026-08-06): this TR's single page fell well short of
    a 200-day moving average's needed history, so app.execution.market_regime's
    domestic Faber signal never once computed successfully. Fixed via the
    same `_paginate` helper `fetch_balance` already uses."""
    page_calls = 0
    page1 = {
        "rt_cd": "0", "msg1": "OK", "output1": {},
        "output2": [{"stck_bsop_date": "20260101", "bstp_nmix_prpr": "2500.00"}],
    }
    page2 = {
        "rt_cd": "0", "msg1": "OK", "output1": {},
        "output2": [{"stck_bsop_date": "20260102", "bstp_nmix_prpr": "2510.00"}],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal page_calls
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        page_calls += 1
        if page_calls == 1:
            return httpx.Response(200, json=page1, headers={"tr_cont": "M"})
        assert request.headers["tr_cont"] == "N"
        return httpx.Response(200, json=page2, headers={"tr_cont": "D"})

    broker = make_broker(tmp_path, handler)
    bars = await broker.fetch_index_daily_chart("0001", "20260101", "20260102")
    assert page_calls == 2
    assert {(b.day, b.close) for b in bars} == {("20260101", 2500.0), ("20260102", 2510.0)}


async def test_index_daily_chart_dedupes_rows_by_day(tmp_path):
    """Defensive: if this TR's continuation ever fails to actually advance
    (re-serving the same page instead of the next one), duplicate rows for
    the same day must not silently corrupt the Faber SMA with repeated
    closes — see fetch_index_daily_chart's own docstring."""
    same_page = {
        "rt_cd": "0", "msg1": "OK", "output1": {},
        "output2": [{"stck_bsop_date": "20260101", "bstp_nmix_prpr": "2500.00"}],
    }
    page_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal page_calls
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        page_calls += 1
        # "M" on every page - the pathological "never advances" case.
        tr_cont = "M" if page_calls < 3 else "D"
        return httpx.Response(200, json=same_page, headers={"tr_cont": tr_cont})

    broker = make_broker(tmp_path, handler)
    bars = await broker.fetch_index_daily_chart("0001", "20260101", "20260101")
    assert page_calls == 3
    assert bars == [IndexDailyBar(day="20260101", close=2500.0)]


async def test_missing_access_token_raises_with_bounded_body(tmp_path):
    # Regression: the error message used to embed the full response body
    # unbounded, unlike the HTTP-status-error branch just above it, which
    # already truncates to 200 chars.
    huge_body = {"error_description": "x" * 5_000}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=huge_body)

    broker = make_broker(tmp_path, handler)
    with pytest.raises(KISBrokerError, match="no access_token") as exc_info:
        await broker.fetch_balance()
    assert len(str(exc_info.value)) < 400
    assert exc_info.value.status_code == 200


async def test_token_issuance_rate_limit_populates_msg_cd_and_status(tmp_path):
    """2026-08-26: token-issuance failures used to raise a bare KISBrokerError
    with no msg_cd/status_code, so app.credentials.validation couldn't tell a
    rate-limited/down KIS apart from an actually-wrong app key."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/oauth2/tokenP"
        return httpx.Response(403, json={"error_code": "EGW00133", "error_description": "발급 요청이 제한되었습니다."})

    broker = make_broker(tmp_path, handler)
    with pytest.raises(KISBrokerError) as exc_info:
        await broker.fetch_balance()
    assert exc_info.value.msg_cd == "EGW00133"
    assert exc_info.value.status_code == 403


async def test_token_issuance_server_error_populates_status_code(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/oauth2/tokenP"
        return httpx.Response(500, text="Internal Server Error")

    broker = make_broker(tmp_path, handler)
    with pytest.raises(KISBrokerError) as exc_info:
        await broker.fetch_balance()
    assert exc_info.value.status_code == 500
    assert exc_info.value.msg_cd == ""


async def test_refused_response_raises(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        return httpx.Response(200, json={"rt_cd": "1", "msg_cd": "EGW00123", "msg1": "기간이 만료된 token 입니다."})

    broker = make_broker(tmp_path, handler)
    with pytest.raises(KISBrokerError, match="EGW00123"):
        await broker.fetch_balance()


async def test_submit_market_buy_order(tmp_path):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        assert request.url.path == "/uapi/domestic-stock/v1/trading/order-cash"
        captured["tr_id"] = request.headers["tr_id"]
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "rt_cd": "0",
                "msg_cd": "40600000",
                "msg1": "모의투자 매수주문이 완료 되었습니다.",
                "output": {"KRX_FWDG_ORD_ORGNO": "00950", "ODNO": "0000117057", "ORD_TMD": "090512"},
            },
        )

    broker = make_broker(tmp_path, handler)
    result = await broker.submit_order("005930", "buy", 1)

    assert captured["tr_id"] == "VTTC0012U"  # 모의투자 매수
    assert captured["body"]["PDNO"] == "005930"
    assert captured["body"]["ORD_QTY"] == "1"  # strings, per KIS POST rules
    assert captured["body"]["ORD_DVSN"] == "01"  # market order
    assert captured["body"]["ORD_UNPR"] == "0"
    assert result.order_no == "0000117057"
    assert result.org_no == "00950"  # captured for a later cancel


async def test_cancel_order_sends_full_cancel(tmp_path):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        assert request.url.path == "/uapi/domestic-stock/v1/trading/order-rvsecncl"
        captured["tr_id"] = request.headers["tr_id"]
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "rt_cd": "0",
                "msg1": "모의투자 취소주문이 완료 되었습니다.",
                "output": {"KRX_FWDG_ORD_ORGNO": "00950", "ODNO": "0000117099", "ORD_TMD": "091530"},
            },
        )

    broker = make_broker(tmp_path, handler)
    result = await broker.cancel_order("0000117057", "00950", 1, price=200000)

    assert captured["tr_id"] == "VTTC0013U"
    assert captured["body"]["ORGN_ODNO"] == "0000117057"
    assert captured["body"]["KRX_FWDG_ORD_ORGNO"] == "00950"
    assert captured["body"]["RVSE_CNCL_DVSN_CD"] == "02"  # cancel
    assert captured["body"]["QTY_ALL_ORD_YN"] == "Y"
    assert result.order_no == "0000117099"

    with pytest.raises(ValueError):
        await broker.cancel_order("", "00950", 1)  # missing order_no
    with pytest.raises(ValueError):
        await broker.cancel_order("0000117057", "", 1)  # missing org_no


async def test_submit_sell_uses_sell_tr_and_rejects_bad_input(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        assert request.headers["tr_id"] == "VTTC0011U"  # 모의투자 매도
        body = json.loads(request.content)
        assert body["ORD_DVSN"] == "00" and body["ORD_UNPR"] == "70000"  # limit order
        return httpx.Response(200, json={"rt_cd": "0", "msg1": "ok", "output": {"ODNO": "1", "ORD_TMD": "1"}})

    broker = make_broker(tmp_path, handler)
    await broker.submit_order("005930", "sell", 2, price=70000)
    with pytest.raises(ValueError):
        await broker.submit_order("005930", "hold", 1)
    with pytest.raises(ValueError):
        await broker.submit_order("005930", "buy", 0)


async def test_fetch_buyable_prefers_cash_only_figures(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        assert request.headers["tr_id"] == "VTTC8908R"
        return httpx.Response(
            200,
            json={
                "rt_cd": "0",
                "output": {
                    "ord_psbl_cash": "30000000",
                    "nrcvb_buy_amt": "30000000",
                    "nrcvb_buy_qty": "422",
                    "max_buy_amt": "75000000",
                    "max_buy_qty": "1055",
                },
            },
        )

    broker = make_broker(tmp_path, handler)
    buyable = await broker.fetch_buyable("005930")
    assert buyable.cash == 30000000
    assert buyable.max_qty == 422  # cash-only, not the leveraged ceiling
    assert buyable.max_amount == 30000000


async def test_fetch_sellable_returns_orderable_quantity(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        assert request.url.path == "/uapi/domestic-stock/v1/trading/inquire-psbl-sell"
        assert request.headers["tr_id"] == "VTTC8408R"
        assert request.url.params["PDNO"] == "005930"
        return httpx.Response(
            200,
            json={
                "rt_cd": "0",
                "output": {
                    "pdno": "005930",
                    "cblc_qty": "10",
                    "sll_qty": "3",
                    "ord_psbl_qty": "7",  # holding minus already-committed sells
                },
            },
        )

    broker = make_broker(tmp_path, handler)
    sellable = await broker.fetch_sellable("005930")
    assert sellable.sellable_qty == 7  # the safe ceiling, not the 10 held
    assert sellable.holding_qty == 10
    assert sellable.already_sold_qty == 3


async def test_revise_order_sends_revise_division(tmp_path):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        assert request.url.path == "/uapi/domestic-stock/v1/trading/order-rvsecncl"
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"rt_cd": "0", "msg1": "정정완료",
                                          "output": {"KRX_FWDG_ORD_ORGNO": "00950", "ODNO": "0000200", "ORD_TMD": "1"}})

    broker = make_broker(tmp_path, handler)
    res = await broker.revise_order("0000100", "00950", 1, 68000)
    assert captured["body"]["RVSE_CNCL_DVSN_CD"] == "01"  # 정정
    assert captured["body"]["ORD_UNPR"] == "68000" and captured["body"]["ORGN_ODNO"] == "0000100"
    assert res.order_no == "0000200"
    with pytest.raises(ValueError):
        await broker.revise_order("", "00950", 1, 68000)


async def test_revise_order_rejects_market_order(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        raise AssertionError("KIS should not be called for a market-order revise")

    broker = make_broker(tmp_path, handler)
    with pytest.raises(ValueError, match="market order"):
        await broker.revise_order("0000100", "00950", 1, 68000, ord_dvsn="01")


async def test_fetch_sector_indices_rejects_unknown_market(tmp_path):
    broker = make_broker(tmp_path, lambda r: httpx.Response(200, json=TOKEN_BODY))
    with pytest.raises(ValueError, match="market"):
        await broker.fetch_sector_indices("9999")


async def test_fetch_vi_status_rejects_bad_params(tmp_path):
    broker = make_broker(tmp_path, lambda r: httpx.Response(200, json=TOKEN_BODY))
    with pytest.raises(ValueError, match="market"):
        await broker.fetch_vi_status(market="Z")
    with pytest.raises(ValueError, match="rank_sort"):
        await broker.fetch_vi_status(rank_sort="9")
    with pytest.raises(ValueError, match="div_cls"):
        await broker.fetch_vi_status(div_cls="9")


async def test_fetch_sector_indices(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        assert request.url.path == "/uapi/domestic-stock/v1/quotations/inquire-index-category-price"
        assert request.headers["tr_id"] == "FHPUP02140000"
        return httpx.Response(200, json={"rt_cd": "0", "output1": [
            {"bstp_cls_code": "0001", "hts_kor_isnm": "종합", "bstp_nmix_prpr": "6806.93", "bstp_nmix_prdy_ctrt": "-8.95"},
            {"bstp_cls_code": "0021", "hts_kor_isnm": "은행", "bstp_nmix_prpr": "300.5", "bstp_nmix_prdy_ctrt": "1.20"},
            {"hts_kor_isnm": ""},  # dropped
        ]})

    broker = make_broker(tmp_path, handler)
    secs = await broker.fetch_sector_indices()
    assert len(secs) == 2
    assert secs[0].name == "종합" and secs[0].change_pct == pytest.approx(-8.95)
    assert secs[1].name == "은행" and secs[1].value == pytest.approx(300.5)


async def test_fetch_sector_indices_kosdaq_filters_foreign_codes_and_dedupes(tmp_path):
    # Observed defect: fetch_sector_indices("1001") returned KOSPI industry
    # rows (00NN codes) mixed into the KOSDAQ response plus duplicates. The
    # market-prefix filter + dedupe-by-code + code sort must make it clean
    # and deterministic.
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        assert request.headers["tr_id"] == "FHPUP02140000"
        return httpx.Response(200, json={"rt_cd": "0", "output1": [
            {"bstp_cls_code": "1028", "hts_kor_isnm": "코스닥 제조", "bstp_nmix_prpr": "100.0", "bstp_nmix_prdy_ctrt": "1.0"},
            {"bstp_cls_code": "0025", "hts_kor_isnm": "보험", "bstp_nmix_prpr": "300.0", "bstp_nmix_prdy_ctrt": "2.0"},
            {"bstp_cls_code": "1005", "hts_kor_isnm": "음식료·담배", "bstp_nmix_prpr": "200.0", "bstp_nmix_prdy_ctrt": "-1.0"},
            {"bstp_cls_code": "1028", "hts_kor_isnm": "코스닥 제조", "bstp_nmix_prpr": "100.0", "bstp_nmix_prdy_ctrt": "1.0"},
            {"bstp_cls_code": "0006", "hts_kor_isnm": "섬유·의류", "bstp_nmix_prpr": "50.0", "bstp_nmix_prdy_ctrt": "0.5"},
        ]})

    broker = make_broker(tmp_path, handler)
    secs = await broker.fetch_sector_indices("1001")
    assert [s.code for s in secs] == ["1005", "1028"]  # sorted, deduped, KOSPI codes dropped
    assert all(s.code.startswith("10") for s in secs)


async def test_fetch_sector_indices_diagnostics_report_other_market_rows(tmp_path, caplog):
    # A KOSPI (0001) call whose response also carries KOSDAQ (10NN) rows: the
    # scan log must report a non-zero other-market count, and the RETURNED list
    # must be exactly what the prefix filter + dedupe + sort already produce.
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        assert request.headers["tr_id"] == "FHPUP02140000"
        return httpx.Response(200, json={"rt_cd": "0", "output1": [
            {"bstp_cls_code": "0001", "hts_kor_isnm": "종합", "bstp_nmix_prpr": "6806.93", "bstp_nmix_prdy_ctrt": "-8.95"},
            {"bstp_cls_code": "0021", "hts_kor_isnm": "은행", "bstp_nmix_prpr": "300.5", "bstp_nmix_prdy_ctrt": "1.20"},
            {"bstp_cls_code": "1005", "hts_kor_isnm": "코스닥 음식료", "bstp_nmix_prpr": "200.0", "bstp_nmix_prdy_ctrt": "1.0"},
            {"bstp_cls_code": "1028", "hts_kor_isnm": "코스닥 제조", "bstp_nmix_prpr": "100.0", "bstp_nmix_prdy_ctrt": "1.0"},
        ]})

    broker = make_broker(tmp_path, handler)
    with caplog.at_level(logging.INFO, logger="app.brokers.kis.domestic.info"):
        secs = await broker.fetch_sector_indices("0001")
    assert [s.code for s in secs] == ["0001", "0021"]  # returned list unchanged
    scan = next(r.getMessage() for r in caplog.records if "sector indices scan:" in r.getMessage())
    assert "other_market_prefix_rows=2" in scan
    assert "market=0001" in scan and "kept=2" in scan


async def test_fetch_sector_indices_uses_per_market_blng_cls_code(tmp_path):
    # §섹터breadth 2단계 (2026-09-09): FID_BLNG_CLS_CODE means a DIFFERENT
    # enum per market for this TR, and KOSDAQ's "0" (전업종) was found to
    # return a randomly-sampled, 100-row-capped subset every call — "3"
    # (일반구분) is stable and cap-free. KOSPI is unaffected and keeps "0".
    # This is the only thing this fix actually changes, so it is the one
    # regression this test exists to catch.
    captured_params: dict[str, dict] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        captured_params[request.url.params["FID_INPUT_ISCD"]] = dict(request.url.params)
        return httpx.Response(200, json={"rt_cd": "0", "output1": []})

    broker = make_broker(tmp_path, handler)
    await broker.fetch_sector_indices("0001")
    await broker.fetch_sector_indices("1001")

    assert captured_params["0001"]["FID_MRKT_CLS_CODE"] == "K"
    assert captured_params["0001"]["FID_BLNG_CLS_CODE"] == "0"
    assert captured_params["1001"]["FID_MRKT_CLS_CODE"] == "Q"
    assert captured_params["1001"]["FID_BLNG_CLS_CODE"] == "3"


async def test_fetch_sector_name_parses_industry_and_market(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        assert request.url.path == "/uapi/domestic-stock/v1/quotations/inquire-price"
        assert request.headers["tr_id"] == "FHKST01010100"
        return httpx.Response(200, json={"rt_cd": "0", "output": {
            "bstp_kor_isnm": "보험", "rprs_mrkt_kor_name": "KOSPI200",
        }})

    broker = make_broker(tmp_path, handler)
    name, market = await broker.fetch_sector_name("088350")
    assert name == "보험" and market == "KOSPI200"


async def test_fetch_market_warning_parses_code(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        assert request.url.path == "/uapi/domestic-stock/v1/quotations/inquire-price"
        assert request.headers["tr_id"] == "FHKST01010100"
        return httpx.Response(200, json={"rt_cd": "0", "output": {"mrkt_warn_cls_code": "2"}})

    broker = make_broker(tmp_path, handler)
    assert await broker.fetch_market_warning("082850") == "2"


async def test_fetch_market_warning_empty_when_absent(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        return httpx.Response(200, json={"rt_cd": "0", "output": {"mrkt_warn_cls_code": ""}})

    broker = make_broker(tmp_path, handler)
    assert await broker.fetch_market_warning("005930") == ""


async def test_fetch_designations_uses_price_tr(tmp_path):
    """§4-5: same TR/endpoint as fetch_market_warning (FHKST01010100,
    주식현재가 시세) — this is the whole reason it costs zero extra network
    round-trips. Switching to a different TR (e.g. 시세2/FHPST01010000 for
    low-liquidity) is a deliberate future change, not this one."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        assert request.url.path == "/uapi/domestic-stock/v1/quotations/inquire-price"
        assert request.headers["tr_id"] == "FHKST01010100"
        return httpx.Response(200, json={"rt_cd": "0", "output": {}})

    broker = make_broker(tmp_path, handler)
    await broker.fetch_designations("005930")


async def test_fetch_designations_parses_all_flags(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        return httpx.Response(200, json={"rt_cd": "0", "output": {
            "mrkt_warn_cls_code": "2", "short_over_yn": "Y", "sltr_yn": "Y",
            "mang_issu_cls_code": "1",
        }})

    broker = make_broker(tmp_path, handler)
    d = await broker.fetch_designations("082850")
    assert d.code == "082850"
    assert d.market_warning == "2"
    assert d.short_overheat is True
    assert d.liquidation is True
    assert d.administrative_raw == "1"


async def test_fetch_designations_clean_stock(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        return httpx.Response(200, json={"rt_cd": "0", "output": {
            "mrkt_warn_cls_code": "", "short_over_yn": "N", "sltr_yn": "N",
            "mang_issu_cls_code": "0",
        }})

    broker = make_broker(tmp_path, handler)
    d = await broker.fetch_designations("005930")
    assert d.market_warning == ""
    assert d.short_overheat is False
    assert d.liquidation is False
    assert d.administrative_raw == "0"  # raw, not interpreted — see StockDesignations' docstring


async def test_fetch_designations_missing_fields_are_clean(tmp_path):
    """fail-open direction: an absent field must never be read as a
    designation — the core regression this filter (a cost-saving
    pre-filter, not a safety gate) must never introduce."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        return httpx.Response(200, json={"rt_cd": "0", "output": {"stck_prpr": "70000"}})

    broker = make_broker(tmp_path, handler)
    d = await broker.fetch_designations("005930")
    assert d.market_warning == ""
    assert d.short_overheat is False
    assert d.liquidation is False
    assert d.administrative_raw == ""


async def test_fetch_designations_lowercase_and_whitespace_yn(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        return httpx.Response(200, json={"rt_cd": "0", "output": {
            "short_over_yn": "y", "sltr_yn": " Y ",
        }})

    broker = make_broker(tmp_path, handler)
    d = await broker.fetch_designations("005930")
    assert d.short_overheat is True
    assert d.liquidation is True


async def test_fetch_daily_executions_maps_sides(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        return httpx.Response(
            200,
            json={
                "rt_cd": "0",
                "ctx_area_fk100": "",
                "ctx_area_nk100": "",
                "output1": [
                    {
                        "ord_dt": "20260713", "odno": "0000117057", "pdno": "005930",
                        "prdt_name": "삼성전자", "sll_buy_dvsn_cd": "02", "ord_qty": "1",
                        "tot_ccld_qty": "1", "rmn_qty": "0", "avg_prvs": "74300",
                        "tot_ccld_amt": "74300",
                    },
                    {
                        "ord_dt": "20260713", "odno": "0000117058", "pdno": "000660",
                        "prdt_name": "SK하이닉스", "sll_buy_dvsn_cd": "01", "ord_qty": "2",
                        "tot_ccld_qty": "0", "rmn_qty": "2", "avg_prvs": "0",
                        "tot_ccld_amt": "0",
                    },
                ],
                "output2": {},
            },
        )

    broker = make_broker(tmp_path, handler)
    execs = await broker.fetch_daily_executions("20260710", "20260713")
    assert len(execs) == 2
    assert execs[0].side == "buy" and execs[0].filled_qty == 1
    assert execs[1].side == "sell" and execs[1].remaining_qty == 2
    # output1 has no fee field at all — always 0.0 until
    # fetch_domestic_fee_tax's separate probe fills it in.
    assert execs[0].fee_tax == 0.0 and execs[1].fee_tax == 0.0


# ---------------------------------------------------------------------------
# fetch_domestic_fee_tax (2026-09-07) — prsm_tlex_smtl lives in output2 (one
# summary object per call), never in output1's per-fill rows, so this probes
# one (ticker, order date, side) group at a time via a narrowed
# inquire_daily_ccld call. See app.brokers.kis.domestic.order's own
# docstrings for the full reasoning; these tests pin the call shape,
# proportional split, validity guards, and fail-open behavior.
# ---------------------------------------------------------------------------


def _exec(code="005930", order_date="20260901", side="buy", qty=10.0, amount=1000000.0, order_no="1"):
    return BrokerExecution(
        order_date=order_date, order_no=order_no, code=code, name="테스트종목", side=side,
        order_qty=qty, filled_qty=qty, remaining_qty=0.0, avg_price=amount / qty if qty else 0.0,
        filled_amount=amount,
    )


def _fee_probe_handler(fee_by_group: dict, calls: list):
    """Shared fake handler: `fee_by_group[(code, date, side)]` supplies the
    output2 body a probe for that exact group should see; any group not in
    the dict returns fee=0 (the "field genuinely empty" shape)."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        params = request.url.params
        key = (params["PDNO"], params["INQR_STRT_DT"], "sell" if params["SLL_BUY_DVSN_CD"] == "01" else "buy")
        calls.append(key)
        spec = fee_by_group.get(key, {})
        qty = spec.get("qty", 0.0)
        return httpx.Response(
            200,
            json={
                "rt_cd": "0",
                "output1": [],
                "output2": {
                    "tot_ccld_qty": str(qty),
                    "tot_ccld_amt": str(spec.get("amount", 0.0)),
                    "prsm_tlex_smtl": str(spec.get("fee", 0.0)),
                },
            },
            headers=spec.get("headers", {}),
        )
    return handler


async def test_fee_probe_issues_exactly_one_call_per_code_date_side_group(tmp_path):
    calls: list = []
    fees = {
        ("005930", "20260901", "buy"): {"qty": 10.0, "amount": 1000000.0, "fee": 1500.0},
        ("000660", "20260902", "sell"): {"qty": 5.0, "amount": 500000.0, "fee": 1200.0},
    }
    broker = make_broker(tmp_path, _fee_probe_handler(fees, calls))
    executions = [
        _exec(code="005930", order_date="20260901", side="buy", qty=10.0, amount=1000000.0),
        _exec(code="000660", order_date="20260902", side="sell", qty=5.0, amount=500000.0),
        _exec(code="005930", order_date="20260903", side="buy", qty=0.0, amount=0.0),  # filled_qty=0 -> no group
    ]

    unresolved = await broker.fetch_domestic_fee_tax(
        executions, codes={"005930", "000660"}, max_groups=30, today="20260907",
    )

    assert unresolved == 0
    assert sorted(calls) == [("000660", "20260902", "sell"), ("005930", "20260901", "buy")]
    assert executions[0].fee_tax == 1500.0
    assert executions[1].fee_tax == 1200.0
    assert executions[2].fee_tax == 0.0  # never probed


async def test_fee_probe_splits_group_fee_proportionally(tmp_path):
    fees = {("005930", "20260901", "sell"): {"qty": 10.0, "amount": 1000000.0, "fee": 1000.0}}
    broker = make_broker(tmp_path, _fee_probe_handler(fees, []))
    executions = [
        _exec(code="005930", order_date="20260901", side="sell", qty=6.0, amount=600000.0, order_no="1"),
        _exec(code="005930", order_date="20260901", side="sell", qty=4.0, amount=400000.0, order_no="2"),
    ]

    unresolved = await broker.fetch_domestic_fee_tax(
        executions, codes={"005930"}, max_groups=30, today="20260907",
    )

    assert unresolved == 0
    assert executions[0].fee_tax == pytest.approx(600.0)  # 60% of notional
    assert executions[1].fee_tax == pytest.approx(400.0)  # 40% of notional


async def test_fee_probe_separates_buy_and_sell_on_the_same_day(tmp_path):
    calls: list = []
    fees = {
        ("005930", "20260901", "buy"): {"qty": 10.0, "amount": 1000000.0, "fee": 150.0},
        ("005930", "20260901", "sell"): {"qty": 10.0, "amount": 1000000.0, "fee": 2000.0},
    }
    broker = make_broker(tmp_path, _fee_probe_handler(fees, calls))
    executions = [
        _exec(code="005930", order_date="20260901", side="buy", qty=10.0, amount=1000000.0),
        _exec(code="005930", order_date="20260901", side="sell", qty=10.0, amount=1000000.0),
    ]

    await broker.fetch_domestic_fee_tax(executions, codes={"005930"}, max_groups=30, today="20260907")

    assert executions[0].fee_tax == 150.0  # buy-side fee never contaminated by sell-side tax
    assert executions[1].fee_tax == 2000.0
    assert len(calls) == 2  # separate probes, never merged into one


async def test_fee_probe_rejects_implausible_fee(tmp_path, caplog):
    # A mislabeled/shifted field could hand back something the size of the
    # notional itself (e.g. total filled amount) under the fee key — this
    # must never be believed at face value.
    fees = {("005930", "20260901", "sell"): {"qty": 10.0, "amount": 1000000.0, "fee": 1000000.0}}
    broker = make_broker(tmp_path, _fee_probe_handler(fees, []))
    executions = [_exec(code="005930", order_date="20260901", side="sell", qty=10.0, amount=1000000.0)]

    with caplog.at_level(logging.WARNING):
        unresolved = await broker.fetch_domestic_fee_tax(
            executions, codes={"005930"}, max_groups=30, today="20260907",
        )

    assert unresolved == 1
    assert executions[0].fee_tax == 0.0
    assert "implausible" in caplog.text


async def test_fee_probe_rejects_quantity_mismatch(tmp_path, caplog):
    fees = {("005930", "20260901", "sell"): {"qty": 3.0, "amount": 300000.0, "fee": 500.0}}
    broker = make_broker(tmp_path, _fee_probe_handler(fees, []))
    executions = [_exec(code="005930", order_date="20260901", side="sell", qty=10.0, amount=1000000.0)]

    with caplog.at_level(logging.WARNING):
        unresolved = await broker.fetch_domestic_fee_tax(
            executions, codes={"005930"}, max_groups=30, today="20260907",
        )

    assert unresolved == 1
    assert executions[0].fee_tax == 0.0
    assert "does not match" in caplog.text


async def test_fee_probe_ignores_multipage_response(tmp_path, caplog):
    fees = {
        ("005930", "20260901", "sell"): {
            "qty": 10.0, "amount": 1000000.0, "fee": 1500.0, "headers": {"tr_cont": "F"},
        },
    }
    broker = make_broker(tmp_path, _fee_probe_handler(fees, []))
    executions = [_exec(code="005930", order_date="20260901", side="sell", qty=10.0, amount=1000000.0)]

    with caplog.at_level(logging.WARNING):
        unresolved = await broker.fetch_domestic_fee_tax(
            executions, codes={"005930"}, max_groups=30, today="20260907",
        )

    assert unresolved == 1
    assert executions[0].fee_tax == 0.0
    assert "multi-page" in caplog.text


async def test_fee_probe_failure_keeps_report_alive(tmp_path, caplog):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        return httpx.Response(500, text="server error")

    broker = make_broker(tmp_path, handler)
    executions = [_exec(code="005930", order_date="20260901", side="sell", qty=10.0, amount=1000000.0)]

    with caplog.at_level(logging.WARNING):
        unresolved = await broker.fetch_domestic_fee_tax(
            executions, codes={"005930"}, max_groups=30, today="20260907",
        )

    assert unresolved == 1
    assert executions[0].fee_tax == 0.0  # no exception propagates


async def test_fee_probe_respects_max_groups(tmp_path):
    calls: list = []
    fees = {
        ("005930", "20260905", "sell"): {"qty": 1.0, "amount": 10000.0, "fee": 20.0},
        ("000660", "20260904", "sell"): {"qty": 1.0, "amount": 10000.0, "fee": 20.0},
        ("035420", "20260903", "sell"): {"qty": 1.0, "amount": 10000.0, "fee": 20.0},
    }
    broker = make_broker(tmp_path, _fee_probe_handler(fees, calls))
    executions = [
        _exec(code="005930", order_date="20260905", side="sell", qty=1.0, amount=10000.0),
        _exec(code="000660", order_date="20260904", side="sell", qty=1.0, amount=10000.0),
        _exec(code="035420", order_date="20260903", side="sell", qty=1.0, amount=10000.0),
    ]

    unresolved = await broker.fetch_domestic_fee_tax(
        executions, codes={"005930", "000660", "035420"}, max_groups=2, today="20260907",
    )

    assert unresolved == 1  # the oldest group, dropped past the cap
    assert len(calls) == 2
    # Most-recent-order-date-first: the 20260903 group must be the one skipped.
    assert ("035420", "20260903", "sell") not in calls
    assert executions[2].fee_tax == 0.0
    assert executions[0].fee_tax == 20.0
    assert executions[1].fee_tax == 20.0


async def test_fee_probe_caches_closed_days_but_not_today(tmp_path):
    calls: list = []
    fees = {
        ("005930", "20260901", "sell"): {"qty": 1.0, "amount": 10000.0, "fee": 20.0},  # closed day
        ("005930", "20260907", "sell"): {"qty": 1.0, "amount": 10000.0, "fee": 20.0},  # "today"
    }
    broker = make_broker(tmp_path, _fee_probe_handler(fees, calls))
    closed_day_exec = [_exec(code="005930", order_date="20260901", side="sell", qty=1.0, amount=10000.0)]
    today_exec = [_exec(code="005930", order_date="20260907", side="sell", qty=1.0, amount=10000.0)]

    for _ in range(2):
        await broker.fetch_domestic_fee_tax(closed_day_exec, codes={"005930"}, max_groups=30, today="20260907")
        await broker.fetch_domestic_fee_tax(today_exec, codes={"005930"}, max_groups=30, today="20260907")

    assert calls.count(("005930", "20260901", "sell")) == 1  # cached after the first probe
    assert calls.count(("005930", "20260907", "sell")) == 2  # today's own fills never cached


async def test_fetch_price_and_candles(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        if request.url.path.endswith("inquire-price"):
            assert request.url.params["FID_INPUT_ISCD"] == "005930"
            return httpx.Response(200, json={"rt_cd": "0", "output": {"stck_prpr": "74300"}})
        return httpx.Response(
            200,
            json={
                "rt_cd": "0",
                "output2": [
                    {"stck_bsop_date": "20260710", "stck_oprc": "73800", "stck_hgpr": "74600",
                     "stck_lwpr": "73500", "stck_clpr": "74300", "acml_vol": "12345678"},
                    {"stck_bsop_date": "20260709", "stck_oprc": "73000", "stck_hgpr": "73900",
                     "stck_lwpr": "72800", "stck_clpr": "73800", "acml_vol": "9876543"},
                    {"stck_bsop_date": "", "stck_clpr": ""},  # blank padding row must be dropped
                ],
            },
        )

    broker = make_broker(tmp_path, handler)
    assert await broker.fetch_price("005930") == 74300
    candles = await broker.fetch_daily_candles("005930", "20260709", "20260710")
    assert [c.day for c in candles] == ["20260709", "20260710"]  # oldest first
    assert candles[1].low == 73500


async def test_fetch_candles_period_passes_through(tmp_path):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        captured["period"] = request.url.params["FID_PERIOD_DIV_CODE"]
        return httpx.Response(200, json={"rt_cd": "0", "output2": [
            {"stck_bsop_date": "20260101", "stck_oprc": "1", "stck_hgpr": "2",
             "stck_lwpr": "1", "stck_clpr": "2", "acml_vol": "10"}]})

    broker = make_broker(tmp_path, handler)
    await broker.fetch_daily_candles("005930", "20260101", "20261231", period="W")
    assert captured["period"] == "W"
    with pytest.raises(ValueError):
        await broker.fetch_daily_candles("005930", "a", "b", period="X")


async def test_fetch_vi_status_and_is_triggered_when_already_released(tmp_path):
    # fetch_vi_status returns the day's full VI history, not just active
    # halts — a row with a non-empty vi_cncl_hour (해제시간) means this VI
    # already ended, so is_vi_triggered must report False even though the
    # code is present in the response.
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        assert request.url.path == "/uapi/domestic-stock/v1/quotations/inquire-vi-status"
        assert request.headers["tr_id"] == "FHPST01390000"
        assert request.url.params["FID_COND_SCR_DIV_CODE"] == "20139"
        return httpx.Response(
            200,
            json={
                "rt_cd": "0",
                "output": [
                    {
                        "mksc_shrn_iscd": "005930", "hts_kor_isnm": "삼성전자",
                        "vi_cls_code": "1", "vi_kind_code": "1", "cntg_vi_hour": "101530",
                        "vi_cncl_hour": "101730", "vi_prc": "260500", "vi_count": "1",
                    }
                ],
            },
        )

    broker = make_broker(tmp_path, handler)
    statuses = await broker.fetch_vi_status(code="005930")
    assert len(statuses) == 1
    assert statuses[0].code == "005930" and statuses[0].vi_price == 260500
    assert statuses[0].triggered_at == "101530"
    assert statuses[0].released_at == "101730"
    assert await broker.is_vi_triggered("005930") is False  # already released, not still halted
    assert await broker.is_vi_triggered("000660") is False  # not in the returned list


async def test_is_vi_triggered_true_while_unresolved(tmp_path):
    # Empty vi_cncl_hour (해제시간) means the halt hasn't been lifted yet.
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        return httpx.Response(
            200,
            json={
                "rt_cd": "0",
                "output": [
                    {
                        "mksc_shrn_iscd": "005930", "hts_kor_isnm": "삼성전자",
                        "vi_cls_code": "1", "vi_kind_code": "1", "cntg_vi_hour": "101530",
                        "vi_cncl_hour": "", "vi_prc": "260500", "vi_count": "1",
                    }
                ],
            },
        )

    broker = make_broker(tmp_path, handler)
    assert await broker.is_vi_triggered("005930") is True


async def test_fetch_minute_candles(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        assert request.url.path == "/uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice"
        assert request.headers["tr_id"] == "FHKST03010200"
        return httpx.Response(200, json={"rt_cd": "0", "output2": [
            {"stck_cntg_hour": "153000", "stck_oprc": "254600", "stck_hgpr": "254700",
             "stck_lwpr": "254400", "stck_prpr": "254500", "cntg_vol": "8285914"},
            {"stck_cntg_hour": "152900", "stck_oprc": "255000", "stck_hgpr": "255100",
             "stck_lwpr": "254500", "stck_prpr": "254600", "cntg_vol": "5120000"},
            {"stck_cntg_hour": ""},  # dropped
        ]})

    broker = make_broker(tmp_path, handler)
    candles = await broker.fetch_minute_candles("005930")
    assert [c.time for c in candles] == ["152900", "153000"]  # oldest first
    assert candles[-1].close == 254500 and candles[-1].volume == 8285914


async def test_fetch_volume_rank(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        assert request.url.path == "/uapi/domestic-stock/v1/quotations/volume-rank"
        assert request.headers["tr_id"] == "FHPST01710000"
        return httpx.Response(200, json={"rt_cd": "0", "output": [
            {"data_rank": "1", "mksc_shrn_iscd": "252670", "hts_kor_isnm": "KODEX 인버스",
             "stck_prpr": "103", "prdy_ctrt": "19.77", "acml_vol": "10370599614"},
            {"data_rank": "2", "mksc_shrn_iscd": "005930", "hts_kor_isnm": "삼성전자",
             "stck_prpr": "254500", "prdy_ctrt": "-10.70", "acml_vol": "31882652"},
        ]})

    broker = make_broker(tmp_path, handler)
    rows = await broker.fetch_volume_rank(10)
    assert len(rows) == 2
    assert rows[0].rank == 1 and rows[0].code == "252670" and rows[0].change_pct == pytest.approx(19.77)
    assert rows[1].name == "삼성전자" and rows[1].volume == 31882652


async def test_fetch_trading_value_rank(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        assert request.url.path == "/uapi/domestic-stock/v1/quotations/volume-rank"
        assert request.headers["tr_id"] == "FHPST01710000"
        assert request.url.params["FID_BLNG_CLS_CODE"] == "3"  # 거래대금순
        return httpx.Response(200, json={"rt_cd": "0", "output": [
            {"data_rank": "1", "mksc_shrn_iscd": "000660", "hts_kor_isnm": "SK하이닉스",
             "stck_prpr": "1919000", "prdy_ctrt": "4.86", "acml_vol": "4081543"},
            {"data_rank": "2", "mksc_shrn_iscd": "005930", "hts_kor_isnm": "삼성전자",
             "stck_prpr": "270000", "prdy_ctrt": "3.65", "acml_vol": "16000165"},
        ]})

    broker = make_broker(tmp_path, handler)
    rows = await broker.fetch_trading_value_rank(10)
    assert len(rows) == 2
    assert rows[0].code == "000660" and rows[0].name == "SK하이닉스"
    assert rows[1].code == "005930"


async def test_fetch_asking_price(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        assert request.url.path == "/uapi/domestic-stock/v1/quotations/inquire-asking-price-exp-ccn"
        assert request.headers["tr_id"] == "FHKST01010200"
        assert request.url.params["FID_INPUT_ISCD"] == "005930"
        return httpx.Response(200, json={"rt_cd": "0", "output1": {
            "bidp1": "208500", "askp1": "209000", "stck_prpr": "208500",
        }, "output2": {}})

    broker = make_broker(tmp_path, handler)
    quote = await broker.fetch_asking_price("005930")
    assert quote.code == "005930" and quote.bid == 208500.0 and quote.ask == 209000.0


async def test_fetch_overseas_asking_price(tmp_path):
    # KIS puts pbid1/pask1 in output2 for this TR; last/open/high/low in output1.
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        assert request.url.path == "/uapi/overseas-price/v1/quotations/inquire-asking-price"
        assert request.headers["tr_id"] == "HHDFS76200100"
        assert request.url.params["SYMB"] == "AAPL"
        return httpx.Response(200, json={"rt_cd": "0",
            "output1": {"last": "233.50"},
            "output2": {"pbid1": "233.40", "pask1": "233.60"},
        })

    broker = make_broker(tmp_path, handler)
    quote = await broker.fetch_overseas_asking_price("AAPL")
    assert quote.code == "AAPL" and quote.bid == 233.40 and quote.ask == 233.60
    assert quote.last == 233.50


async def test_fetch_overseas_asking_price_bid_ask_in_output1(tmp_path):
    # If KIS ever populates output1 instead, the shallow merge still reads it.
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        return httpx.Response(200, json={"rt_cd": "0", "output1": {
            "pbid1": "233.40", "pask1": "233.60", "last": "233.50",
        }})

    broker = make_broker(tmp_path, handler)
    quote = await broker.fetch_overseas_asking_price("AAPL")
    assert quote.bid == 233.40 and quote.ask == 233.60


async def test_fetch_overseas_asking_price_missing_bid_ask_warns(tmp_path, caplog):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        return httpx.Response(200, json={"rt_cd": "0", "output1": {"last": "233.50"}})

    broker = make_broker(tmp_path, handler)
    with caplog.at_level("WARNING"):
        quote = await broker.fetch_overseas_asking_price("AAPL")
    assert quote.bid == 0.0 and quote.ask == 0.0 and quote.last == 233.50
    assert any("no bid/ask fields" in r.message for r in caplog.records)


async def test_fetch_market_cap_rank(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        assert request.url.path == "/uapi/domestic-stock/v1/ranking/market-cap"
        assert request.headers["tr_id"] == "FHPST01740000"
        assert request.url.params["FID_INPUT_ISCD"] == "0000"
        return httpx.Response(200, json={"rt_cd": "0", "output": [
            {"data_rank": "1", "mksc_shrn_iscd": "005930", "hts_kor_isnm": "삼성전자",
             "stck_prpr": "233000", "prdy_ctrt": "5.91", "acml_vol": "20000000"},
            {"data_rank": "2", "mksc_shrn_iscd": "000660", "hts_kor_isnm": "SK하이닉스",
             "stck_prpr": "1919000", "prdy_ctrt": "4.86", "acml_vol": "4081543"},
        ]})

    broker = make_broker(tmp_path, handler)
    rows = await broker.fetch_market_cap_rank(10)
    assert len(rows) == 2
    assert rows[0].code == "005930" and rows[0].name == "삼성전자" and rows[0].rank == 1
    assert rows[1].code == "000660"


async def test_fetch_fluctuation_rank(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        assert request.url.path == "/uapi/domestic-stock/v1/ranking/fluctuation"
        assert request.headers["tr_id"] == "FHPST01700000"
        assert request.url.params["FID_RANK_SORT_CLS_CODE"] == "0"  # rising
        return httpx.Response(200, json={"rt_cd": "0", "output": [
            {"data_rank": "1", "stck_shrn_iscd": "123450", "hts_kor_isnm": "아이에이",
             "stck_prpr": "3430", "prdy_ctrt": "29.92", "acml_vol": "236673"},
        ]})

    broker = make_broker(tmp_path, handler)
    rows = await broker.fetch_fluctuation_rank(rising=True)
    assert len(rows) == 1
    assert rows[0].code == "123450" and rows[0].change_pct == pytest.approx(29.92)


async def test_fetch_fluctuation_rank_falling_sets_sort_code(tmp_path):
    # The existing test above only covers rising=True — this pins the
    # losing-side sort code, which krx_fluctuation_losers_n depends on.
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        assert request.url.params["FID_RANK_SORT_CLS_CODE"] == "1"  # falling
        return httpx.Response(200, json={"rt_cd": "0", "output": []})

    broker = make_broker(tmp_path, handler)
    await broker.fetch_fluctuation_rank(rising=False)


async def test_fetch_fluctuation_rank_default_min_volume_is_unchanged(tmp_path):
    # Regression guard: a call with no min_volume passed (the pre-existing
    # UI caller's exact call shape) must still send FID_VOL_CNT="" — the
    # 2026-09-02 min_volume param must not change wire behavior for anyone
    # who doesn't opt into it.
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        assert request.url.params["FID_VOL_CNT"] == ""
        return httpx.Response(200, json={"rt_cd": "0", "output": []})

    broker = make_broker(tmp_path, handler)
    await broker.fetch_fluctuation_rank(rising=True)


async def test_fetch_fluctuation_rank_min_volume_sets_floor(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        assert request.url.params["FID_VOL_CNT"] == "100000"
        return httpx.Response(200, json={"rt_cd": "0", "output": []})

    broker = make_broker(tmp_path, handler)
    await broker.fetch_fluctuation_rank(rising=True, min_volume=100_000)


async def test_fetch_disparity_rank(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        assert request.url.path == "/uapi/domestic-stock/v1/ranking/disparity"
        assert request.headers["tr_id"] == "FHPST01780000"
        assert request.url.params["FID_COND_SCR_DIV_CODE"] == "20178"
        assert request.url.params["FID_RANK_SORT_CLS_CODE"] == "1"  # default: low (이격도하위순)
        assert request.url.params["FID_HOUR_CLS_CODE"] == "20"  # default window
        assert request.url.params["FID_INPUT_ISCD"] == "0000"
        return httpx.Response(200, json={"rt_cd": "0", "output": [
            {"data_rank": "1", "mksc_shrn_iscd": "005930", "hts_kor_isnm": "삼성전자",
             "stck_prpr": "70000", "prdy_ctrt": "-3.21", "acml_vol": "12345678"},
            {"data_rank": "2", "mksc_shrn_iscd": "000660", "hts_kor_isnm": "SK하이닉스",
             "stck_prpr": "180000", "prdy_ctrt": "-2.05", "acml_vol": "5432100"},
        ]})

    broker = make_broker(tmp_path, handler)
    rows = await broker.fetch_disparity_rank()
    assert len(rows) == 2
    assert rows[0].code == "005930" and rows[0].name == "삼성전자" and rows[0].rank == 1
    assert rows[0].change_pct == pytest.approx(-3.21)
    assert rows[1].code == "000660"


async def test_fetch_disparity_rank_high_flips_sort_code(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        assert request.url.params["FID_RANK_SORT_CLS_CODE"] == "0"  # high
        return httpx.Response(200, json={"rt_cd": "0", "output": []})

    broker = make_broker(tmp_path, handler)
    await broker.fetch_disparity_rank(low=False)


async def test_fetch_disparity_rank_window_days_selects_moving_average(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        assert request.url.params["FID_HOUR_CLS_CODE"] == "60"
        return httpx.Response(200, json={"rt_cd": "0", "output": []})

    broker = make_broker(tmp_path, handler)
    await broker.fetch_disparity_rank(window_days=60)


async def test_fetch_disparity_rank_min_volume_zero_sends_blank(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        assert request.url.params["FID_VOL_CNT"] == ""
        return httpx.Response(200, json={"rt_cd": "0", "output": []})

    broker = make_broker(tmp_path, handler)
    await broker.fetch_disparity_rank(min_volume=0)


async def test_fetch_disparity_rank_min_volume_sets_floor(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        assert request.url.params["FID_VOL_CNT"] == "100000"
        return httpx.Response(200, json={"rt_cd": "0", "output": []})

    broker = make_broker(tmp_path, handler)
    await broker.fetch_disparity_rank(min_volume=100_000)


# ---------------------------------------------------------------------------
# max_price (FID_INPUT_PRICE_2) — 2026-09-02 (screening-improvement-research
# #3 follow-up). Each TR's own pre-existing "no filter" spelling for this
# field ("0" for the shared volume-rank TR, "" for the other three) must be
# byte-identical when max_price is omitted, and _price_ceiling_param must
# floor rather than round when it isn't.
# ---------------------------------------------------------------------------


async def test_fetch_trading_value_rank_default_max_price_is_unchanged(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        assert request.url.params["FID_INPUT_PRICE_2"] == "0"
        return httpx.Response(200, json={"rt_cd": "0", "output": []})

    broker = make_broker(tmp_path, handler)
    await broker.fetch_trading_value_rank()


async def test_fetch_trading_value_rank_max_price_sets_ceiling(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        assert request.url.params["FID_INPUT_PRICE_2"] == "400000"
        return httpx.Response(200, json={"rt_cd": "0", "output": []})

    broker = make_broker(tmp_path, handler)
    await broker.fetch_trading_value_rank(max_price=400_000)


async def test_fetch_volume_rank_max_price_threads_through_shared_tr(tmp_path):
    # fetch_volume_rank shares _fetch_volume_rank_tr with
    # fetch_trading_value_rank -- confirms the pass-through works on this
    # caller too, not just the one exercised above.
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        assert request.url.params["FID_INPUT_PRICE_2"] == "400000"
        return httpx.Response(200, json={"rt_cd": "0", "output": []})

    broker = make_broker(tmp_path, handler)
    await broker.fetch_volume_rank(max_price=400_000)


async def test_fetch_market_cap_rank_default_max_price_is_unchanged(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        assert request.url.params["FID_INPUT_PRICE_2"] == ""
        return httpx.Response(200, json={"rt_cd": "0", "output": []})

    broker = make_broker(tmp_path, handler)
    await broker.fetch_market_cap_rank()


async def test_fetch_market_cap_rank_max_price_sets_ceiling(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        assert request.url.params["FID_INPUT_PRICE_2"] == "400000"
        return httpx.Response(200, json={"rt_cd": "0", "output": []})

    broker = make_broker(tmp_path, handler)
    await broker.fetch_market_cap_rank(max_price=400_000)


async def test_fetch_fluctuation_rank_default_max_price_is_unchanged(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        assert request.url.params["FID_INPUT_PRICE_2"] == ""
        return httpx.Response(200, json={"rt_cd": "0", "output": []})

    broker = make_broker(tmp_path, handler)
    await broker.fetch_fluctuation_rank(rising=True)


async def test_fetch_fluctuation_rank_max_price_sets_ceiling(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        assert request.url.params["FID_INPUT_PRICE_2"] == "400000"
        return httpx.Response(200, json={"rt_cd": "0", "output": []})

    broker = make_broker(tmp_path, handler)
    await broker.fetch_fluctuation_rank(rising=True, max_price=400_000)


async def test_fetch_disparity_rank_default_max_price_is_unchanged(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        assert request.url.params["FID_INPUT_PRICE_2"] == ""
        return httpx.Response(200, json={"rt_cd": "0", "output": []})

    broker = make_broker(tmp_path, handler)
    await broker.fetch_disparity_rank()


async def test_fetch_disparity_rank_max_price_sets_ceiling(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        assert request.url.params["FID_INPUT_PRICE_2"] == "400000"
        return httpx.Response(200, json={"rt_cd": "0", "output": []})

    broker = make_broker(tmp_path, handler)
    await broker.fetch_disparity_rank(max_price=400_000)


def test_price_ceiling_param_floors_rather_than_rounds():
    from app.brokers.kis.domestic.ranking import _price_ceiling_param

    assert _price_ceiling_param(399_999.7, blank="") == "399999"
    assert _price_ceiling_param(0, blank="") == ""
    assert _price_ceiling_param(-1, blank="") == ""
    assert _price_ceiling_param(None, blank="") == ""
    assert _price_ceiling_param(None, blank="0") == "0"


async def test_fetch_overseas_trading_value_rank(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        assert request.url.path == "/uapi/overseas-stock/v1/ranking/trade-pbmn"
        assert request.headers["tr_id"] == "HHDFS76320010"
        assert request.url.params["EXCD"] == "NAS"
        return httpx.Response(200, json={"rt_cd": "0", "output1": {"nrec": "1"}, "output2": [
            {"rank": "1", "symb": "MU", "name": "마이크론 테크놀로지", "ename": "MICRON TECHNOLOGY INC",
             "last": "972.47", "rate": "+1.35", "tvol": "23495"},
        ]})

    broker = make_broker(tmp_path, handler)
    rows = await broker.fetch_overseas_trading_value_rank()
    assert len(rows) == 1
    assert rows[0].code == "MU" and rows[0].name == "마이크론 테크놀로지" and rows[0].volume == 23495


async def test_fetch_overseas_market_cap_rank(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        assert request.url.path == "/uapi/overseas-stock/v1/ranking/market-cap"
        assert request.headers["tr_id"] == "HHDFS76350100"
        assert request.url.params["EXCD"] == "NAS"
        return httpx.Response(200, json={"rt_cd": "0", "output1": {"nrec": "1"}, "output2": [
            {"rank": "1", "symb": "AAPL", "name": "애플", "ename": "APPLE INC",
             "last": "233.5", "rate": "+1.2", "tvol": "50000000", "tomv": "3500000000000"},
        ]})

    broker = make_broker(tmp_path, handler)
    rows = await broker.fetch_overseas_market_cap_rank()
    assert len(rows) == 1
    assert rows[0].code == "AAPL" and rows[0].name == "애플" and rows[0].rank == 1


async def test_fetch_overseas_index_quote(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        assert request.url.path == "/uapi/overseas-price/v1/quotations/inquire-daily-chartprice"
        assert request.headers["tr_id"] == "FHKST03030100"
        assert request.url.params["FID_COND_MRKT_DIV_CODE"] == "N"
        assert request.url.params["FID_INPUT_ISCD"] == "SPX"
        return httpx.Response(200, json={"rt_cd": "0", "output1": {
            "ovrs_nmix_prpr": "7600.50", "ovrs_nmix_prdy_vrss": "45.20", "prdy_ctrt": "0.60",
        }, "output2": []})

    broker = make_broker(tmp_path, handler)
    q = await broker.fetch_overseas_index_quote("SPX", "S&P 500", "20260728", "20260804")
    assert q.name == "S&P 500" and q.value == pytest.approx(7600.50)
    assert q.change == pytest.approx(45.20) and q.change_pct == pytest.approx(0.60)


async def test_fetch_index_price(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        assert request.url.path == "/uapi/domestic-stock/v1/quotations/inquire-index-price"
        assert request.headers["tr_id"] == "FHPUP02100000"
        assert request.url.params["FID_INPUT_ISCD"] == "0001"
        return httpx.Response(200, json={"rt_cd": "0", "output": {
            "bstp_nmix_prpr": "6806.93", "bstp_nmix_prdy_vrss": "-669.01", "bstp_nmix_prdy_ctrt": "-8.95"}})

    broker = make_broker(tmp_path, handler)
    q = await broker.fetch_index_price("0001")
    assert q.name == "KOSPI" and q.value == pytest.approx(6806.93)
    assert q.change == pytest.approx(-669.01) and q.change_pct == pytest.approx(-8.95)


async def test_fetch_is_market_open_on_trading_day(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        assert request.url.path == "/uapi/domestic-stock/v1/quotations/chk-holiday"
        assert request.headers["tr_id"] == "CTCA0903R"
        assert request.url.params["BASS_DT"] == "20260804"
        return httpx.Response(200, json={"rt_cd": "0", "output": [{"bass_dt": "20260804", "opnd_yn": "Y"}]})

    broker = make_broker(tmp_path, handler)
    assert await broker.fetch_is_market_open_on("20260804") is True


async def test_fetch_is_market_open_on_holiday(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        return httpx.Response(200, json={"rt_cd": "0", "output": [{"bass_dt": "20260101", "opnd_yn": "N"}]})

    broker = make_broker(tmp_path, handler)
    assert await broker.fetch_is_market_open_on("20260101") is False


async def test_fetch_is_market_open_on_raises_when_no_rows(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        return httpx.Response(200, json={"rt_cd": "0", "output": []})

    broker = make_broker(tmp_path, handler)
    with pytest.raises(KISBrokerError, match="no matching bass_dt row"):
        await broker.fetch_is_market_open_on("20260101")


async def test_fetch_investor_trend_parses_net_buy(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        assert request.url.path == "/uapi/domestic-stock/v1/quotations/inquire-investor"
        assert request.headers["tr_id"] == "FHKST01010900"
        assert request.url.params["FID_INPUT_ISCD"] == "005930"
        return httpx.Response(
            200,
            json={
                "rt_cd": "0",
                "output": [
                    {"stck_bsop_date": "20260713", "stck_clpr": "260500",
                     "prsn_ntby_qty": "-12000", "frgn_ntby_qty": "8000", "orgn_ntby_qty": "4000"},
                    {"stck_bsop_date": "", "stck_clpr": ""},  # padding row dropped
                ],
            },
        )

    broker = make_broker(tmp_path, handler)
    flows = await broker.fetch_investor_trend("005930")
    assert len(flows) == 1
    assert flows[0].foreign_net == 8000 and flows[0].institution_net == 4000
    assert flows[0].individual_net == -12000 and flows[0].close == 260500


async def test_fetch_investor_estimate_parses(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        assert request.url.path == "/uapi/domestic-stock/v1/quotations/investor-trend-estimate"
        assert request.headers["tr_id"] == "HHPTJ04160200"
        assert request.url.params["MKSC_SHRN_ISCD"] == "005930"
        return httpx.Response(
            200,
            json={
                "rt_cd": "0",
                "output2": [
                    {"bsop_hour_gb": "1", "frgn_fake_ntby_qty": "5000",
                     "orgn_fake_ntby_qty": "-2000", "sum_fake_ntby_qty": "3000"},
                ],
            },
        )

    broker = make_broker(tmp_path, handler)
    est = await broker.fetch_investor_estimate("005930")
    assert len(est) == 1
    assert est[0].foreign_net == 5000 and est[0].institution_net == -2000 and est[0].total_net == 3000


async def test_fetch_kr_financial_statements(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        path = request.url.path
        if path.endswith("/finance/income-statement"):
            assert request.headers["tr_id"] == "FHKST66430200"
            return httpx.Response(200, json={"rt_cd": "0", "output": [
                {"stac_yymm": "202512", "sale_account": "3000000", "sale_totl_prfi": "900000",
                 "bsop_prti": "500000", "thtr_ntin": "400000"},
                {"stac_yymm": ""},  # padding dropped
            ]})
        if path.endswith("/finance/balance-sheet"):
            assert request.headers["tr_id"] == "FHKST66430100"
            return httpx.Response(200, json={"rt_cd": "0", "output": [
                {"stac_yymm": "202512", "total_aset": "5000000", "total_lblt": "2000000",
                 "total_cptl": "3000000", "cras": "1800000", "flow_lblt": "1200000"},
            ]})
        assert path.endswith("/finance/financial-ratio")
        assert request.headers["tr_id"] == "FHKST66430300"
        return httpx.Response(200, json={"rt_cd": "0", "output": [
            {"stac_yymm": "202512", "grs": "12.5", "bsop_prfi_inrt": "8.0", "ntin_inrt": "6.0",
             "roe_val": "13.3", "eps": "5000", "bps": "40000", "lblt_rate": "66.7"},
        ]})

    broker = make_broker(tmp_path, handler)
    inc = await broker.fetch_kr_income_statement("005930")
    assert len(inc) == 1 and inc[0].revenue == 3000000 and inc[0].net_income == 400000
    bal = await broker.fetch_kr_balance_sheet("005930")
    assert bal[0].total_equity == 3000000 and bal[0].current_assets == 1800000
    ratio = await broker.fetch_kr_financial_ratio("005930")
    assert ratio[0].roe == pytest.approx(13.3) and ratio[0].debt_ratio == pytest.approx(66.7)


async def test_fetch_kr_additional_ratio_statements(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        path = request.url.path
        if path.endswith("/finance/stability-ratio"):
            assert request.headers["tr_id"] == "FHKST66430600"
            return httpx.Response(200, json={"rt_cd": "0", "output": [
                {"stac_yymm": "202603", "lblt_rate": "30.15", "bram_depn": "4.44",
                 "crnt_rate": "253.91", "quck_rate": "0.00"},
            ]})
        if path.endswith("/finance/profit-ratio"):
            assert request.headers["tr_id"] == "FHKST66430400"
            return httpx.Response(200, json={"rt_cd": "0", "output": [
                {"stac_yymm": "202603", "cptl_ntin_rate": "14.65", "self_cptl_ntin_inrt": "19.16",
                 "sale_ntin_rate": "35.28", "sale_totl_rate": "61.19"},
            ]})
        if path.endswith("/finance/growth-ratio"):
            assert request.headers["tr_id"] == "FHKST66430800"
            return httpx.Response(200, json={"rt_cd": "0", "output": [
                {"stac_yymm": "202603", "grs": "69.16", "bsop_prfi_inrt": "756.10",
                 "equt_inrt": "19.68", "totl_aset_inrt": "22.65"},
            ]})
        assert path.endswith("/finance/other-major-ratios")
        assert request.headers["tr_id"] == "FHKST66430500"
        return httpx.Response(200, json={"rt_cd": "0", "output": [
            {"stac_yymm": "202603", "payout_rate": "0.00", "eva": "0.00",
             "ebitda": "695994.00", "ev_ebitda": "6.88"},
        ]})

    broker = make_broker(tmp_path, handler)
    stab = await broker.fetch_kr_stability_ratio("005930")
    assert stab[0].borrowing_dependency == pytest.approx(4.44) and stab[0].current_ratio == pytest.approx(253.91)
    prof = await broker.fetch_kr_profitability_ratio("005930")
    assert prof[0].net_margin == pytest.approx(35.28) and prof[0].gross_margin == pytest.approx(61.19)
    growth = await broker.fetch_kr_growth_ratio_detail("005930")
    assert growth[0].equity_growth == pytest.approx(19.68) and growth[0].asset_growth == pytest.approx(22.65)
    other = await broker.fetch_kr_other_major_ratios("005930")
    assert other[0].ebitda == pytest.approx(695994.00) and other[0].ev_ebitda == pytest.approx(6.88)


async def test_fetch_stock_status_parses_name_fields(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        assert request.url.path == "/uapi/domestic-stock/v1/quotations/search-stock-info"
        assert request.headers["tr_id"] == "CTPF1002R"
        assert request.url.params["PDNO"] == "005930"
        return httpx.Response(200, json={"rt_cd": "0", "output": {
            "tr_stop_yn": "N", "admn_item_yn": "N", "nxt_tr_stop_yn": "N",
            "prdt_abrv_name": "삼성전자", "prdt_eng_abrv_name": "SamsungElec",
            "scty_grp_id_cd": "ST",
        }})

    broker = make_broker(tmp_path, handler)
    status = await broker.fetch_stock_status("005930")
    assert status.korean_name == "삼성전자" and status.english_name == "SamsungElec"
    assert status.trading_halted is False
    assert status.security_group_code == "ST"


async def test_fetch_overseas_company_name_parses_name_fields(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        assert request.url.path == "/uapi/overseas-price/v1/quotations/search-info"
        assert request.headers["tr_id"] == "CTPF1702R"
        assert request.url.params["PDNO"] == "AAPL"
        assert request.url.params["PRDT_TYPE_CD"] == "512"
        return httpx.Response(200, json={"rt_cd": "0", "output": {
            "prdt_name": "애플", "prdt_eng_name": "Apple",
        }})

    broker = make_broker(tmp_path, handler)
    korean_name, english_name = await broker.fetch_overseas_company_name("AAPL")
    assert korean_name == "애플" and english_name == "Apple"


async def test_fetch_overseas_news_titles_normalizes_single_article_dict(tmp_path):
    # outblock1 comes back as a bare dict, not a list, when exactly one
    # article matches — iterating a dict yields its keys and crashes on
    # item.get(...) without normalization first.
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        assert request.url.path == "/uapi/overseas-price/v1/quotations/news-title"
        return httpx.Response(200, json={"rt_cd": "0", "outblock1": {
            "data_dt": "20260807", "data_tm": "093000", "title": "Apple beats", "source": "Reuters",
        }})

    broker = make_broker(tmp_path, handler)
    titles = await broker.fetch_overseas_news_titles("AAPL")
    assert len(titles) == 1
    assert titles[0].title == "Apple beats" and titles[0].source == "Reuters"


async def test_fetch_kr_earnings_estimate(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        assert request.url.path == "/uapi/domestic-stock/v1/quotations/estimate-perform"
        assert request.headers["tr_id"] == "HHKST668300C0"
        assert request.url.params["SHT_CD"] == "005930"
        return httpx.Response(200, json={
            "rt_cd": "0",
            "output1": {"sht_cd": "A005930", "item_kor_nm": "삼성전자", "rcmd_name": "매수"},
            "output2": [
                {"data1": "2589355", "data2": "3008709"},
                {"data1": "-143", "data2": "162"},
                {"data1": "65670", "data2": "327260"},
                {"data1": "-849", "data2": "3983"},
                {"data1": "144734", "data2": "336214"},
                {"data1": "-736", "data2": "1323"},
            ],
            "output3": [],
            "output4": [{"dt": "2023.12"}, {"dt": "2024.12E"}],
        })

    broker = make_broker(tmp_path, handler)
    rows = await broker.fetch_kr_earnings_estimate("005930")
    assert len(rows) == 2
    assert rows[0].period == "2023.12" and rows[0].revenue == 2589355 and rows[0].revenue_growth == -143
    assert rows[0].operating_profit == 65670 and rows[0].net_income == 144734
    assert rows[1].period == "2024.12E" and rows[1].revenue == 3008709 and rows[1].net_income_growth == 1323


async def test_fetch_kr_earnings_estimate_returns_empty_when_rows_too_short(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        return httpx.Response(200, json={"rt_cd": "0", "output1": {}, "output2": [], "output3": [], "output4": []})

    broker = make_broker(tmp_path, handler)
    rows = await broker.fetch_kr_earnings_estimate("005930")
    assert rows == []


async def test_fetch_invest_opinion(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        assert request.url.path == "/uapi/domestic-stock/v1/quotations/invest-opinion"
        assert request.headers["tr_id"] == "FHKST663300C0"
        assert request.url.params["FID_INPUT_ISCD"] == "005930"
        return httpx.Response(200, json={"rt_cd": "0", "output": [
            {"stck_bsop_date": "20260710", "invt_opnn": "매수", "rgbf_invt_opnn": "중립",
             "hts_goal_prc": "380000", "stck_prdy_clpr": "285000"},
            {"stck_bsop_date": ""},  # padding dropped
        ]})

    broker = make_broker(tmp_path, handler)
    ops = await broker.fetch_invest_opinion("005930", "20260101", "20260713")
    assert len(ops) == 1
    assert ops[0].opinion == "매수" and ops[0].prev_opinion == "중립"
    assert ops[0].target_price == 380000 and ops[0].prev_close == 285000


async def test_fetch_daily_short_sale_and_program_trade(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        if request.url.path.endswith("/quotations/daily-short-sale"):
            assert request.headers["tr_id"] == "FHPST04830000"
            return httpx.Response(200, json={"rt_cd": "0", "output1": {"stck_prpr": "254500"}, "output2": [
                {"stck_bsop_date": "20260713", "stck_clpr": "254500", "ssts_cntg_qty": "330953",
                 "ssts_vol_rlim": "1.04", "acml_ssts_cntg_qty_rlim": "3.43"},
                {"stck_bsop_date": ""},  # padding dropped
            ]})
        assert request.url.path.endswith("/quotations/program-trade-by-stock")
        assert request.headers["tr_id"] == "FHPPG04650101"
        return httpx.Response(200, json={"rt_cd": "0", "output": [
            {"bsop_hour": "153000", "whol_smtn_ntby_qty": "-125000", "whol_smtn_ntby_tr_pbmn": "-31000000000"},
        ]})

    broker = make_broker(tmp_path, handler)
    shorts = await broker.fetch_daily_short_sale("005930", "20260601", "20260713")
    assert len(shorts) == 1 and shorts[0].short_ratio == pytest.approx(1.04)
    assert shorts[0].short_volume == 330953 and shorts[0].accum_short_ratio == pytest.approx(3.43)
    prog = await broker.fetch_program_trade("005930")
    assert len(prog) == 1 and prog[0].net_buy_qty == -125000


async def test_fetch_news_titles(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        assert request.url.path == "/uapi/domestic-stock/v1/quotations/news-title"
        assert request.headers["tr_id"] == "FHKST01011800"
        assert request.url.params["FID_INPUT_ISCD"] == "005930"
        return httpx.Response(200, json={"rt_cd": "0", "output1": [
            {"data_dt": "20260713", "data_tm": "150000", "dorg": "연합뉴스",
             "hts_pbnt_titl_cntt": "삼성전자, 신형 메모리 양산 발표"},
            {"data_dt": "20260713", "data_tm": "", "dorg": "", "hts_pbnt_titl_cntt": "  "},  # blank dropped
        ]})

    broker = make_broker(tmp_path, handler)
    news = await broker.fetch_news_titles("005930")
    assert len(news) == 1
    assert news[0].title == "삼성전자, 신형 메모리 양산 발표" and news[0].source == "연합뉴스"


async def test_fetch_dividend_schedule(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        assert request.url.path == "/uapi/domestic-stock/v1/ksdinfo/dividend"
        assert request.headers["tr_id"] == "HHKDB669102C0"
        assert request.url.params["SHT_CD"] == "005930"
        return httpx.Response(200, json={"rt_cd": "0", "output1": [
            {"record_date": "20261231", "divi_kind": "결산배당", "per_sto_divi_amt": "361",
             "divi_rate": "2.10", "divi_pay_dt": "20270401"},
            {"record_date": ""},  # dropped
        ]})

    broker = make_broker(tmp_path, handler)
    divs = await broker.fetch_dividend_schedule("005930", "20260101", "20261231")
    assert len(divs) == 1
    assert divs[0].kind == "결산배당" and divs[0].cash_amount == 361 and divs[0].pay_date == "20270401"


async def test_submit_overseas_limit_buy(tmp_path):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        assert request.url.path == "/uapi/overseas-stock/v1/trading/order"
        captured["tr_id"] = request.headers["tr_id"]
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"rt_cd": "0", "msg1": "모의투자 매수주문이 완료 되었습니다.",
                  "output": {"ODNO": "0030089601", "ORD_TMD": "223045"}},
        )

    broker = make_broker(tmp_path, handler)
    result = await broker.submit_overseas_order("AAPL", "buy", 1, 315.50)

    assert captured["tr_id"] == "VTTT1002U"  # 미국 모의 매수
    assert captured["body"]["OVRS_EXCG_CD"] == "NASD"
    assert captured["body"]["ORD_DVSN"] == "00"  # limit only on paper
    assert captured["body"]["OVRS_ORD_UNPR"] == "315.50"
    assert captured["body"]["SLL_TYPE"] == ""
    assert result.order_no == "0030089601"

    with pytest.raises(ValueError, match="limit-only"):
        await broker.submit_overseas_order("AAPL", "buy", 1, None)


async def test_submit_overseas_order_rejects_genuinely_fractional_quantity(tmp_path):
    """A REAL fraction (e.g. 111.5, not a whole number stored as float) must
    be rejected loudly rather than silently floored — silently flooring
    here would just move the 2026-08-19-style regression one layer down
    instead of surfacing it."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        raise AssertionError("a fractional ORD_QTY must never reach the KIS request")

    broker = make_broker(tmp_path, handler)
    with pytest.raises(ValueError, match="whole number of shares"):
        await broker.submit_overseas_order("AAPL", "buy", 111.5, 315.50)
    with pytest.raises(ValueError, match="whole number of shares"):
        await broker.cancel_overseas_order("0030089601", "AAPL", 5.5)


async def test_submit_overseas_order_sends_integer_ord_qty_not_float_string(tmp_path):
    """2026-08-19 incident, exact reproduction: a WHOLE-number float
    quantity (e.g. real_shares_held leaking 111.0 instead of int 111) must
    serialize as ORD_QTY="111", never "111.0" — the malformed string KIS
    rejected with IGW00013/IGW00007. An int quantity gets the same
    treatment (this is what every legitimate call site already passes)."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200, json={"rt_cd": "0", "msg1": "ok", "output": {"ODNO": "1", "ORD_TMD": "090000"}},
        )

    broker = make_broker(tmp_path, handler)
    await broker.submit_overseas_order("AXTI", "sell", 111.0, 73.04)  # the exact 2026-08-19 shape
    assert captured["body"]["ORD_QTY"] == "111"

    await broker.submit_overseas_order("AXTI", "sell", 111, 73.04)
    assert captured["body"]["ORD_QTY"] == "111"


async def test_domestic_submit_order_rejects_genuinely_fractional_quantity(tmp_path):
    """Symmetric guard on the domestic path (2026-08-19) — parsing is
    already integer-only there, so this should never trigger in practice,
    but the defense line shouldn't exist on only one of the two paths."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        raise AssertionError("a fractional ORD_QTY must never reach the KIS request")

    broker = make_broker(tmp_path, handler)
    with pytest.raises(ValueError, match="whole number of shares"):
        await broker.submit_order("005930", "buy", 10.5)


async def test_domestic_revise_order_rejects_genuinely_fractional_quantity(tmp_path):
    """2026-08-26: revise_order was the one of the three domestic order
    methods (submit_order/cancel_order/revise_order) missing this guard —
    a Stage 3 app/web audit finding, not the original 2026-08-19 incident."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        raise AssertionError("a fractional ORD_QTY must never reach the KIS request")

    broker = make_broker(tmp_path, handler)
    with pytest.raises(ValueError, match="whole number of shares"):
        await broker.revise_order("0000100", "00950", 10.5, 68000)


async def test_domestic_revise_order_sends_integer_ord_qty_not_float_string(tmp_path):
    """Same 2026-08-19 shape as test_submit_overseas_order_sends_integer_
    ord_qty_not_float_string above, applied to revise_order: a WHOLE-number
    float quantity must serialize as ORD_QTY="111", never "111.0"."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"rt_cd": "0", "msg1": "정정완료",
                                          "output": {"KRX_FWDG_ORD_ORGNO": "00950", "ODNO": "0000200", "ORD_TMD": "1"}})

    broker = make_broker(tmp_path, handler)
    await broker.revise_order("0000100", "00950", 111.0, 68000)
    assert captured["body"]["ORD_QTY"] == "111"

    await broker.revise_order("0000100", "00950", 111, 68000)
    assert captured["body"]["ORD_QTY"] == "111"


async def test_fetch_overseas_balance_floors_fractional_quantity_and_warns(tmp_path, caplog):
    """A fractional ovrs_cblc_qty (e.g. a split/dividend remainder) must be
    floored to an int for BrokerPosition.quantity, never passed through as
    a float — that float previously ended up verbatim in ORD_QTY and KIS
    rejected it (2026-08-19 AXTI/GOOG/GOOGL incident)."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        return httpx.Response(
            200,
            json={
                "rt_cd": "0", "ctx_area_fk200": "", "ctx_area_nk200": "",
                "output1": [
                    {"ovrs_pdno": "AXTI", "ovrs_item_name": "AXT INC", "ovrs_cblc_qty": "111.5",
                     "pchs_avg_pric": "73.04", "now_pric2": "73.00",
                     "ovrs_stck_evlu_amt": "8103.0", "frcr_evlu_pfls_amt": "0", "evlu_pfls_rt": "0"},
                ],
                "output2": {"frcr_pchs_amt1": "0", "tot_evlu_pfls_amt": "0"},
            },
        )

    broker = make_broker(tmp_path, handler)
    with caplog.at_level("WARNING"):
        bal = await broker.fetch_overseas_balance()
    assert bal.positions[0].quantity == 111
    assert isinstance(bal.positions[0].quantity, int)
    assert "fractional quantity" in caplog.text


async def test_fetch_overseas_balance_quantity_is_int(tmp_path):
    """BrokerPosition.quantity must be a real int, not a float that happens
    to equal a whole number (str -> int, not str -> float)."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        return httpx.Response(
            200,
            json={
                "rt_cd": "0", "ctx_area_fk200": "", "ctx_area_nk200": "",
                "output1": [
                    {"ovrs_pdno": "AAPL", "ovrs_item_name": "APPLE INC", "ovrs_cblc_qty": "111",
                     "pchs_avg_pric": "315.50", "now_pric2": "318.20",
                     "ovrs_stck_evlu_amt": "318.20", "frcr_evlu_pfls_amt": "2.70", "evlu_pfls_rt": "0.85"},
                ],
                "output2": {"frcr_pchs_amt1": "315.50", "tot_evlu_pfls_amt": "2.70"},
            },
        )

    broker = make_broker(tmp_path, handler)
    bal = await broker.fetch_overseas_balance()
    assert bal.positions[0].quantity == 111
    assert isinstance(bal.positions[0].quantity, int)


async def test_checked_body_captures_msg_cd_for_permanent_refusal_classification(tmp_path):
    """2026-08-19: KISBrokerError previously carried msg_cd only baked into
    its message string, so a caller couldn't tell a malformed-request
    refusal (IGW00013) from a transient one. This is what let AXTI's
    stop-loss retry the same doomed order 17 times."""
    from app.brokers.kis.const import KISBrokerError, is_permanent_refusal

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        return httpx.Response(
            200, json={"rt_cd": "1", "msg_cd": "IGW00013", "msg1": "주문 수량을 확인해주세요."},
        )

    broker = make_broker(tmp_path, handler)
    with pytest.raises(KISBrokerError) as excinfo:
        await broker.submit_overseas_order("AXTI", "sell", 111, 73.04)
    assert excinfo.value.msg_cd == "IGW00013"
    assert is_permanent_refusal(excinfo.value)

    # An unrecognized code defaults to transient/retryable — a wrong guess
    # here may cost an extra retry, never suppress a needed one.
    other = KISBrokerError("x", msg_cd="EGW00201")
    assert not is_permanent_refusal(other)
    # And a KISBrokerError constructed the old way (no msg_cd) is also
    # treated as transient, not permanent.
    assert not is_permanent_refusal(KISBrokerError("legacy call site"))


async def test_fetch_overseas_balance_drops_zero_qty(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        assert request.headers["tr_id"] == "VTTS3012R"
        return httpx.Response(
            200,
            json={
                "rt_cd": "0", "ctx_area_fk200": "", "ctx_area_nk200": "",
                "output1": [
                    {"ovrs_pdno": "AAPL", "ovrs_item_name": "APPLE INC", "ovrs_cblc_qty": "1",
                     "pchs_avg_pric": "315.5000", "now_pric2": "318.20",
                     "ovrs_stck_evlu_amt": "318.20", "frcr_evlu_pfls_amt": "2.70", "evlu_pfls_rt": "0.85"},
                    {"ovrs_pdno": "TSLA", "ovrs_item_name": "TESLA INC", "ovrs_cblc_qty": "0",
                     "pchs_avg_pric": "0", "now_pric2": "0", "ovrs_stck_evlu_amt": "0",
                     "frcr_evlu_pfls_amt": "0", "evlu_pfls_rt": "0"},
                ],
                "output2": {"frcr_pchs_amt1": "315.50", "tot_evlu_pfls_amt": "2.70"},
            },
        )

    broker = make_broker(tmp_path, handler)
    bal = await broker.fetch_overseas_balance()
    assert len(bal.positions) == 1
    assert bal.positions[0].code == "AAPL"
    assert bal.total_eval_amount == pytest.approx(318.20)
    assert bal.total_pnl_amount == pytest.approx(2.70)


async def test_fetch_overseas_present_balance(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        assert request.url.path == "/uapi/overseas-stock/v1/trading/inquire-present-balance"
        assert request.headers["tr_id"] == "VTRP6504R"
        return httpx.Response(200, json={
            "rt_cd": "0",
            "output2": [
                {"crcy_cd": "CNY", "frcr_dncl_amt_2": "0", "frcr_drwg_psbl_amt_1": "0",
                 "frcr_evlu_amt2": "0", "frst_bltn_exrt": "220.00"},
                # frcr_dncl_amt_2 reads 0 here on purpose — live-verified to
                # be unreliable per-currency; output3 below is the real source.
                {"crcy_cd": "USD", "frcr_dncl_amt_2": "0", "frcr_drwg_psbl_amt_1": "1000.00",
                 "frcr_evlu_amt2": "1500.00", "frst_bltn_exrt": "1300.00"},
            ],
            "output3": {"frcr_evlu_tota": "1604928"},  # 1234.56 USD * 1300.00
        })

    broker = make_broker(tmp_path, handler)
    cash = await broker.fetch_overseas_present_balance("USD")
    assert cash.currency == "USD" and cash.deposit == pytest.approx(1234.56)
    assert cash.orderable == pytest.approx(1000.0) and cash.eval_amount == pytest.approx(1500.0)
    assert cash.matched is True


async def test_fetch_overseas_present_balance_no_matching_currency_row(tmp_path):
    # Regression: previously this silently returned a zeroed OverseasCash
    # indistinguishable from a genuine $0 balance — callers now check
    # cash.matched instead of trusting the zero.
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        return httpx.Response(200, json={
            "rt_cd": "0",
            "output2": [
                {"crcy_cd": "CNY", "frcr_dncl_amt_2": "0", "frcr_drwg_psbl_amt_1": "0",
                 "frcr_evlu_amt2": "0", "frst_bltn_exrt": "220.00"},
            ],
            "output3": {"frcr_evlu_tota": "0"},
        })

    broker = make_broker(tmp_path, handler)
    cash = await broker.fetch_overseas_present_balance("USD")
    assert cash.matched is False
    assert cash.orderable == 0.0 and cash.deposit == 0.0


async def test_cancel_overseas_order(tmp_path):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        assert request.url.path.endswith("/trading/order-rvsecncl")
        assert request.headers["tr_id"] == "VTTT1004U"
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"rt_cd": "0", "msg1": "취소완료",
                                          "output": {"ODNO": "0030100", "ORD_TMD": "223500"}})

    broker = make_broker(tmp_path, handler)
    res = await broker.cancel_overseas_order("0030099", "AAPL", 2)
    assert res.order_no == "0030100"
    assert captured["body"]["RVSE_CNCL_DVSN_CD"] == "02" and captured["body"]["ORGN_ODNO"] == "0030099"
    assert captured["body"]["OVRS_ORD_UNPR"] == "0"  # KIS spec requires "0" on a cancel, not the real price


async def test_fetch_overseas_price_and_executions(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        if request.url.path.endswith("/quotations/price"):
            assert request.url.params["EXCD"] == "NAS"
            return httpx.Response(200, json={"rt_cd": "0", "output": {"last": "318.20"}})
        assert request.headers["tr_id"] == "VTTS3035R"
        return httpx.Response(
            200,
            json={
                "rt_cd": "0", "ctx_area_fk200": "", "ctx_area_nk200": "",
                "output": [
                    {"ord_dt": "20260713", "odno": "0030089601", "pdno": "AAPL",
                     "prdt_name": "APPLE INC", "sll_buy_dvsn_cd": "02", "ft_ord_qty": "1",
                     "ft_ccld_qty": "1", "nccs_qty": "0", "ft_ccld_unpr3": "315.5000",
                     "ft_ccld_amt3": "315.50"},
                ],
            },
        )

    broker = make_broker(tmp_path, handler)
    assert await broker.fetch_overseas_price("AAPL") == pytest.approx(318.20)
    execs = await broker.fetch_overseas_executions("20260713", "20260713")
    assert len(execs) == 1
    assert execs[0].side == "buy" and execs[0].filled_qty == 1
    assert execs[0].avg_price == pytest.approx(315.50)


@pytest.mark.parametrize(
    "extra, expected",
    [
        ({}, False),  # fields absent entirely — the conservative default
        ({"rjct_rson": "", "rjct_rson_name": ""}, False),  # present but empty
        ({"rjct_rson": "0", "rjct_rson_name": ""}, False),  # "0" means "no rejection"
        ({"rjct_rson": "40250000"}, True),
        ({"rjct_rson_name": "주문가능금액이 부족합니다"}, True),
        ({"rjct_rson": "40250000", "rjct_rson_name": "주문가능금액이 부족합니다"}, True),
    ],
)
async def test_fetch_overseas_executions_flags_rejected_orders(tmp_path, extra, expected):
    # A KIS-rejected overseas order (e.g. 주문가능금액 부족) still reports the full
    # quantity as 미체결(nccs_qty) with cncl_yn absent, so without reading
    # rjct_rson/rjct_rson_name it looks like a permanently-open order.
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        return httpx.Response(200, json={
            "rt_cd": "0", "ctx_area_fk200": "", "ctx_area_nk200": "",
            "output": [
                {"ord_dt": "20260818", "odno": "0031000", "pdno": "SNDK",
                 "prdt_name": "SANDISK", "sll_buy_dvsn_cd": "02", "ft_ord_qty": "6",
                 "ft_ccld_qty": "0", "nccs_qty": "6", "ft_ccld_unpr3": "0",
                 "ft_ccld_amt3": "0", **extra},
            ],
        })

    broker = make_broker(tmp_path, handler)
    execs = await broker.fetch_overseas_executions("20260818", "20260818")
    assert len(execs) == 1
    assert execs[0].remaining_qty == 6  # unchanged either way — only the flag differs
    assert execs[0].rejected is expected


async def test_fetch_overseas_executions_all_filter_sentinel_differs_paper_vs_real(tmp_path):
    # KIS's own overseas executions TR docs (해외주식 주문체결내역) say the
    # "all products/all exchanges" sentinel is "" on paper but "%" on real —
    # confirmed live 2026-08-05 the wrong direction returns rt_cd=0 with an
    # empty output (looks like "no trades", not an error), which reconciliation
    # then reports as every overseas trade never having reached KIS.
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        seen["PDNO"] = request.url.params["PDNO"]
        seen["OVRS_EXCG_CD"] = request.url.params["OVRS_EXCG_CD"]
        return httpx.Response(200, json={"rt_cd": "0", "ctx_area_fk200": "", "ctx_area_nk200": "", "output": []})

    paper = make_broker(tmp_path, handler)
    await paper.fetch_overseas_executions("20260713", "20260713")
    assert seen["PDNO"] == "" and seen["OVRS_EXCG_CD"] == ""

    real = KISBroker(
        app_key="test-app-key-12345678", app_secret="test-secret", account_8="12345678",
        base_url="https://real.test", token_cache_path=tmp_path / "real_token.json",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), min_request_interval=0,
        is_real=True,
    )
    await real.fetch_overseas_executions("20260713", "20260713")
    assert seen["PDNO"] == "%" and seen["OVRS_EXCG_CD"] == "%"


# ---------------------------------------------------------------------------
# EGW00215 (ledger-layer per-second inquiry cap) — 2026-09-01 real-account
# incident. Retried with an escalating backoff, distinct from EGW00201.
# ---------------------------------------------------------------------------

async def test_egw00215_retried_with_escalating_backoff(tmp_path, monkeypatch):
    sleeps: list[float] = []

    async def fake_sleep(s):
        sleeps.append(s)

    monkeypatch.setattr("app.brokers.kis.client.asyncio.sleep", fake_sleep)

    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        attempts["n"] += 1
        if attempts["n"] <= 2:
            return httpx.Response(500, json={"rt_cd": "1", "msg_cd": "EGW00215",
                                             "msg1": "원장에서 허용 가능한 초당 거래건수를 초과하였습니다."})
        return httpx.Response(200, json=BALANCE_BODY)

    broker = make_broker(tmp_path, handler)
    balance = await broker.fetch_balance()
    assert isinstance(balance, BrokerBalance)
    assert attempts["n"] == 3          # 2 rejected + 1 success
    # Escalating bases 1.0, 2.0 — jittered ×uniform(0.7, 1.4) since 2026-09-03
    # (decision id=1179) so a burst of concurrent rejections de-synchronizes.
    assert len(sleeps) == 2
    for base, actual in zip((1.0, 2.0), sleeps):
        assert base * 0.7 <= actual <= base * 1.4


async def test_egw00215_gives_up_after_two_retries_and_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("app.brokers.kis.client.asyncio.sleep", lambda s: _noop())
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        attempts["n"] += 1
        return httpx.Response(200, json={"rt_cd": "1", "msg_cd": "EGW00215", "msg1": "초과"})

    broker = make_broker(tmp_path, handler)
    with pytest.raises(KISBrokerError) as exc_info:
        await broker.fetch_balance()
    assert exc_info.value.msg_cd == "EGW00215"
    assert attempts["n"] == 3  # initial + 2 retries, then gives up


async def test_egw00215_detected_on_http_200_body_too(tmp_path, monkeypatch):
    monkeypatch.setattr("app.brokers.kis.client.asyncio.sleep", lambda s: _noop())
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        attempts["n"] += 1
        if attempts["n"] == 1:
            # HTTP 200 with rt_cd != "0" — must still be retried, not raised.
            return httpx.Response(200, json={"rt_cd": "1", "msg_cd": "EGW00215", "msg1": "초과"})
        return httpx.Response(200, json=BALANCE_BODY)

    broker = make_broker(tmp_path, handler)
    await broker.fetch_balance()
    assert attempts["n"] == 2


async def test_egw00201_retries_twice_with_jittered_backoff(tmp_path, monkeypatch):
    # 2026-09-03 (decision id=1179): EGW00201 was widened from a single fixed
    # 1.0s retry to two attempts (bases 0.6, 1.8) jittered ×uniform(0.7, 1.4),
    # so ~9 concurrent Fundamentals-Analyst calls all rejected in one tick
    # stop waking together (thundering herd).
    sleeps: list[float] = []

    async def fake_sleep(s):
        sleeps.append(s)

    monkeypatch.setattr("app.brokers.kis.client.asyncio.sleep", fake_sleep)
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        attempts["n"] += 1
        if attempts["n"] == 1:
            return httpx.Response(500, text='{"rt_cd":"1","msg_cd":"EGW00201"}')
        return httpx.Response(200, json=BALANCE_BODY)

    broker = make_broker(tmp_path, handler)
    await broker.fetch_balance()
    assert attempts["n"] == 2           # recovered on the first retry
    assert len(sleeps) == 1
    assert 0.6 * 0.7 <= sleeps[0] <= 0.6 * 1.4


async def test_order_submission_is_not_slowed_by_a_held_account_lane(tmp_path):
    # submit_order must NOT take the account lane — a stop-loss sell can't
    # queue behind inquiry traffic.
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        return httpx.Response(200, json={"rt_cd": "0", "msg_cd": "OK", "msg1": "정상",
                                         "output": {"ODNO": "0001", "ORD_TMD": "090000"}})

    broker = make_broker(tmp_path, handler)
    # jam the lane hard
    broker._account_min_interval = 100.0
    async with broker.account_lane():
        import asyncio as _a
        loop = _a.get_running_loop()
        t0 = loop.time()
        await _a.wait_for(broker.submit_order("005930", "buy", 1, price=70000), timeout=2.0)
        assert loop.time() - t0 < 1.0


async def _noop():
    return None


async def test_broker_from_credentials_intervals_track_creds_not_env(monkeypatch):
    # A config/DB mismatch on active mode must not run a real process on
    # paper timing (or vice versa) — the intervals come from creds.is_real,
    # same source as base_url/is_real.
    from app.brokers.kis.broker import broker_from_credentials
    from app.config import settings
    from app.domain.models import KISCredentials

    monkeypatch.setattr(settings, "kis_account_mode", "paper")  # env says paper...

    real_creds = KISCredentials(
        user_id="default", account_mode="real", app_key="k", app_secret="s",
        account_8="12345678", account_prod="01", hts_id=None,
    )
    b = broker_from_credentials(real_creds)  # ...but these creds are real
    assert b._min_request_interval == settings.kis_min_request_interval_real
    assert b._account_min_interval == settings.kis_account_inquiry_min_interval_real

    paper_creds = KISCredentials(
        user_id="default", account_mode="paper", app_key="k", app_secret="s",
        account_8="12345678", account_prod="01", hts_id=None,
    )
    bp = broker_from_credentials(paper_creds)
    assert bp._min_request_interval == settings.kis_min_request_interval_paper
    assert bp._account_min_interval == 0.0


async def test_balance_pagination_takes_the_account_lane_once(tmp_path):
    # 3 pages of a single balance inquiry share ONE lane acquisition — no
    # deadlock. FIX-3 (2026-09-03 id=1187): pages within that one hold ARE
    # now spaced via `_AccountLaneHold.space()` — see the dedicated test
    # below for that. This test just pins "one lane ACQUISITION, not 3": the
    # cost bound below is the one-acquisition-plus-per-page-spacing budget
    # (< 3 * interval), comfortably below what 3 SEPARATE acquisitions of a
    # jammed lane would cost (each paying account_lane()'s own entry wait
    # against a globally-jammed `_account_lane_last`, not just this
    # sub-interval).
    page = {"rt_cd": "0", "msg_cd": "OK", "msg1": "정상", "output1": [],
            "output2": [{"dnca_tot_amt": "1", "tot_evlu_amt": "1",
                         "evlu_pfls_smtl_amt": "0", "pchs_amt_smtl_amt": "0"}],
            "ctx_area_fk100": "fk", "ctx_area_nk100": "nk"}
    seen = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        seen["n"] += 1
        hdr = {"tr_cont": "M"} if seen["n"] < 3 else {"tr_cont": "D"}
        return httpx.Response(200, json=page, headers=hdr)

    broker = make_broker(tmp_path, handler)
    broker._account_min_interval = 0.05
    import asyncio as _a
    loop = _a.get_running_loop()
    t0 = loop.time()
    await _a.wait_for(broker.fetch_balance(), timeout=2.0)
    assert seen["n"] == 3
    assert loop.time() - t0 < 0.15


async def test_balance_pagination_spaces_pages_within_one_lane_hold(tmp_path):
    # FIX-3 (2026-09-03 id=1187 audit): account_lane() itself only ever
    # spaced successive HOLDS of the lane against each other — nothing spaced
    # the individual wire requests fired WITHIN one hold, so `_paginate`
    # could (and, on id=1187, did) fire up to MAX_PAGINATION_PAGES ledger TRs
    # back-to-back inside a single hold, fast enough to trip KIS's own
    # per-second ledger limit (EGW00215) despite the lane being "held".
    page = {"rt_cd": "0", "msg_cd": "OK", "msg1": "정상", "output1": [],
            "output2": [{"dnca_tot_amt": "1", "tot_evlu_amt": "1",
                         "evlu_pfls_smtl_amt": "0", "pchs_amt_smtl_amt": "0"}],
            "ctx_area_fk100": "fk", "ctx_area_nk100": "nk"}
    seen = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        seen["n"] += 1
        hdr = {"tr_cont": "M"} if seen["n"] < 3 else {"tr_cont": "D"}
        return httpx.Response(200, json=page, headers=hdr)

    broker = make_broker(tmp_path, handler)
    broker._account_min_interval = 0.05
    import asyncio as _a
    loop = _a.get_running_loop()
    # A very recent prior stamp makes even the FIRST page's `space()` call
    # (page_no=1, the second page) wait — otherwise a fresh broker's unset
    # `_account_lane_last` would let just that one call through for free,
    # making the timing assertion below flaky.
    broker._account_lane_last = loop.time()
    t0 = loop.time()
    await _a.wait_for(broker.fetch_balance(), timeout=2.0)
    assert seen["n"] == 3
    # 3 pages -> 2 inter-page `space()` calls, each >= account_min_interval.
    assert loop.time() - t0 >= 0.08
