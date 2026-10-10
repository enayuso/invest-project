"""KISHttpClient's per-msg_cd rate-limit retry (app/brokers/kis/client.py) —
httpx.MockTransport, no network. 2026-09-03 decision id=1179: EGW00201 was
widened to two retries and the sleep jittered so a burst of concurrent
rejections doesn't wake together."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import pytest

from app.brokers.kis import KISBrokerError
from app.brokers.kis.client import KISHttpClient

TOKEN_BODY = {
    "access_token": "test-token-abc",
    "access_token_token_expired": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"),
    "token_type": "Bearer",
    "expires_in": 86400,
}

GOOD_BODY = {"rt_cd": "0", "msg_cd": "0000", "msg1": "OK", "output": {"x": "1"}}
EGW00201_BODY = {"msg_cd": "EGW00201", "msg1": "초당 거래건수를 초과하였습니다."}


def _client(tmp_path: Path, handler) -> KISHttpClient:
    return KISHttpClient(
        app_key="test-app-key-12345678",
        app_secret="test-secret",
        base_url="https://paper.test",
        token_cache_path=tmp_path / "token.json",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        min_request_interval=0,
    )


async def _noop_sleep(*_a, **_k):
    return None


async def test_egw00201_retries_twice_then_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", _noop_sleep)
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        calls += 1
        return httpx.Response(500, json=EGW00201_BODY)

    client = _client(tmp_path, handler)
    resp = await client._request("GET", "/x", "TR")
    assert calls == 3  # initial + 2 retries
    with pytest.raises(KISBrokerError):
        client._checked_body(resp, "x")


async def test_egw00201_recovers_on_second_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", _noop_sleep)
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        calls += 1
        if calls < 3:
            return httpx.Response(500, json=EGW00201_BODY)
        return httpx.Response(200, json=GOOD_BODY)

    client = _client(tmp_path, handler)
    resp = await client._request("GET", "/x", "TR")
    assert calls == 3
    assert client._checked_body(resp, "x")["output"] == {"x": "1"}


async def test_rate_limit_sleep_is_jittered(tmp_path, monkeypatch):
    slept: list[float] = []

    async def fake_sleep(seconds=0):
        slept.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        return httpx.Response(500, json=EGW00201_BODY)

    client = _client(tmp_path, handler)
    await client._request("GET", "/x", "TR")

    # Two retries: base sleeps are 0.6 and 1.8, each scaled by uniform(0.7, 1.4).
    assert len(slept) == 2
    for base, actual in zip((0.6, 1.8), slept):
        assert base * 0.7 <= actual <= base * 1.4
        assert actual != base  # jittered, not the bare base value


# --- 2026-10-02 진단 로그: 거부 시 직전 요청/호출자 컨텍스트 1줄 ---
EGW00215_BODY = {"msg_cd": "EGW00215", "msg1": "원장 초당 건수 초과"}


def _ctx_records(caplog):
    return [r.getMessage() for r in caplog.records if "rate-limit context" in r.getMessage()]


async def test_egw00215_context_logged_once_with_caller(tmp_path, monkeypatch, caplog):
    import logging

    from app.brokers.kis import client as kc

    monkeypatch.setattr(asyncio, "sleep", _noop_sleep)
    kc._LEDGER_RECENT.clear()
    kc._GENERAL_RECENT.clear()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        return httpx.Response(500, json=EGW00215_BODY)

    client = _client(tmp_path, handler)
    with caplog.at_level(logging.WARNING, logger=kc.logger.name):
        await client._request("GET", "/uapi/domestic-stock/v1/trading/inquire-balance", "TR", lane="account")
    msgs = _ctx_records(caplog)
    assert len(msgs) == 1  # 3 attempts, one context line
    assert "EGW00215 on /uapi/domestic-stock/v1/trading/inquire-balance" in msgs[0]
    assert f"inquire-balance <- {__name__}:test_egw00215_context_logged_once_with_caller" in msgs[0]
    assert "last 1 ledger TRs within 5s" in msgs[0]


async def test_egw00201_general_context_logged(tmp_path, monkeypatch, caplog):
    import logging

    from app.brokers.kis import client as kc

    monkeypatch.setattr(asyncio, "sleep", _noop_sleep)
    kc._GENERAL_RECENT.clear()
    state = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/tokenP":
            return httpx.Response(200, json=TOKEN_BODY)
        state["n"] += 1
        return httpx.Response(500, json=EGW00201_BODY) if state["n"] == 1 else httpx.Response(200, json=GOOD_BODY)

    client = _client(tmp_path, handler)
    with caplog.at_level(logging.WARNING, logger=kc.logger.name):
        await client._request("GET", "/quotes", "TR")
    msgs = _ctx_records(caplog)
    assert len(msgs) == 1
    assert "EGW00201 on /quotes" in msgs[0] and "within 2s" in msgs[0]
    assert f"quotes <- {__name__}:test_egw00201_general_context_logged" in msgs[0]


async def test_context_excludes_entries_older_than_window(monkeypatch):
    import time

    from app.brokers.kis import client as kc

    kc._LEDGER_RECENT.clear()
    now = time.monotonic()
    kc._LEDGER_RECENT.append((now - 9.0, "old", "m:old"))
    kc._LEDGER_RECENT.append((now - 1.0, "new", "m:new"))
    out = kc._format_recent(kc._LEDGER_RECENT, 5.0)
    assert "new <- m:new" in out and "old" not in out
