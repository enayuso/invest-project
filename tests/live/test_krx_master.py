"""app.live.krx_master — fixed-width row parsing and the loaded/get cache
contract. No network: _download is monkeypatched everywhere."""

from __future__ import annotations

import asyncio

import pytest

from app.live import krx_master


def _row(code: str, std: str, name: str) -> str:
    # 단축코드(9) + 표준코드(12) + 한글명(padded) + 228-char flag block,
    # same layout as the real KOSPI/KOSDAQ master files.
    head = code.ljust(9) + std.ljust(12) + name.ljust(28)
    return head + "0" * krx_master._FLAG_SUFFIX_LEN


@pytest.fixture(autouse=True)
def _reset_cache():
    krx_master._cache.clear()
    krx_master._loaded = False
    krx_master._last_failed_at = None
    yield
    krx_master._cache.clear()
    krx_master._loaded = False
    krx_master._last_failed_at = None


def test_parse_extracts_code_and_name():
    text = _row("028050", "KR7028050003", "삼성E&A") + "\n" + _row("005380", "KR7005380001", "현대차")
    result = krx_master._parse(text)
    assert result == {"028050": "삼성E&A", "005380": "현대차"}


def test_parse_skips_short_malformed_rows():
    assert krx_master._parse("short\n") == {}


async def test_ensure_loaded_populates_cache_from_both_markets(monkeypatch):
    async def fake_download(market):
        return {"028050": "삼성E&A"} if market == "kospi" else {"247540": "에코프로비엠"}

    monkeypatch.setattr(krx_master, "_download", fake_download)
    await krx_master.ensure_loaded()
    assert krx_master.get("028050.KS") == "삼성E&A"
    assert krx_master.get("247540.KS") == "에코프로비엠"


async def test_ensure_loaded_is_a_no_op_on_second_call(monkeypatch):
    calls = 0

    async def fake_download(market):
        nonlocal calls
        calls += 1
        return {}

    monkeypatch.setattr(krx_master, "_download", fake_download)
    await krx_master.ensure_loaded()
    await krx_master.ensure_loaded()
    assert calls == 2  # one kospi + one kosdaq call, not four


async def test_ensure_loaded_survives_download_failure(monkeypatch):
    async def failing_download(market):
        raise RuntimeError("network down")

    monkeypatch.setattr(krx_master, "_download", failing_download)
    await krx_master.ensure_loaded()  # must not raise
    assert krx_master.get("028050.KS") is None
    assert krx_master._loaded is False  # a failed attempt is not a successful load


async def test_ensure_loaded_does_not_retry_within_the_backoff_window(monkeypatch):
    # A down mirror shouldn't cost every subsequent ticker lookup a fresh
    # 15s-timeout download attempt — but see the next test, it must not
    # latch the failure forever either.
    async def failing_download(market):
        raise RuntimeError("network down")

    monkeypatch.setattr(krx_master, "_download", failing_download)
    await krx_master.ensure_loaded()

    calls = 0

    async def fake_download(market):
        nonlocal calls
        calls += 1
        return {}

    monkeypatch.setattr(krx_master, "_download", fake_download)
    await krx_master.ensure_loaded()  # still inside the backoff window
    assert calls == 0


async def test_ensure_loaded_retries_after_the_backoff_window_elapses(monkeypatch):
    async def failing_download(market):
        raise RuntimeError("network down")

    monkeypatch.setattr(krx_master, "_download", failing_download)
    await krx_master.ensure_loaded()
    # Simulate the backoff window having already elapsed, rather than a real
    # sleep — same technique as app.memory.db's identical backoff pattern.
    krx_master._last_failed_at -= krx_master._RETRY_BACKOFF_SECONDS + 1

    async def fake_download(market):
        return {"028050": "삼성E&A"} if market == "kospi" else {}

    monkeypatch.setattr(krx_master, "_download", fake_download)
    await krx_master.ensure_loaded()
    assert krx_master.get("028050.KS") == "삼성E&A"
    assert krx_master._loaded is True


async def test_ensure_loaded_does_not_mark_loaded_until_the_download_succeeds(monkeypatch):
    # The concurrency bug this guards against: `_loaded` used to flip to
    # True BEFORE the download started, so a concurrent second caller could
    # observe "loaded" with an empty cache while the first download was
    # still in flight.
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_download(market):
        started.set()
        await release.wait()
        return {"028050": "삼성E&A"} if market == "kospi" else {}

    monkeypatch.setattr(krx_master, "_download", slow_download)
    task = asyncio.create_task(krx_master.ensure_loaded())
    await started.wait()
    assert krx_master._loaded is False  # still in flight — must not look loaded yet
    release.set()
    await task
    assert krx_master._loaded is True
    assert krx_master.get("028050.KS") == "삼성E&A"


def test_get_returns_none_for_non_krx_ticker():
    krx_master._cache["AAPL"] = "Apple"
    assert krx_master.get("AAPL") is None


def test_get_returns_none_when_not_loaded():
    assert krx_master.get("028050.KS") is None


def test_search_matches_name_substring_case_insensitive():
    krx_master._cache["005930"] = "삼성전자"
    krx_master._cache["005935"] = "삼성전자우"
    krx_master._cache["000660"] = "SK하이닉스"
    assert krx_master.search("삼성") == [("005930", "삼성전자"), ("005935", "삼성전자우")]
    assert krx_master.search("sk") == [("000660", "SK하이닉스")]


def test_search_matches_bare_code():
    krx_master._cache["005930"] = "삼성전자"
    assert krx_master.search("0059") == [("005930", "삼성전자")]


def test_search_respects_limit():
    for i in range(5):
        krx_master._cache[f"00{i}"] = f"테스트{i}"
    assert len(krx_master.search("테스트", limit=2)) == 2


def test_search_empty_query_returns_nothing():
    krx_master._cache["005930"] = "삼성전자"
    assert krx_master.search("") == []


def test_search_returns_nothing_when_not_loaded():
    assert krx_master.search("삼성") == []
