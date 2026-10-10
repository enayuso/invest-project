"""app.live.overseas_master — tab-separated NASDAQ row parsing and the
loaded/get cache contract. No network: _download is monkeypatched
everywhere."""

from __future__ import annotations

import asyncio

import pytest

from app.live import overseas_master


def _row(symbol: str, korean_name: str) -> str:
    # 24 tab-separated columns, symbol at index 4, Korean name at index 6 —
    # same layout as the real nasmst.cod file.
    fields = ["col"] * 24
    fields[overseas_master._SYMBOL_COLUMN] = symbol
    fields[overseas_master._KOREAN_NAME_COLUMN] = korean_name
    return "\t".join(fields)


@pytest.fixture(autouse=True)
def _reset_cache():
    overseas_master._cache.clear()
    overseas_master._loaded = False
    overseas_master._last_failed_at = None
    yield
    overseas_master._cache.clear()
    overseas_master._loaded = False
    overseas_master._last_failed_at = None


def test_parse_extracts_symbol_and_korean_name():
    text = _row("AAPL", "애플") + "\n" + _row("TSLA", "테슬라")
    result = overseas_master._parse(text)
    assert result == {"AAPL": "애플", "TSLA": "테슬라"}


def test_parse_skips_rows_with_wrong_column_count():
    assert overseas_master._parse("too\tfew\tcolumns\n") == {}


async def test_ensure_loaded_populates_cache(monkeypatch):
    async def fake_download():
        return {"AAPL": "애플"}

    monkeypatch.setattr(overseas_master, "_download", fake_download)
    await overseas_master.ensure_loaded()
    assert overseas_master.get("AAPL") == "애플"


async def test_ensure_loaded_is_a_no_op_on_second_call(monkeypatch):
    calls = 0

    async def fake_download():
        nonlocal calls
        calls += 1
        return {}

    monkeypatch.setattr(overseas_master, "_download", fake_download)
    await overseas_master.ensure_loaded()
    await overseas_master.ensure_loaded()
    assert calls == 1


async def test_ensure_loaded_survives_download_failure(monkeypatch):
    async def failing_download():
        raise RuntimeError("network down")

    monkeypatch.setattr(overseas_master, "_download", failing_download)
    await overseas_master.ensure_loaded()  # must not raise
    assert overseas_master.get("AAPL") is None
    assert overseas_master._loaded is False  # a failed attempt is not a successful load


async def test_ensure_loaded_does_not_retry_within_the_backoff_window(monkeypatch):
    async def failing_download():
        raise RuntimeError("network down")

    monkeypatch.setattr(overseas_master, "_download", failing_download)
    await overseas_master.ensure_loaded()

    calls = 0

    async def fake_download():
        nonlocal calls
        calls += 1
        return {}

    monkeypatch.setattr(overseas_master, "_download", fake_download)
    await overseas_master.ensure_loaded()  # still inside the backoff window
    assert calls == 0


async def test_ensure_loaded_retries_after_the_backoff_window_elapses(monkeypatch):
    async def failing_download():
        raise RuntimeError("network down")

    monkeypatch.setattr(overseas_master, "_download", failing_download)
    await overseas_master.ensure_loaded()
    overseas_master._last_failed_at -= overseas_master._RETRY_BACKOFF_SECONDS + 1

    async def fake_download():
        return {"AAPL": "애플"}

    monkeypatch.setattr(overseas_master, "_download", fake_download)
    await overseas_master.ensure_loaded()
    assert overseas_master.get("AAPL") == "애플"
    assert overseas_master._loaded is True


async def test_ensure_loaded_does_not_mark_loaded_until_the_download_succeeds(monkeypatch):
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_download():
        started.set()
        await release.wait()
        return {"AAPL": "애플"}

    monkeypatch.setattr(overseas_master, "_download", slow_download)
    task = asyncio.create_task(overseas_master.ensure_loaded())
    await started.wait()
    assert overseas_master._loaded is False
    release.set()
    await task
    assert overseas_master._loaded is True
    assert overseas_master.get("AAPL") == "애플"


def test_get_returns_none_when_not_loaded():
    assert overseas_master.get("AAPL") is None


def test_search_matches_korean_name():
    overseas_master._cache["AAPL"] = "애플"
    overseas_master._cache["MSFT"] = "마이크로소프트"
    assert overseas_master.search("애플") == [("AAPL", "애플")]


def test_search_matches_symbol_case_insensitive():
    overseas_master._cache["AAPL"] = "애플"
    assert overseas_master.search("aapl") == [("AAPL", "애플")]


def test_search_empty_query_returns_nothing():
    overseas_master._cache["AAPL"] = "애플"
    assert overseas_master.search("") == []
