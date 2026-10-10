"""Reddit sentiment provider — httpx mocked via monkeypatch, no real network (B-stage)."""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest

from app.clock import FakeClock, LiveClock, set_clock
from app.dataflows import reddit_provider

_ATOM_NS = "http://www.w3.org/2005/Atom"


@pytest.fixture(autouse=True)
def restore_clock():
    yield
    set_clock(LiveClock())


def _feed_xml(titles: list[str]) -> str:
    entries = "".join(f"<entry><title>{t}</title></entry>" for t in titles)
    return f'<feed xmlns="{_ATOM_NS}">{entries}</feed>'


def _dated_feed_xml(entries: list[tuple[str, str | None]]) -> str:
    """entries: list of (title, iso-date-or-None) — None omits <updated> entirely."""
    parts = []
    for title, when in entries:
        updated = f"<updated>{when}</updated>" if when else ""
        parts.append(f"<entry><title>{title}</title>{updated}</entry>")
    return f'<feed xmlns="{_ATOM_NS}">{"".join(parts)}</feed>'


class _FakeResponse:
    def __init__(self, text: str, status: int = 200):
        self.text = text
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("GET", "https://www.reddit.com")
            raise httpx.HTTPStatusError("error", request=request, response=httpx.Response(self.status_code, request=request))


async def test_aggregates_posts_across_subreddits(monkeypatch):
    async def fake_get(self, url, headers=None, **kwargs):
        if "/r/stocks/" in url:
            return _FakeResponse(_feed_xml(["AAPL beats earnings"]))
        return _FakeResponse(_feed_xml(["AAPL discussion thread"]))

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    result = await reddit_provider._reddit_sentiment("AAPL")
    assert "Reddit discussion:" in result
    assert "r/stocks: AAPL beats earnings" in result
    assert "r/investing: AAPL discussion thread" in result


async def test_one_subreddit_failing_does_not_block_the_other(monkeypatch):
    async def fake_get(self, url, headers=None, **kwargs):
        if "/r/stocks/" in url:
            return _FakeResponse("", status=500)
        return _FakeResponse(_feed_xml(["still works"]))

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    result = await reddit_provider._reddit_sentiment("AAPL")
    assert "r/investing: still works" in result


async def test_raises_when_no_posts_found_anywhere(monkeypatch):
    async def fake_get(self, url, headers=None, **kwargs):
        return _FakeResponse(_feed_xml([]))

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    with pytest.raises(RuntimeError, match="no reddit posts found"):
        await reddit_provider._reddit_sentiment("AAPL")


async def test_dated_post_renders_date_prefix(monkeypatch):
    set_clock(FakeClock(datetime(2026, 9, 11, 12, 0, tzinfo=timezone.utc)))

    async def fake_get(self, url, headers=None, **kwargs):
        if "/r/stocks/" in url:
            return _FakeResponse(_dated_feed_xml([("AAPL beats earnings", "2026-09-10T12:00:00+00:00")]))
        return _FakeResponse(_feed_xml([]))

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    result = await reddit_provider._reddit_sentiment("AAPL")
    assert "[2026-09-10] r/stocks: AAPL beats earnings" in result


async def test_post_older_than_max_age_is_dropped(monkeypatch):
    set_clock(FakeClock(datetime(2026, 9, 11, 12, 0, tzinfo=timezone.utc)))

    async def fake_get(self, url, headers=None, **kwargs):
        if "/r/stocks/" in url:
            return _FakeResponse(_dated_feed_xml([("ancient AAPL thread", "2024-01-01T00:00:00+00:00")]))
        return _FakeResponse(_feed_xml([]))

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    with pytest.raises(RuntimeError, match="no reddit posts found"):
        await reddit_provider._reddit_sentiment("AAPL")


async def test_cross_subreddit_ordering_is_newest_first(monkeypatch):
    set_clock(FakeClock(datetime(2026, 9, 11, 12, 0, tzinfo=timezone.utc)))

    async def fake_get(self, url, headers=None, **kwargs):
        if "/r/stocks/" in url:
            return _FakeResponse(_dated_feed_xml([("older stocks post", "2026-08-01T00:00:00+00:00")]))
        return _FakeResponse(_dated_feed_xml([("newer investing post", "2026-09-05T00:00:00+00:00")]))

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    result = await reddit_provider._reddit_sentiment("AAPL")
    lines = [ln for ln in result.splitlines() if ln.startswith("[")]
    assert "newer investing post" in lines[0]
    assert "older stocks post" in lines[1]


async def test_staleness_banner_fires_when_newest_dated_post_is_old(monkeypatch):
    set_clock(FakeClock(datetime(2026, 9, 11, 12, 0, tzinfo=timezone.utc)))

    async def fake_get(self, url, headers=None, **kwargs):
        if "/r/stocks/" in url:
            return _FakeResponse(_dated_feed_xml([("stale AAPL post", "2026-06-01T00:00:00+00:00")]))
        return _FakeResponse(_feed_xml([]))

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    result = await reddit_provider._reddit_sentiment("AAPL")
    assert "days old" in result.splitlines()[0]
    assert "do not read this as current sentiment" in result


async def test_no_staleness_banner_when_all_posts_undated(monkeypatch):
    set_clock(FakeClock(datetime(2026, 9, 11, 12, 0, tzinfo=timezone.utc)))

    async def fake_get(self, url, headers=None, **kwargs):
        if "/r/stocks/" in url:
            return _FakeResponse(_feed_xml(["AAPL beats earnings"]))
        return _FakeResponse(_feed_xml([]))

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    result = await reddit_provider._reddit_sentiment("AAPL")
    assert "days old" not in result
    assert result.startswith("Reddit discussion:")
    assert "[date unknown] r/stocks: AAPL beats earnings" in result
