"""Reddit sentiment source — keyless public RSS search feed.

Pre-fetched and injected as literal text, never exposed to the Sentiment
Analyst as a live tool call: an LLM told to "analyze social sentiment" with
no real tool for it will fabricate posts under pressure, so the fetch always
happens before the LLM is invoked.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

import httpx

from app.clock import get_clock
from app.dataflows.router import router

_RSS = "https://www.reddit.com/r/{sub}/search.rss?{qs}"
_UA = "invest-project/0.1 (standalone research tool)"
_ATOM_NS = "{http://www.w3.org/2005/Atom}"
_SUBREDDITS = ["stocks", "investing"]

# 2026-09-11: a post older than this is background noise, not something a
# Sentiment Analyst should weigh as current discussion — 180 days (~6 months)
# is generous enough to keep a slow-moving thread while still excluding
# multi-year-old posts that `sort=new`+keyword search can surface.
_MAX_POST_AGE_DAYS = 180
# 2026-09-11: if even the NEWEST dated post is older than this, the whole
# feed is stale enough to warrant an explicit warning rather than trusting
# the reader to notice — 30 days is roughly "older than a month", well past
# what anyone would call "current" sentiment.
_STALE_AFTER_DAYS = 30


def _parse_entry_date(entry: ET.Element) -> datetime | None:
    for tag in ("updated", "published"):
        text = entry.findtext(f"{_ATOM_NS}{tag}", default="")
        if not text:
            continue
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            continue
        # Always return an AWARE datetime. Reddit's Atom feed emits RFC3339
        # with an explicit offset, so this normally changes nothing — but a
        # naive value reaching the newest-first sort would be read by
        # `.timestamp()` as the SERVER's local time, and this server runs KST,
        # so a naive UTC stamp would sort 9 hours off and could reorder posts.
        # (`.timestamp()` on a naive pre-1970 value also raises OSError on
        # Windows, which would escape `_reddit_sentiment` — the per-subreddit
        # try/except only wraps the fetch — and silently drop the whole Reddit
        # category through the router's broad catch.) Assuming UTC is the
        # right default: this feed is UTC, and the alternative (local time)
        # is the one that is definitely wrong.
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
    return None


async def _fetch_subreddit(
    client: httpx.AsyncClient, sub: str, ticker: str
) -> list[tuple[datetime | None, str]]:
    qs = urlencode({"q": ticker, "restrict_sr": "on", "sort": "new", "limit": 10})
    resp = await client.get(_RSS.format(sub=sub, qs=qs), headers={"User-Agent": _UA})
    resp.raise_for_status()
    root = ET.fromstring(resp.text)
    posts = []
    for entry in root.findall(f"{_ATOM_NS}entry"):
        title = entry.findtext(f"{_ATOM_NS}title", default="")
        if title:
            posts.append((_parse_entry_date(entry), f"r/{sub}: {title}"))
    return posts


async def _reddit_sentiment(ticker: str) -> str:
    async with httpx.AsyncClient(timeout=10.0) as client:
        posts: list[tuple[datetime | None, str]] = []
        for sub in _SUBREDDITS:
            try:
                posts.extend(await _fetch_subreddit(client, sub, ticker))
            except (httpx.HTTPError, ET.ParseError):
                continue

        now = get_clock().today()
        max_age = timedelta(days=_MAX_POST_AGE_DAYS)
        kept: list[tuple[datetime | None, str]] = []
        for when, body in posts:
            if when is not None and (now - when.date()) > max_age:
                continue  # drop stale dated posts; undated posts are never dropped
            kept.append((when, body))
        if not kept:
            raise RuntimeError(f"no reddit posts found for {ticker!r}")

        # Newest-first across subreddits, undated posts sorted last; stable
        # so equal dates (or all-undated feeds) keep their fetch order.
        kept.sort(key=lambda p: (p[0] is None, -p[0].timestamp() if p[0] else 0))

        lines = [
            f"[{when.date()}] {body}" if when else f"[date unknown] {body}"
            for when, body in kept[:15]
        ]

        dated = [when for when, _ in kept if when is not None]
        text = "Reddit discussion:\n" + "\n".join(lines)
        if dated:
            newest = max(dated).date()
            if (now - newest) > timedelta(days=_STALE_AFTER_DAYS):
                days_old = (now - newest).days
                text = (
                    f"[Warning: most recent discussion found is {days_old} days old — "
                    "do not read this as current sentiment]\n" + text
                )
        return text


router.register("reddit_sentiment", [_reddit_sentiment], optional=True)


async def fetch_reddit(ticker: str) -> str:
    return await router.call("reddit_sentiment", ticker=ticker)
