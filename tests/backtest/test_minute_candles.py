"""app.backtest.minute_candles.fetch_minute_day_from_kis — the 2026-09-11
fix for `_num` silently turning a blank/malformed KIS numeric field into
0.0, which then got cached PERMANENTLY (minute_candles table) and corrupted
every future reader of that day (a fabricated fill price, a division by
zero, a fabricated -100% bar)."""

from __future__ import annotations

import logging
from datetime import date

import pytest

from app.backtest import minute_candles


def _bar(time: str, o="100", h="101", l="99", c="100.5", v="1000") -> dict:
    return {
        "stck_cntg_hour": time,
        "stck_oprc": o, "stck_hgpr": h, "stck_lwpr": l, "stck_prpr": c, "cntg_vol": v,
    }


class _FakeBroker:
    """Returns one page of `output2` per call, in order; empty after that."""

    def __init__(self, pages: list[list[dict]]):
        self._pages = list(pages)
        self.calls = 0

    async def _request(self, *a, **kw):
        return object()  # opaque; _checked_body below ignores it

    def _checked_body(self, resp, what):
        self.calls += 1
        page = self._pages.pop(0) if self._pages else []
        return {"output2": page}


async def test_blank_price_field_is_dropped_not_cached_as_zero(caplog):
    rows = [_bar("090200"), _bar("090100", o=""), _bar("090000")]  # newest-first, matches real KIS paging
    broker = _FakeBroker([rows])
    with caplog.at_level(logging.WARNING):
        result = await minute_candles.fetch_minute_day_from_kis(broker, "005930", date(2026, 1, 2))
    times = {c.time for c in result}
    assert times == {"090000", "090200"}
    assert "090100" not in times
    assert any("dropping malformed bar" in r.message for r in caplog.records)
    assert any("090100" in r.message for r in caplog.records)


async def test_zero_price_field_is_dropped_not_cached_as_zero(caplog):
    rows = [_bar("090100"), _bar("090000", c="0")]
    broker = _FakeBroker([rows])
    with caplog.at_level(logging.WARNING):
        result = await minute_candles.fetch_minute_day_from_kis(broker, "005930", date(2026, 1, 2))
    assert {c.time for c in result} == {"090100"}


async def test_inverted_bar_is_dropped():
    # high < low is structurally impossible -- must never be cached even
    # though every individual field parses as a plausible-looking float.
    rows = [_bar("090100"), _bar("090000", h="90", l="99")]
    broker = _FakeBroker([rows])
    result = await minute_candles.fetch_minute_day_from_kis(broker, "005930", date(2026, 1, 2))
    assert {c.time for c in result} == {"090100"}


async def test_valid_zero_volume_is_kept_not_dropped():
    # Volume 0 is a real, legitimate value (a minute with no trades) --
    # unlike a price field, it must NOT be treated as malformed.
    rows = [_bar("090000", v="0")]
    broker = _FakeBroker([rows])
    result = await minute_candles.fetch_minute_day_from_kis(broker, "005930", date(2026, 1, 2))
    assert len(result) == 1
    assert result[0].volume == 0


async def test_price_helper_rejects_non_positive_and_non_finite():
    assert minute_candles._price("100") == 100.0
    assert minute_candles._price("0") is None
    assert minute_candles._price("-5") is None
    assert minute_candles._price("") is None
    assert minute_candles._price(None) is None
    assert minute_candles._price("nan") is None
    assert minute_candles._price("inf") is None


async def test_volume_helper_keeps_zero_and_floors_negative():
    assert minute_candles._volume("0") == 0
    assert minute_candles._volume("1000") == 1000
    assert minute_candles._volume("") == 0
    assert minute_candles._volume("-5") == 0
