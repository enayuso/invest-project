"""Korean money-unit formatting (spec S:3-1).

Every KRW amount this project hands back to the model is a
`{"display": ..., "krw": ...}` pair -- never a bare number, and never an
English scale word (million/billion). This is a direct response to a
real, recurring failure in the sibling trading system: 억원<->billion
confusion propagated all the way to the Trader node over roughly a year
of incidents (see handoff/knowledge's reference to
`llm-numeric-hallucination-root-causes`). Structurally preventing that
here means: the tool never emits the string "billion" or "million"
anywhere, for any value, ever.
"""

from __future__ import annotations

_EOK = 100_000_000  # 1억 = 1e8
_JO = 1_000_000_000_000  # 1조 = 1e12


def format_krw(value: float | int | None) -> dict[str, object] | None:
    """Returns {"display": "333.6조원", "krw": 333605938000000} or None.

    `display` is the primary representation a model should quote in
    prose. `krw` is the exact raw integer, kept only for verification --
    tool descriptions tell the model not to narrate this raw value
    directly (that's exactly how eok/billion confusion starts: a model
    sees a 15-digit integer and free-associates a scale word for it).

    `display` always renders in 억원 or 조원, at every magnitude -- a bare
    "...원" reading (no unit conversion) forces the model to convert units
    in its head to compare against a neighbouring 억/조 value, which is
    exactly the error class this module exists to prevent. Sub-억 amounts
    use 2 decimals (there is no 만원 tier -- a third unit makes comparison
    harder, not easier) and round to "0억원" with no sign rather than a
    misleading "-0.00억원".

    The unit is chosen *after* rounding, not before: rounding a value like
    9,999.96억 to 1 decimal can tip it to "10000.0억원", which is not a
    valid display (10000억 == 1조) -- so a value that rounds up to the next
    tier is re-rendered in that tier's unit instead. The same check applies
    one tier down, so a sub-억 value that rounds up to a full 1억 (e.g.
    99,996,000) renders "1.0억원" (that tier's own precision), not a
    2-decimal "1.00억원" that looks like it is still short of 1억.
    """
    if value is None:
        return None
    sign = "-" if value < 0 else ""
    av = abs(value)

    def _jo_display() -> str:
        return f"{sign}{av / _JO:.1f}조원"

    if av >= _JO:
        display = _jo_display()
    elif av >= _EOK:
        eok = round(av / _EOK, 1)
        display = _jo_display() if eok >= 10000.0 else f"{sign}{eok:.1f}억원"
    else:
        eok_2dp = round(av / _EOK, 2)
        if eok_2dp == 0:
            display = "0억원"
        elif eok_2dp >= 1.0:
            eok_1dp = round(av / _EOK, 1)
            display = _jo_display() if eok_1dp >= 10000.0 else f"{sign}{eok_1dp:.1f}억원"
        else:
            display = f"{sign}{eok_2dp:.2f}억원"
    return {"display": display, "krw": int(value)}


def format_pct(value: float | None) -> float | None:
    """Round a raw ratio to a percentage with 2 decimals.

    Always multiplies by 100 -- there is no "already a percent" mode.
    An `already_percent` escape hatch used to exist here; it was removed
    because every one of its 3 call sites was a 100x error: the call site
    passed in a raw ratio (e.g. 0.1951) but set `already_percent=True`, so
    this function's ×100 was skipped -- producing a percentage 100x too
    small (a +19.51% move reported as 0.2). The flag didn't paper over an
    upstream double-multiplication; it caused the under-multiplication
    itself. A flag whose entire observed usage was incorrect is not a
    feature, it's a trap. If you have a value that is genuinely already a
    percentage, do not route it through this function at all.

    Kept separate from format_krw because percentages (RSI, ROE, 부채비율,
    change_pct) are not money and must never go through the 억/조 path.
    """
    if value is None:
        return None
    return round(value * 100, 2)
