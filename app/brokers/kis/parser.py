# app/brokers/kis/parser.py
# Shared response-parsing helpers for broker.py and kis_ws.py.
# broker.py / kis_ws.py 공용 응답 파싱 유틸

from app.domain.models import OrderResult, RankedStock


def _num(value: str | None) -> float:
    """KIS numeric fields arrive as strings (or missing); coerce safely.
    KIS 숫자 필드는 문자열로 오거나 누락될 수 있어 안전하게 변환"""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _kis_datetime_to_iso(date_str: str, time_str: str) -> str:
    """KIS news TRs (domestic FID_INPUT_DATE_1/HOUR_1, overseas DATA_DT/TM)
    return bare "YYYYMMDD"/"HHMMSS" strings, no separators, no timezone —
    reformatted here so the frontend's `new Date(published_at)` (used for
    formatTimeAgo) can actually parse it instead of getting `Invalid Date`.
    Falls back to "" (frontend already treats a falsy published_at as
    "time unknown") when either half is missing or malformed."""
    if len(date_str) != 8 or len(time_str) < 6:
        return ""
    try:
        return (
            f"{date_str[0:4]}-{date_str[4:6]}-{date_str[6:8]}T"
            f"{time_str[0:2]}:{time_str[2:4]}:{time_str[4:6]}"
        )
    except IndexError:
        return ""


def _as_rows(value: dict | list | None) -> list[dict]:
    """Normalizes a KIS response envelope that comes back as a bare dict
    (exactly one matching row) instead of a list of rows — confirmed on the
    domestic/overseas news TRs when exactly one article matches; iterating
    an un-normalized dict yields its keys (strings), not rows, and crashes
    on the first `.get(...)` call. Missing/falsy `value` -> []."""
    if isinstance(value, list):
        return value
    return [value] if value else []


def _ranked_rows(
    rows: list[dict],
    count: int,
    *,
    code_keys: tuple[str, ...],
    name_keys: tuple[str, ...] = ("name", "ename"),
    rank_key: str = "rank",
    price_key: str = "last",
    change_pct_key: str = "rate",
    volume_key: str = "tvol",
) -> list[RankedStock]:
    """Shared row-builder behind every *_rank fetcher (domestic + overseas) —
    same six-field RankedStock construction, skip-if-no-code guard, and
    [:count] truncation, previously duplicated 7 times (3 domestic + 4
    overseas ranking methods) with only the KIS field names differing per TR.
    `code_keys`/`name_keys` are tried in order, first non-empty wins (the
    domestic fluctuation-rank TR uses a different code field than its volume/
    market-cap siblings; the overseas TRs fall back from `name` to `ename`)."""
    out: list[RankedStock] = []
    for row in rows[:count]:
        code = next((row.get(k) for k in code_keys if row.get(k)), "")
        if not code:
            continue
        name = next((row.get(k) for k in name_keys if row.get(k)), "")
        out.append(RankedStock(
            rank=int(_num(row.get(rank_key))),
            code=code,
            name=name.strip(),
            price=_num(row.get(price_key)),
            change_pct=_num(row.get(change_pct_key)),
            volume=int(_num(row.get(volume_key))),
        ))
    return out


def _order_result(parsed: dict) -> OrderResult:
    """Builds an OrderResult from a submit/cancel/revise response body — the
    same three fields (ODNO/ORD_TMD/msg1) plus org_no (present for domestic
    orders, absent — defaults to "" — for overseas ones), assembled
    identically across submit_order/cancel_order/revise_order (domestic) and
    submit_overseas_order/cancel_overseas_order.
    주문 제출/취소/정정 5곳에서 동일하게 조립되던 OrderResult 생성 로직 통합"""
    # KIS sometimes returns an explicit "output": null / "msg1": null (not
    # just a missing key) — `.get(key, default)` only falls back on a
    # missing key, so a present-but-null value used to slip through and
    # crash `.get`/`.strip()` on None right after an order had already gone
    # through, leaving it unrecorded in the local ledger.
    out = parsed.get("output") or {}
    return OrderResult(
        order_no=out.get("ODNO", ""),
        order_time=out.get("ORD_TMD", ""),
        message=(parsed.get("msg1") or "").strip(),
        org_no=out.get("KRX_FWDG_ORD_ORGNO", ""),
    )
