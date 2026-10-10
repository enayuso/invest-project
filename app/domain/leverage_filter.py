"""Name-based futures/leveraged/inverse product filter.

Pure domain logic (no I/O, no dependencies) per this project's Domain-First
principle — every layer that discovers or admits a ticker (screening
universe, trade allowlist add, the KIS order choke point) calls the same function
here instead of each re-implementing its own keyword list.

Semi-auto/full-auto modes remove the human's per-trade judgment call, and
futures/leveraged/inverse products carry margin-call/liquidation risk and
decay/volatility behavior that this system's long-only, no-leverage scope
(see config.py::sizing_target_atr_pct's neighboring comments) was never
designed to size or stop correctly. Matched on the product NAME, not the
ticker code — KRX leveraged/inverse ETFs are ordinary-looking 6-digit codes
(e.g. KODEX 레버리지 = 122630.KS) indistinguishable from a regular stock by
format alone, and the same is true of US leveraged ETF tickers (TQQQ, SOXL).
"""

from __future__ import annotations

import re

_KEYWORDS = (
    # Korean: KRX leveraged/inverse ETF and futures-linked product naming.
    "레버리지",
    "인버스",
    "선물",
    # English: the standard leveraged/inverse ETF family naming conventions
    # (ProShares Ultra*, Direxion Daily *Bull/Bear*). "leverage" (not
    # "leveraged") so it also catches naming like "LEVERAGE SHARES ..." that
    # doesn't use the "-ed" form.
    "leverage",
    "inverse",
    "ultra",
    " bull ",
    " bear ",
    "futures",
    "short",
)

# Any explicit leverage/direction multiplier — "2x"/"3x", but also "-1x"/
# "-1.5x"/"1.25x" — the KIS-name convention for the "-1X"/"-1.5X" family
# (e.g. "DIREXION NVDA DAILY -1X") that plain "bear"/"inverse" keywords
# missed: KIS names these by factor, not by the bull/bear wording yfinance
# uses (confirmed live against the NASDAQ master file — NVDD/TSLS/AAPD/SARK
# and others all escaped the old keyword-only check this way). Deliberately
# broad on purpose: a false positive here (e.g. blocking a real KRX name
# that happens to contain "10X") is far cheaper than a false negative that
# lets an actual leveraged/inverse product trade unsized for its risk.
_MULTIPLIER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?\s*x\b")

# Korean-language counterpart to _MULTIPLIER_RE: a digit prefix immediately
# before "배" (the Korean multiplier word), e.g. "2배", "-1.5배", "2.5배" —
# mirrors KRX's own leveraged-ETF naming convention ("TIGER 200선물레버리지"
# family; some products say "2배" instead of "2X"). Digit-anchored the same
# way _MULTIPLIER_RE is, for the same reason: measured against the live KIS
# NASDAQ master file (5,237 tickers, Korean-name column) on 2026-08-27 — 0
# matches today (every leveraged/inverse product in that file is still
# English-named and already caught by _KEYWORDS/_MULTIPLIER_RE) and 0
# collision risk (no ordinary company name in the dataset has a digit
# immediately followed by "배"). This is defensive/future-proofing only, in
# case KIS later Korean-translates a NASDAQ leveraged product using this
# wording. Bare Korean keywords (배/불/베어/숏/울트라 with no digit anchor)
# were deliberately NOT added — the same measurement found 16 real false
# positives on ordinary NASDAQ names (위불/Webull BULL, 울트라제닉스/
# Ultragenyx RARE, 울트라 클린 홀딩스/Ultra Clean Holdings UCTT, 베어드
# 메디컬/Baird Medical BDMD, and others) and 0 genuine leveraged/inverse
# catches — do not re-add them without re-measuring against the live master
# file.
#
# 2026-09-02 left-boundary fix: measured for the first time against the live
# KRX master (app.live.krx_master, 3,552 plain 6-digit codes). The OLD regex
# (no left anchor) had exactly 2 hits, BOTH false positives — "429000 TIGER
# 미국S&P500배당귀족" (a plain dividend equity ETF; product_filter.
# out_of_scope_reason returns None for it, so this regex was its ONLY block
# across every gate) and "276970 KODEX 미국S&P500배당귀족커버드콜(합성 H)".
# Zero genuine leveraged/inverse catches. Root cause: no left boundary, so
# "S&P500배당" parsed as "500배" (the "500" belongs to the index name
# "S&P500"; the "배" is the first syllable of 배당/dividend). Enumeration
# evidence: all 55 KRX names containing "배" were checked — NONE use "N배"
# multiplier notation (they are 배당/배출권/배터리/배럴/문배철강). KRX names
# leverage only as "레버리지" or "2X", and the 8 real KRX leveraged/inverse
# ETFs are already double-caught by _MULTIPLIER_RE + _KEYWORDS. The regex is
# nonetheless KEPT: its target is KIS's Korean *translations* of NASDAQ
# leveraged products, which are not bound by KRX naming conventions. NASDAQ
# master (5,228 rows) re-measured 2026-09-02: still 0 hits, so the
# 2026-08-27 conclusion holds. The fix adds a left lookbehind
# `(?<![0-9A-Za-z])` plus `(?!당)`: the lookbehind rejects "S&P500배당", and
# `(?!당)` additionally guards a Hangul-prefixed case (e.g. a hypothetical
# "코스피200배당성장" where "200" is preceded by Hangul "피" so the left
# anchor alone would pass it). NOT `(?![가-힣])` — that would make the regex
# self-defeating on real targets like "TIGER 2배레버리지" (배 followed by 레,
# which is inside 가-힣); those are caught by the 레버리지 keyword anyway.
# 한글 배수 표기("2배", "-1.5배")는 숫자가 왼쪽 경계 뒤에 바로 붙고 오른쪽이
# "당"이 아닐 때만 매칭 — 좌측 경계가 없어 "S&P500배당"이 "500배"로 읽히던
# 오탐(라이브 KRX 실측 2건) 수정. 접두사 없는 배/불/베어/숏/울트라는 실제
# 종목명 오탐(16건) 확인되어 제외.
_KOREAN_MULTIPLIER_RE = re.compile(r"(?<![0-9A-Za-z])[-+]?\d+(?:\.\d+)?\s*배(?!당)")


def is_leveraged_or_inverse(name: str) -> bool:
    """True if `name` (a company/product display name) looks like a futures,
    leveraged, or inverse product, by keyword or by an explicit multiplier
    pattern (see `_MULTIPLIER_RE`). Empty/blank names never match — an
    unknown name isn't evidence either way, and the ticker-format checks
    upstream already reject unroutable symbols. (`get_ticker_display_info`'s
    callers must NOT rely on that "unknown isn't evidence" leniency to mean
    "an unresolvable name is safe to trade" — see their own guards for why
    an unresolvable name is refused before it ever reaches this function.)"""
    if not name:
        return False
    lowered = f" {name.lower()} "
    if any(keyword in lowered for keyword in _KEYWORDS):
        return True
    return bool(_MULTIPLIER_RE.search(lowered) or _KOREAN_MULTIPLIER_RE.search(lowered))


# yfinance's own instrument classification ("EQUITY", "ETF", "MUTUALFUND",
# "ETN", ...) — a plain stock or ETF is what this system's multi-agent
# analysis (fundamentals, news, PER/PBR comparisons, ATR-based stop-loss
# sizing) is built for. A mutual fund/ETN doesn't fit that model (no
# meaningful "company earnings" to analyze) and, per the incident that
# prompted this check (218150.KS, quoteType MUTUALFUND), tends to come with
# much worse yfinance data coverage too — see
# app.dataflows.yfinance.market_data.get_ticker_display_info's own
# name-validation safeguard for that same incident's other half.
_SUPPORTED_QUOTE_TYPES = frozenset({"EQUITY", "ETF"})

# yfinance returns the literal string "NONE" (not an empty string) for some
# tickers' quoteType — confirmed live 2026-09-03 (014950.KS, 483350.KS) —
# which used to slip past the `if not quote_type` unknown-value exemption
# below and get blocked as an "unsupported product" despite carrying no
# actual evidence of one. Treated the same as empty/missing.
_UNKNOWN_QUOTE_TYPE_VALUES = frozenset({"NONE", "UNKNOWN"})


def is_unsupported_product_type(quote_type: str) -> bool:
    """True if `quote_type` (see above) isn't a plain stock or ETF.
    Empty/unknown quote_type never matches — same "unknown isn't evidence
    either way" convention as `is_leveraged_or_inverse`."""
    if not quote_type or quote_type.upper() in _UNKNOWN_QUOTE_TYPE_VALUES:
        return False
    return quote_type not in _SUPPORTED_QUOTE_TYPES
