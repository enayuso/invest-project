"""Name-based out-of-scope product filter — cash/bond funds AND
derivative-income (covered-call / option-income) funds.

Pure domain logic (no I/O, no dependencies), same Domain-First convention as
`app.domain.leverage_filter` — every layer that discovers or admits a ticker
(screening universe, trade allowlist add, the KIS order choke point, the
dashboard search box) calls this module instead of each re-implementing its
own keyword list.

Kept a SEPARATE module from `leverage_filter`, not a new function inside it:
that module's name, docstring, and every comment are bound to one concept —
"leveraged/inverse/futures = margin-call, liquidation, and decay/volatility
risk this long-only system was never designed to size or stop." Neither a
cash/bond fund nor a covered-call fund carries that risk. They're excluded
for a different reason, closer to `is_unsupported_product_type`: the
multi-agent pipeline here (fundamentals, news, an analyst debate, ATR-based
stop sizing) has nothing to analyze — there are no company earnings, no
company news, and for these products momentum/valuation/ATR analysis is
actively misleading (the blended ATR of a cash/bond fund is misleadingly
low; a covered-call fund's return is an option-delta/theta payoff, not a
company's).

TWO predicates, deliberately NOT folded into one. `is_cash_or_bond_fund`
must keep meaning exactly what its name says. A 커버드콜 ETF like
498400.KS "KODEX 200타겟위클리커버드콜" is 100% equity with a written call
overlay — zero bond sleeve, zero cash sleeve. Calling it a "cash or bond
fund" would be a lie, and that exact misread was the 2026-09-02 judgment
failure: the PM/risk camp reasoned about the payoff as if it were a bond
proxy and inverted the covered-call profile (capped upside, full downside
minus premium). So derivative-income is its own predicate
(`is_derivative_income_fund`), and callers that want "is this in scope at
all?" use `is_out_of_scope_product` / `out_of_scope_reason`.

Matched on the product NAME (same as `leverage_filter` — a KRX bond ETF or
covered-call ETF is an ordinary-looking 6-digit code). yfinance classifies
these as quoteType "ETF", so `is_unsupported_product_type` can't catch them;
a name filter is the only handle.

Built for two incidents:

2026-09-01: the SEMI_AUTO cycle surfaced 481050.KS
"KODEX CD1년금리플러스액티브(합성)" — a CD-rate cash-parking ETF — as a
candidate, because these funds routinely rank near the top of KRX 거래대금
(institutions move large sums through them). Three things went wrong:
(1) every analyst misread it as a "corporate bond ETF" and the Portfolio
Manager's Hold rationale argued about credit spreads widening in an equity
drawdown — a risk that does not exist for a CD-rate product; (2) its 14-day
ATR is 0.0%, so `pipeline._atr_context` told the model "a Buy gets its full
100% base allocation" (only cash-insufficiency actually stopped a Buy from
being sized at 100% of equity); (3) it burned a full ~4-minute multi-agent
LLM analysis pass on a product with no earnings, no news, no fundamentals.

2026-09-02 (corporate account, first FULL_AUTO live cycle): 498400.KS
"KODEX 200타겟위클리커버드콜" passed all 3 gates (product_filter,
leverage_filter, is_unsupported_product_type), ran the full ~17-LLM-call
company-analysis pass, and the Trader issued a Buy. The PM flipped it to
Hold, but on wrong grounds (the distribution yield was fabricated ~2x too
high, and the PM/risk camp understood the covered-call payoff backwards —
treating the call premium as free carry rather than sold-away upside). No
company to analyze: the fund holds KOSPI 200 and writes weekly calls; its
earnings/news are the index's, its return is the option overlay's. audit
id=1170.

채권혼합(bond-mix) funds — roughly 30% equity + 70% bonds — are deliberately
INCLUDED by `is_cash_or_bond_fund` (the "채권" keyword catches all of them
on the live KRX file). A product like "KODEX 테슬라커버드콜채권혼합액티브"
is caught by "채권" as well as by the covered-call predicate. Pure-equity
covered-call ETFs have no bond keyword at all — that gap is what the new
`is_derivative_income_fund` predicate now closes.

현금성(CD금리/KOFR/머니마켓)·채권·채권혼합형 상품에 더해 커버드콜·옵션인컴형
상품도 상품명으로 걸러냄 — 실적·뉴스가 없고(분석할 회사가 없음), 커버드콜은
수익 구조가 옵션 델타/세타라 모멘텀·밸류에이션·ATR 분석이 오히려 오도함
(2026-09-01 481050.KS, 2026-09-02 498400.KS 사고).
"""

from __future__ import annotations

import re

# Substring keywords. Measured 2026-09-01 against the live master files
# (app.live.krx_master / app.live.overseas_master):
#
#   KRX (KOSPI+KOSDAQ, 3,553 plain 6-digit codes): 161 matches, ALL of them
#   bond/rate ETFs under a fund-issuer brand prefix (KODEX/TIGER/RISE/PLUS/
#   ACE/SOL/HANARO/KIWOOM/…). Zero operating-company or equity-ETF matches.
#   481050 "KODEX CD1년금리플러스액티브(합성)" matches via "금리".
#   Per-keyword KRX hit counts: 채권 45, 국채 38, 국고채 23, 금리 21, 회사채 16,
#   머니마켓 11, 통안 6, 하이일드 3, 단기자금 1.
#
#   NASDAQ (5,228 rows): 252 matches. Exactly one measured false positive —
#   "OBAI  OUR BOND INC", a real operating company whose name contains
#   "bond". Accepted the same way leverage_filter accepts its broad
#   multiplier regex: one missed candidate costs far less than a missed
#   cash/bond fund burning an LLM pass and mis-sizing at 100%. The other 251
#   are Treasury/muni/corp-bond ETFs plus ~51 NASDAQ-listed baby bonds whose
#   KIS name comes back in Korean ("T 모바일 USA 일반채권(2070-03-01 5.500%)")
#   — which is why the Korean keywords are applied to overseas names too.
#
# Rejected keywords, each after a measured false positive — do NOT re-add
# without re-measuring against the live file:
#   "yield" alone   -> "FIRST TRUST BLOOMBERG SHAREHOLDER YIELD" (a dividend
#                      equity ETF) + 46 GraniteShares YieldBoost single-stock
#                      leveraged names. Use "high yield" only.
#   "혼합" alone     -> "183710 RISE 주식혼합" (an equity-mix fund). "채권"
#                      already covers all 59 채권혼합 funds, so "혼합" adds
#                      nothing but this false positive.
#   "캐시"/"cash"    -> "465670 TIGER 미국캐시카우100" (an equity ETF).
#   "bill" substring -> use the word-boundary regex below instead.
#
# "금리" is load-bearing: it catches CD금리/KOFR금리/SOFR금리/단기변동금리부채권
# in one token, with zero KRX operating-company false positives ("금융" does
# NOT contain "금리").
_CASH_BOND_KEYWORDS = (
    # Korean — also applied to overseas names (KIS returns NASDAQ baby bonds
    # with Korean names).
    "채권",
    "국채",
    "국고채",
    "회사채",
    "금리",
    "머니마켓",
    "통안",
    "단기자금",
    "하이일드",
    # English
    "bond",
    "treasury",
    "t-bill",
    "money market",
    "municipal",
    "high yield",
    "aggregate bond",
    "commercial paper",
    "short duration",
    "ultrashort",
    "ultra short",
)

# "bill"/"bills" as a whole word only — "T. Rowe Price ... Bill" / "US
# Treasury 3M Bill" without also matching a company with "bill" inside
# another word.
_BILL_RE = re.compile(r"\bbills?\b")

# Derivative-income (covered-call / option-income) keywords. Measured
# 2026-09-02 against the live master files (KRX 3,552 / NASDAQ 5,228):
#
#   커버드콜        KRX 33 hits. 5 of those are ALSO caught by 채권/국채 (the
#                  채권혼합 / 국채커버드콜 families), so this predicate is
#                  net-new for 27 pure-equity covered-call ETFs
#                  (TIGER 200커버드콜OTM, RISE 200고배당커버드콜ATM,
#                  KODEX 200타겟위클리커버드콜, PLUS 고배당주위클리커버드콜, …).
#   covered call   NASDAQ 8 hits (GLOBAL X … COVERED CALL family, etc.).
#   buywrite       NASDAQ 2 hits (FIRST TRUST BUYWRITE INCOME, …).
#   premium income NASDAQ 9 hits (JPMORGAN … EQUITY PREMIUM INCOME family, …).
#   option income  NASDAQ 4 hits. 1 ("YQQQ") is ALREADY blocked via "short",
#                  net-new 3.
#   False positives across all five: 0.
#
# Rejected keywords, each after a measured false positive — do NOT re-add
# without re-measuring against the live file:
#   "옵션" alone   -> "OPCH  옵션 케어 헬스" (Option Care Health, a real
#                    operating company).
#   "프리미엄" alone -> "PC  프리미엄 케이터링 홀딩스" (a real operating company).
#   "인컴" alone   -> "GIPR 제너레이션 인컴 프라퍼티스", "OPI 오피스 프로퍼티스
#                    인컴 트러스트" (real REITs) + "460960 ACE 글로벌인컴TOP10",
#                    "475380 RISE 글로벌리얼티인컴" (equity ETFs) — 4 false
#                    positives.
#   "위클리" / "타겟위클리" / "커버드" -> no false positives, but every hit is a
#                    strict subset of a "커버드콜" hit (net-new 0), so not
#                    added — same logic as the rejected "혼합".
_DERIVATIVE_INCOME_KEYWORDS = ("커버드콜", "premium income", "option income")

# "covered call" / "covered-call" / "coveredcall" and "buy write" / "buy-write"
# / "buywrite" — hyphen or joined or space-separated, all one pattern.
_COVERED_CALL_RE = re.compile(r"covered[\s-]?call|buy[\s-]?write")


def is_cash_or_bond_fund(name: str) -> bool:
    """True if `name` (a fund/product display name) looks like a cash-like
    (CD-rate, KOFR, money-market), bond, or bond-mix fund — see the module
    docstring for why these are out of scope for this system's
    company-analysis pipeline. Empty/blank names never match — same "unknown
    isn't evidence either way" convention as
    `leverage_filter.is_leveraged_or_inverse` (and see
    `app.live.universe`'s docstring for why a discovery-stage filter is
    deliberately lenient on unresolved names).

    Deliberately does NOT match pure-equity covered-call funds — those have
    no cash or bond sleeve and calling them a "cash or bond fund" would be a
    lie. Use `is_derivative_income_fund` (or `is_out_of_scope_product`) for
    those."""
    if not name:
        return False
    lowered = f" {name.lower()} "
    if any(keyword in lowered for keyword in _CASH_BOND_KEYWORDS):
        return True
    return bool(_BILL_RE.search(lowered))


def is_derivative_income_fund(name: str) -> bool:
    """True if `name` looks like a covered-call / option-income (buy-write,
    equity-premium-income) fund. These are typically 100% equity with a
    written-option overlay — no bond sleeve, so `is_cash_or_bond_fund` does
    not (and must not) catch them — but they are still out of scope for the
    same reason: there is no company to analyze (the fund's earnings/news are
    its underlying index's), and the return is an option-delta/theta payoff
    that momentum/valuation/ATR analysis reads backwards (2026-09-02
    498400.KS incident, audit id=1170).

    Empty/blank names never match — same convention as
    `is_cash_or_bond_fund`."""
    if not name:
        return False
    lowered = f" {name.lower()} "
    if any(keyword in lowered for keyword in _DERIVATIVE_INCOME_KEYWORDS):
        return True
    return bool(_COVERED_CALL_RE.search(lowered))


def out_of_scope_reason(name: str) -> str | None:
    """`"cash_or_bond_fund"`, `"derivative_income_fund"`, or `None` — the
    single classification helper every caller should use instead of calling
    the two predicates separately (so a new out-of-scope category added here
    later reaches every gate at once). cash-or-bond is checked first: a
    bond-sleeve covered-call fund (e.g. "…커버드콜채권혼합…") is reported as
    `cash_or_bond_fund`, matching the pre-2026-09-02 behavior for those."""
    if is_cash_or_bond_fund(name):
        return "cash_or_bond_fund"
    if is_derivative_income_fund(name):
        return "derivative_income_fund"
    return None


def is_out_of_scope_product(name: str) -> bool:
    """True if `name` is any product this system's company-analysis pipeline
    has no business analyzing — cash-like/bond/bond-mix OR covered-call/
    option-income. `out_of_scope_reason(name) is not None`."""
    return out_of_scope_reason(name) is not None
