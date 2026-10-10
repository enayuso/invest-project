"""Verifies an analyst report actually names the ticker it was asked to
analyze, catching hallucinated/mistranslated company names that the prompt
grounding alone (`app.agents.tool_loop.IDENTITY_GROUNDING_NOTE`) doesn't
reliably prevent — the model can still ignore a verified label and
substitute its own guess.

Built for the 2026-08-20 incident: 316140.KS (우리금융지주/Woori Financial
Group) was mislabeled "KB Financial Group" (a different, unrelated company)
by the Fundamentals Analyst; the Bear Researcher then dismissed real news
about 316140 as "about a different company", flipping a Hold decision to
Underweight. A 20-day/276-decision sample found the same class of error
recurring across many tickers (both affiliate-name collapse — e.g. 006660.KS
삼성공조 mislabeled "삼성전자" — and transliteration corruption — e.g. QQQ
mislabeled "인베스코 QQK 트러스트"). This module is the detection net for
whatever the source-level fixes (see kis_provider.get_ticker_display_info's
US-market English-preference fix) don't already prevent at the source.

Deliberately NOT a general name validator: it only checks the analyzed
ticker's own identity, using an allowlist of names actually sourced for that
exact ticker (KRX/overseas master file, KIS's own name, yfinance's name) —
never a fuzzy/similarity check that could flag a legitimately worded report.

2026-08-28 addendum: a live audit found this gate's own false-positive rate
at 98% (44/45 same-day reports), drowning out the real misattributions it
exists to catch. Root cause (confirmed by reproducing every false positive
against real yfinance data): `_normalize` used to DELETE punctuation rather
than replace it with a space, which fused KRX-style "Co.,Ltd." into a single
"coltd" token `_strip_suffix` could never recognize — silently breaking
suffix-stripping for most large-cap correct English names. Fixed here
(normalize/suffix/prefix/abbreviation/ETF-brand-token handling below).

A `.KS` ticker still has no AUTHORITATIVE English name source, though —
only yfinance's (weak-alias-only, see `_company_identity`'s own docstring
for why) — and yfinance itself sometimes returns nothing usable at all
(confirmed for 196170.KS/알테오젠: `longName` was `None`). For that
narrower remaining case, Detector B degrades gracefully (skips rather than
false-positives — see `check_identity`'s own comment), but Detector A does
NOT get the same skip and can still false-positive on it — a deliberate,
accepted trade-off (see the comment directly above Detector A's block for
why extending the skip there would silently un-fix the original 2026-08-20
incident this module exists for). Closing that residual gap for real needs
an actual English name from KIS itself — `app.brokers.kis.domestic.quote`
already parses `prdt_eng_abrv_name` into `StockStatus.english_name` but
nothing consumes it yet, and the TR it comes from (CTPF1002R) is rejected
outright on a paper
account (EGW02006) — so promoting it to an authoritative alias is real-
account-only work, not done in this pass.

2026-09-01 addendum: a live KRX cycle produced 10 violations across 4
decisions, all false positives. Three causes, two fixed here: (1) the
Sentiment Analyst's fixed skip report carried no company name at all, so
Detector B flagged EVERY .KS decision — fixed in sentiment_analyst.py by
grounding the template with `ticker_label(...)`. (2) a Latin<->Hangul
spacing mismatch ("KBI메탈" vs a correct "KBI 메탈") — fixed by `_script_join`
(applied additively, never inside `_normalize`). The third, romanization /
transliteration variance ("Geumho Electric" vs "Kumho", "메탈" vs "Metal"),
normalization cannot bridge and is left as a residual (still ~3
violations/cycle) pending the KIS-English-name work noted above — now
technically reachable since the system runs on a real account. Also: a
report that simply omits the name (Detector B -> UNNAMED) is now routed to a
neutral note, not a "misnamed the company" banner (that phrasing was itself
false for an omission), and to pipeline.py's `identity_unnamed` list rather
than `identity_violations` — an omission is a different failure from a
substitution and does not gate the analyst-report cache.

2026-09-03 addendum (id=1207): a live report for 270520.KS (앱튼, KOSDAQ
bio) was, in its entirety, a Fundamentals Analyst report for a completely
different company on a completely different exchange — Appen Ltd. (APPN),
listed on NASDAQ (previously ASX). Detector A never fired because the
report doesn't parenthesize our ticker at all — it parenthesizes "(APPN)"
instead, so there is no "(270520.KS)" mention for the 90-char window check
to even look at. The Hangul-candidate check added 2026-09-01 also can't
see it — the report is pure English prose, with zero Hangul runs to
inspect. Both detectors are structurally blind to "report is entirely
about someone else's foreign listing", so it fell through to UNNAMED — a
false "no name was given" when a name (the WRONG one) was given
repeatedly, in detail.

Detector C (`_foreign_listing_offender`) closes this specific gap. It only
ever runs when Detector A/B/Hangul-candidate have all found nothing AND
the ticker is a KRX ticker (`_is_krx_ticker`) — every legitimately foreign
ticker (AAPL, QQQ, ...) analyzed by this system is correctly, constantly
awash in USD figures and NASDAQ/NYSE mentions, so Detector C must never
even look at those reports, or it would flag every one of them. KRX ETFs
tracking a foreign index (e.g. a KODEX/TIGER fund holding AAPL/NVDA) are
also legitimately full of USD-denominated foreign names, so they are
carved out via the same `_ETF_BRAND_TOKENS` alias check the rest of this
module already relies on.

Within that KRX-only, non-ETF slice, Detector C requires three
independent signals to ALL agree before flagging anything (a 3-way AND,
deliberately conservative to keep the false-positive rate this module has
been tuned against since 2026-08-28 near zero):
  1. Subjecthood — the offending symbol/code must appear either exchange-
     qualified ("NASDAQ: APPN") or within the report's first
     `_SUBJECT_HEAD_CHARS` characters, where a report conventionally
     states what it's about. A peer mentioned only in passing, mid-body,
     with no exchange qualifier, is not "subject".
  2. Exchange evidence — either the matched candidate was itself
     exchange-qualified, or some foreign-exchange name/abbreviation
     appears anywhere in the report (`_FOREIGN_EXCHANGE_WORDS` —
     deliberately excludes KOSPI/KOSDAQ/KRX, our own market).
  3. Currency dominance — the report must contain zero KRW-shaped figures
     and at least `_MIN_USD_MENTIONS` USD-shaped figures. A single
     `$`-denominated peer comparison inside an otherwise KRW-denominated
     KRX report is normal equity-research prose, not a wrong-listing
     report; a report that is USD-only, cover to cover, is not.
Candidate symbols matching `_TICKER_LIKE_STOPWORDS` (financial/technical/
macro acronyms like EBIT, ROE, FDA, NASDAQ-adjacent index tickers, etc.)
or any known alias for the analyzed ticker are never treated as an
offending listing.

2026-09-04 addendum (romanization amnesty): a live measurement found ~50%
(13/26) of recent KRX decisions carrying a false-positive violation banner
for the OPPOSITE reason Detector C exists — the analyst correctly
Romanized/translated the Korean name into English (파인텍->"FineTech",
라온피플->"Laon People", 탑코미디어->"Topcomedia") and no Latin alias had
ever been sourced for it, so the substring check had nothing to match.
`_romanization_plausible` recognizes this narrow shape: it converts the
ticker's own registered Hangul name(s) to a rough romanization
(`_romanize_hangul`), reduces both that and the analyst's candidate text to
a coarse "phonetic skeleton" (`_phonetic_skeleton` — collapse similar
consonants, drop vowels), and accepts a fuzzy match only above
`_ROMANIZATION_AMNESTY_THRESHOLD` with a minimum skeleton length on both
sides (guards a KODEX-200-shaped short skeleton from fuzzy-matching an
unrelated short word like "index"). This is consulted ONLY after
`_romanization_amnesty_guard_ok` passes: the ticker must be KRX-listed, the
offending/candidate text must contain zero Hangul (a Hangul-vs-Hangul
mismatch — e.g. 삼성공조 vs 삼성전자 at 0.86 similarity, 두산퓨얼셀 vs
범한퓨얼셀 at 0.71 — must NEVER get amnesty regardless of score; both are
real, different companies, not a script/translation variance), there must
be no existing AUTHORITATIVE Latin alias for the ticker (a mismatch against
a confirmed-correct English name is meaningful, same precedent as
196170.KS/AlteoGen below), and the name isn't an ETF brand token (already
handled by `_etf_brand_alias`). At both Detector A and Detector B/C's
failure-return points, consulted strictly AFTER Detector C — see
`check_identity`'s own comments for exactly where. A deliberate, sanctioned
side effect: 196170.KS/AlteoGen (the module's own long-documented residual
false positive, see the comment on Detector A's Latin-alias skip below)
now scores 0.75 and IS cleared by this — the test that used to pin that
false positive as accepted/unfixed is inverted, not silently changed.

2026-09-04 addendum (KIS English name, exact-despaced): ahead of the
phonetic fallback above, a first, higher-confidence gate now runs at both
amnesty sites — `_kis_english_name_present` checks whether KIS's own
authoritative `prdt_eng_abrv_name` (threaded in as `kis_english_name` from
`pipeline._company_identity`, real-account only) appears in the candidate
text after reducing both sides to `[a-z0-9]` only (so KIS "LaonPeople"
matches an analyst's "Laon People"). It is EXACT substring, never fuzzy: a
fuzzy match against this field would clear a wrong name that merely shares
a generic corporate suffix with the real one — measured, the 2026-08-20
flagship's "KB Financial Group" vs KIS "WooriFinancialGroup" phonetic
skeletons score 0.777 (> the 0.75 amnesty threshold) and would flip that
regression test, whereas "kbfinancialgroup" is not a despaced substring of
anything containing "woorifinancialgroup". Gated by the SAME
`_romanization_amnesty_guard_ok` call as the phonetic check it precedes.

2026-09-14 addendum (short-Korean-name romanization blind spot): the
09-04 phonetic-skeleton amnesty above has a structural floor
(`_ROMANIZATION_SKELETON_MIN_LEN` = 4) that exists to stop a short
skeleton like "KODEX 200" -> "KTS" from fuzzy-matching an unrelated short
word ("index" -> "NTS"), but the same floor also blocks every SHORT
Korean name from ever getting amnesty, no matter how perfect the
romanization — 서남 -> "seonam" reduces to skeleton "SNM", only 3
characters, so a correctly-written "Seonam" was flagged as a wrong
company. Replaying the current logic over 143 distinct (ticker, Korean
name, offending-span) pairs from live records confirmed the gap: 29
pairs already cleared, 89 still flagged (40 of them short-skeleton).
Fix: when the Korean name's skeleton is shorter than
`_ROMANIZATION_SKELETON_MIN_LEN` (so the skeleton path can never accept
it), `_romanization_plausible` now ALSO compares at the romanization
level — `_despace_alnum_lower(_romanize_hangul(name))` against a
romanization-level candidate (vowels retained, unlike the skeleton) — at
a separate, higher threshold, `_ROMANIZATION_SHORT_SKELETON_THRESHOLD` =
0.80, with both reduced strings required to be at least
`_ROMANIZATION_LEVEL_MIN_LEN` = 4 characters.

A first version of this fallback built its candidate from the LARGEST
available Latin-token window slid across every position in
`window_or_report`. That shipped as a no-op: Detector A's real input
(`first_ungrounded_window`) is the ~90 raw characters of report prose
immediately before a "(TICKER)" mention, e.g. "**Equity Research Note:
Seonam " — a ~12-token window whose only 4-token (the configured max)
sub-strings are things like "equityresearchnoteseonam", which never
scores near 0.80 against a short Korean romanization like "seonam". The
"all positions, largest window" rule was also independently wrong on the
merits: with vowels kept, a smaller sub-n-gram can be a perfect PREFIX
match of a longer, wrong name ("Seonam" inside "Seonam Pharmaceutical"
scores 1.0 against Korean "seonam"), which would have silently cleared a
real hallucination that merely shares a root with the correct short
name — measured directly on 서남/"Seonam Pharmaceutical".

The candidate is instead built from TRAILING n-grams: Detector A's window
ends immediately before the ticker parenthetical, so the named company
sits at the END of the window, not the start or the middle. Latin-bearing
tokens are taken, trailing corporate-suffix tokens are trimmed using the
existing `_SUFFIXES` set (each token reduced to `[a-z0-9&]` before the
membership test, so "Jin Yeong Co., Ltd." reduces to "jinyeong" —
`_SUFFIXES` deliberately holds only legal-form words, never industry
words, which is exactly why "Seonam Pharmaceutical" keeps "Pharmaceutical"
and stays blocked), and then for n in 1..`_ROMANIZATION_NGRAM_MAX_WORDS`
the LAST n tokens (`toks[-n:]`) are reduced and compared — the best ratio
across those n wins. Anchoring to the end (rather than sliding a window
across every position) is what keeps this safe: a sub-n-gram can only
extend the compared string backward from the fixed tail, so an extra word
appended AFTER the real name (as in "Seonam Pharmaceutical") is never
droppable the way a leading-prefix match would drop it. Vowels are kept
precisely because that is what makes this level more discriminating than
the skeleton, which is what makes a higher threshold safe on short
strings: sweeping the romanization-level ratio against the same 40
short-skeleton pairs showed 0.80 clears them (all verifiably the same
company) with ZERO effect on any long-skeleton pair, while the
highest-scoring GENUINE hallucination in the whole 143-pair dataset
(유티아이 -> "Utility AI", an invented expansion of the initials UTI)
sits at 0.667 — a 0.133 margin below 0.80. 서남's OTHER offending span,
"Seonam Pharmaceutical" (the real id=1237 hallucination), scores only
0.462 at this level and correctly stays blocked even though it shares
the same short Korean name as the now-cleared "Seonam" — the trailing-
anchor rule is what tells the two spans apart, not the Korean name
alone. 누보 vs "Nobo" (0.750) is a known, accepted residual that stays
blocked at this threshold — deliberately not chased by lowering the
threshold further, since that would erode the margin above the 0.667
hallucination. Only `_romanization_plausible` changed;
`_romanization_amnesty_guard_ok` is untouched, so every existing
precondition (KRX-listed ticker only, zero Hangul in the offending text,
no existing authoritative Latin alias, not an ETF brand token) still
gates this new branch exactly as it gates the 09-04 one.

Limitation (not a regression): Detector B calls this with the WHOLE
report as `window_or_report`, so the trailing tokens are the end of the
report rather than the company name — this fallback is effective for
Detector A only. Short-skeleton names never got amnesty at Detector B
before this fix either; that gap is unchanged and must not be "fixed" by
reverting to all-position n-grams (see above for why that is unsafe).

요약: 짧은 한글 상호(스켈레톤 4자 미만)는 기존 로직상 절대 사면받을 수
없었던 구조적 사각지대였고, 모음을 보존하는 로마자 표기 수준에서 더 높은
임계값(0.80)으로 비교하도록 보완함. 최초 구현("전 위치 슬라이딩 중
최대폭")은 Detector A의 실제 입력(티커 괄호 직전 90자 원문)에서 4단어
윈도우가 절대 짧은 한글 이름과 매칭될 수 없어 실질적으로 무동작이었고,
설계상으로도 접미사가 붙은 오탐(서남/Seonam Pharmaceutical)을 걸러낼 수
없었음 — 윈도우 끝(티커 직전 회사명 위치)에 고정한 트레일링 n-gram +
법인접미사 트리밍으로 교체함.
"""

from __future__ import annotations

import difflib
import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

# Corporate-suffix noise stripped before alias comparison so "Woori Financial
# Group Inc." and "Woori Financial Group Inc" both match the same normalized
# form, in either language. Deliberately NOT "holdings"/"group"/"heavy"/
# "industries"/"biologics"/"e&c" — those are frequently load-bearing parts of
# a Korean conglomerate's actual legal name (e.g. "Woori Financial Group" is
# the real name, not "Woori Financial" with a suffix attached), so stripping
# them would make "Woori Financial" (a truncated, WRONG name) match as if it
# were the real one.
_SUFFIXES = {"inc", "corp", "corporation", "company", "co", "ltd", "limited", "plc", "주식회사", "(주)", "㈜"}

# Substring replacements applied to a suffix-stripped alias to add the
# common abbreviated form actual reports use — 2026-08-28: "Hyundai
# Engineering & Construction Co.,Ltd." (yfinance longName, correct) never
# matched a report correctly abbreviating it "Hyundai E&C", a false-positive
# violation confirmed live for 000720.KS. `_prefix_aliases` below already
# covers a report using a truncated PREFIX of the full name (e.g. "Hyosung
# Heavy" from "Hyosung Heavy Industries") — this table is only for a
# non-prefix abbreviation. Extend as more are confirmed by a live audit;
# never add an entry for something that could also be a DIFFERENT company's
# name.
_ABBREV_ALIASES = {
    "engineering&construction": "e&c",
    # 2026-09-17 confirmed live false positive (042370.KS, 비츠로테크 / KIS
    # English name "Bitzro Tech"): both the Market Analyst and News+Macro
    # Analyst wrote "Bitro Tech", dropping the 'z' — an unusual consonant
    # cluster ("tz") an LLM is expected to keep mis-rendering for this
    # specific ticker. The fuzzy romanization fallback scores the dropped-'z'
    # spelling at 0.727 against the Korean skeleton, just under
    # `_ROMANIZATION_AMNESTY_THRESHOLD` (0.75), so it fell through to the
    # banner. Fixed narrowly here rather than by teaching `_phonetic_skeleton`
    # a "tz" digraph rule, which would be a GLOBAL change affecting every
    # string ever skeletonized (quartz/waltz/blitz/pretzel/Ritz), risking the
    # false-negative calibration the two romanization thresholds were tuned
    # against. "bitzro" is specific enough that this substring could not
    # plausibly collide with any other company's name.
    "bitzro": "bitro",
}

# ETF issuer brand tokens — a Korean ETF's actual name (e.g. "KODEX 200") is
# usually rendered in an English report as some paraphrase of the underlying
# index rather than a literal translation ("KODEX US S&P 500 ETF" vs
# yfinance's "Samsung KODEX S&P500 ETF") — false-positive confirmed live for
# 379800.KS. Requiring only the brand token (not the whole name) to appear is
# safe specifically for ETFs: a genuine misattribution to a DIFFERENT
# company's report never contains one of these by chance (they are unrelated
# fund-issuer trademarks, never real operating-company names), so this can't
# mask 069500.KS/102110.KS being mislabeled "Samsung Electronics" — that
# offending text has no ETF brand token in it either.
_ETF_BRAND_TOKENS = frozenset({
    "kodex", "tiger", "kbstar", "arirang", "hanaro", "sol", "ace", "plus", "rise", "timefolio",
})

# --- Detector C: foreign-exchange-listing misattribution (2026-09-03, id=1207) ---
# See the module docstring's 2026-09-03 addendum for the full incident and
# rationale. These constants support `_foreign_listing_offender` only.

_SUBJECT_HEAD_CHARS = 250  # 리포트가 주체를 선언하는 머리말 영역
_MIN_USD_MENTIONS = 3


def _is_krx_ticker(ticker: str) -> bool:
    """kis_provider._is_krx와 동일한 형태 판정(순환 import 회피 위해 이 모듈에
    독립 구현) — 이 저장소는 KOSDAQ도 ".KS"를 쓴다(".KQ"는 방어적으로만 수용)."""
    base, _, suf = ticker.rpartition(".")
    return suf.upper() in ("KS", "KQ") and base.isdigit()


_FOREIGN_EXCHANGE_WORDS = frozenset({
    "nasdaq", "nyse", "nyse american", "amex", "asx", "australian securities exchange",
    "tsx", "tsxv", "toronto stock exchange", "lse", "london stock exchange",
    "hkex", "sehk", "hong kong stock exchange", "sgx", "singapore exchange",
    "euronext", "xetra", "tokyo stock exchange", "shanghai stock exchange",
    "shenzhen stock exchange", "taiwan stock exchange", "otc markets",
})   # KOSPI/KOSDAQ/KRX는 우리 시장이므로 절대 포함하지 않는다 — 포함하면 국내
     # 리포트 전부가 이 신호를 트리거하게 됨

_EXCH_QUALIFIED_SYMBOL_RE = re.compile(
    r"\b(NASDAQ|NYSE|AMEX|ASX|TSX|TSXV|LSE|HKEX|SEHK|SGX|TSE|OTC|Euronext)\s*[:：]\s*([A-Z]{1,6})\b")
_PAREN_SYMBOL_RE = re.compile(r"[(\[（]\s*([A-Z]{2,5})\s*[)\]）]")
_PROSE_TICKER_RE = re.compile(r"ticker(?:\s+symbol)?\s*(?:is\s*)?[“”\"']?([A-Z]{2,5})[“”\"']?")
_PAREN_KRX_CODE_RE = re.compile(r"[(\[（]\s*(\d{6})(?:\.(?:KS|KQ))?\s*[)\]）]")

_USD_AMOUNT_RE = re.compile(r"\$\s?\d|\bUSD\b")
_KRW_AMOUNT_RE = re.compile(
    r"₩|\bKRW\b|\d[\d,.]*\s*(?:원|억원|조원|만원|천원)|\d[\d,.]*\s*won\b|Korean won", re.I)

# 정당하게 괄호 병기되는 2~5자 대문자 약어 — 재무·기술지표·바이오·매크로 전반.
# 2선 방어일 뿐(주체성+거래소근거+통화지배 3중 AND가 1선 방어) — 여기 없는
# 약어라도 저 3중 조건을 다 통과하지 않으면 위반으로 안 잡힌다.
_TICKER_LIKE_STOPWORDS = frozenset({
    "ATR", "RSI", "MACD", "OBV", "ADX", "SMA", "EMA", "VWAP", "BB", "EPS", "PER", "PBR",
    "PSR", "PEG", "ROE", "ROA", "ROIC", "WACC", "DCF", "EBIT", "EBITDA", "CAGR", "CAPEX",
    "COGS", "FCF", "NAV", "AUM", "DPS", "BPS", "NIM", "OPM", "TTM", "QOQ", "YOY", "YTD",
    "H1", "H2", "Q1", "Q2", "Q3", "Q4", "FY", "GAAP", "IFRS", "ESG", "ETF", "IPO", "SPAC",
    "CEO", "CFO", "CTO", "COO", "IR", "HQ", "AI", "ML", "NLP", "LLM", "GPU", "HBM", "OLED",
    "LCD", "LED", "DRAM", "NAND", "SSD", "PCB", "ODM", "OEM", "EMS", "ESS", "EV", "CDMO",
    "CMO", "CRO", "ADC", "NDA", "FDA", "LOI", "MOU", "MSCI", "FED", "FOMC", "BOJ", "BOK",
    "CPI", "PPI", "PMI", "PCE", "GDP", "WTI", "OPEC", "IEA", "EIA", "IMF", "OECD", "ECB",
    "PBOC", "GDPR", "WTO", "FTA", "USD", "KRW", "JPY", "EUR", "GBP", "CNY", "HKD", "AUD",
    "CAD", "SGD", "TWD", "INR", "TAM", "SAM", "SOM", "ARPU", "MAU", "DAU", "MRR", "ARR",
    "CAC", "LTV", "NPS", "SKU", "SLA", "RFP", "KPI", "SOTP", "CAPM", "TSR", "GMV", "NPL",
    "LNG", "SEC",
})

# Tokens trimmed off the left edge of a name guessed from the text
# immediately before a "(TICKER)" mention — these are sentence fragments
# ("...report on Woori Financial Group") that would otherwise get folded
# into the "offending name" logged/banner-quoted for a violation.
_LEADING_STOPWORDS = {
    "for", "of", "the", "on", "about", "analyzing", "analyze", "ticker",
    "report", "regarding", "covering",
}

_WINDOW_CHARS = 90
_MAX_NAME_TOKENS = 6
# Shortest prefix `_prefix_aliases` will generate — a single leading token
# ("hyundai", "samsung") would match almost every Korean conglomerate's
# report and defeat the whole gate, so a prefix alias must keep at least 2
# tokens of the real name.
_MIN_PREFIX_TOKENS = 2


def _normalize(s: str) -> str:
    """NFKC first (folds full-width（）into ASCII (), full-width Latin into
    ASCII) then casefold + punctuation-to-space + whitespace collapse — so a
    comparison is blind to script-width, case, and punctuation-style
    variance, the things a translating/transliterating model (or a different
    vendor's own formatting convention) changes without changing the
    underlying name.

    2026-08-28 fix: punctuation used to be DELETED, not replaced with a
    space — "Samsung Biologics Co.,Ltd." (yfinance's actual, correct,
    comma-no-space KRX rendering convention) normalized to
    "samsung biologics coltd", a single fused "coltd" token `_strip_suffix`
    could never recognize as the "co"+"ltd" suffix pair it's built to strip.
    That silently broke the suffix-strip for most large-cap KRX names
    (Yahoo's punctuation convention here is inconsistent per ticker — only
    005930.KS happened to come back with a space, "Co., Ltd.", which is why
    it alone passed clean in a same-day sample where 4 other tickers with
    verified-correct English names all false-positived). Confirmed by
    reproducing all 5 live 2026-08-28 false positives against real yfinance
    data before this fix and clean after.

    `&` is also re-collapsed with no surrounding space ("S & P" / "S&P" /
    "S & P 500" all normalize the same way) and a letter immediately
    followed by a bare digit has the space between them removed ("S&P 500"
    -> "s&p500") — both needed for ETF names, which vary on exactly this
    formatting across sources."""
    s = unicodedata.normalize("NFKC", s).casefold()
    s = re.sub(r"['’\"]", "", s)  # possessive/quote marks never split a token
    s = re.sub(r"[.,/()\[\]{}·:;|–—-]+", " ", s)
    s = re.sub(r"\s*&\s*", "&", s)
    s = re.sub(r"([a-z])\s+(\d)", r"\1\2", s)
    return re.sub(r"\s+", " ", s).strip()


_HANGUL = r"가-힣"
# Drop the space at a Latin/digit <-> Hangul boundary, BOTH directions.
# 2026-09-01: display name "KBI메탈" (KIS's own, no space) vs a report
# correctly writing "KBI 메탈" (space) — a real false positive. Also
# "SK 하이닉스" vs "SK하이닉스". Applied ADDITIVELY (a joined alias alongside
# the spaced one, see build_alias_set / check_identity) — NEVER folded into
# `_normalize` itself: that fuses "TIGER 미국나스닥100" into one token and
# `_etf_brand_alias` then loses the "tiger" brand token (confirmed: 379800.KS
# drops from a strong pass to weak-only). Measured 2026-09-01 against the live
# KRX master file (3,553 plain 6-digit codes): 670 names have a spaced
# Latin<->Hangul boundary this touches (SK하이닉스, CJ대한통운, LG생활건강,
# DB하이텍, JW중외제약, 신세계 I&C, plus many "TIGER 미국..." ETFs). Exactly
# ONE new cross-company alias-containment results: "TIGER 200 에너지화학" ⊂
# "TIGER 200에너지화학레버리지" — same issuer, already mutually passable via
# the "tiger" brand token, and the leveraged one is filtered out of the
# universe anyway. Net marginal weakening: nil.
# 라틴/숫자 ↔ 한글 경계 공백 제거 — "KBI메탈" vs "KBI 메탈" 오탐. _normalize엔
# 넣지 않고 가법 적용(ETF 브랜드 토큰 파괴 방지).
_SCRIPT_JOIN_RES = (
    re.compile(rf"(?<=[0-9a-z&])\s+(?=[{_HANGUL}])"),
    re.compile(rf"(?<=[{_HANGUL}])\s+(?=[0-9a-z&])"),
)


def _script_join(normalized: str) -> str:
    """`normalized` (already through `_normalize`) with the space removed at
    every Latin/digit <-> Hangul boundary. Returns the input unchanged when
    there is no such boundary."""
    for rx in _SCRIPT_JOIN_RES:
        normalized = rx.sub("", normalized)
    return normalized


_LATIN_ALIAS_RE = re.compile(r"^[0-9a-z&. ]+$")


def _alias_in(alias: str, form: str) -> bool:
    """Containment test for one normalized alias against one normalized
    report/window form. A Latin/digit-only alias must match on token
    boundaries (a 2-char alias like "bl" must not match inside "problem");
    a Hangul-bearing alias keeps raw substring containment (needed for
    Korean particle attachment like "모아라이프플러스는" and for a correctly
    prefixed name like "GC녹십자웰빙" containing "녹십자웰빙")."""
    if _LATIN_ALIAS_RE.match(alias):
        return re.search(rf"(?<![0-9a-z&]){re.escape(alias)}(?![0-9a-z&])", form) is not None
    return alias in form


def _strip_suffix(normalized: str) -> str:
    tokens = normalized.split()
    while tokens and tokens[-1] in _SUFFIXES:
        tokens = tokens[:-1]
    return " ".join(tokens)


def _prefix_aliases(stripped: str) -> set[str]:
    """Every leading-token prefix of `stripped` down to `_MIN_PREFIX_TOKENS`
    tokens, EXCLUDING the full string itself (already an alias) — 2026-08-28:
    "Hyosung Heavy Industries Corporation" (yfinance longName, correct)
    never matched a report correctly truncating it to "Hyosung Heavy",
    false-positived live for 298040.KS. Deliberately PREFIX-only, never a
    suffix or arbitrary substring: this keeps the 2026-08-20 316140.KS
    incident this gate exists to catch still caught, since "KB Financial
    Group" is not a prefix of "Woori Financial Group" (nor vice versa) — a
    wrong company's name never becomes a false pass just because both names
    happen to share a tail. The accepted side-effect is that a report
    truncating the real name at some OTHER prefix boundary than a full legal
    name now also passes (e.g. "Woori Financial" alone, without "Group") —
    a truncation of the right answer, not a different answer, which is
    exactly what this gate is meant to tell apart."""
    tokens = stripped.split()
    if len(tokens) <= _MIN_PREFIX_TOKENS:
        return set()
    return {" ".join(tokens[:n]) for n in range(_MIN_PREFIX_TOKENS, len(tokens))}


def _abbreviated_aliases(stripped: str) -> set[str]:
    """Applies `_ABBREV_ALIASES`' substring replacements to `stripped`,
    returning only the entries that actually changed something (so a name
    with no matching substring contributes nothing)."""
    out: set[str] = set()
    for old, new in _ABBREV_ALIASES.items():
        if old in stripped:
            out.add(stripped.replace(old, new))
    return out


def _etf_brand_alias(stripped: str) -> set[str]:
    """If `stripped` contains one of `_ETF_BRAND_TOKENS`, that single token
    is added as its own alias — see `_ETF_BRAND_TOKENS`'s own comment for
    why a 1-token alias is safe specifically for this closed set of fund-
    issuer trademarks and not a general relaxation."""
    return {t for t in stripped.split() if t in _ETF_BRAND_TOKENS}


def build_alias_set(*names: str | None) -> frozenset[str]:
    """Normalized name, its suffix-stripped form, every leading-token prefix
    of the stripped form, any table-abbreviated form, and any ETF brand
    token found in it — for every non-empty name given. Callers pass every
    legitimately-sourced NAME for one ticker (display name, master-file
    name, yfinance name) — never names from a different ticker, and never
    the ticker string or its bare KRX code (see check_identity's ticker_forms
    subtraction for why those are not aliases)."""
    aliases: set[str] = set()
    for name in names:
        if not name:
            continue
        norm = _normalize(name)
        if not norm:
            continue
        aliases.add(norm)
        stripped = _strip_suffix(norm)
        if not stripped:
            continue
        aliases.add(stripped)
        aliases |= _prefix_aliases(stripped)
        aliases |= _abbreviated_aliases(stripped)
        aliases |= _etf_brand_alias(stripped)
    # Additive Latin<->Hangul-join variant of every alias built so far (see
    # `_script_join`). Kept separate from `_normalize` on purpose.
    aliases |= {j for a in list(aliases) if (j := _script_join(a)) != a}
    return frozenset(aliases)


def _extract_name_guess(window: str) -> str:
    """Best-effort human-readable label for whatever name preceded a
    "(TICKER)" mention that didn't match any alias — used only for the log
    line and the correction banner, not for the pass/fail decision itself,
    so approximate is fine."""
    tokens = window.strip().split()
    # `while ... pop(0)`, not `for i, tok in enumerate(tokens): tokens =
    # tokens[i+1:]` — the old loop rebound `tokens` while continuing to
    # iterate with the ORIGINAL index, so a second leading stopword sliced
    # past the name entirely. Confirmed 2026-08-25: "the report on KB
    # Financial Group" (three stopwords) yielded "(unrecognized)" while the
    # single-stopword "a note about KB Financial Group" yielded the right
    # answer — meaning the log line and banner lost the actual hallucinated
    # name in exactly the wordy cases most likely to produce one.
    while tokens and tokens[0].casefold().rstrip(".,:;") in _LEADING_STOPWORDS:
        tokens.pop(0)
    tokens = tokens[-_MAX_NAME_TOKENS:]
    return " ".join(tokens) if tokens else "(unrecognized)"


def _ticker_paren_pattern(ticker: str) -> re.Pattern[str]:
    alternatives = [re.escape(ticker)]
    if ticker.endswith(".KS"):
        alternatives.append(re.escape(ticker[:-3]) + r"(?:\.KS)?")
    return re.compile(r"[(\[]\s*(?:" + "|".join(alternatives) + r")\s*[)\]]")


class ForeignListingName(str):
    """Detector C가 반환하는 offending 이름 — 값은 평범한 str이므로
    pipeline.py의 `!= UNNAMED` 라우팅과 jsonb 직렬화가 그대로 동작하고,
    enforce_identity만 isinstance로 배너 문구를 갈라 쓴다. `symbol`은
    리포트가 실제로 선언한 다른 거래소의 티커."""

    symbol: str

    def __new__(cls, value: str, symbol: str) -> "ForeignListingName":
        obj = super().__new__(cls, value)
        obj.symbol = symbol
        return obj


def _foreign_listing_offender(
    report: str, ticker: str, all_aliases: frozenset[str]
) -> ForeignListingName | None:
    """2026-09-03 (id=1207, 270520.KS/앱튼): Fundamentals Analyst가 나스닥/ASX
    상장 Appen Ltd.(APPN) 리포트를 통째로 써냈다. Detector A는 매치할
    "(270520.KS)"가 리포트에 아예 없어 발동조차 못 하고, 한글 후보 검사도
    리포트가 순수 영문이라 0건 -> UNNAMED("이름 미기재")로 오분류됐다. 이름은
    명백히 있었고, 다른 회사(다른 거래소) 것이었다. KRX 티커 리포트에서만
    발동(_is_krx_ticker) — 해외종목 분석은 USD/NASDAQ가 정상이라 이 검사가
    영원히 안 걸려야 한다. ETF는 카브아웃(해외 기초자산을 USD로 서술하는 게
    정상)."""
    if not _is_krx_ticker(ticker):
        return None
    norm_report = unicodedata.normalize("NFKC", report)

    # ETF 카브아웃 — 우리 별칭 집합에 ETF 브랜드 토큰이 있으면 해외 기초자산을
    # USD로 서술하는 게 정상이므로 스킵.
    if any(t in _ETF_BRAND_TOKENS for a in all_aliases for t in a.split()):
        return None

    head = norm_report[:_SUBJECT_HEAD_CHARS]

    # C-1: 머리말에 우리 코드와 다른 KRX 6자리 코드가 괄호 병기됨
    our_code = ticker[:-3] if ticker.endswith((".KS", ".KQ")) else None
    for m in _PAREN_KRX_CODE_RE.finditer(head):
        code = m.group(1)
        if code != our_code:
            name_guess = _extract_name_guess(norm_report[max(0, m.start() - _WINDOW_CHARS):m.start()])
            return ForeignListingName(f"{name_guess} ({code}) — foreign listing", code)

    # C-2: 외국 상장 심볼 — 주체성 + 거래소 근거 + 통화 지배, 셋 다 필요
    candidates: list[tuple[str, int, bool]] = []  # (symbol, pos, is_exchange_qualified)
    for m in _EXCH_QUALIFIED_SYMBOL_RE.finditer(norm_report):
        candidates.append((m.group(2), m.start(), True))
    for m in _PAREN_SYMBOL_RE.finditer(head):
        candidates.append((m.group(1), m.start(), False))
    for m in _PROSE_TICKER_RE.finditer(head):
        candidates.append((m.group(1), m.start(), False))

    our_ticker_letters = ticker.split(".")[0]
    subject_symbol: tuple[str, int] | None = None
    for sym, pos, is_qualified in candidates:
        if sym == our_ticker_letters or sym in _TICKER_LIKE_STOPWORDS:
            continue
        if any(_alias_in(a, _normalize(sym)) or _alias_in(_normalize(sym), a) for a in all_aliases):
            continue
        if is_qualified or pos < _SUBJECT_HEAD_CHARS:
            subject_symbol = (sym, pos)
            break

    if subject_symbol is None:
        return None

    has_exchange_word = any(w in norm_report.casefold() for w in _FOREIGN_EXCHANGE_WORDS)
    is_qualified_match = any(sym == subject_symbol[0] and q for sym, _, q in candidates)
    if not (has_exchange_word or is_qualified_match):
        return None

    krw_hits = len(_KRW_AMOUNT_RE.findall(norm_report))
    usd_hits = len(_USD_AMOUNT_RE.findall(norm_report))
    if krw_hits > 0 or usd_hits < _MIN_USD_MENTIONS:
        return None

    symbol, pos = subject_symbol
    name_guess = _extract_name_guess(norm_report[max(0, pos - _WINDOW_CHARS):pos])
    return ForeignListingName(f"{name_guess} ({symbol}) — foreign listing", symbol)


_HANGUL_NAME_RUN = re.compile(r"[0-9A-Za-z&]*[가-힣][가-힣0-9A-Za-z&]*")
_KR_NUMERIC_UNIT = re.compile(r"^[0-9,.]*(원|억|억원|조|조원|만원|천원|주|배|퍼센트|년|월|일|분기|개월|프로)$")
_KR_INFLECTIONS = (
    "를", "을", "은", "는", "이", "가", "의", "에", "에서", "으로", "로", "도", "만", "와", "과",
    "다", "요", "지", "고", "며", "서", "나", "까", "죠", "네", "군", "자", "라", "면", "야",
)
_MIN_HANGUL_NAME_SYLLABLES = 3


def _hangul_name_candidates(report: str) -> list[str]:
    """Hangul runs in `report` shaped like a company name: at least
    `_MIN_HANGUL_NAME_SYLLABLES` Hangul syllables, not a Korean numeric/unit
    token (억원/조원/주 etc.), and not ending in a Korean grammatical
    particle or verb/adjective ending — those are what an ordinary inflected
    prose word or a unit-suffixed number look like, not a proper noun.
    Best-effort heuristic used only to decide whether Detector B has
    something name-shaped to check — false negatives here just mean "still
    skip", never a wrong verdict."""
    candidates = []
    for m in _HANGUL_NAME_RUN.finditer(report):
        text = m.group(0)
        hangul_only = re.sub(r"[^가-힣]", "", text)
        if len(hangul_only) < _MIN_HANGUL_NAME_SYLLABLES:
            continue
        if _KR_NUMERIC_UNIT.match(text):
            continue
        if any(text.endswith(suf) for suf in _KR_INFLECTIONS):
            continue
        candidates.append(text)
    return candidates


# --- Romanization amnesty (2026-09-04) --- see the module docstring's
# 2026-09-04 addendum for the full incident/rationale. Supports
# `_romanization_plausible` and its `_romanization_amnesty_guard_ok` guard
# only; never consulted anywhere else.

_RR_INITIALS = ("g", "kk", "n", "d", "tt", "r", "m", "b", "pp", "s", "ss", "",
                "j", "jj", "ch", "k", "t", "p", "h")
_RR_MEDIALS = ("a", "ae", "ya", "yae", "eo", "e", "yeo", "ye", "o", "wa", "wae",
               "oe", "yo", "u", "wo", "we", "wi", "yu", "eu", "ui", "i")
# Finals are neutralized to their dominant surface sound (e.g. ㄷ/ㅈ/ㅊ/ㅌ/ㅎ
# all -> "t", ㄱ/ㄲ/ㅋ all -> "k") — fine here because `_phonetic_skeleton`
# collapses further anyway; this table only needs to be phonetically
# plausible, not a publication-quality romanization.
_RR_FINALS = ("", "k", "k", "k", "n", "n", "n", "t", "l", "k", "m", "l", "l",
              "l", "p", "l", "m", "p", "p", "t", "t", "ng", "t", "t", "k",
              "t", "p", "t")
_HANGUL_SYLLABLE_BASE = 0xAC00
_HANGUL_SYLLABLE_COUNT = 19 * 21 * 28


def _despace_alnum_lower(s: str) -> str:
    """Lowercase, keep only [a-z0-9]. For an exact "same name modulo spacing
    and capitalisation" comparison — 'LaonPeople' and 'Laon People' both
    reduce to 'laonpeople'."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


_KIS_ENGLISH_MIN_LEN = 4  # 'SPG'/'VECT'-length abbreviations reduced to <4 alnum chars are too collision-prone to trust as an exact-match needle


def _kis_english_name_present(kis_english_name: str, text: str) -> bool:
    """True if `kis_english_name` (KIS's authoritative prdt_eng_abrv_name)
    appears in `text` ignoring spacing/punctuation/case. Exact substring, not
    fuzzy — a fuzzy match against this field would clear a wrong name that
    shares a generic corporate suffix with the real one (measured: the
    2026-08-20 flagship's 'KB Financial Group' vs KIS 'WooriFinancialGroup'
    skeletons score 0.777). Needle must be >= _KIS_ENGLISH_MIN_LEN alnum
    chars after reduction."""
    if not kis_english_name:
        return False
    needle = _despace_alnum_lower(kis_english_name)
    if len(needle) < _KIS_ENGLISH_MIN_LEN:
        return False
    return needle in _despace_alnum_lower(text)


def _romanize_hangul(text: str) -> str:
    """Minimal Revised-Romanization-flavored transliteration of Hangul
    syllables in `text` — good enough to feed `_phonetic_skeleton`'s fuzzy
    reduction, not a general-purpose/publication-quality romanizer. No
    romanization library is a dependency of this repo (checked
    pyproject.toml) so this is hand-rolled and deliberately small.
    Non-Hangul characters (including already-Latin text) pass through
    unchanged."""
    out: list[str] = []
    for ch in text:
        code = ord(ch) - _HANGUL_SYLLABLE_BASE
        if 0 <= code < _HANGUL_SYLLABLE_COUNT:
            initial, rem = divmod(code, 21 * 28)
            medial, final = divmod(rem, 28)
            out.append(_RR_INITIALS[initial] + _RR_MEDIALS[medial] + _RR_FINALS[final])
        else:
            out.append(ch)
    return "".join(out)


_SKELETON_VOWELS = frozenset("aeiouy")
_SKELETON_CONSONANTS = {
    "g": "K", "k": "K", "q": "K", "c": "K",
    "b": "P", "p": "P", "f": "P", "v": "P",
    "d": "T", "t": "T",
    "j": "J", "z": "J",
    "s": "S", "x": "S",
    "r": "L", "l": "L",
}


def _phonetic_skeleton(text: str) -> str:
    """Lowercase, collapse consonant classes (`_SKELETON_CONSONANTS`; the
    two-char "ch" digraph is collapsed to the same J class as a single unit,
    handled via substitution BEFORE the single-char rules below it run),
    drop every vowel (a,e,i,o,u,y) and every non-alpha character (spaces,
    punctuation, digits), and collapse consecutive duplicate skeleton
    characters. Deliberately approximate — only needs a legitimate
    romanization/translation of the SAME name to land close to the
    original's skeleton; see `_romanization_plausible`'s guard conditions
    for what keeps this from ever weakening a genuine wrong-company
    detection."""
    text = text.lower().replace("ch", "\x00")
    out: list[str] = []
    for ch in text:
        if ch == "\x00":
            mapped = "J"
        elif ch in _SKELETON_VOWELS:
            continue
        elif not ch.isalpha():
            continue
        else:
            mapped = _SKELETON_CONSONANTS.get(ch, ch.upper())
        if out and out[-1] == mapped:
            continue
        out.append(mapped)
    return "".join(out)


_ROMANIZATION_AMNESTY_THRESHOLD = 0.75
# Shortest skeleton either side of the comparison may have and still count —
# without this, a short skeleton like "KODEX 200"'s ("KTS") fuzzy-matches
# unrelated short words ("index" -> "NTS") at an inflated ratio. Verified
# empirically while developing this: with the floor removed, that pair
# scored above threshold on some tokenizations; with it, both skeletons are
# too short to ever qualify.
_ROMANIZATION_SKELETON_MIN_LEN = 4
_ROMANIZATION_NGRAM_MAX_WORDS = 4
_LATIN_LETTER_RE = re.compile(r"[A-Za-z]")

# 2026-09-14: for a Korean name whose skeleton is too short for the
# skeleton path above to ever accept (< _ROMANIZATION_SKELETON_MIN_LEN),
# compare at the romanization level instead (vowels retained, so more
# discriminating -- see the module docstring's 2026-09-14 addendum for the
# measurement behind this threshold and its margin below the worst
# observed genuine hallucination, 0.667).
_ROMANIZATION_SHORT_SKELETON_THRESHOLD = 0.80
_ROMANIZATION_LEVEL_MIN_LEN = 4


def _romanization_plausible(korean_names: "list[str] | frozenset[str]", window_or_report: str) -> bool:
    """True if some registered Korean name in `korean_names`, once
    romanized and reduced to a phonetic skeleton, fuzzy-matches (>=
    `_ROMANIZATION_AMNESTY_THRESHOLD` via `difflib.SequenceMatcher`) the
    skeleton of some 1-to-4-word Latin-script n-gram found in
    `window_or_report` — with both skeletons required to be at least
    `_ROMANIZATION_SKELETON_MIN_LEN` characters long. As a 2026-09-14
    fallback, a Korean name whose skeleton is too short for that path to
    ever accept is ALSO compared at the romanization level (vowels
    retained) against TRAILING n-grams (anchored to the end of
    `window_or_report`, after trimming trailing corporate-suffix tokens),
    at `_ROMANIZATION_SHORT_SKELETON_THRESHOLD` — see the module
    docstring's 2026-09-14 addendum. Only ever meaningful to call after
    `_romanization_amnesty_guard_ok` has passed — this function does no
    guarding itself (no KRX check, no Hangul-offender check, no
    authoritative-alias check, no ETF check), it is purely the
    fuzzy-similarity computation."""
    korean_skeletons = {
        skel for name in korean_names
        if name and (skel := _phonetic_skeleton(_romanize_hangul(name)))
    }
    # Romanization-level (vowels retained) reduction of each Korean name,
    # kept separately from the skeletons above -- only consulted below for
    # names whose skeleton is too short for the skeleton path to ever
    # accept. See the module docstring's 2026-09-14 addendum.
    korean_romanized = {
        r for name in korean_names
        if name and (r := _despace_alnum_lower(_romanize_hangul(name)))
    }
    if not korean_skeletons and not korean_romanized:
        return False

    latin_tokens = [t for t in window_or_report.split() if _LATIN_LETTER_RE.search(t)]
    candidate_skeletons: set[str] = set()
    for n in range(1, _ROMANIZATION_NGRAM_MAX_WORDS + 1):
        for i in range(len(latin_tokens) - n + 1):
            skel = _phonetic_skeleton(" ".join(latin_tokens[i:i + n]))
            if skel:
                candidate_skeletons.add(skel)

    # Romanization-level candidates for the short-skeleton fallback below:
    # deliberately NOT the same all-sizes-1..N-slid-across-every-position
    # enumeration as the skeleton candidates above, and deliberately
    # TRAILING (anchored to the END of `window_or_report`), not slid across
    # every position. Detector A's window is the report prose immediately
    # BEFORE a "(TICKER)" mention, so the company the report is naming sits
    # at the END of that window; taking n-grams from every position is what
    # let a PREFIX of a longer, wrong name match (e.g. "Seonam" inside
    # "Seonam Pharmaceutical" scores 1.0 against Korean "seonam") — measured
    # on the 143-pair dataset, this reproduces 서남/"Seonam Pharmaceutical"
    # -> 0.462 (stays blocked) while 서남/"...Seonam " -> 1.000 (clears).
    #
    # Before slicing, trailing corporate-suffix tokens are trimmed using the
    # module's existing `_SUFFIXES` set (each token reduced to [a-z0-9&]
    # before the membership test) so "Jin Yeong Co., Ltd." reduces to
    # "jinyeong" — `_SUFFIXES` deliberately contains only legal-form words
    # (co, ltd, inc, ...), never industry words, which is exactly why
    # "Seonam Pharmaceutical" keeps "Pharmaceutical" and stays blocked; do
    # not add industry words to `_SUFFIXES` to chase a residual here.
    #
    # NOTE (Detector B limitation): when `window_or_report` is the whole
    # report (Detector B's call), the trailing tokens are the END OF THE
    # REPORT rather than the company name, so this fallback is only
    # effective for Detector A. That is pre-existing status quo — a
    # short-skeleton name never got amnesty at Detector B either — not a
    # regression introduced here, and it must not be "fixed" by reverting
    # to all-position n-grams (see above for why that reopens the
    # Seonam/Seonam-Pharmaceutical confusion).
    trimmed_tokens = list(latin_tokens)
    while trimmed_tokens and re.sub(r"[^a-z0-9&]", "", trimmed_tokens[-1].lower()) in _SUFFIXES:
        trimmed_tokens.pop()

    candidate_romanized: set[str] = set()
    for n in range(1, min(len(trimmed_tokens), _ROMANIZATION_NGRAM_MAX_WORDS) + 1):
        rom = _despace_alnum_lower(" ".join(trimmed_tokens[-n:]))
        if rom:
            candidate_romanized.add(rom)

    best_ratio, best_kr_len, best_cand_len = 0.0, 0, 0
    for kr_skel in korean_skeletons:
        for cand_skel in candidate_skeletons:
            r = difflib.SequenceMatcher(None, kr_skel, cand_skel).ratio()
            if r > best_ratio:
                best_ratio, best_kr_len, best_cand_len = r, len(kr_skel), len(cand_skel)

    if (
        best_ratio >= _ROMANIZATION_AMNESTY_THRESHOLD
        and best_kr_len >= _ROMANIZATION_SKELETON_MIN_LEN
        and best_cand_len >= _ROMANIZATION_SKELETON_MIN_LEN
    ):
        return True

    # Short-skeleton fallback: only for Korean names the skeleton path can
    # never accept (skeleton shorter than _ROMANIZATION_SKELETON_MIN_LEN).
    # A long-skeleton name never reaches this branch -- it either already
    # passed above or its skeleton comparison is the authoritative answer.
    short_korean_romanized = {
        rom for name in korean_names
        if name
        and len(_phonetic_skeleton(_romanize_hangul(name))) < _ROMANIZATION_SKELETON_MIN_LEN
        and (rom := _despace_alnum_lower(_romanize_hangul(name)))
    }
    for kr_rom in short_korean_romanized:
        if len(kr_rom) < _ROMANIZATION_LEVEL_MIN_LEN:
            continue
        for cand_rom in candidate_romanized:
            if len(cand_rom) < _ROMANIZATION_LEVEL_MIN_LEN:
                continue
            r = difflib.SequenceMatcher(None, kr_rom, cand_rom).ratio()
            if r >= _ROMANIZATION_SHORT_SKELETON_THRESHOLD:
                return True

    return False


def _hangul_aliases(alias_set: frozenset[str]) -> list[str]:
    """Every alias in `alias_set` that contains at least one Hangul
    character — used as the `korean_names` input to
    `_romanization_plausible` (the registered Korean name(s) are already
    present, unmodified, in the alias set `build_alias_set` produces)."""
    return [a for a in alias_set if re.search(f"[{_HANGUL}]", a)]


# --- Semantic-translation amnesty (2026-09-10) --- see this comment block's
# own incident writeup. Distinct from the phonetic romanization amnesty
# above: that one bridges an honest TRANSLITERATION (파인텍->FineTech, same
# sound, different script). This bridges an honest TRANSLATION (반도체-
# >Semiconductor, same MEANING, unrelated sound) — _phonetic_skeleton
# structurally cannot and must never be asked to match these, so this is a
# separate, narrow, table-driven check, the same "EN keyword <-> KR
# substring" shape app.agents.business_description_guard._EN_TO_KR_AMNESTY
# already uses for a different purpose (there: is a hallucinated name's
# sector keyword justified by the REAL name; here: is the REAL name's own
# business-word component honestly translated in the candidate text).
#
# 2026-09-10 (id=1376, 080220.KS/제주반도체, a REAL EXECUTED Buy order): all
# three analysts correctly wrote "Jeju Semiconductor" — "Jeju" phonetically
# transliterates 제주 (a place name), "Semiconductor" SEMANTICALLY
# translates 반도체 (no phonetic relation to "bandoche" whatsoever) — and
# _romanization_plausible's whole-name phonetic-skeleton comparison
# structurally cannot bridge that, so all three reports false-positived:
#   Market Analyst: "**Technical Analysis Report: Jeju Semiconductor"
#   Fundamentals Analyst: "**Fundamentals Report: Jeju Semiconductor"
#   News + Macro Analyst: "**Analysis of Jeju Semiconductor"
#
# Deliberately narrow and exact, NOT fuzzy: requires the literal EN
# translation word present in the candidate text (case-insensitive
# substring), AND the REMAINDER of the real name (with the matched KR
# word removed) to romanize to an exact despaced-lowercase substring
# of the candidate text — reusing `_kis_english_name_present`'s "exact
# substring after despacing" philosophy, not `_romanization_plausible`'s
# fuzzy threshold (a short remainder like "제주" romanizes to "jeju", a
# 4-character skeleton-collapse of "J" that would never clear
# `_ROMANIZATION_SKELETON_MIN_LEN`'s fuzzy floor — exact substring
# matching sidesteps that entirely and is safe here because the EN-word
# gate already did the hard work of narrowing down which table entry
# applies).
#
# Must NEVER clear the 2026-08-20 flagship incident (316140.KS, real name
# 우리금융지주 containing "금융" -> EN table has "financial"/"finance", and
# the offending "KB Financial Group" DOES contain "Financial") — verify
# your implementation rejects this: the remainder after removing "금융"
# from "우리금융지주" is "우리지주", which romanizes to something with no
# relation to "kbfinancialgroup" at all, so the remainder check correctly
# fails even though the EN-word gate alone would have passed. This
# two-stage design (EN-word gate, then exact remainder substring) is why
# a single coincidental keyword match is not enough — do not weaken this
# to an EN-word-only check.
_KR_SEMANTIC_EN_AMNESTY = {
    "반도체": ("semiconductor", "semiconductors"),
    "전자": ("electronics", "electronic"),
    "화학": ("chemical", "chemicals"),
    "제약": ("pharmaceutical", "pharmaceuticals", "pharma"),
    "금융": ("financial", "finance"),
    "은행": ("bank", "banking"),
    "증권": ("securities", "brokerage"),
    "보험": ("insurance",),
    "건설": ("construction", "builder", "builders"),
    "철강": ("steel",),
    "조선": ("shipbuilding",),
    "해운": ("shipping", "marine"),
    "통신": ("telecom", "telecommunications"),
    "중공업": ("heavy industries",),
    "산업": ("industries", "industrial"),
}

# Minimum romanized-remainder length to accept as an exact substring match —
# guards against a near-empty remainder (e.g. a name that IS just the
# semantic word, with nothing left over) producing a needle so short it
# could coincidentally appear in unrelated text. A near-empty remainder
# (len 0) is handled separately below (the whole name WAS the semantic
# word — accepted outright, no substring check needed at all).
_SEMANTIC_REMAINDER_MIN_LEN = 3


def _semantic_translation_plausible(korean_names: "list[str] | frozenset[str]", candidate_text: str) -> bool:
    """True if some registered Korean name in `korean_names` contains a
    business-word substring from `_KR_SEMANTIC_EN_AMNESTY` whose English
    translation appears in `candidate_text`, AND the remainder of that name
    (the word removed) romanizes to an exact despaced-lowercase substring of
    `candidate_text`. See the module comment block above `_KR_SEMANTIC_EN_
    AMNESTY` for the full incident and the flagship-incident negative case
    this must never clear. Only ever meaningful to call after
    `_romanization_amnesty_guard_ok` has passed and after
    `_romanization_plausible` has already been tried and failed — this
    function does no guarding itself."""
    candidate_despaced = _despace_alnum_lower(candidate_text)
    candidate_lower = candidate_text.lower()
    for name in korean_names:
        if not name:
            continue
        for kr_word, en_words in _KR_SEMANTIC_EN_AMNESTY.items():
            if kr_word not in name:
                continue
            if not any(en_word in candidate_lower for en_word in en_words):
                continue
            remainder = name.replace(kr_word, "", 1)
            if not remainder:
                return True
            needle = _despace_alnum_lower(_romanize_hangul(remainder))
            if len(needle) >= _SEMANTIC_REMAINDER_MIN_LEN and needle in candidate_despaced:
                return True
    return False


def _romanization_amnesty_guard_ok(
    ticker: str,
    offending_text: str,
    authoritative_aliases: frozenset[str],
    all_aliases: frozenset[str],
) -> bool:
    """Every condition here must hold before `_romanization_plausible` may
    even be called, at either Detector A's or Detector B/C's failure-return
    point — see the module docstring's 2026-09-04 addendum.

    `authoritative_aliases` must be the caller's STRONG/authoritative set
    only (never weak) — an existing confirmed-correct Latin name means a
    mismatch is meaningful (same precedent as the 196170.KS/AlteoGen
    Detector-A comment elsewhere in this file: when we DO have a real
    answer, amnesty must not paper over a wrong one). `offending_text` is
    the Latin-candidate text under test (the ungrounded window for Detector
    A, the full report for Detector B) — if it contains ANY Hangul, amnesty
    never applies, full stop: a Hangul-vs-Hangul mismatch is never a
    script/translation variance, it's two different real companies (see the
    module docstring for the two measured danger cases this guards)."""
    if not _is_krx_ticker(ticker):
        return False
    if re.search(f"[{_HANGUL}]", offending_text):
        return False
    if any(_LATIN_LETTER_RE.search(a) for a in authoritative_aliases):
        return False
    if any(t in _ETF_BRAND_TOKENS for a in all_aliases for t in a.split()):
        return False
    return True


def check_identity(
    report: str,
    ticker: str,
    aliases: frozenset[str],
    *,
    weak_aliases: frozenset[str] = frozenset(),
    language: str = "en",
    kis_english_name: str = "",
) -> tuple[str | None, bool]:
    """(offending_name_or_None, passed_only_via_weak_alias).

    Detector A: for every "(TICKER)"/"(CODE.KS)" mention in the report, the
    90 characters immediately before it must contain some alias — this
    catches "KB Financial Group (316140.KS)" without the false positives a
    single fixed-shape regex has (a multi-word name with an internal space
    still matches, because this checks substring-in-window rather than
    trying to capture the exact name span).

    Detector A does not require EVERY parenthesized mention to be grounded —
    only at least one. `app.agents.tool_loop.IDENTITY_GROUNDING_NOTE`
    mandates the supplied name appear verbatim in the first sentence, after
    which referring to the ticker alone (e.g. a "## Conclusion" section) is
    correct, instructed behavior — requiring an alias near every mention
    flagged reports that never mismatched anything. The 2026-08-20 incident
    this detector exists to catch has ZERO grounded mentions, not merely one
    ungrounded one, so the weaker quantifier still catches it.

    Detector B (only when the report never parenthesizes the ticker at all,
    which Detector A can't see): the report's full text must contain some
    alias somewhere. Skipped when the alias set has no Latin-script entry
    and the report is in English — an honest English rendering of a
    Korean-only name is indistinguishable from a substitution in that case,
    so checking would produce false positives rather than catch anything.
    Detector A deliberately does NOT get this same skip — see the comment
    directly above its block (2026-08-28) for why extending it there would
    silently un-fix the original 2026-08-20 incident this module exists for.
    """
    # The ticker and its bare KRX code are NEVER aliases for the pass/fail
    # test, though pipeline._company_identity passed them into
    # build_alias_set until 2026-08-25. Both detectors accept "some alias
    # appears" as a pass, and the ticker appears in essentially every report
    # BY CONSTRUCTION (the analyst is told which ticker to analyze), so with
    # the ticker in the set the gate was self-satisfying. Confirmed against
    # the shipped code — all three of these returned "no violation":
    #   "KB Financial Group, listed as 316140.KS, posted strong growth."
    #   "The stock 316140 rose 3%. We rate it Overweight."
    #   "Ticker: 316140.KS\n\nKB Financial Group (316140.KS) reported…"
    # i.e. the exact 2026-08-20 incident shape passed clean whenever the
    # model wrote the correct ticker near its own wrong company name, which
    # is ordinary report-writing behavior, not an unusual one. A ticker code
    # says nothing about WHICH company the report thinks it is about — the
    # only question this module asks.
    #
    # Subtracted here rather than only at the call site: a caller re-adding
    # the ticker "for completeness" would silently reopen the hole, and this
    # function is the one place that knows what the alias set is FOR.
    ticker_forms = {_normalize(ticker)}
    if ticker.endswith(".KS"):
        ticker_forms.add(_normalize(ticker[:-3]))
    aliases = aliases - ticker_forms
    weak_aliases = weak_aliases - ticker_forms
    all_aliases = aliases | weak_aliases

    # Fail OPEN when nothing is known, not closed. Reachable exactly when
    # every upstream name lookup failed: pipeline._company_identity then
    # falls back to `display_name = ticker`, so the alias set is nothing but
    # ticker forms and the subtraction above empties it. Detector A would
    # otherwise flag every report (no alias can appear) and stamp a banner
    # reading "The correct company for 316140.KS is ." onto four downstream
    # agents' prompts — asserting a falsehood is strictly worse than not
    # checking, and there is no ground truth here to check against.
    # (Pre-2026-08-25 this case was also INCONSISTENT: Detector A
    # fail-closed, Detector B fail-open in `en` and fail-closed in `ko`.)
    if not all_aliases:
        logger.warning(
            "identity_gate: no company name known for %s — every name lookup "
            "failed; skipping the identity check rather than banner-flagging a "
            "report there is nothing to check against",
            ticker,
        )
        return None, False

    # 2026-08-28: considered moving this Latin-alias-availability skip (see
    # its Detector B application below) to also cover Detector A, to close a
    # false positive confirmed live for 196170.KS/알테오젠 (yfinance returned
    # no usable English name, so a correctly-written "AlteoGen" had nothing
    # to match against). Reverted after it broke `test_check_identity_flags_
    # hallucinated_name_near_ticker` — the ORIGINAL 2026-08-20 316140.KS
    # incident this whole module was built for is the exact same shape
    # (English text before "(TICKER)", no Latin alias sourced for a
    # Korean-only-named company) EXCEPT the written name is a real, wrong,
    # DIFFERENT company. Detector A's substring match cannot distinguish an
    # honest un-sourced translation from a wrong-company substitution when
    # neither has a Latin alias to check against — skipping here to fix one
    # would silently un-fix the other. Left as a known, accepted, narrower
    # residual (Detector A still flags 196170-shaped reports) rather than
    # reopening the flagship incident; closing it for real needs an actual
    # English name source (see the module docstring's Phase 2 KIS plan).
    norm_report = unicodedata.normalize("NFKC", report)
    pattern = _ticker_paren_pattern(ticker)
    matches = list(pattern.finditer(norm_report))

    def window_ok(window: str) -> tuple[bool, bool]:
        # Check both the plain normalized window and its Latin<->Hangul-join
        # form, so a report writing "KBI 메탈" matches a "KBI메탈" alias (and
        # vice versa via the joined aliases build_alias_set already added).
        forms = (n := _normalize(window)), _script_join(n)
        hit_strong = any(_alias_in(a, f) for a in aliases for f in forms)
        hit_weak = any(_alias_in(a, f) for a in weak_aliases for f in forms)
        return hit_strong or hit_weak, hit_strong

    if matches:
        grounded_via_strong = False
        grounded_via_weak = False
        first_ungrounded: str | None = None
        first_ungrounded_window: str | None = None
        for m in matches:
            window = norm_report[max(0, m.start() - _WINDOW_CHARS):m.start()]
            ok, via_strong = window_ok(window)
            if ok:
                grounded_via_strong = grounded_via_strong or via_strong
                grounded_via_weak = grounded_via_weak or not via_strong
            elif first_ungrounded is None:
                first_ungrounded = _extract_name_guess(window)
                first_ungrounded_window = window
        if grounded_via_strong or grounded_via_weak:
            if first_ungrounded is not None:
                logger.info(
                    "identity_gate: %s has a grounded ticker mention but a later "
                    "mention with no nearby alias (nearest text %r) — not flagged; "
                    "this is the expected 'name once, abbreviate after' shape",
                    ticker, first_ungrounded,
                )
            return None, (grounded_via_weak and not grounded_via_strong)
        # 2026-09-04 romanization amnesty — see the module docstring's
        # addendum. Only reachable when NOTHING grounded any parenthesized
        # ticker mention (the shape above already returns for a report that
        # names the company correctly even once), so this cannot mask the
        # 2026-08-20 flagship incident shape (a genuinely different, wrong
        # company name) — that incident's own regression test
        # (test_check_identity_flags_hallucinated_name_near_ticker) must
        # keep failing this fuzzy check, which it does: "KB Financial Group"
        # is not a phonetic match for "우리금융지주"'s romanization.
        if (
            first_ungrounded_window is not None
            and _romanization_amnesty_guard_ok(ticker, first_ungrounded_window, aliases, all_aliases)
        ):
            if _kis_english_name_present(kis_english_name, first_ungrounded_window):
                logger.info(
                    "identity_gate: %s — KIS English name %r found in report (exact, despaced) — not flagged",
                    ticker, kis_english_name,
                )
                return None, True
            if _romanization_plausible(_hangul_aliases(all_aliases), first_ungrounded_window):
                logger.info(
                    "identity_gate: %s — %r accepted as a romanization of %r",
                    ticker, first_ungrounded, _hangul_aliases(all_aliases),
                )
                return None, True
            if _semantic_translation_plausible(_hangul_aliases(all_aliases), first_ungrounded_window):
                logger.info(
                    "identity_gate: %s — %r accepted as a semantic translation of %r",
                    ticker, first_ungrounded, _hangul_aliases(all_aliases),
                )
                return None, True
        return first_ungrounded or "(unrecognized)", False

    # Detector B — no parenthesized ticker mention anywhere in the report.
    # The ticker/code exclusion computed above applies here too. Skipped
    # when the alias set has no Latin-script entry and the report is in
    # English — an honest English rendering of a Korean-only name is
    # indistinguishable from a substitution in that case, so checking would
    # produce false positives rather than catch anything. (Detector A does
    # NOT get this same skip — see the comment above Detector A's own block
    # for why extending it there reopens the 2026-08-20 incident.)
    has_latin_alias = any(re.search(r"[A-Za-z]{2,}", a) for a in all_aliases)
    full_forms = (n := _normalize(norm_report)), _script_join(n)
    hit_strong = any(_alias_in(a, f) for a in aliases for f in full_forms)
    hit_weak = any(_alias_in(a, f) for a in weak_aliases for f in full_forms)
    if hit_strong or hit_weak:
        return None, (hit_weak and not hit_strong)

    # No alias matched anywhere in the report. Before falling back to the
    # Latin-alias-availability skip, check whether the report actually wrote
    # a Hangul name that LOOKS like a company name but isn't in the alias
    # set at all — that's a wrong-Hangul-name substitution (audit #6,
    # id=1193: 142760.KS/모아라이프플러스 -> "한국전자기술"), not an honest
    # un-sourced translation, and skipping here would silently miss it.
    for candidate in _hangul_name_candidates(report):
        norm_candidate = _normalize(candidate)
        if not any(_alias_in(a, norm_candidate) or _alias_in(norm_candidate, a) for a in all_aliases):
            return candidate, False

    # 2026-09-03 (id=1207, 270520.KS/앱튼): Fundamentals Analyst가 나스닥/ASX
    # 상장 Appen Ltd.(APPN) 리포트를 통째로 써냈다. Detector A는 매치할
    # "(270520.KS)"가 리포트에 아예 없어 발동조차 못 하고, 한글 후보 검사도
    # 리포트가 순수 영문이라 0건 -> UNNAMED로 오분류. 이름은 명백히 있었고,
    # 다른 회사 것이었다. Detector C는 우리 이름이 어디에도 없을 때, 리포트가
    # 선언한 주체가 다른 거래소/다른 티커인지를 별도로 본다.
    # has_latin_alias 스킵보다 앞에 두는 이유: 다른 거래소 티커가 명시된
    # 리포트는 "한국어 이름의 정직한 영어 표기"가 아니므로 알테오젠형
    # degradation에 가려지면 안 된다.
    foreign = _foreign_listing_offender(report, ticker, all_aliases)
    if foreign is not None:
        return foreign, False

    # 2026-09-04 romanization amnesty — see the module docstring's addendum.
    # Consulted STRICTLY AFTER Detector C above: a report entirely about a
    # different exchange's listing (e.g. id=1207's 앱튼/Appen Ltd.) must get
    # first refusal, since that is a real misattribution, not a
    # romanization/translation variance (defense in depth — 앱튼's own
    # romanization only scores ~0.57 against "Appen", safely under
    # `_ROMANIZATION_AMNESTY_THRESHOLD`, but the ordering still matters).
    if _romanization_amnesty_guard_ok(ticker, norm_report, aliases, all_aliases):
        if _kis_english_name_present(kis_english_name, norm_report):
            logger.info(
                "identity_gate: %s — KIS English name %r found in report (exact, despaced) — not flagged",
                ticker, kis_english_name,
            )
            return None, True
        if _romanization_plausible(_hangul_aliases(all_aliases), norm_report):
            logger.info(
                "identity_gate: %s — report text accepted as a romanization of %r (Detector B)",
                ticker, _hangul_aliases(all_aliases),
            )
            return None, True
        if _semantic_translation_plausible(_hangul_aliases(all_aliases), norm_report):
            logger.info(
                "identity_gate: %s — report text accepted as a semantic translation of %r (Detector B)",
                ticker, _hangul_aliases(all_aliases),
            )
            return None, True

    # Skipped when the alias set has no Latin-script entry and the report is
    # in English — an honest English rendering of a Korean-only name is
    # indistinguishable from a substitution in that case, so checking would
    # produce false positives rather than catch anything. (Detector A does
    # NOT get this same skip — see the comment above Detector A's own block
    # for why extending it there reopens the 2026-08-20 incident.)
    if language == "en" and not has_latin_alias:
        logger.info(
            "identity_gate: skipping Detector B for %s — no Latin-script alias "
            "available, so an honest English rendering of a Korean-only name is "
            "indistinguishable from a substitution",
            ticker,
        )
        return None, False
    return UNNAMED, False


# Detector B's "the report never names any company" outcome — a module
# constant (not a bare literal) because pipeline.py compares against it to
# route a name OMISSION away from the name-MISATTRIBUTION path.
UNNAMED = "(no recognized company name found in report)"

# Wrapper-noise stripped off the front of a noisy `offending` guess before a
# scrub candidate is derived from it — a short leading run of words (1-4)
# immediately followed by a colon, e.g. "Fundamentals Report:", "Technical
# Analysis:", "Analysis Report:". Only matched at the very start; junk that
# appears mid-string (e.g. "Electronics) ### Technical Analysis: ...") is
# NOT this shape and is left for the last-N-words candidate fallback below
# to work around instead.
_LEADING_LABEL_RE = re.compile(
    r"^\s*[*#]*\s*[A-Za-z][\w\-]*(?:\s+[A-Za-z][\w\-]*){0,3}\s*:\s*", re.IGNORECASE
)

# Trailing legal-entity suffixes the body prose usually drops after first
# mention — stripped from the scrub candidate so the shorter recurring form
# ("Samsung Electronics") is what gets matched, not the one-off full-legal
# form ("Samsung Electronics Co., Ltd.") that appears only in the offending
# guess itself.
_TRAILING_LEGAL_SUFFIX_RE = re.compile(
    r"[,.]?\s*(?:Co\.,?\s*Ltd\.?|Corporation|Corp\.?|Inc\.?|주식회사|그룹)\s*$",
    re.IGNORECASE,
)

# Generic words that must never be used as a scrub target on their own, even
# if they survive candidate derivation (e.g. a degenerate `offending` guess
# reduced to a bare label word). Checked against the WHOLE candidate.
_SCRUB_STOPLIST = {"the", "this", "company", "analysis", "report", "co", "ltd", "corp", "group"}

_ASCII_LETTERS_ONLY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9 .,&'\-]*[A-Za-z0-9]$")


def _clean_scrub_candidates(offending: str) -> list[str]:
    """Longest-to-shortest candidate strings derived from a noisy
    `offending` guess (see `_extract_name_guess`'s own docstring: it is
    best-effort and approximate, never a clean repeatable name). Candidates
    are the full cleaned string, then its last 3 words, then its last 2
    words — later, shorter candidates are skipped if identical to one
    already produced."""
    cleaned = offending.strip().strip("*# ").strip()
    cleaned = _LEADING_LABEL_RE.sub("", cleaned).strip()
    cleaned = cleaned.strip(" \t\"'“”")
    cleaned = _TRAILING_LEGAL_SUFFIX_RE.sub("", cleaned).strip()
    cleaned = cleaned.rstrip(".,;: \t")
    if not cleaned:
        return []

    candidates: list[str] = []
    seen: set[str] = set()

    def _add(candidate: str) -> None:
        candidate = candidate.strip()
        if candidate and candidate not in seen:
            seen.add(candidate)
            candidates.append(candidate)

    _add(cleaned)
    words = cleaned.split()
    if len(words) > 3:
        _add(" ".join(words[-3:]))
    if len(words) > 2:
        _add(" ".join(words[-2:]))
    return candidates


def _is_eligible_scrub_candidate(candidate: str, display_name: str) -> bool:
    if len(candidate.strip()) < 2:
        return False
    cf = candidate.casefold()
    if cf in _SCRUB_STOPLIST:
        return False
    display_cf = display_name.casefold()
    if cf in display_cf or display_cf in cf:
        return False
    return True


def _count_occurrences(candidate: str, report: str) -> tuple[re.Pattern[str], int]:
    """(compiled pattern, count). Word-boundary-aware for a purely-ASCII
    candidate (avoids matching inside an unrelated longer word); a plain
    case-insensitive substring pattern otherwise (CJK/mixed-script has no
    reliable word-boundary notion here)."""
    if _ASCII_LETTERS_ONLY_RE.match(candidate):
        pattern = re.compile(r"(?<!\w)" + re.escape(candidate) + r"(?!\w)", re.IGNORECASE)
    else:
        pattern = re.compile(re.escape(candidate), re.IGNORECASE)
    return pattern, len(pattern.findall(report))


def scrub_offending_name(report: str, offending: str, display_name: str) -> tuple[str, int]:
    """(scrubbed_report, occurrence_count). Replaces recurring mentions of the
    wrong company name inside `report`'s body with `display_name`, so the
    wrong name doesn't keep echoing after the correction banner (which is
    prepended separately, by the caller, and still shows the raw `offending`
    value for an accurate "here's what was wrong" message — this function
    only touches the body).

    `offending` is `_extract_name_guess`'s best-effort output, not a clean
    repeatable name (e.g. "Electronics) ### Technical Analysis: Samsung
    Electronics" or "Analysis Report: Samsung Electronics Co., Ltd.") — a
    naive `report.replace(offending, display_name)` would scrub nothing.
    This derives progressively shorter candidates (full cleaned string, its
    last 3 words, its last 2 words), skips any candidate that is empty/too
    short, collides with `display_name`, or is on a generic stoplist, then
    picks whichever surviving candidate actually recurs most in `report`
    (ties favor the longer/earlier candidate). If nothing recurs, the
    report is returned unchanged — this function never guesses.

    The first matched occurrence is replaced with
    `f'{display_name} [name corrected from "<as found in text>"]'` so a
    reader can see what was corrected; every later occurrence becomes bare
    `display_name`.
    """
    candidates = _clean_scrub_candidates(offending)
    best_pattern: re.Pattern[str] | None = None
    best_count = 0
    for candidate in candidates:
        if not _is_eligible_scrub_candidate(candidate, display_name):
            continue
        pattern, count = _count_occurrences(candidate, report)
        if count > best_count:
            best_pattern, best_count = pattern, count

    if best_pattern is None or best_count == 0:
        return report, 0

    state = {"first": True}

    def _replace(match: re.Match[str]) -> str:
        if state["first"]:
            state["first"] = False
            return f'{display_name} [name corrected from "{match.group(0)}"]'
        return display_name

    scrubbed = best_pattern.sub(_replace, report)
    return scrubbed, best_count


_BANNER = {
    "ko": (
        '⚠️ 신원 오류 정정: 이 리포트는 회사명을 "{offending}"(으)로 잘못 표기했습니다. '
        "{ticker}의 정확한 회사명은 {display_name}입니다. 아래의 시세·재무 수치는 "
        "{ticker}에 대해 조회된 것이므로 유효합니다. 다만 이 회사가 '무엇을 하는 "
        "회사인지'에 대한 서술(업종·섹터·제품·사업모델)은 잘못된 이름에서 추론된 "
        "것일 수 있으며 검증되지 않았습니다. 리포트 내의 업종·사업 설명은 근거로 "
        "사용하지 마십시오.\n\n"
    ),
    "en": (
        '⚠️ IDENTITY CORRECTION: this report misnamed the company as "{offending}". '
        "The correct company for {ticker} is {display_name}. The market data, prices "
        "and financial figures below were fetched for {ticker} and are valid. However, "
        "any statement below that describes what this company DOES — its industry, "
        "sector, products, or business model — may have been inferred from the wrong "
        "name and is NOT verified. Do not treat a business or industry description in "
        "this report as evidence.\n\n"
    ),
}

# A report that simply never states a company name (Detector B -> UNNAMED) is
# not the same failure as one that states the WRONG company: the 2026-08-20
# downstream damage (a Bear dismissing real data as "about a different
# company") needs a wrong name to exist. So this note is neutral — it adds
# the identity, it does not accuse the report of an error. 2026-09-01: the
# News + Macro Analyst's pure-macro reports and (before its own fix) the
# Sentiment Analyst's skip report both legitimately omit the name.
# 회사명 미기재 ≠ 오표기 — 중립 안내만, 비난 문구 없음.
_BANNER_UNNAMED = {
    "ko": (
        "ℹ️ 신원 안내: 이 리포트에는 회사명이 명시되어 있지 않습니다. "
        "{ticker}의 회사명은 {display_name}입니다. 아래 데이터와 분석은 모두 "
        "{ticker}에 대한 것입니다.\n\n"
    ),
    "en": (
        "ℹ️ IDENTITY NOTE: this report does not state a company name. "
        "The company for {ticker} is {display_name}. All data and analysis below "
        "is for {ticker}.\n\n"
    ),
}


_BANNER_FOREIGN = {
    "ko": (
        '⚠️ 신원 오류 정정: 이 리포트는 {ticker}가 아니라 다른 거래소에 상장된 '
        '다른 회사({offending})에 대해 작성되었습니다. {ticker}의 회사명은 '
        '{display_name}이며, 아래 리포트의 회사명·티커·통화·재무수치는 모두 '
        '{ticker}와 무관하므로 근거로 사용하지 마십시오.\n\n'
    ),
    "en": (
        '⚠️ IDENTITY CORRECTION: this report is about a DIFFERENT company listed on a '
        'DIFFERENT exchange ({offending}), not {ticker}. The company for {ticker} is '
        '{display_name}. Every company name, ticker, currency figure and financial '
        'number below belongs to that other listing and must NOT be used as evidence '
        'for {ticker}.\n\n'
    ),
}


def enforce_identity(
    report: str,
    ticker: str,
    display_name: str,
    aliases: frozenset[str],
    *,
    weak_aliases: frozenset[str] = frozenset(),
    language: str = "en",
    kis_english_name: str = "",
    agent_name: str = "",
) -> tuple[str, str | None]:
    """(possibly banner-prefixed report, offending_name_or_None). Never
    retries the LLM call — see identity-misattribution design decision
    2026-08-20 (retry-on-violation left off by default, flag-gated, not
    implemented in this pass). A violation is always logged with a stable
    `identity_gate:` prefix so a later audit can grep for it instead of
    re-deriving the 2026-08-20 incident's manual sample by hand."""
    offending, weak_only = check_identity(
        report, ticker, aliases, weak_aliases=weak_aliases, language=language, kis_english_name=kis_english_name
    )
    if offending is None:
        if weak_only:
            logger.info(
                "identity_gate: %s (%s) passed only via an unverified yfinance-sourced alias — "
                "authoritative name is %r",
                agent_name, ticker, display_name,
            )
        return report, None
    if isinstance(offending, ForeignListingName):
        logger.warning(
            "identity_gate: %s report for %s is about a different listing (%s / %r), expected %r "
            "— foreign-listing banner applied",
            agent_name, ticker, offending.symbol, str(offending), display_name,
        )
        banner = _BANNER_FOREIGN.get(language, _BANNER_FOREIGN["en"]).format(
            offending=offending, ticker=ticker, display_name=display_name)
        return banner + report, offending
    if offending == UNNAMED:
        # Name omission, not misattribution — INFO not WARNING, and a neutral
        # note rather than a correction banner (see _BANNER_UNNAMED). Detector
        # B can't tell "no name" from "no name + a wrong name buried in prose",
        # so the check itself still fires; only the phrasing is softened.
        logger.info(
            "identity_gate: %s report for %s never names the company (expected %r) — neutral note applied",
            agent_name, ticker, display_name,
        )
        note = _BANNER_UNNAMED.get(language, _BANNER_UNNAMED["en"]).format(ticker=ticker, display_name=display_name)
        return note + report, offending
    scrubbed_report, scrub_count = scrub_offending_name(report, offending, display_name)
    logger.warning(
        "identity_gate: %s report for %s named %r, expected %r — banner applied, "
        "%d occurrence(s) scrubbed from the body",
        agent_name, ticker, offending, display_name, scrub_count,
    )
    banner = _BANNER.get(language, _BANNER["en"]).format(offending=offending, ticker=ticker, display_name=display_name)
    return banner + scrubbed_report, offending
