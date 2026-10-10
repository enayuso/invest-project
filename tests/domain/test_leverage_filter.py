"""app.domain.leverage_filter — pure functions, no I/O."""

from __future__ import annotations

from app.domain.leverage_filter import is_leveraged_or_inverse, is_unsupported_product_type


def test_is_leveraged_or_inverse_matches_korean_keyword():
    assert is_leveraged_or_inverse("KODEX 레버리지") is True


def test_is_leveraged_or_inverse_matches_english_keyword():
    assert is_leveraged_or_inverse("ProShares UltraPro Short QQQ") is True


def test_is_leveraged_or_inverse_false_for_ordinary_name():
    assert is_leveraged_or_inverse("Apple Inc.") is False


def test_is_leveraged_or_inverse_false_for_empty_name():
    assert is_leveraged_or_inverse("") is False


# ---------------------------------------------------------------------------
# Multiplier-pattern matching (2026-08-08) — KIS names these NASDAQ inverse
# ETFs by factor ("DAILY -1X"), not by yfinance's "Bear"/"Inverse" wording.
# Confirmed live against the real NASDAQ master file: every one of these
# escaped the old keyword-only check before this fix.
# ---------------------------------------------------------------------------


def test_matches_negative_one_x_direxion_style_naming():
    assert is_leveraged_or_inverse("DIREXION NVDA DAILY -1X") is True


def test_matches_fractional_multiplier():
    assert is_leveraged_or_inverse("TRADR NVDA DAILY -1.5X") is True
    assert is_leveraged_or_inverse("GRANITESHARES TSLA DAILY 1.25X") is True


def test_matches_leverage_without_ed_suffix():
    # "LEVERAGE SHARES ..." — the old keyword was "leveraged", which this
    # naming doesn't contain at all.
    assert is_leveraged_or_inverse("LEVERAGE SHARES 100% TSLA AND 100% SPCX DAILY") is True


def test_matches_plain_2x_3x_as_before():
    assert is_leveraged_or_inverse("KODEX 200선물인버스2X") is True
    assert is_leveraged_or_inverse("ProShares UltraPro QQQ 3X") is True


def test_accepted_false_positive_on_a_numeric_prefixed_name():
    # Deliberate tradeoff (block-first): a real KRX company name containing
    # a bare "<digits>X" pattern gets caught too. Documented here so a
    # future reader sees it's a known, accepted cost, not a regression.
    assert is_leveraged_or_inverse("10X Genomics") is True


# ---------------------------------------------------------------------------
# Korean multiplier pattern (2026-08-27) — digit-anchored counterpart to the
# "X" multiplier regex above. Measured against the live KIS NASDAQ master
# file (5,237 tickers): 0 matches today, 0 collision risk. Bare Korean
# keywords (배/불/베어/숏/울트라 without a digit anchor) were deliberately
# NOT added — same measurement found 16 real false positives and 0 genuine
# catches, so the regression cases below must stay False.
# ---------------------------------------------------------------------------


def test_matches_korean_digit_multiplier():
    assert is_leveraged_or_inverse("KODEX 미국나스닥100 3배") is True


def test_matches_korean_decimal_multiplier():
    assert is_leveraged_or_inverse("어떤상품 2.5배") is True


def test_no_false_positive_on_bare_korean_multiplier_words():
    # These are real NASDAQ company names containing 배/불/베어/숏/울트라 as
    # substrings of ordinary words, with no digit prefix — must stay False.
    assert is_leveraged_or_inverse("위불") is False
    assert is_leveraged_or_inverse("울트라제닉스 파머슈티컬") is False
    assert is_leveraged_or_inverse("베어드 메디컬 인베스트먼트 홀딩스") is False
    assert is_leveraged_or_inverse("아메리칸 배터리 테크놀로지") is False


# ---------------------------------------------------------------------------
# Left-boundary fix (2026-09-02) — "S&P500배당" was read as "500배"
# Measured against the live KRX master (3,552 codes): the old no-left-anchor
# regex had 2 hits, both false positives, 0 genuine catches. For 429000 this
# regex was the SOLE block across every gate (product_filter.
# out_of_scope_reason returns None for it). Fix: left lookbehind
# `(?<![0-9A-Za-z])` + `(?!당)`.
# ---------------------------------------------------------------------------


def test_no_false_positive_on_sp500_dividend_etf():
    # 429000 TIGER 미국S&P500배당귀족 — "500배" is the tail of the index name
    # "S&P500" + first syllable of 배당. This regex was its only block.
    assert is_leveraged_or_inverse("TIGER 미국S&P500배당귀족") is False


def test_no_false_positive_on_sp500_dividend_covered_call_etf():
    assert is_leveraged_or_inverse("KODEX 미국S&P500배당귀족커버드콜(합성 H)") is False


def test_no_false_positive_on_hangul_prefixed_dividend_growth():
    # The (?!당) case: "200" is preceded by Hangul "피", so the left anchor
    # alone would pass it — (?!당) rejects "배당".
    assert is_leveraged_or_inverse("코스피200배당성장") is False


def test_matches_korean_multiplier_after_hangul_prefix():
    # Future-proofing still works: "2배" where 2 is preceded by Hangul "년".
    assert is_leveraged_or_inverse("KODEX 미국채30년2배") is True


def test_matches_korean_multiplier_tiger_2x_leverage():
    assert is_leveraged_or_inverse("TIGER 2배레버리지") is True


def test_real_leveraged_inverse_names_still_blocked():
    assert is_leveraged_or_inverse("KODEX 레버리지") is True
    assert is_leveraged_or_inverse("KODEX 200선물인버스2X") is True


def test_is_unsupported_product_type_true_for_mutual_fund():
    assert is_unsupported_product_type("MUTUALFUND") is True


def test_is_unsupported_product_type_false_for_equity():
    assert is_unsupported_product_type("EQUITY") is False


def test_is_unsupported_product_type_false_for_etf():
    assert is_unsupported_product_type("ETF") is False


def test_is_unsupported_product_type_false_for_unknown():
    # Empty/unknown isn't evidence either way — same convention as
    # is_leveraged_or_inverse's empty-name case.
    assert is_unsupported_product_type("") is False


def test_is_unsupported_product_type_false_for_literal_none_string():
    # yfinance returns the literal string "NONE" (not "") for some tickers'
    # quoteType — confirmed live 2026-09-03 (014950.KS, 483350.KS). Must be
    # treated the same as the empty-string case above, not as an unsupported
    # product type.
    assert is_unsupported_product_type("NONE") is False
    assert is_unsupported_product_type("none") is False


def test_is_unsupported_product_type_false_for_unknown_string():
    assert is_unsupported_product_type("UNKNOWN") is False
