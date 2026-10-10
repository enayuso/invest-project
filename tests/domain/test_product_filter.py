"""app.domain.product_filter — pure functions, no I/O."""

from __future__ import annotations

from app.domain.product_filter import (
    is_cash_or_bond_fund,
    is_derivative_income_fund,
    is_out_of_scope_product,
    out_of_scope_reason,
)


# --- true positives: real KRX / NASDAQ fund names ---------------------------

def test_matches_cd_rate_cash_etf():
    # The 2026-09-01 incident ticker (481050.KS).
    assert is_cash_or_bond_fund("KODEX CD1년금리플러스액티브(합성)") is True


def test_matches_kofr_and_money_market():
    assert is_cash_or_bond_fund("TIGER KOFR금리액티브(합성)") is True
    assert is_cash_or_bond_fund("KODEX 머니마켓액티브") is True
    assert is_cash_or_bond_fund("히어로즈 CD금리액티브(합성)") is True


def test_matches_bond_etfs():
    assert is_cash_or_bond_fund("KODEX 국고채30년액티브") is True
    assert is_cash_or_bond_fund("KBSTAR 종합채권(A-이상)액티브") is True
    assert is_cash_or_bond_fund("TIGER 미국채10년") is True
    assert is_cash_or_bond_fund("ACE 단기통안채") is True
    assert is_cash_or_bond_fund("KIWOOM 단기자금") is True


def test_matches_bond_mix_funds():
    # 채권혼합: ~30% equity + ~70% bonds — deliberately in scope (module
    # docstring). "채권" catches them; "혼합" alone is NOT used (it would
    # also catch "RISE 주식혼합", an equity-mix fund — see below).
    assert is_cash_or_bond_fund("RISE 테슬라애플아마존채권혼합") is True
    assert is_cash_or_bond_fund("TIGER 테슬라채권혼합Fn") is True
    assert is_cash_or_bond_fund("KODEX 테슬라커버드콜채권혼합액티브") is True


def test_matches_english_bond_names():
    assert is_cash_or_bond_fund("ISHARES 20+Y TREASURY BOND") is True
    assert is_cash_or_bond_fund("VANGUARD SHORT TERM CORPORATE BOND") is True
    assert is_cash_or_bond_fund("BULLETSHARES 2030 HIGH YIELD CORPORATE BOND") is True
    assert is_cash_or_bond_fund("T. ROWE PRICE INTERMEDIATE MUNICIPAL INCOME") is True
    assert is_cash_or_bond_fund("VANGUARD ULTRA SHORT TREASURY") is True


def test_matches_bill_only_as_a_whole_word():
    assert is_cash_or_bond_fund("RBB US TREASURY 3M BILL") is True


def test_matches_korean_named_nasdaq_baby_bond():
    # KIS's overseas master/ranking TR returns NASDAQ-listed baby bonds with
    # a Korean name — which is why the Korean keywords apply to overseas
    # names too.
    assert is_cash_or_bond_fund("T 모바일 USA 일반채권(2070-03-01 5.500%)") is True


# --- true negatives: real names that a rejected keyword would have hit ------

def test_does_not_match_ordinary_stocks():
    assert is_cash_or_bond_fund("삼성전자") is False
    assert is_cash_or_bond_fund("NAVER") is False
    assert is_cash_or_bond_fund("APPLE INC") is False


def test_does_not_match_equity_etfs():
    assert is_cash_or_bond_fund("KODEX 200") is False
    assert is_cash_or_bond_fund("TIGER 2차전지테마") is False
    # "혼합" alone would catch this equity-mix fund — that's why only "채권"
    # (not "혼합") is a keyword.
    assert is_cash_or_bond_fund("RISE 주식혼합") is False
    # "캐시"/"cash" would catch this equity ETF — rejected keyword.
    assert is_cash_or_bond_fund("TIGER 미국캐시카우100") is False
    # "yield" alone would catch this dividend equity ETF — rejected; only
    # "high yield" is a keyword.
    assert is_cash_or_bond_fund("FIRST TRUST BLOOMBERG SHAREHOLDER YIELD") is False


def test_does_not_match_single_stock_leveraged_name():
    # Already caught by leverage_filter; not this filter's job, and "yield"
    # alone would have mis-caught the GraniteShares YieldBoost family.
    assert is_cash_or_bond_fund("GRANITESHARES YIELDBOOST NVDA 2X") is False


def test_empty_name_never_matches():
    assert is_cash_or_bond_fund("") is False


def test_known_accepted_false_positive_is_documented():
    # "OUR BOND INC" (NASDAQ: OBAI) is a real operating company whose name
    # contains "bond". This IS a match — an accepted trade-off (module
    # docstring), documented here so a future reader doesn't file it as a bug.
    assert is_cash_or_bond_fund("OUR BOND INC") is True


# --- derivative-income (covered-call / option-income) funds (2026-09-02) ----
# 498400.KS "KODEX 200타겟위클리커버드콜" passed all 3 gates and burned a full
# ~17-LLM-call company-analysis pass; the Trader issued Buy, the PM flipped it
# to Hold but on wrong grounds (yield fabricated ~2x, covered-call payoff
# understood backwards). audit id=1170.

def test_derivative_income_matches_incident_ticker():
    assert is_derivative_income_fund("KODEX 200타겟위클리커버드콜") is True


def test_incident_ticker_is_not_a_cash_or_bond_fund():
    # Pin: is_cash_or_bond_fund must not lie — a 100%-equity covered-call ETF
    # has no bond/cash sleeve. It is out of scope via the OTHER predicate.
    assert is_cash_or_bond_fund("KODEX 200타겟위클리커버드콜") is False


def test_derivative_income_matches_pure_equity_covered_call_reals():
    for name in (
        "TIGER 200커버드콜OTM",
        "RISE 200고배당커버드콜ATM",
        "PLUS 고배당주위클리커버드콜",
        "KODEX 미국나스닥100데일리커버드콜OTM",
    ):
        assert is_derivative_income_fund(name) is True, name
        assert is_cash_or_bond_fund(name) is False, name


def test_bond_sleeve_covered_call_matches_both_predicates():
    for name in (
        "KODEX 테슬라커버드콜채권혼합액티브",
        "TIGER 미국30년국채커버드콜액티브(H)",
    ):
        assert is_cash_or_bond_fund(name) is True, name
        assert is_derivative_income_fund(name) is True, name
        # cash-or-bond wins the 3-way classification.
        assert out_of_scope_reason(name) == "cash_or_bond_fund", name


def test_derivative_income_matches_english_names():
    for name in (
        "GLOBAL X NASDAQ 100 COVERED CALL",
        "FIRST TRUST BUYWRITE INCOME",
        "JP MORGAN NASDAQ EQUITY PREMIUM INCOME",
        "INFRASTRUCTURE CAPITAL NASDAQ OPTION INCOME",
    ):
        assert is_derivative_income_fund(name) is True, name


def test_derivative_income_regex_handles_hyphen_and_joined():
    for name in ("SOME COVERED-CALL FUND", "SOME COVEREDCALL FUND", "SOME BUY-WRITE FUND"):
        assert is_derivative_income_fund(name) is True, name


def test_derivative_income_rejected_keyword_false_positive_locks():
    # Measured false positives for the rejected 옵션/프리미엄/인컴 keywords —
    # all real operating companies or equity ETFs. Must stay False.
    for name in (
        "옵션 케어 헬스",
        "프리미엄 케이터링 홀딩스",
        "제너레이션 인컴 프라퍼티스",
        "오피스 프로퍼티스 인컴 트러스트",
        "ACE 글로벌인컴TOP10",
        "RISE 글로벌리얼티인컴",
    ):
        assert is_derivative_income_fund(name) is False, name


def test_derivative_income_leaves_ordinary_names_alone():
    for name in ("삼성전자", "KODEX 200", "TIGER 200", "TIGER 미국캐시카우100"):
        assert is_derivative_income_fund(name) is False, name


def test_derivative_income_empty_name_never_matches():
    assert is_derivative_income_fund("") is False


def test_out_of_scope_reason_three_way_and_consistency():
    cases = {
        "KODEX CD1년금리플러스액티브(합성)": "cash_or_bond_fund",
        "KODEX 200타겟위클리커버드콜": "derivative_income_fund",
        "KODEX 테슬라커버드콜채권혼합액티브": "cash_or_bond_fund",
        "삼성전자": None,
        "KODEX 200": None,
        "": None,
    }
    for name, expected in cases.items():
        assert out_of_scope_reason(name) == expected, name
        assert is_out_of_scope_product(name) is (
            is_cash_or_bond_fund(name) or is_derivative_income_fund(name)
        ), name
