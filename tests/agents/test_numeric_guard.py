"""app.agents.numeric_guard — the unit+ratio composer and its node prefixing.

2026-09-01 first-real-account audit: decision id=1157 published
"-785.2 billion KRW" (10x wrong) in a risk-debate turn that matched
`_BILLION_RE` exactly but sat in a node the guards were never wired to —
"right regex shape, wrong node".
"""

from __future__ import annotations

from app.agents.numeric_guard import enforce_numeric_guards


def test_right_regex_wrong_node_is_caught_and_node_prefixed():
    # 785.2억원 -> true billion value ~78.5; "-785.2 billion KRW" is 10x large.
    text = "At a -785.2 billion KRW equivalent loss the position impact is minimal."
    guarded, descriptions = enforce_numeric_guards(
        text, node="risk_debate[r1 Aggressive]", kr_block="loss=785억원",
    )
    assert len(descriptions) == 1
    assert descriptions[0].startswith("risk_debate[r1 Aggressive]: billion_10x:")
    assert "78.5 billion KRW" in guarded


def test_ratio_label_is_also_run_and_prefixed():
    text = "The firm carries a 372.75% debt-to-assets ratio."
    guarded, descriptions = enforce_numeric_guards(text, node="Market Analyst")
    assert descriptions == ["Market Analyst: debt_ratio_denominator: '372.75% debt-to-assets' -> "
                            + descriptions[0].split(" -> ", 1)[1]]
    assert "total EQUITY" in guarded


def test_clean_text_passes_through_untouched():
    text = "Revenue reached 95,154.2 billion KRW; leverage is 78.8% liabilities/assets."
    guarded, descriptions = enforce_numeric_guards(
        text, node="News + Macro Analyst", kr_block="revenue=951542억원",
    )
    assert guarded == text
    assert descriptions == []


def test_empty_text_is_a_noop():
    assert enforce_numeric_guards("", node="x") == ("", [])
    assert enforce_numeric_guards(None, node="x") == (None, [])  # type: ignore[arg-type]


def test_banner_false_suppresses_the_leading_banner():
    text = "매출 ₩625,389 billion 기록"
    guarded, descriptions = enforce_numeric_guards(
        text, node="portfolio_manager.executive_summary", kr_block="x=625389억원", banner=False,
    )
    assert "62,538.9 billion KRW" in guarded
    assert "UNIT-SCALE CORRECTION" not in guarded
    assert '[unit corrected from "' in guarded  # inline marker still present
    assert descriptions[0].startswith("portfolio_manager.executive_summary: billion_10x:")


def test_inline_note_false_gives_bare_value_for_card_text():
    text = "밸류 매력적, ₩625,389 billion 매출 성장"
    guarded, _ = enforce_numeric_guards(
        text, node="portfolio_manager.card_reason_ko", kr_block="x=625389억원",
        banner=False, inline_note=False,
    )
    assert "62,538.9 billion KRW" in guarded
    assert "[unit corrected from" not in guarded


def test_scans_out_is_keyed_by_node():
    scans: dict = {}
    enforce_numeric_guards(
        "revenue ₩625,389 billion", node="Sentiment Analyst", kr_block="x=625389억원",
        scans_out=scans,
    )
    assert "Sentiment Analyst" in scans
    assert scans["Sentiment Analyst"]["corrected"] == 1


def test_currency_check_is_off_by_default():
    # id=1207 (270520.KS/앱튼): a Fundamentals report entirely about Appen Ltd.
    # (NASDAQ: APPN) — USD cover to cover, zero KRW, while a full 억원 block was
    # supplied and ignored. The currency-composition net must stay silent
    # unless a caller explicitly opts in via `currency_check` — every OTHER
    # call site (pipeline `_numeric_turn_guard`, `_guard_fields`, the analyst
    # loop) must keep the default False.
    from app.agents.unit_scale_guard import check_unit_scale, scan_unit_scale

    eok = frozenset(float(1000 + i) for i in range(39))
    kr_block = "\n".join(f"x{i}={1000 + i}억원" for i in range(39))
    report = (
        "Appen Limited (NASDAQ: APPN) full-year revenue was $273.0 million, down "
        "from $389.4 million, with an EBITDA loss of $45.5 million and cash of "
        "$52.0 million. The placement raised $47.0 million at $1.85 per share. "
        "The analyst price target is $1.20 versus a $1.05 close; the institutional "
        "stake is $9.0 million and short interest is $6.0 million. Free cash flow "
        "was negative $31.0 million with $19.0 million of capitalised development "
        "and $12.0 million of lease liabilities. Fair value is around $0.90 per "
        "share on our estimates, and we would not add here given the dilution "
        "risk and the absence of any near-term catalyst for a re-rating of the "
        "shares over the next several reporting periods on current guidance."
    )
    # thin wrapper: no currency_check parameter at all -> silent
    assert check_unit_scale(report, eok) == []
    # scan default: judgment off, but the observation counts are still recorded
    v_default, scan_default = scan_unit_scale(report, eok)
    assert v_default == []
    assert scan_default.currency_inverted is False
    assert scan_default.usd_figures >= 3
    # composer default (no currency_check kwarg) -> silent
    _, descriptions = enforce_numeric_guards(
        report, node="Fundamentals Analyst", kr_block=kr_block,
    )
    assert descriptions == []
    # explicit opt-in flips it on
    v_on, _ = scan_unit_scale(report, eok, currency_check=True)
    assert [v.kind for v in v_on] == ["currency_inverted"]


def test_idempotent_second_pass_finds_nothing():
    text = "매출 ₩625,389 billion"
    once, _ = enforce_numeric_guards(text, node="A", kr_block="x=625389억원")
    twice, descriptions = enforce_numeric_guards(once, node="A", kr_block="x=625389억원")
    assert descriptions == []
    assert twice == once


def test_soft_eok_block_threads_to_the_unit_guard():
    # 2026-09-07 audit #3: a headline 억 amount reported 10x large is flagged
    # (detect-only) via soft_eok_block, node-prefixed like every other finding.
    news = "[20260904] 서남, 초전도 선재 공급 계약 6.6억원 (매출액대비 19.56 %)"
    report = "A ₩6.6 billion contract was announced."
    guarded, descriptions = enforce_numeric_guards(
        report, node="research_debate[r1 Bear]", kr_block="", soft_eok_block=news,
    )
    assert any("news_eok_10x" in d for d in descriptions)
    assert all(d.startswith("research_debate[r1 Bear]: ") for d in descriptions)
    assert "₩6.6 billion contract" in guarded  # not rewritten


def test_soft_eok_block_default_empty_is_a_noop():
    report = "A ₩6.6 billion contract was announced."
    a, da = enforce_numeric_guards(report, node="A", kr_block="")
    b, db = enforce_numeric_guards(report, node="A", kr_block="", soft_eok_block="")
    assert (a, da) == (b, db)
