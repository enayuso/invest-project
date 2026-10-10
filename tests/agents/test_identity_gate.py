"""app.agents.identity_gate — no network, no LLM (A-stage)."""

from __future__ import annotations

import logging
import re

from app.agents.identity_gate import (
    UNNAMED,
    ForeignListingName,
    _foreign_listing_offender,
    build_alias_set,
    check_identity,
    enforce_identity,
)


def _aliases(*names):
    return build_alias_set(*names)


def test_build_alias_set_normalizes_case_and_suffix():
    aliases = _aliases("Woori Financial Group Inc.")
    assert "woori financial group inc" in aliases
    assert "woori financial group" in aliases  # suffix-stripped form also present


def test_build_alias_set_skips_empty_names():
    # "." is punctuation REPLACED WITH A SPACE by normalization (2026-08-28 —
    # see _normalize's own docstring for why deleting it outright was wrong),
    # so "316140.KS" normalizes to "316140 ks", not "316140ks".
    aliases = _aliases("우리금융지주", None, "", "316140.KS")
    assert aliases == frozenset({"우리금융지주", "316140 ks"})


def test_check_identity_clean_report_passes():
    aliases = _aliases("우리금융지주", "316140.KS")
    report = "우리금융지주 (316140.KS)에 대한 리포트입니다. 실적이 견고합니다."
    offending, weak_only = check_identity(report, "316140.KS", aliases, language="ko")
    assert offending is None
    assert weak_only is False


def test_check_identity_flags_hallucinated_name_near_ticker():
    # The actual 2026-08-20 incident shape: a different, unrelated company's
    # name immediately preceding the correct ticker in parentheses.
    aliases = _aliases("우리금융지주", "316140.KS")
    report = "KB Financial Group (316140.KS) reported strong insurance segment growth."
    offending, weak_only = check_identity(report, "316140.KS", aliases, language="en")
    assert offending is not None
    assert "KB Financial" in offending or "Financial Group" in offending


def test_check_identity_multiword_english_alias_does_not_false_positive():
    # A single fixed-shape "[^\\s(]{2,60}?\\(TICKER\\)" regex would fail this
    # (the space inside "KB Financial" would break a no-space capture) —
    # this is the specific defect the window-substring approach avoids.
    aliases = _aliases("Woori Financial Group Inc.", "316140.KS")
    report = "Woori Financial Group Inc. (316140.KS) posted a strong quarter."
    offending, weak_only = check_identity(report, "316140.KS", aliases, language="en")
    assert offending is None


def test_check_identity_handles_full_width_parens():
    aliases = _aliases("우리금융지주", "316140.KS")
    report = "우리금융지주（316140.KS）의 실적을 살펴봅니다."
    offending, weak_only = check_identity(report, "316140.KS", aliases, language="ko")
    assert offending is None


def test_check_identity_parenthetical_english_rendering_passes():
    # The P0-2 grounding-note format: verbatim Korean name, then an English
    # rendering in a second parenthetical, then the ticker.
    aliases = _aliases("우리금융지주", "316140.KS")
    report = "우리금융지주 (Woori Financial Group) (316140.KS) delivered solid results."
    offending, weak_only = check_identity(report, "316140.KS", aliases, language="en")
    assert offending is None


def test_check_identity_detector_b_flags_missing_name_when_ticker_never_parenthesized():
    aliases = _aliases("우리금융지주", "316140.KS", "Woori Financial Group")
    report = "This company posted strong insurance segment growth this quarter."
    offending, weak_only = check_identity(report, "316140.KS", aliases, language="en")
    assert offending is not None
    # ...and it's the UNNAMED sentinel specifically — a NAME OMISSION, which
    # pipeline.py routes to identity_unnamed / a neutral note, not the
    # wrong-company path. (2026-09-01)
    assert offending == UNNAMED


def test_check_identity_detector_b_skipped_when_no_latin_alias_and_english_report():
    # Only a Korean-script alias is known (yfinance lookup failed) and the
    # report is in English — an honest English rendering would be
    # indistinguishable from a substitution, so Detector B must not fire.
    aliases = _aliases("우리금융지주", "316140.KS")
    report = "This company posted strong insurance segment growth this quarter."
    offending, weak_only = check_identity(report, "316140.KS", aliases, language="en")
    assert offending is None


def test_check_identity_weak_alias_only_is_reported():
    strong = _aliases("우리금융지주", "316140.KS")
    weak = _aliases("KB Financial Group")  # pretend yfinance itself is cross-confused
    report = "KB Financial Group (316140.KS) reported strong results."
    offending, weak_only = check_identity(report, "316140.KS", strong, weak_aliases=weak, language="en")
    assert offending is None
    assert weak_only is True


def test_enforce_identity_prepends_banner_and_reports_offending_name():
    aliases = _aliases("우리금융지주", "316140.KS")
    report = "KB Financial Group (316140.KS) reported strong results."
    fixed, offending = enforce_identity(
        report, "316140.KS", "우리금융지주", aliases, language="en", agent_name="Fundamentals Analyst",
    )
    assert offending is not None
    assert "IDENTITY CORRECTION" in fixed
    assert "우리금융지주" in fixed
    # 2026-09-11: the body's own wrong-name mention is now scrubbed (see
    # scrub_offending_name) rather than left to recur verbatim — the
    # surrounding sentence (ticker + rest of the text) is still intact.
    assert "(316140.KS) reported strong results." in fixed
    assert "우리금융지주 [name corrected from" in fixed


def test_enforce_identity_leaves_clean_report_untouched():
    aliases = _aliases("우리금융지주", "316140.KS")
    report = "우리금융지주 (316140.KS)에 대한 리포트입니다."
    fixed, offending = enforce_identity(report, "316140.KS", "우리금융지주", aliases, language="ko")
    assert offending is None
    assert fixed == report


# ---------------------------------------------------------------------------
# 2026-08-25 — the ticker/code themselves used to be valid aliases, making
# both detectors self-satisfying (the ticker appears in essentially every
# report by construction). check_identity now subtracts ticker_forms from
# whatever alias set it's given, regardless of what a caller passes.
# ---------------------------------------------------------------------------


def test_ticker_alone_never_satisfies_the_identity_check():
    # The 2026-08-20 incident shape, in three phrasings that all used to pass
    # clean because "316140"/"316140ks" were themselves members of the alias
    # set built at the pipeline call site. A real English name ("Woori
    # Financial Group") is included so Detector B's own has_latin_alias skip
    # (unrelated to this bug — see test_check_identity_detector_b_skipped_
    # when_no_latin_alias_and_english_report) doesn't mask the ticker-alone
    # bypass this test targets.
    aliases = _aliases("우리금융지주", "316140.KS", "Woori Financial Group")
    for report in (
        "KB Financial Group, listed as 316140.KS, posted strong insurance growth.",
        "The stock 316140 rose 3%. We rate it Overweight.",
        "Ticker: 316140.KS\n\nKB Financial Group (316140.KS) reported strong results.",
    ):
        offending, _ = check_identity(report, "316140.KS", aliases, language="en")
        assert offending is not None, report


def test_bare_krx_code_alone_never_satisfies_the_identity_check():
    aliases = _aliases("우리금융지주", "316140.KS", "Woori Financial Group")
    report = "The stock 316140 rose 3% today on heavy volume."
    offending, _ = check_identity(report, "316140.KS", aliases, language="en")
    assert offending is not None


def test_us_ticker_alone_never_satisfies_the_identity_check():
    aliases = _aliases("Invesco QQQ Trust", "QQQ")
    report = "QQQ gained 1.2% on the session."
    offending, _ = check_identity(report, "QQQ", aliases, language="en")
    assert offending is not None


def test_ticker_reference_after_first_grounded_mention_does_not_false_positive():
    # IDENTITY_GROUNDING_NOTE-compliant shape: name the company once up
    # front, then refer to the ticker alone later — must not flag.
    aliases = _aliases("우리금융지주", "316140.KS")
    report = (
        "우리금융지주 (Woori Financial Group) (316140.KS) delivered a solid quarter.\n\n"
        "## Valuation\nTrailing P/E stands at 5.2x versus a domestic banking peer median.\n\n"
        "## Conclusion\nWe rate the shares Overweight (316140.KS)."
    )
    offending, _ = check_identity(report, "316140.KS", aliases, language="en")
    assert offending is None


def test_wrong_name_flagged_even_with_ticker_repeated_throughout():
    aliases = _aliases("우리금융지주", "316140.KS")
    report = (
        "KB Financial Group (316140.KS) reported strong results. 316140.KS shares "
        "rallied. We rate 316140.KS Overweight."
    )
    offending, _ = check_identity(report, "316140.KS", aliases, language="en")
    assert offending is not None


def test_extract_name_guess_survives_multiple_leading_stopwords():
    from app.agents.identity_gate import _extract_name_guess

    assert _extract_name_guess("the report on KB Financial Group") == "KB Financial Group"


def test_empty_alias_set_fails_open_and_warns(caplog):
    with caplog.at_level(logging.WARNING):
        offending_en, _ = check_identity("Some report text.", "316140.KS", frozenset(), language="en")
    assert offending_en is None
    assert any("identity_gate:" in msg for msg in caplog.messages)

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        offending_ko, _ = check_identity("일부 리포트 텍스트입니다.", "316140.KS", frozenset(), language="ko")
    assert offending_ko is None


def test_enforce_identity_never_emits_a_banner_with_an_empty_display_name():
    fixed, offending = enforce_identity(
        "Some report text with no known identity.", "316140.KS", "316140.KS", frozenset(), language="en",
    )
    assert offending is None
    assert fixed == "Some report text with no known identity."


def test_weak_alias_only_pass_survives_the_any_mention_relaxation():
    strong = _aliases("우리금융지주")
    weak = _aliases("Woori Financial Group")
    report = (
        "우리금융지주 (316140.KS) delivered a solid quarter.\n\n"
        "## Conclusion\nWoori Financial Group (316140.KS) remains attractively valued."
    )
    offending, weak_only = check_identity(report, "316140.KS", strong, weak_aliases=weak, language="en")
    assert offending is None


# ---------------------------------------------------------------------------
# 2026-08-28 domestic AI audit — this gate's own false-positive rate measured
# at 98% (44/45 same-day reports) live, root-caused to `_normalize` deleting
# (not spacing) punctuation, which fused yfinance's KRX-convention
# "Co.,Ltd." into a single "coltd" token `_strip_suffix` could never
# recognize. Each case below reproduces one of the 5 sampled false positives
# (as strong/weak alias sets matching what `_company_identity` would have
# built from the same real KRX/yfinance names) plus the true misattributions
# that must remain caught. See `_normalize`/`_prefix_aliases`/
# `_abbreviated_aliases`/`_etf_brand_alias`'s own docstrings for the exact
# mechanism each fix uses.
# ---------------------------------------------------------------------------


def test_samsung_biologics_no_longer_false_positives():
    # 207940.KS — yfinance longName "Samsung Biologics Co.,Ltd." (correct,
    # no space before the comma — the exact pattern that broke suffix
    # stripping pre-fix).
    strong = _aliases("삼성바이오로직스")
    weak = _aliases("Samsung Biologics Co.,Ltd.")
    report = "Samsung Biologics (207940.KS) reported strong CDMO order backlog growth."
    offending, _ = check_identity(report, "207940.KS", strong, weak_aliases=weak, language="en")
    assert offending is None


def test_hyundai_e_and_c_no_longer_false_positives():
    # 000720.KS — needs the abbreviation table (_ABBREV_ALIASES), not just
    # suffix-stripping: "E&C" is not a prefix of "Engineering & Construction".
    strong = _aliases("현대건설")
    weak = _aliases("Hyundai Engineering & Construction Co.,Ltd.")
    report = "Hyundai E&C (000720.KS) posted a wider-than-expected overseas order backlog."
    offending, _ = check_identity(report, "000720.KS", strong, weak_aliases=weak, language="en")
    assert offending is None


def test_bitzro_tech_dropped_z_typo_no_longer_false_positives():
    # 042370.KS/비츠로테크, confirmed live 2026-09-17: both the Market Analyst
    # and News+Macro Analyst wrote "Bitro Tech" (dropped the 'z' from KIS's
    # official "Bitzro Tech") — an unusual "tz" cluster expected to recur for
    # this ticker. The fuzzy romanization fallback scored this at 0.727,
    # under `_ROMANIZATION_AMNESTY_THRESHOLD` (0.75), so it fell through to
    # the banner pre-fix. `_ABBREV_ALIASES["bitzro"] = "bitro"` must catch
    # this via the ordinary weak-alias path — built the same way
    # `app.orchestrator.pipeline._company_identity` builds its weak alias set
    # (`build_alias_set(yf_name, yf_short_name, kis_english)`) — before the
    # fuzzy romanization path is ever reached.
    strong = _aliases("비츠로테크")
    weak = _aliases("Bitzro Tech")
    report = "Bitro Tech (042370.KS) posted improved order backlog for its industrial equipment segment."
    offending, passed_via_weak = check_identity(report, "042370.KS", strong, weak_aliases=weak, language="en")
    assert offending is None
    assert passed_via_weak is True


def test_hyosung_heavy_no_longer_false_positives():
    # 298040.KS — needs prefix-alias handling: "Hyosung Heavy" is a genuine
    # leading-token prefix of "Hyosung Heavy Industries Corporation".
    strong = _aliases("효성중공업")
    weak = _aliases("Hyosung Heavy Industries Corporation")
    report = "Hyosung Heavy (298040.KS) benefited from robust power-equipment demand."
    offending, _ = check_identity(report, "298040.KS", strong, weak_aliases=weak, language="en")
    assert offending is None


def test_kodex_etf_no_longer_false_positives():
    # 379800.KS — the report paraphrases the underlying index differently
    # from yfinance's own name; only the "KODEX" brand token is common to
    # both, which _etf_brand_alias now accepts as sufficient for an ETF.
    strong = _aliases("KODEX 미국S&P500")
    weak = _aliases("Samsung KODEX S&P500 ETF")
    report = "KODEX US S&P 500 ETF (379800.KS) tracks large-cap US equities."
    offending, _ = check_identity(report, "379800.KS", strong, weak_aliases=weak, language="en")
    assert offending is None


def test_hyosung_heavy_ticker_still_catches_a_real_misattribution():
    # Same 298040.KS alias set as the false-positive test above, but this
    # report actually names a DIFFERENT real company (the 2026-08-28 audit's
    # own confusion with 267260.KS/HD현대일렉트릭) — none of the false-
    # positive fixes may relax this into a pass.
    strong = _aliases("효성중공업")
    weak = _aliases("Hyosung Heavy Industries Corporation")
    report = "HD Hyundai Electric (298040.KS) posted record transformer order intake."
    offending, _ = check_identity(report, "298040.KS", strong, weak_aliases=weak, language="en")
    assert offending is not None


def test_index_etf_mislabeled_samsung_electronics_still_caught():
    # 069500.KS/102110.KS — an index ETF's report analyzing its largest
    # constituent (Samsung Electronics) as if IT were the fund, a real
    # 2026-08-28 misattribution. _etf_brand_alias must not mask this: the
    # offending text has no ETF brand token in it, so it still fails to
    # match.
    strong = _aliases("KODEX 200")
    weak = _aliases("Samsung KODEX 200 ETF")
    report = "Samsung Electronics (069500.KS) is expected to benefit from AI-driven memory demand."
    offending, _ = check_identity(report, "069500.KS", strong, weak_aliases=weak, language="en")
    assert offending is not None


def test_hyundai_mobis_mislabeled_samsung_electronics_still_caught():
    # 012330.KS — the audit's most severe incident: a report concluded the
    # supplied data "must belong to" Samsung Electronics and wrote the
    # entire report under that name. Must still be caught regardless of the
    # false-positive fixes above.
    strong = _aliases("현대모비스")
    weak = _aliases("Hyundai Mobis Co.,Ltd.")
    report = "Fundamentals Report: Samsung Electronics (012330.KS) — the data provided is unequivocally Samsung's."
    offending, _ = check_identity(report, "012330.KS", strong, weak_aliases=weak, language="en")
    assert offending is not None


def test_alteogen_no_latin_alias_now_cleared_by_romanization_amnesty():
    # INTENTIONAL BEHAVIOR CHANGE (2026-09-04): this test used to be named
    # `test_alteogen_no_latin_alias_remains_a_known_residual_false_positive`
    # and asserted the OPPOSITE — that 196170.KS/알테오젠 stayed flagged as a
    # documented, accepted residual false positive (yfinance returned no
    # usable English name live, so the correct "AlteoGen" had no alias to
    # match). The romanization-amnesty fix targets exactly this shape:
    # "AlteoGen" scores 0.75 (at the threshold) against 알테오젠's
    # romanization skeleton, and is now cleared rather than flagged. This is
    # a deliberate, sanctioned inversion of the old assertion, not an
    # accidental edit — see the module docstring's 2026-09-04 addendum.
    strong = _aliases("알테오젠")
    report = "AlteoGen (196170.KS) advanced its ALT-B4 hyaluronidase platform licensing deals."
    offending, weak_only = check_identity(report, "196170.KS", strong, language="en")
    assert offending is None
    assert weak_only is True  # passed only via the fuzzy romanization amnesty path


# ---------------------------------------------------------------------------
# 2026-09-01 KRX audit — 10 false positives across 4 decisions. Latin<->Hangul
# spacing (_script_join) and the name-omission / misattribution split.
# ---------------------------------------------------------------------------

def test_kbi_metal_spaced_hangul_no_longer_false_positives():
    # 024840.KS — display name "KBI메탈" (KIS's own, no space); the News +
    # Macro Analyst correctly wrote "KBI 메탈" (with a space) and got flagged.
    aliases = _aliases("KBI메탈")
    report = (
        "Synthesis of Macro Data: As a KOSDAQ-listed small/mid-cap, KBI 메탈 is "
        "exposed to this broader risk-off sentiment. Small caps typically suffer more."
    )
    offending, weak_only = check_identity(report, "024840.KS", aliases, language="en")
    assert offending is None
    assert weak_only is False


def test_script_join_matches_when_report_omits_the_space_and_alias_has_it():
    # The other direction: alias "SK 하이닉스", report "SK하이닉스".
    aliases = _aliases("SK 하이닉스")
    report = "SK하이닉스 (000660.KS) posted record HBM revenue this quarter."
    offending, _ = check_identity(report, "000660.KS", aliases, language="en")
    assert offending is None


def test_etf_still_passes_via_strong_alias_after_script_join():
    # Pin against a regression where _script_join gets folded into _normalize:
    # that fuses "TIGER 미국나스닥100" into one token and _etf_brand_alias
    # loses the "tiger" brand token, dropping this from a strong pass to
    # weak-only.
    strong = _aliases("TIGER 미국나스닥100")
    weak = _aliases("Mirae Asset TIGER US Nasdaq 100 ETF")
    report = "TIGER US Nasdaq 100 ETF (379800.KS) tracks the Nasdaq-100 index."
    offending, weak_only = check_identity(report, "379800.KS", strong, weak_aliases=weak, language="en")
    assert offending is None
    assert weak_only is False  # matched a STRONG alias, not just the weak one


def test_enforce_identity_unnamed_uses_a_neutral_note_not_an_accusation():
    aliases = _aliases("우리금융지주", "316140.KS", "Woori Financial Group")
    report = "This company posted strong insurance segment growth this quarter."
    fixed, offending = enforce_identity(
        report, "316140.KS", "우리금융지주", aliases, language="en", agent_name="News + Macro Analyst"
    )
    assert offending == UNNAMED
    assert "IDENTITY CORRECTION" not in fixed
    assert "misnamed" not in fixed
    assert "disregard" not in fixed
    assert "우리금융지주" in fixed          # the identity is still supplied
    assert report in fixed                  # original content untouched


def test_enforce_identity_wrong_company_still_gets_the_correction_banner():
    # The 2026-08-20 flagship shape — must be unchanged by the omission split.
    aliases = _aliases("우리금융지주", "316140.KS")
    report = "KB Financial Group (316140.KS) posted strong growth."
    fixed, offending = enforce_identity(report, "316140.KS", "우리금융지주", aliases, language="en")
    assert offending is not None and offending != UNNAMED
    assert "IDENTITY CORRECTION" in fixed


def test_enforce_identity_banner_no_longer_claims_content_is_verified():
    # 2026-09-07, id=1237: the old wording ("disregard the wrong name only,
    # not the content") told downstream readers the report's CONTENT was
    # still trustworthy — it wasn't (business_description_guard exists
    # because of this). The new wording must instead say a business/industry
    # description is NOT verified.
    aliases = _aliases("우리금융지주", "316140.KS")
    report = "KB Financial Group (316140.KS) posted strong growth."
    fixed, offending = enforce_identity(report, "316140.KS", "우리금융지주", aliases, language="en")
    assert offending is not None
    assert "NOT verified" in fixed
    assert "disregard the wrong name only, not the content" not in fixed

    fixed_ko, offending_ko = enforce_identity(report, "316140.KS", "우리금융지주", aliases, language="ko")
    assert offending_ko is not None
    assert "업종·사업 설명은 근거로 사용하지 마십시오" in fixed_ko


# ---------------------------------------------------------------------------
# 2026-09-03 audit #6 (id=1193): 142760.KS/모아라이프플러스 was mislabeled
# "한국전자기술" (a completely different company) and the gate reported
# identity_violations=[]. Two independent root causes:
#  (1-A) Detector B's Latin-alias skip looked only at whether the ALIAS set
#        had a Latin entry, not whether the REPORT itself was Latin-script —
#        fixed by checking for a name-shaped Hangul run in the report before
#        falling back to that skip.
#  (1-B) yfinance's stale pre-rename name ("BL Corp") produced a 2-char weak
#        alias "bl" that matched as a raw substring inside ordinary English
#        words ("problem", "table") — fixed by requiring token-boundary
#        matching for any Latin/digit-only alias (`_alias_in`).
# ---------------------------------------------------------------------------


def test_detector_b_flags_a_wrong_hangul_name_when_only_hangul_aliases_exist():
    aliases = _aliases("모아라이프플러스")
    report = "**한국전자기술** is in a dangerous technical posture and momentum is fading. SELL/AVOID."
    offending, weak_only = check_identity(report, "142760.KS", aliases, language="en")
    assert offending is not None
    assert offending != UNNAMED
    assert "한국전자기술" in offending


def test_detector_b_still_skips_a_pure_ascii_report_with_no_latin_alias():
    # 196170.KS/알테오젠-shaped case must still degrade to a skip, not a
    # false positive, when the report never writes any Hangul at all.
    aliases = _aliases("알테오젠")
    report = "This company advanced international licensing deals with several new distribution partners."
    offending, _ = check_identity(report, "196170.KS", aliases, language="en")
    assert offending is None


def test_detector_b_hangul_prose_and_units_are_not_name_candidates():
    # 억원/조원/매수/한글 조사가 붙은 평범한 서술은 "이름 후보"가 아니어야 하고,
    # 그 결과 오귀속이 아니라 기존 UNNAMED 경로로 남아야 한다.
    aliases = _aliases("모아라이프플러스")
    report = "150억원 규모의 매수세가 유입되었습니다. 3조원 시장에서 리포트를 생성하지 못했습니다."
    offending, _ = check_identity(report, "142760.KS", aliases, language="ko")
    assert offending == UNNAMED


def test_short_latin_weak_alias_does_not_match_inside_an_english_word():
    # yfinance's stale "BL Corp" (pre-rename) produces the 2-char weak alias
    # "bl" — must not match inside "problem"/"table".
    weak = _aliases("BL Corp")
    report = "This is a problem that shows up in the data table below, not the company itself."
    offending, _ = check_identity(report, "999999.KS", frozenset(), weak_aliases=weak, language="en")
    assert offending is not None  # must not silently pass


def test_latin_alias_still_matches_at_token_boundaries():
    # Guard against over-tightening: existing token-shaped aliases (an
    # abbreviation, an ETF brand token, an "&"-joined name) must still match
    # inside a normal, correctly-written report.
    strong = _aliases("Hyundai Engineering & Construction Co.,Ltd.")
    report = "Hyundai E&C (000720.KS) posted a wider-than-expected overseas order backlog."
    offending, _ = check_identity(report, "000720.KS", strong, language="en")
    assert offending is None

    strong = _aliases("KODEX 미국S&P500")
    weak = _aliases("Samsung KODEX S&P500 ETF")
    report = "KODEX US S&P 500 ETF (379800.KS) tracks large-cap US equities."
    offending, _ = check_identity(report, "379800.KS", strong, weak_aliases=weak, language="en")
    assert offending is None

    strong = _aliases("KODEX 200")
    weak = _aliases("Samsung KODEX 200 ETF")
    report = "KODEX 200 (069500.KS) closely tracks the KOSPI 200 index."
    offending, _ = check_identity(report, "069500.KS", strong, weak_aliases=weak, language="en")
    assert offending is None


def test_prefixed_correct_hangul_name_still_passes():
    # Hangul aliases keep raw substring containment (no boundary-matching) —
    # "GC녹십자웰빙" must still contain "녹십자웰빙".
    aliases = _aliases("녹십자웰빙")
    report = "GC녹십자웰빙에 대한 리포트입니다. 실적이 견고합니다."
    offending, _ = check_identity(report, "034300.KS", aliases, language="ko")
    assert offending is None


def test_lotte_wellbeing_misattribution_still_caught():
    # Regression guard: the false-positive relief above must not also mask a
    # genuine misattribution to a real, differently-named company.
    strong = _aliases("녹십자웰빙")
    weak = _aliases("GC Wellbeing")
    report = "**Fundamentals Report: Lotte Wellbeing Co., Ltd. (234690.KS)**\n\nRevenue grew steadily this quarter."
    offending, _ = check_identity(report, "234690.KS", strong, weak_aliases=weak, language="en")
    assert offending is not None and offending != UNNAMED


# ---------------------------------------------------------------------------
# 2026-09-03 (id=1207, 270520.KS/앱튼) — Detector C: foreign-exchange-listing
# misattribution. The Fundamentals Analyst wrote an entire report for Appen
# Ltd. (APPN, NASDAQ/formerly ASX), a completely different company on a
# completely different exchange. Neither Detector A (no "(270520.KS)" mention
# to check a window before) nor the Hangul-candidate check (the report is
# pure English) can see this — it fell through to UNNAMED, a false "no name
# given" when a name (the wrong one) was given repeatedly and in detail.
# ---------------------------------------------------------------------------

_APPEN_REPORT = (
    "**Fundamentals Report: Appen Ltd. (APPN)**\n"
    "*Note: The ticker symbol APPN refers to Appen Limited, an "
    "Australian-founded AI data company now traded on NASDAQ. Appen was "
    "previously listed on the Australian Securities Exchange.*\n\n"
    "Q2 2026 Revenue: $48.2 million. Cash & Equivalents: $112 million. "
    "Shareholders' Equity: $290 million. Goldman Sachs set a price target "
    "of $1.80."
)


def test_detector_c_flags_a_foreign_listing_report_id_1207():
    strong = build_alias_set("앱튼")
    weak = build_alias_set("G Enone Energy Co Ltd") - strong
    offending, _ = check_identity(_APPEN_REPORT, "270520.KS", strong, weak_aliases=weak, language="en")
    assert offending is not None
    assert offending != UNNAMED
    assert isinstance(offending, ForeignListingName)
    assert offending.symbol == "APPN"


def test_enforce_identity_foreign_listing_banner_is_not_the_unnamed_note():
    strong = build_alias_set("앱튼")
    weak = build_alias_set("G Enone Energy Co Ltd") - strong
    fixed, offending = enforce_identity(
        _APPEN_REPORT, "270520.KS", "앱튼", strong, weak_aliases=weak, language="en",
        agent_name="Fundamentals Analyst",
    )
    assert offending is not None
    assert "IDENTITY NOTE" not in fixed
    assert "does not state a company name" not in fixed
    assert "DIFFERENT exchange" in fixed
    assert "앱튼" in fixed
    assert _APPEN_REPORT in fixed


def test_detector_c_never_fires_for_a_us_ticker():
    # Legitimately foreign tickers (AAPL, QQQ, ...) are constantly awash in
    # USD figures and NASDAQ mentions — Detector C must never even look at
    # them, or it would flag every one of them.
    aliases = build_alias_set("Apple Inc.")
    report = (
        "Apple Inc. (AAPL) trades on NASDAQ. Q2 2026 Revenue: $94.8 billion. "
        "Cash & Equivalents: $61 million. Shareholders' Equity: $74 billion. "
        "Price target: $250."
    )
    assert _foreign_listing_offender(report, "AAPL", aliases) is None
    assert _foreign_listing_offender(_APPEN_REPORT, "AAPL", aliases) is None


def test_detector_c_ignores_a_peer_mention_with_krw_figures_present():
    # A single USD-denominated peer comparison inside an otherwise
    # KRW-denominated KRX report is normal equity-research prose.
    aliases = build_alias_set("삼성전자")
    report = (
        "삼성전자 (005930.KS) is expected to benefit from AI-driven memory "
        "demand. Peer Micron (NASDAQ: MU) trades near $95, but Samsung's own "
        "shares closed at ₩71,000 today."
    )
    assert _foreign_listing_offender(report, "005930.KS", aliases) is None


def test_detector_c_ignores_a_mid_body_peer_symbol_with_no_krw():
    # No KRW figure at all, but the peer symbol is mentioned mid-body (past
    # the subject-declaring head region) with no exchange qualifier — not
    # "subject", so it must not be flagged.
    aliases = build_alias_set("삼성전자")
    padding = "실적 전망이 매우 밝고 견조한 흐름이 이어지고 있습니다. " * 6
    report = (
        f"삼성전자 (005930.KS)에 대한 상세 리포트입니다. {padding}"
        "한편 경쟁사 Micron (MU)의 실적도 긍정적인 흐름을 보이고 있습니다."
    )
    assert len(report) > 250  # sanity: the peer mention really is past the head
    assert _foreign_listing_offender(report, "005930.KS", aliases) is None


def test_detector_c_skips_a_korean_etf_tracking_a_foreign_index():
    # A KODEX/TIGER-branded ETF holding Apple/Nvidia is legitimately full of
    # USD-denominated foreign names — carved out via the ETF brand token.
    strong = build_alias_set("KODEX 미국나스닥100")
    report = (
        "KODEX US Nasdaq 100 ETF (379810.KS) holds Apple (AAPL) $230 and "
        "Nvidia (NVDA) as its top constituents, both trading on NASDAQ."
    )
    assert _foreign_listing_offender(report, "379810.KS", strong) is None


def test_detector_c_ignores_financial_acronyms_in_parentheses():
    aliases = build_alias_set("삼성전자")
    report = (
        "삼성전자 (005930.KS)에 대한 리포트입니다. Operating profit remained solid "
        "on an (EBIT) basis, with (ROE) improving and a five-year (CAGR) of 8%. "
        "No pending (FDA) filings are relevant to this business."
    )
    assert _foreign_listing_offender(report, "005930.KS", aliases) is None


def test_detector_c_flags_a_wrong_krx_code_declared_in_the_header():
    aliases = build_alias_set("앱튼")
    report = (
        "**Fundamentals Report: 삼성전자 (005930.KS)**\n\n"
        "Revenue grew steadily this quarter driven by memory demand."
    )
    offending = _foreign_listing_offender(report, "270520.KS", aliases)
    assert offending is not None
    assert isinstance(offending, ForeignListingName)
    assert offending.symbol == "005930"


def test_detector_c_does_not_fire_on_a_wrong_krx_code_mentioned_only_as_a_peer():
    aliases = build_alias_set("앱튼")
    padding = "이 회사의 최근 실적 흐름은 견조하며 향후 전망도 밝습니다. " * 6
    report = (
        f"앱튼 (270520.KS)에 대한 리포트입니다. {padding}"
        "참고로 동종업계 삼성전자 (005930.KS)의 실적도 양호합니다."
    )
    assert len(report) > 250
    assert _foreign_listing_offender(report, "270520.KS", aliases) is None


# ---------------------------------------------------------------------------
# 2026-09-04 — romanization amnesty. Live measurement: ~50% (13/26 recent)
# domestic decisions carried a false-positive violation banner because the
# analyst correctly Romanized/translated the Korean name into English (e.g.
# 파인텍->"FineTech") and no Latin alias had ever been sourced for it.
# `_romanization_plausible` recognizes this narrow shape via a fuzzy
# phonetic-skeleton comparison, gated by `_romanization_amnesty_guard_ok`
# (KRX-only, zero Hangul in the offending text, no existing authoritative
# Latin alias, not an ETF brand token) — see the module docstring's
# 2026-09-04 addendum for the full rationale.
# ---------------------------------------------------------------------------


def test_romanization_amnesty_clears_laon_people_id_1213():
    aliases = _aliases("라온피플", "300120.KS")
    report = "Laon People (300120.KS) posted strong quarterly earnings growth."
    offending, weak_only = check_identity(report, "300120.KS", aliases, language="en")
    assert offending is None
    assert weak_only is True


def test_romanization_amnesty_clears_topcomedia_id_1208():
    aliases = _aliases("탑코미디어", "900001.KS")
    report = "Topcomedia (900001.KS) reported a rebound in advertising revenue this quarter."
    offending, weak_only = check_identity(report, "900001.KS", aliases, language="en")
    assert offending is None
    assert weak_only is True


def test_romanization_amnesty_clears_finetech_id_1210():
    aliases = _aliases("파인텍", "900002.KS")
    report = "FineTech (900002.KS) posted steady order growth in its display equipment segment."
    offending, weak_only = check_identity(report, "900002.KS", aliases, language="en")
    assert offending is None
    assert weak_only is True


def test_romanization_amnesty_clears_tinkware_id_1216():
    aliases = _aliases("팅크웨어", "900003.KS")
    report = "Tinkware (900003.KS) saw firmware licensing revenue climb this quarter."
    offending, weak_only = check_identity(report, "900003.KS", aliases, language="en")
    assert offending is None
    assert weak_only is True


def test_romanization_amnesty_clears_metaplus_id_1230():
    aliases = _aliases("메타랩스", "900004.KS")
    report = "Metaplus (900004.KS) expanded its enterprise VR licensing partnerships."
    offending, weak_only = check_identity(report, "900004.KS", aliases, language="en")
    assert offending is None
    assert weak_only is True


def test_romanization_amnesty_does_not_clear_the_2026_08_20_flagship():
    # THE mandatory guard test — byte-identical fixture to
    # test_check_identity_flags_hallucinated_name_near_ticker. "KB Financial
    # Group" is a real, different, wrong company, not a phonetic rendering of
    # "우리금융지주" — this must stay flagged after the amnesty fix exactly as
    # it did before.
    aliases = _aliases("우리금융지주", "316140.KS")
    report = "KB Financial Group (316140.KS) reported strong insurance segment growth."
    offending, weak_only = check_identity(report, "316140.KS", aliases, language="en")
    assert offending is not None
    assert "KB Financial" in offending or "Financial Group" in offending


def test_romanization_amnesty_does_not_clear_hyundai_mobis_samsung_electronics():
    # 012330.KS — the 2026-08-28 audit's most severe incident: a report
    # concluded the data "must belong to" Samsung Electronics. Must still be
    # caught with amnesty wired in.
    strong = _aliases("현대모비스")
    weak = _aliases("Hyundai Mobis Co.,Ltd.")
    report = "Fundamentals Report: Samsung Electronics (012330.KS) — the data provided is unequivocally Samsung's."
    offending, _ = check_identity(report, "012330.KS", strong, weak_aliases=weak, language="en")
    assert offending is not None


def test_romanization_amnesty_does_not_clear_index_etf_samsung_electronics():
    # 069500.KS/102110.KS — an index ETF's report analyzing its largest
    # constituent (Samsung Electronics) as if IT were the fund.
    strong = _aliases("KODEX 200")
    weak = _aliases("Samsung KODEX 200 ETF")
    report = "Samsung Electronics (069500.KS) is expected to benefit from AI-driven memory demand."
    offending, _ = check_identity(report, "069500.KS", strong, weak_aliases=weak, language="en")
    assert offending is not None


def test_romanization_amnesty_never_applies_to_a_hangul_offender():
    # 두산퓨얼셀 (336260.KS) mislabeled 범한퓨얼셀 scores 0.71 phonetic
    # similarity — both real, different companies, both Hangul-vs-Hangul.
    # Amnesty must never apply regardless of score: this is caught upstream
    # by the existing Hangul-name-candidate check (audit #6, id=1193), which
    # runs before Detector C/amnesty are ever consulted. (The other measured
    # danger case, 삼성공조/006660.KS mislabeled 삼성전자 at 0.86 similarity,
    # is exercised directly against the guard function below instead of via
    # check_identity's full Hangul-candidate pipeline — "삼성전자" happens to
    # end in a syllable, "자", that a pre-existing, unrelated heuristic in
    # `_hangul_name_candidates` filters as a Korean particle ending, which
    # would make this test about that heuristic rather than about the
    # romanization-amnesty guard it's meant to pin.)
    aliases = _aliases("두산퓨얼셀")
    report = "**범한퓨얼셀**은 최근 수주 모멘텀이 강화되고 있으며 목표주가를 상향합니다."
    offending, _ = check_identity(report, "336260.KS", aliases, language="ko")
    assert offending is not None
    assert offending != UNNAMED
    assert "범한퓨얼셀" in offending


def test_romanization_amnesty_guard_rejects_hangul_offender_text_directly():
    from app.agents.identity_gate import _romanization_amnesty_guard_ok

    aliases = build_alias_set("삼성공조")
    assert _romanization_amnesty_guard_ok("006660.KS", "삼성전자", aliases, aliases) is False


def test_romanization_amnesty_skipped_when_a_latin_authoritative_alias_exists():
    # A confirmed authoritative Latin alias ("AlteoGen") is present — a
    # mismatch against it is meaningful and must not be waved through by
    # amnesty just because the wrong name is also phonetically plausible.
    strong = _aliases("알테오젠", "AlteoGen")
    report = "Altezza Bio (196170.KS) advanced its hyaluronidase licensing platform."
    offending, _ = check_identity(report, "196170.KS", strong, language="en")
    assert offending is not None


def test_romanization_amnesty_runs_after_detector_c():
    # id=1207 replay — a report entirely about a different exchange's
    # listing (Appen Ltd. / APPN) must be caught by Detector C, never
    # reaching (or being masked by) the romanization amnesty path.
    strong = build_alias_set("앱튼")
    weak = build_alias_set("G Enone Energy Co Ltd") - strong
    offending, weak_only = check_identity(_APPEN_REPORT, "270520.KS", strong, weak_aliases=weak, language="en")
    assert isinstance(offending, ForeignListingName)
    assert offending.symbol == "APPN"
    assert weak_only is False


def test_romanization_plausible_matches_a_genuine_romanization():
    from app.agents.identity_gate import _romanization_plausible

    assert _romanization_plausible(["라온피플"], "Laon People posted strong earnings.") is True


def test_romanization_plausible_enforces_minimum_skeleton_length():
    from app.agents.identity_gate import _romanization_plausible

    # A short Korean skeleton (코스 -> "KS") must never fuzzy-match a short
    # unrelated word just because both skeletons are tiny.
    assert _romanization_plausible(["코스"], "KS fund tracks the broad index.") is False


def test_romanization_plausible_kodex_200_does_not_match_index():
    from app.agents.identity_gate import _romanization_plausible

    # The real measured false-amnesty risk this guards against: a short ETF
    # name's skeleton fuzzy-matching an unrelated short word like "index".
    assert _romanization_plausible(["코덱스 200"], "This fund benchmarks a broad index of many equities.") is False


# ---------------------------------------------------------------------------
# 2026-09-14 — short-skeleton romanization-level fallback. The skeleton path
# above requires both skeletons >= _ROMANIZATION_SKELETON_MIN_LEN (4), which
# structurally blocks every short Korean name from ever getting amnesty no
# matter how perfect the romanization (서남 -> "seonam" -> skeleton "SNM" is
# only 3 chars). For a Korean name whose skeleton is that short,
# `_romanization_plausible` now also compares at the romanization level
# (vowels retained) at `_ROMANIZATION_SHORT_SKELETON_THRESHOLD` = 0.80. All
# pairs below are real live (ticker, Korean name, offending span) pairs from
# the 143-pair measurement referenced in the module docstring's 2026-09-14
# addendum.
# ---------------------------------------------------------------------------


def test_romanization_plausible_clears_short_korean_names_now():
    from app.agents.identity_gate import _romanization_plausible

    must_clear = [
        ("서산", "Seosan"),
        ("서남", "Seonam"),
        ("혜인", "Hyein"),
        ("제닉", "Jenik"),
        ("진영", "Jin Yeong"),
        ("누보", "Nubo"),
        ("나무가", "Namugha"),
        ("나무가", "Namo-ga"),
        ("베셀", "Bessel"),
        ("베셀", "Beasel"),
        ("피노", "Pinon"),
        ("카프로", "Kapro"),
        ("티움바이오", "Tiom Bio"),  # 0.800 -- the exact boundary, pinned
    ]
    for korean_name, candidate in must_clear:
        assert _romanization_plausible([korean_name], candidate) is True, (korean_name, candidate)


def test_romanization_plausible_short_korean_name_fallback_still_blocks_hallucinations():
    from app.agents.identity_gate import _romanization_plausible

    # The decisive control: 서남/"Seonam" (above) must now clear, but
    # 서남/"Seonam Pharmaceutical" is the REAL id=1237 hallucination and
    # shares the same short Korean name -- it must stay blocked. The
    # trailing-n-gram candidate construction (anchored to the END of the
    # text, not slid across every position) is what tells these two spans
    # apart: with only "Seonam Pharmaceutical" as input, the n=1 trailing
    # window is "Pharmaceutical" (scores low) and the n=2 trailing window is
    # the full "Seonam Pharmaceutical" (scores 0.462) -- neither n isolates
    # "Seonam" alone the way an all-positions search would.
    assert _romanization_plausible(["서남"], "Seonam Pharmaceutical") is False

    must_stay_blocked = [
        ("유티아이", "Utility AI"),  # 0.667 -- highest genuine hallucination in the dataset
        ("샘표", "Samuel Cellophane"),
        ("우리바이오", "Silla Pharmaceutical"),
    ]
    for korean_name, candidate in must_stay_blocked:
        assert _romanization_plausible([korean_name], candidate) is False, (korean_name, candidate)

    # 누보/"Nobo" (0.750) is a correct variant that stays blocked at this
    # threshold -- a known, accepted residual (see the module docstring's
    # 2026-09-14 addendum). NOT chased by lowering the threshold, since that
    # would erode the margin above 유티아이/"Utility AI"'s 0.667.
    assert _romanization_plausible(["누보"], "Nobo") is False


def test_romanization_plausible_short_skeleton_fallback_never_reached_by_long_skeleton_flagship():
    from app.agents.identity_gate import _romanization_plausible

    # 우리금융지주's skeleton is long, so the new short-skeleton branch must
    # never even apply to it -- the 2026-08-20 flagship incident must stay
    # exactly as blocked as it always has been.
    assert _romanization_plausible(["우리금융지주"], "KB Financial Group") is False


# ---------------------------------------------------------------------------
# 2026-09-14 (second pass) -- end-to-end through check_identity, not just
# direct _romanization_plausible calls. A prior round of this fix passed
# every unit test above (they hand _romanization_plausible pre-cleaned spans
# like "Seonam") but was a NO-OP in production: check_identity's real
# Detector A input is `first_ungrounded_window`, the raw ~90 characters of
# report prose immediately before the "(TICKER)" mention (e.g. "**Equity
# Research Note: Seonam "), not a bare candidate string. These tests exercise
# that real path so a future regression here can't hide behind a
# too-narrow unit test again.
# ---------------------------------------------------------------------------


def test_check_identity_clears_seonam_written_correctly_end_to_end():
    aliases = _aliases("서남", "294630.KS")
    report = (
        "**Equity Research Note: Seonam (294630.KS)**\n\n"
        "Seonam is a superconducting wire manufacturer with a strong order backlog."
    )
    offending, weak_only = check_identity(
        report, "294630.KS", aliases, weak_aliases=frozenset(), language="en",
    )
    assert offending is None
    assert weak_only is True  # only cleared via the fuzzy romanization amnesty path


def test_check_identity_still_flags_seonam_pharmaceutical_end_to_end():
    # The decisive control: same Korean name (서남), same ticker, same
    # report shape -- but the offending span names a DIFFERENT, wrong
    # company ("Seonam Pharmaceutical", the real id=1237 hallucination). The
    # trailing-n-gram fallback must not clear this just because it shares a
    # root with the correct short name.
    aliases = _aliases("서남", "294630.KS")
    report = (
        "**Equity Research Note: Seonam Pharmaceutical (294630.KS)**\n\n"
        "Seonam Pharmaceutical is a superconducting wire manufacturer with a strong order backlog."
    )
    offending, weak_only = check_identity(
        report, "294630.KS", aliases, weak_aliases=frozenset(), language="en",
    )
    assert offending is not None
    assert weak_only is False


def test_check_identity_clears_jin_yeong_with_corporate_suffix_end_to_end():
    # Corporate-suffix trimming: "Jin Yeong Co., Ltd." must reduce to
    # "jinyeong" (dropping "Co.," and "Ltd.") before the trailing n-gram
    # comparison runs, or the added suffix tokens dilute the ratio below
    # threshold.
    aliases = _aliases("진영", "999999.KS")
    report = (
        "**Fundamentals Report: Jin Yeong Co., Ltd. (999999.KS)**\n\n"
        "Jin Yeong Co., Ltd. reported stable margins this quarter."
    )
    offending, weak_only = check_identity(
        report, "999999.KS", aliases, weak_aliases=frozenset(), language="en",
    )
    assert offending is None
    assert weak_only is True


# ---------------------------------------------------------------------------
# 2026-09-04 — KIS English name, exact-despaced amnesty gate. Ahead of the
# phonetic fallback: KIS's own authoritative prdt_eng_abrv_name, matched as
# an EXACT substring after reducing both sides to [a-z0-9] only. Never fuzzy
# (the flagship "KB Financial Group" vs KIS "WooriFinancialGroup" skeletons
# score 0.777, over the amnesty threshold).
# ---------------------------------------------------------------------------


def test_despace_alnum_lower_and_kis_english_name_present_units():
    from app.agents.identity_gate import _despace_alnum_lower, _kis_english_name_present

    assert _despace_alnum_lower("Laon People") == "laonpeople"
    assert _kis_english_name_present("LaonPeople", "... about Laon People (300120.KS) ...") is True
    assert _kis_english_name_present("", "Laon People report") is False
    # needle "spg" is 3 alnum chars, below _KIS_ENGLISH_MIN_LEN
    assert _kis_english_name_present("SPG", "the SPG report") is False


def test_kis_english_name_clears_detector_a_false_positive(caplog):
    # Ungrounded "(ticker)" window naming "Laon People" for a KRX ticker with
    # NO authoritative Latin alias — KIS's own English name clears it.
    aliases = _aliases("라온피플", "300120.KS")
    report = "Laon People (300120.KS) posted strong quarterly earnings growth."
    with caplog.at_level(logging.INFO, logger="app.agents.identity_gate"):
        offending, weak_only = check_identity(
            report, "300120.KS", aliases, language="en", kis_english_name="LaonPeople"
        )
    assert offending is None
    assert weak_only is True
    assert any("KIS English name" in r.message for r in caplog.records)


def test_kis_english_name_clears_detector_b_false_positive():
    # Pure-English prose about "Topcomedia", no "(ticker)" mention anywhere,
    # KRX ticker, no authoritative Latin alias. Despaced "topcomedia" ==
    # despaced KIS "Topco Media".
    aliases = _aliases("탑코미디어", "900001.KS")
    report = "Topco Media is a Korean advertising and content company that rebounded this quarter."
    offending, weak_only = check_identity(
        report, "900001.KS", aliases, language="en", kis_english_name="Topco Media"
    )
    assert offending is None
    assert weak_only is True


def test_kis_english_name_does_not_clear_the_flagship_wrong_company():
    # THE critical guard. 316140.KS — report names the real, different, WRONG
    # company "KB Financial Group" next to the correct ticker. KIS's own
    # English name for 316140 is "WooriFinancialGroup". An EXACT despaced
    # check ("kbfinancialgroup" is not a substring of anything containing
    # "woorifinancialgroup") leaves this flagged; a fuzzy one (skeleton
    # similarity 0.777 > 0.75) would wrongly clear it.
    aliases = _aliases("우리금융지주", "316140.KS")
    report = "KB Financial Group (316140.KS) reported strong insurance segment growth."
    offending, weak_only = check_identity(
        report, "316140.KS", aliases, language="en", kis_english_name="WooriFinancialGroup"
    )
    assert offending is not None
    assert "KB Financial" in offending or "Financial Group" in offending
    checked, banner_offending = enforce_identity(
        report, "316140.KS", "우리금융지주", aliases, language="en",
        kis_english_name="WooriFinancialGroup",
    )
    assert banner_offending is not None
    assert checked.startswith("⚠️")


def test_kis_english_name_ignored_when_candidate_text_contains_hangul():
    # A report writing a DIFFERENT Hangul name must stay flagged even with
    # kis_english_name set — the romanization-amnesty guard rejects any
    # Hangul-bearing offending text before the KIS check can run.
    aliases = _aliases("두산퓨얼셀")
    report = "**범한퓨얼셀**은 최근 수주 모멘텀이 강화되고 있으며 목표주가를 상향합니다."
    offending, _ = check_identity(
        report, "336260.KS", aliases, language="ko", kis_english_name="DoosanFuelCell"
    )
    assert offending is not None
    assert "범한퓨얼셀" in offending


def test_kis_english_name_ignored_when_an_authoritative_latin_alias_exists():
    # A confirmed authoritative Latin alias ("AlteoGen") present — a mismatch
    # against it is meaningful, so the guard blocks the KIS check too.
    strong = _aliases("알테오젠", "AlteoGen")
    report = "Altezza Bio (196170.KS) advanced its hyaluronidase licensing platform."
    offending, _ = check_identity(
        report, "196170.KS", strong, language="en", kis_english_name="AlteoGen"
    )
    assert offending is not None


def test_kis_english_name_default_empty_leaves_existing_behaviour_unchanged():
    # The flagship still flags with the param at its "" default.
    aliases = _aliases("우리금융지주", "316140.KS")
    report = "KB Financial Group (316140.KS) reported strong insurance segment growth."
    offending, _ = check_identity(report, "316140.KS", aliases, language="en")
    assert offending is not None


# ---------------------------------------------------------------------------
# 2026-09-10 (id=1376, 080220.KS/제주반도체) — semantic-translation amnesty.
# Distinct from the phonetic romanization amnesty above: "Jeju Semiconductor"
# is an honest SEMANTIC TRANSLATION of 반도체 ("semiconductor"), not a
# phonetic transliteration ("bandoche") — `_romanization_plausible`'s
# phonetic-skeleton comparison structurally cannot bridge that, so all three
# analyst reports for this real, executed Buy order false-positived. See the
# module comment block above `_KR_SEMANTIC_EN_AMNESTY` for the full
# rationale, including why the 2026-08-20 flagship (316140.KS/"KB Financial
# Group") must never be cleared by this even though "금융"->"financial" is
# in the table.
# ---------------------------------------------------------------------------


def test_semantic_translation_amnesty_accepts_jeju_semiconductor_market_analyst():
    aliases = _aliases("제주반도체", "080220.KS")
    report = (
        "**Technical Analysis Report: Jeju Semiconductor (080220.KS)**\n"
        "**Date:** September 10, 2026"
    )
    fixed, offending = enforce_identity(report, "080220.KS", "제주반도체", aliases, language="en")
    assert offending is None
    assert "IDENTITY CORRECTION" not in fixed


def test_semantic_translation_amnesty_accepts_jeju_semiconductor_fundamentals_analyst():
    aliases = _aliases("제주반도체", "080220.KS")
    report = (
        "**Fundamentals Report: Jeju Semiconductor (080220.KS)**\n"
        "*Date: September 10, 2026*"
    )
    fixed, offending = enforce_identity(report, "080220.KS", "제주반도체", aliases, language="en")
    assert offending is None
    assert "IDENTITY CORRECTION" not in fixed


def test_semantic_translation_amnesty_accepts_jeju_semiconductor_news_macro_analyst():
    aliases = _aliases("제주반도체", "080220.KS")
    report = "**Analysis of Jeju Semiconductor (080220.KS)**"
    fixed, offending = enforce_identity(report, "080220.KS", "제주반도체", aliases, language="en")
    assert offending is None
    assert "IDENTITY CORRECTION" not in fixed


def test_semantic_translation_amnesty_still_rejects_the_2026_08_20_flagship_incident():
    # THE mandatory negative case — "금융"->"financial"/"finance" IS in the
    # table, and the offending text DOES contain "Financial", so the EN-word
    # gate alone would pass. The remainder check must still reject it: the
    # remainder of "우리금융지주" with "금융" removed is "우리지주", which
    # romanizes to something with no relation to "kbfinancialgroup" at all.
    aliases = build_alias_set("우리금융지주", "316140.KS")
    report = "KB Financial Group (316140.KS) reported strong insurance segment growth."
    offending, _ = check_identity(report, "316140.KS", aliases, language="en")
    assert offending is not None


def test_semantic_translation_amnesty_rejects_when_neither_word_nor_remainder_match():
    from app.agents.identity_gate import _semantic_translation_plausible

    # A Korean name containing a table key ("화학"), but the candidate text
    # is an unrelated wrong company name that contains neither the EN
    # translation word nor a romanized-remainder match.
    assert _semantic_translation_plausible(
        ["한화화학"], "Some Other Company Ltd. reported strong quarterly results."
    ) is False


def test_semantic_translation_amnesty_direct_unit_whole_name_is_the_semantic_word():
    from app.agents.identity_gate import _semantic_translation_plausible

    # Remainder is empty — the whole registered name IS the semantic word —
    # accepted outright, no substring check needed.
    assert _semantic_translation_plausible(["반도체"], "Semiconductor Inc. posted a strong quarter.") is True


# ---------------------------------------------------------------------------
# 2026-09-11 — scrub_offending_name: the banner alone never touched the
# report BODY, so the wrong name kept recurring verbatim after it (real
# 2026-09-11 transcripts id=1389/320000.KS and id=1396/003350.KS, both
# "Samsung Electronics" repeated 4+ times after the banner). See this
# module's own docstring on scrub_offending_name for the full design.
# ---------------------------------------------------------------------------

from app.agents.identity_gate import scrub_offending_name  # noqa: E402


def test_id_1389_shape_samsung_electronics_scrubbed_from_body():
    display_name = "한울반도체"
    report = (
        "### Technical Analysis: Samsung Electronics (320000.KS)\n\n"
        "Samsung Electronics has shown resilient price action this week. "
        "Samsung Electronics' trading volume remains elevated. Analysts "
        "covering Samsung Electronics note strong momentum. In summary, "
        "Samsung Electronics remains a technically strong name."
    )
    aliases = _aliases(display_name, "320000.KS")
    fixed, offending_out = enforce_identity(
        report, "320000.KS", display_name, aliases, language="en", agent_name="Technical Analyst",
    )
    assert offending_out is not None
    assert '[name corrected from "' in fixed
    # Reconstruct the exact banner enforce_identity prepended (it embeds the
    # raw, unscrubbed offending guess verbatim by design — see the "before
    # prepending the banner" ordering in enforce_identity's general-case
    # branch) so we can isolate the scrubbed BODY that follows it and check
    # that only the marker's own quoted echo mentions the wrong name there.
    from app.agents.identity_gate import _BANNER

    banner = _BANNER["en"].format(offending=offending_out, ticker="320000.KS", display_name=display_name)
    assert fixed.startswith(banner)
    body = fixed[len(banner):]
    remaining = re.sub(r'\[name corrected from "[^"]*"\]', "", body)
    assert "samsung electronics" not in remaining.lower()


def test_id_1396_shape_samsung_electronics_co_ltd_scrubbed_from_body():
    offending = "Analysis Report: Samsung Electronics Co., Ltd."
    display_name = "한울반도체"
    report = (
        "**Fundamentals Analysis Report: Samsung Electronics Co., Ltd. (003350.KS)**\n\n"
        "Samsung Electronics Co., Ltd. reported stable margins. Samsung Electronics "
        "continues to invest in R&D. The outlook for Samsung Electronics remains "
        "positive given Samsung Electronics' balance sheet strength."
    )
    scrubbed, count = scrub_offending_name(report, offending, display_name)
    assert count > 0
    remaining = re.sub(r'\[name corrected from "[^"]*"\]', "", scrubbed)
    assert "samsung electronics" not in remaining.lower()
    assert display_name in scrubbed
    assert '[name corrected from "' in scrubbed


def test_scrub_offending_name_leaves_report_unchanged_when_candidate_never_recurs():
    offending = "A Completely Unrelated One-Off Guess Ltd."
    display_name = "우리금융지주"
    report = "The company posted solid quarterly results with improving margins."
    scrubbed, count = scrub_offending_name(report, offending, display_name)
    assert count == 0
    assert scrubbed == report


def test_scrub_offending_name_skips_candidate_colliding_with_display_name():
    # Noisy extraction happens to guess a candidate that is a substring of
    # (or contains) the CORRECT display name — must never scrub, since doing
    # so would corrupt or no-op a legitimate mention of the right company.
    offending = "지주"
    display_name = "우리금융지주"
    report = "우리금융지주는 이번 분기 견고한 실적을 발표했다. 우리금융지주 주가는 상승했다."
    scrubbed, count = scrub_offending_name(report, offending, display_name)
    assert count == 0
    assert scrubbed == report


def test_scrub_offending_name_skips_stoplisted_generic_candidate():
    offending = "*** Report: Company"
    display_name = "우리금융지주"
    report = "Company posted results. Company outlook is stable. Company shares rose."
    scrubbed, count = scrub_offending_name(report, offending, display_name)
    assert count == 0
    assert scrubbed == report


def test_scrub_offending_name_direct_unit_markdown_wrapper():
    offending = "Electronics) ### Technical Analysis: Samsung Electronics"
    display_name = "한울반도체"
    report = "Samsung Electronics posted gains. Samsung Electronics rallied further."
    scrubbed, count = scrub_offending_name(report, offending, display_name)
    assert count == 2
    # Only the marker's own quoted echo of the original name remains — no
    # bare/unmarked "Samsung Electronics" mention survives.
    remaining = re.sub(r'\[name corrected from "[^"]*"\]', "", scrubbed)
    assert "samsung electronics" not in remaining.lower()
    assert scrubbed.startswith('한울반도체 [name corrected from "Samsung Electronics"]')
    assert scrubbed.endswith("한울반도체 rallied further.")


def test_scrub_offending_name_direct_unit_trailing_legal_suffix():
    offending = "Analysis Report: Samsung Electronics Co., Ltd."
    display_name = "한울반도체"
    report = "Samsung Electronics is a leader. Samsung Electronics dominates the market."
    scrubbed, count = scrub_offending_name(report, offending, display_name)
    assert count == 2
    remaining = re.sub(r'\[name corrected from "[^"]*"\]', "", scrubbed)
    assert "samsung electronics" not in remaining.lower()


def test_scrub_offending_name_direct_unit_plain_clean_name_no_wrapper():
    offending = "KB Financial Group"
    display_name = "우리금융지주"
    report = "KB Financial Group reported strong results. KB Financial Group shares gained."
    scrubbed, count = scrub_offending_name(report, offending, display_name)
    assert count == 2
    remaining = re.sub(r'\[name corrected from "[^"]*"\]', "", scrubbed)
    assert "kb financial group" not in remaining.lower()
    assert scrubbed.startswith('우리금융지주 [name corrected from "KB Financial Group"]')


def test_enforce_identity_foreign_listing_branch_never_scrubs():
    strong = build_alias_set("앱튼")
    weak = build_alias_set("G Enone Energy Co Ltd") - strong
    report = _APPEN_REPORT
    fixed, offending = enforce_identity(
        report, "270520.KS", "앱튼", strong, weak_aliases=weak, language="en",
        agent_name="Fundamentals Analyst",
    )
    assert isinstance(offending, ForeignListingName)
    # The foreign-listing branch never calls scrub_offending_name — the
    # report body is prepended-only, exactly like before this change.
    assert report in fixed


def test_enforce_identity_unnamed_branch_never_scrubs():
    aliases = _aliases("우리금융지주", "316140.KS", "Woori Financial Group")
    report = "This company posted strong insurance segment growth this quarter."
    fixed, offending = enforce_identity(
        report, "316140.KS", "우리금융지주", aliases, language="en", agent_name="News + Macro Analyst",
    )
    assert offending == UNNAMED
    # The UNNAMED branch never calls scrub_offending_name — nothing to scrub.
    assert report in fixed

