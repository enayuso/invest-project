"""app.agents.ground_truth_claim_guard — no network, no LLM (A-stage).

Real incidents this guards against:
  - id=1394: supplied PBR=0.55, a report attributed "priced at 2.44x P/B"
    to the data — a ~4.4x mismatch.
  - id=1386 (006360.KS): supplied program-trading net-buy 156.5억원 (=15.65
    billion KRW). One debater correctly cited "+15.65 billion KRW"; another
    "corrected" it to "+1.565 billion KRW" while calling the correct figure
    a "fabrication" / "tenfold" error — the wrong figure is the one that
    should be flagged, the correct one should stay clean.
"""

from __future__ import annotations

from app.agents.ground_truth_claim_guard import (
    extract_ground_truth,
    scan_ground_truth_claims,
)

_VALUATION_BLOCK = (
    "Korean financial statements for 006360.KS (source: KIS, most recent periods first)\n"
    "\n[Valuation] (live, source: KIS)\n"
    "  Current price 68,900원 | PER 12.34 | PBR 0.55 | EPS 5,588 | BPS 125,300 | Market cap 12,345억원"
)

_PROGRAM_TRADING_BLOCK = (
    "\n[Program trading] latest net-buy: +406,376 shares (+156.5억원)"
)


def test_extract_ground_truth_parses_valuation_block():
    gt = extract_ground_truth(kr_gt=_VALUATION_BLOCK)
    assert gt["price"] == 68_900.0
    assert gt["per"] == 12.34
    assert gt["pbr"] == 0.55
    assert gt["eps"] == 5_588.0
    assert gt["bps"] == 125_300.0


def test_extract_ground_truth_parses_program_trading_block():
    gt = extract_ground_truth(kr_risk_extra=_PROGRAM_TRADING_BLOCK)
    assert gt["program_net_buy_billion"] == 15.65


def test_extract_ground_truth_empty_blocks_no_exception():
    assert extract_ground_truth() == {}
    assert extract_ground_truth(kr_gt="no data here", kr_risk_extra="nothing here") == {}


def test_pbr_mismatch_shape_id_1394_flagged_when_attributed():
    gt = {"pbr": 0.55}
    text = "Per the data, this name is priced at 2.44x P/B, a rich valuation for the sector."
    findings = scan_ground_truth_claims(text, ground_truth=gt, agent_name="Bull")
    assert len(findings) == 1
    assert "2.44" in findings[0] and "0.55" in findings[0]


def test_pbr_claim_within_threshold_not_flagged():
    gt = {"pbr": 0.55}
    text = "Per the data, this name is priced at 0.6x P/B, in line with peers."
    assert scan_ground_truth_claims(text, ground_truth=gt, agent_name="Bull") == []


def test_program_trading_correct_citation_not_flagged():
    gt = {"program_net_buy_billion": 15.65}
    text = (
        "The Aggressive debater asserts that Program Trading data shows "
        '"+15.65 billion KRW" in buys, consistent with the supplied table.'
    )
    assert scan_ground_truth_claims(text, ground_truth=gt, agent_name="Neutral") == []


def test_program_trading_false_correction_id_1386_flagged():
    gt = {"program_net_buy_billion": 15.65}
    text = (
        'the Aggressive debater asserts that Program Trading data shows "+15.65 billion KRW" '
        'in buys, whereas the provided data table explicitly lists "+156.5억원" (+1.565 billion KRW). '
        "An error of magnitude ten (15.65bn vs 1.56bn) is not a typo; it is a fabrication of material facts."
    )
    findings = scan_ground_truth_claims(text, ground_truth=gt, agent_name="Neutral")
    assert len(findings) == 1
    assert "claimed as 1.565" in findings[0]
    assert "false correction" in findings[0]


def test_unattributed_bare_number_not_flagged():
    gt = {"pbr": 0.55}
    text = "Some analysts elsewhere have cited a 2.44x multiple for comparable names."
    assert scan_ground_truth_claims(text, ground_truth=gt, agent_name="Bull") == []


def test_empty_ground_truth_never_flags():
    text = (
        'Program Trading data shows "+1.565 billion KRW" — this is a fabrication, tenfold off.'
    )
    assert scan_ground_truth_claims(text, ground_truth={}, agent_name="Neutral") == []


# --- 2026-09-14 false positives: "<n> billion KRW" with no program-trading anchor ---

_UNIT_SCALE_BANNER = (
    '⚠️ UNIT-SCALE CORRECTION: this report misstated one or more KRW figures by 10x '
    '(an 억원 figure\'s digits were carried straight into "billion" — 1억원 = 0.1 billion KRW)'
)


def test_unit_scale_correction_banner_not_flagged_as_program_claim():
    # id=1405/id=1408: the banner's explanatory "0.1 billion KRW" was read as a
    # claimed program net-buy, and its "CORRECTION"/"10x" wording tagged it a
    # false correction — systematically, on every unit-scale correction.
    gt = {"program_net_buy_billion": -1.97}
    assert scan_ground_truth_claims(_UNIT_SCALE_BANNER, ground_truth=gt, agent_name="Research Manager") == []


def test_financial_statement_unit_recital_not_flagged_as_program_claim():
    # id=1405: revenue-scale figures discussed during a unit correction.
    gt = {"program_net_buy_billion": -1.97}
    text = (
        "The data shows the income statement listing 4,876.7 billion KRW as 4,876.7 billion KRW "
        '[unit corrected from "48,767 billion KRW"], which is a reporting-unit artifact, not a fabrication.'
    )
    assert scan_ground_truth_claims(text, ground_truth=gt, agent_name="Research Manager") == []


def test_program_trading_claim_with_context_still_flagged():
    gt = {"program_net_buy_billion": -1.97}
    text = (
        "The provided data table shows program trading at a net-buy of -0.197 billion KRW, "
        "a tenfold error versus what was supplied."
    )
    findings = scan_ground_truth_claims(text, ground_truth=gt, agent_name="Neutral")
    assert len(findings) == 1
    assert "-0.197" in findings[0]


def test_quoted_span_not_cut_at_decimal_point():
    # id=1405: spans used to begin mid-number (after the "." in "4,876.7").
    gt = {"program_net_buy_billion": -1.97}
    text = (
        "Per the data, program trading was cited as 4,876.7 billion KRW, which is absurd."
    )
    findings = scan_ground_truth_claims(text, ground_truth=gt, agent_name="Neutral")
    assert len(findings) == 1
    span = findings[0].split(" — ", 1)[-1]
    assert span.startswith("'Per the data, program trading was cited as 4,876.7 billion KRW")


def test_buyback_program_not_flagged_as_program_trading_claim():
    # "program" as a HEAD noun ("buyback program") is not program trading —
    # the anchor only accepts it as a MODIFIER ("program trading"/"program
    # net-buy"), even with an attribution and a correction frame present.
    gt = {"program_net_buy_billion": -1.97}
    text = "Per the data, the company announced a 4.9 billion KRW buyback program, a tenfold increase."
    assert scan_ground_truth_claims(text, ground_truth=gt, agent_name="Bull") == []
