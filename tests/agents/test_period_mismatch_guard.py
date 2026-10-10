"""app.agents.period_mismatch_guard — no network, no LLM.

2026-09-01 audit #5 §4, decision id=1158 (000270.KS/기아): every agent
divided H1 2026's half-year operating profit by FY2024's annual figure and
called it a "62% collapse".

2026-09-03: precision restructure (rules ①-⑧) + Tier-1/Tier-2 split. The two
original true-positive fixtures (Kia / Woori) MUST keep flagging; every
false-positive class gets its own negative test.
"""

from __future__ import annotations

from app.agents.period_mismatch_guard import PeriodMismatchScan, scan_period_mismatch

# The [Income statement] shape get_kr_financials now emits (audit #5 §2).
_KIA_KR_BLOCK = """
[Income statement] (억원, ... never re-derive ...)
  202606 (interim, 6 months — NOT a full year): revenue=625,389억원(62.54조원) gross_profit=1억원 operating_profit=48,337억원(4.83조원) net_income=41,579억원(4.16조원)
    | exchange-reported YoY for this period (vs the same period one year earlier, KIS): revenue_growth=+2.87% operating_profit_growth=+6.51% net_income_growth=-10.79%
  202512 (FY, 12 months): revenue=1,141,400억원 gross_profit=1억원 operating_profit=90,781억원 net_income=75,500억원
  202412 (FY, 12 months): revenue=1,074,500억원 gross_profit=1억원 operating_profit=126,671억원 net_income=97,800억원
  202312 (FY, 12 months): revenue=998,000억원 gross_profit=1억원 operating_profit=1억원 net_income=1억원
  PERIOD WARNING — the top row (202606) is a 6-MONTH interim ...
"""

_WOORI_KR_BLOCK = """
[Income statement] (억원, ...)
  202603 (interim, 3 months — NOT a full year): revenue=152,433억원 gross_profit=1억원 operating_profit=1억원 net_income=1억원
  202512 (FY, 12 months): revenue=415,484억원 gross_profit=1억원 operating_profit=1억원 net_income=1억원
"""


def test_flags_kia_operating_profit_against_two_year_old_fy():
    # 48,337 / 126,671 - 1 = -61.8%  (vs FY2024, the row TWO years back)
    report = "Operating profit shows a sharp decline of roughly -62% versus its 2024 peak, a value trap."
    found = scan_period_mismatch(report, _KIA_KR_BLOCK, agent_name="Fundamentals Analyst")
    assert len(found) == 1
    assert "operating_profit" in found[0] and "202412" in found[0]


def test_flags_woori_revenue_signature():
    report = "Revenue collapsed -63.3% year on year, indicating a structural problem."
    strong: list[str] = []
    found = scan_period_mismatch(report, _WOORI_KR_BLOCK, agent_name="x", strong_out=strong)
    assert len(found) == 1
    assert "revenue" in found[0]
    # Tier-2: not the Fundamentals Analyst, and "year on year" carries no
    # period frame.
    assert strong == []


def test_does_not_flag_a_real_yoy_decline():
    # The report quotes KIS's OWN net_income_growth for the interim period.
    report = "Net income declined -10.79% year over year, per the exchange's figures — a real slowdown."
    assert scan_period_mismatch(report, _KIA_KR_BLOCK, agent_name="x") == []


def test_does_not_flag_a_positive_percentage_without_decline_framing():
    report = "Operating profit grew a healthy 6.51% and revenue rose 2.87%."
    assert scan_period_mismatch(report, _KIA_KR_BLOCK, agent_name="x") == []


def test_does_not_flag_small_forbidden_ratios():
    report = "Operating profit fell about -7% between the two full years."
    assert scan_period_mismatch(report, _KIA_KR_BLOCK, agent_name="x") == []


def test_no_kr_block_is_a_noop():
    assert scan_period_mismatch("a -62% collapse", "", agent_name="x") == []


def test_no_interim_row_is_a_noop():
    block = _KIA_KR_BLOCK.replace("interim, 6 months — NOT a full year", "FY, 12 months")
    assert scan_period_mismatch("a -62% collapse in operating profit vs 2024", block, agent_name="x") == []


def test_logs_a_warning_when_it_flags(caplog):
    import logging

    report = "operating profit -62% collapse vs the 2024 peak"
    with caplog.at_level(logging.WARNING):
        scan_period_mismatch(report, _KIA_KR_BLOCK, agent_name="Neutral")
    assert any("period_mismatch_guard" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# 2026-09-03 — new coverage for the precision rules + the tier split
# ---------------------------------------------------------------------------

def test_flags_unsigned_collapse_phrasing():
    # id=1158's actual phrasing: an UNSIGNED "62%" next to a decline word.
    # The pre-2026-09-03 code could not match this (it required v < 0 or a
    # signed match against -61.8).
    report = "Operating profit suffered a 62% collapse versus the 2024 full-year figure."
    found = scan_period_mismatch(report, _KIA_KR_BLOCK, agent_name="Fundamentals Analyst")
    assert len(found) == 1
    assert found[0].startswith("period_mismatch: '62%'")  # the unsigned span


# ---------------------------------------------------------------------------
# 2026-09-15 — `get_kr_financials`'s new `_eok_or_missing`/
# `_income_revenue_or_missing` (id=1433 / 004310.KS/현대약품, "zero revenue ...
# dormant shell company" fix) render an omitted item as "not reported"
# instead of "<n>억원". `_ITEM_VAL_RE` requires a trailing `억원` token, so a
# row where every item is "not reported" simply contributes no parsed
# values — confirm that degrades cleanly (no KeyError, no crash, no
# spurious forbidden-ratio match) rather than assuming every item is present.
# ---------------------------------------------------------------------------
_ALL_ZERO_KR_BLOCK = """
[Income statement] (억원, ... never re-derive ...)
  202509 (interim, 9 months — NOT a full year): revenue=not reported gross_profit=not reported operating_profit=not reported net_income=not reported
  202412 (FY, 12 months): revenue=not reported gross_profit=not reported operating_profit=not reported net_income=not reported
  202312 (FY, 12 months): revenue=191,800억원 gross_profit=50,000억원 operating_profit=20,000억원 net_income=15,000억원
"""


def test_all_not_reported_income_rows_degrade_cleanly_without_crashing():
    report = "The company shows a sharp -62% collapse in every metric, a dormant shell company."
    scans: list[PeriodMismatchScan] = []
    # Must not raise (no KeyError from an item never populated in the row
    # dict) and must not fabricate a forbidden-ratio match out of rows that
    # carry no parsed item values at all.
    found = scan_period_mismatch(
        report, _ALL_ZERO_KR_BLOCK, agent_name="Fundamentals Analyst", scan_out=scans
    )
    assert found == []
    assert len(scans) == 1
    # The all-"not reported" interim row is still recognized as an interim
    # row (period + kind parse independently of the item values), but it
    # contributes no item values, so there is nothing to divide — the one
    # real FY row (202312) has no interim counterpart with parsed values.
    assert scans[0].interim_period == "202509"
    assert scans[0].fy_rows == 2
    assert scans[0].forbidden_in_band == 0


def test_kia_shape_is_tier_one_from_fundamentals():
    report = "Operating profit shows a sharp decline of roughly -62% versus its 2024 peak."
    strong: list[str] = []
    found = scan_period_mismatch(report, _KIA_KR_BLOCK, agent_name="Fundamentals Analyst", strong_out=strong)
    assert len(found) == 1
    assert len(strong) == 1


_NET_INCOME_LEVEL_BLOCK = """
[Income statement] (억원)
  202606 (interim, 6 months — NOT a full year): revenue=1억원 gross_profit=1억원 operating_profit=1억원 net_income=5,000억원
  202412 (FY, 12 months): revenue=1억원 gross_profit=1억원 operating_profit=1억원 net_income=7,000억원
"""


def test_does_not_flag_a_margin_level():
    # 5,000 / 7,000 - 1 = -28.6% is in the forbidden band, but "-28.6%" here
    # is a margin LEVEL, not an interim/FY division.
    report = "Net margin fell to -28.6% in the interim period, well below peers."
    scan: list[PeriodMismatchScan] = []
    assert scan_period_mismatch(report, _NET_INCOME_LEVEL_BLOCK, agent_name="x", scan_out=scan) == []
    assert scan[0].suppressed.get("level_ctx") == 1


def test_does_not_flag_an_intraday_price_drawdown():
    report = ("The stock collapsed from a session high of 3,995 KRW to 3,295 KRW, "
              "an intraday drawdown of ~62%.")
    scan: list[PeriodMismatchScan] = []
    assert scan_period_mismatch(report, _KIA_KR_BLOCK, agent_name="x", scan_out=scan) == []
    assert scan[0].suppressed.get("price_ctx") == 1


def test_does_not_flag_an_annualized_erosion_rate():
    report = "Book value eroded from 36bn to 14bn, about -62% annually."
    scan: list[PeriodMismatchScan] = []
    assert scan_period_mismatch(report, _KIA_KR_BLOCK, agent_name="x", scan_out=scan) == []
    assert scan[0].suppressed.get("hedge_ctx") == 1


def test_does_not_flag_a_scenario_range_tail():
    report = "A 6.4% drop can extend rapidly to 50-62% if support fails."
    scan: list[PeriodMismatchScan] = []
    assert scan_period_mismatch(report, _KIA_KR_BLOCK, agent_name="x", scan_out=scan) == []
    assert scan[0].suppressed.get("range_tail") == 1


def test_does_not_flag_a_ratio_level():
    report = "Borrowing Dependency deteriorated to 62% in FY 2025 from prior-year levels."
    scan: list[PeriodMismatchScan] = []
    assert scan_period_mismatch(report, _KIA_KR_BLOCK, agent_name="x", scan_out=scan) == []
    assert scan[0].suppressed.get("level_ctx") == 1


_ROUNDED_GROWTH_BLOCK = """
[Income statement] (억원)
  202606 (interim, 6 months — NOT a full year): revenue=200,000억원 gross_profit=1억원 operating_profit=1억원 net_income=1억원
    | exchange-reported YoY (KIS): revenue_growth=-61.79%
  202412 (FY, 12 months): revenue=500,000억원 gross_profit=1억원 operating_profit=1억원 net_income=1억원
"""


def test_does_not_flag_a_rounded_restatement_of_exchange_growth():
    # The report rounds KIS's real revenue_growth=-61.79% to "-62%".
    report = "Revenue declined -62% year over year on the exchange's own figures."
    scan: list[PeriodMismatchScan] = []
    assert scan_period_mismatch(report, _ROUNDED_GROWTH_BLOCK, agent_name="x", scan_out=scan) == []
    assert scan[0].suppressed.get("real_growth_rounded") == 1


def test_does_not_flag_a_percentage_for_a_different_item():
    # -62% matches operating_profit's forbidden ratio, but the sentence is
    # about net income.
    report = "Net income fell -62% on a deteriorating cost base."
    scan: list[PeriodMismatchScan] = []
    assert scan_period_mismatch(report, _KIA_KR_BLOCK, agent_name="x", scan_out=scan) == []
    assert scan[0].suppressed.get("item_mismatch") == 1


_INTERIM_ABOVE_FY_BLOCK = """
[Income statement] (억원)
  202606 (interim, 6 months — NOT a full year): revenue=162,000억원 gross_profit=1억원 operating_profit=1억원 net_income=1억원
  202412 (FY, 12 months): revenue=100,000억원 gross_profit=1억원 operating_profit=1억원 net_income=1억원
"""


def test_positive_forbidden_ratios_are_never_candidates():
    # interim revenue / FY revenue - 1 = +62% — a growth story, never a
    # fabricated collapse.
    report = "Revenue growth of 62% masks a collapsing order book."
    scan: list[PeriodMismatchScan] = []
    assert scan_period_mismatch(report, _INTERIM_ABOVE_FY_BLOCK, agent_name="x", scan_out=scan) == []
    assert scan[0].forbidden_in_band == 0


_ALL_PLACEHOLDER_BLOCK = """
[Income statement] (억원)
  202606 (interim, 6 months — NOT a full year): gross_profit=1억원
  202412 (FY, 12 months): gross_profit=1억원
"""


def test_placeholder_one_eok_rows_are_skipped():
    scan: list[PeriodMismatchScan] = []
    assert scan_period_mismatch("a -62% collapse in profit vs 2024", _ALL_PLACEHOLDER_BLOCK,
                                agent_name="x", scan_out=scan) == []
    assert scan[0].forbidden_in_band == 0


_ZERO_IN_BAND_BLOCK = """
[Income statement] (억원)
  202606 (interim, 6 months — NOT a full year): revenue=390,000억원 gross_profit=1억원 operating_profit=1억원 net_income=1억원
  202412 (FY, 12 months): revenue=415,000억원 gross_profit=1억원 operating_profit=1억원 net_income=1억원
"""

_NO_INTERIM_BLOCK = """
[Income statement] (억원)
  202512 (FY, 12 months): revenue=415,484억원 gross_profit=1억원 operating_profit=1억원 net_income=1억원
  202412 (FY, 12 months): revenue=400,000억원 gross_profit=1억원 operating_profit=1억원 net_income=1억원
"""


def test_scan_record_distinguishes_clean_from_unscannable():
    report = "a -62% collapse in operating profit versus the 2024 peak"

    no_block: list[PeriodMismatchScan] = []
    assert scan_period_mismatch(report, "", agent_name="x", scan_out=no_block) == []
    assert no_block[0].kr_block_present == 0

    no_interim: list[PeriodMismatchScan] = []
    assert scan_period_mismatch(report, _NO_INTERIM_BLOCK, agent_name="x", scan_out=no_interim) == []
    assert no_interim[0].kr_block_present == 1
    assert no_interim[0].interim_period == ""

    zero_band: list[PeriodMismatchScan] = []
    assert scan_period_mismatch(report, _ZERO_IN_BAND_BLOCK, agent_name="x", scan_out=zero_band) == []
    assert zero_band[0].interim_period != ""
    assert zero_band[0].forbidden_in_band == 0

    # dedupe: the same span + forbidden desc twice in one body -> one finding
    dupe_report = (
        "Operating profit shows a sharp decline of roughly -62% versus its 2024 peak. "
        "Again: operating profit -62% decline versus the 2024 peak."
    )
    dupe_scan: list[PeriodMismatchScan] = []
    dupe_found = scan_period_mismatch(dupe_report, _KIA_KR_BLOCK, agent_name="x", scan_out=dupe_scan)
    assert len(dupe_found) == 1
    assert dupe_scan[0].suppressed.get("dedupe") == 1


# --- 2026-09-07 audit #6: the FINANCIAL-DATA RECENCY note must not disturb
# this guard's row parsing or its real-growth exemption set.
def test_staleness_note_does_not_parse_as_an_income_row_or_a_growth_figure():
    from datetime import date

    from app.dataflows.kis_provider import _staleness_note, kr_financials_recency

    recency = kr_financials_recency("202412", "202412 (FY, 12 months)", date(2026, 9, 7))
    note = _staleness_note(recency, date(2026, 9, 7))
    assert note  # 21 months stale -> non-empty

    baseline_interim, baseline_rows = None, None
    from app.agents.period_mismatch_guard import _parse_rows

    baseline_interim, baseline_rows = _parse_rows(_KIA_KR_BLOCK)
    with_note_interim, with_note_rows = _parse_rows(note + "\n" + _KIA_KR_BLOCK)
    assert with_note_interim == baseline_interim
    assert with_note_rows == baseline_rows

    # and the real-YoY exemption set is unchanged (note carries no _growth= token)
    report = "Operating profit shows a sharp decline of roughly -62% versus its 2024 peak."
    assert len(scan_period_mismatch(report, note + "\n" + _KIA_KR_BLOCK, agent_name="Fundamentals Analyst")) == 1
