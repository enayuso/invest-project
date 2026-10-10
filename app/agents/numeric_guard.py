"""Thin composer that runs both numeric report guards over one piece of text
and prefixes every finding with the node that produced it.

`app.agents.unit_scale_guard` and `app.agents.ratio_label_guard` were each
wired at exactly one call site — `app.agents.fundamentals_analyst.analyze`.
The 2026-09-01 first-real-account audit found the same class of error in
nodes the guards never saw:

  - decision id=1157 (009150.KS/삼성전기): a risk-debate Aggressive turn
    wrote "-785.2 billion KRW equivalent" for a figure whose true value was
    ~78.7 billion. That span matched `_BILLION_RE` EXACTLY — right regex
    shape, wrong node. Nothing guarded a debate turn, so it propagated to
    round 2 and the Portfolio Manager unchecked.
  - decision id=1156 (267260.KS/HD현대일렉트릭) and id=1157: the Market and
    News+Macro analysts, the research plan, and the PM's own summary each
    carried magnitude / ratio-label errors that only the Fundamentals report
    was ever checked for.

So this module exists to run the identical unit→ratio chain (same order and
reasoning as `fundamentals_analyst.analyze`'s two chained calls) over every
node that consumes or republishes financial figures, with each finding
labelled `f"{node}: ..."` so a later audit can tell decision 1157's
single bad turn apart from decision 1158's wholesale report failure.

The guards are pure regex over already-in-memory strings (neither module
does any network / DB / LLM call), so the ~15 extra invocations per decision
are sub-millisecond against a pipeline that spends 10-12 LLM calls on the
same decision — zero added API calls, zero added I/O.
"""

from __future__ import annotations

from app.agents.period_mismatch_guard import PeriodMismatchScan, scan_period_mismatch
from app.agents.ratio_label_guard import enforce_ratio_labels
from app.agents.unit_scale_guard import UnitScaleScan, enforce_unit_scale


def enforce_numeric_guards(
    text: str,
    *,
    node: str,
    kr_block: str = "",
    language: str = "en",
    usd_check: bool = False,
    currency_check: bool = False,
    soft_eok_block: str = "",
    supplied_kr_block: bool = False,
    banner: bool = True,
    inline_note: bool = True,
    scans_out: dict[str, dict[str, int]] | None = None,
    period_out: list[str] | None = None,
    period_strong_out: list[str] | None = None,
    period_scans_out: dict[str, dict] | None = None,
) -> tuple[str, list[str]]:
    """(possibly corrected text, node-prefixed violation descriptions).

    `unit_scale_guard` then `ratio_label_guard`, chained so a text corrected
    for a unit-scale error is also checked for a ratio-label error (not
    either/or) — the same order `fundamentals_analyst.analyze` uses. Every
    description is prefixed `f"{node}: "`.

    `kr_block` is the ground-truth 억원 text (empty ⇒ the unit guard falls
    back to its self-contradiction-only check, exactly the documented
    fail-open). `banner` / `inline_note` are threaded to both guards
    (`banner=False` for a structured field whose siblings would each repeat
    the banner; `inline_note=False` for length-budgeted card text — the
    violation is still recorded, so it is an accounted-for silent
    correction). `currency_check` (Fundamentals Analyst only) threads to the
    unit guard's opt-in currency-composition net (id=1207). `soft_eok_block`
    (a `get_kr_news` headline block) threads to the unit guard as its soft
    ground-truth set — a report figure matching a headline's 억/조 amount
    digit-for-digit is flagged `news_eok_10x`, detect-only (2026-09-07 audit
    #3, id=1237). `supplied_kr_block`
    (Fundamentals Analyst only) threads to the unit guard, where it gates the
    id=1158 "SUSPECTED BLIND SPOT" warning to the one node the 억원 block is
    actually supplied to — every other node reuses the same ground-truth text
    without having read it (88/88 false alarms 2026-08-29→09-04; see
    `enforce_unit_scale`'s docstring). `scans_out`, if
    given, gets `{node: {...}}` with the unit
    guard's positive scan counts. `period_out`, if given, is extended with
    `period_mismatch_guard` findings (detect-only, node-prefixed) — kept
    separate from the returned descriptions because they never rewrite the
    text and are recorded in their own PipelineState field.

    `period_strong_out`, if given, is extended with only the Tier-1 (strong)
    period-mismatch findings — the subset that gates the report cache (see
    `period_mismatch_guard` §3 and `PipelineState.period_mismatch_scans`).
    `period_scans_out`, if given, gets `{node: scan.as_dict()}` — the
    positive scan record an empty `period_out` cannot carry (id=1158)."""
    if not text:
        return text, []

    if period_out is not None or period_strong_out is not None or period_scans_out is not None:
        _pm_scans: list[PeriodMismatchScan] = []
        _pm_findings = scan_period_mismatch(
            text, kr_block, agent_name=node,
            strong_out=period_strong_out,
            scan_out=_pm_scans if period_scans_out is not None else None,
        )
        if period_out is not None:
            period_out.extend(f"{node}: {d}" for d in _pm_findings)
        if period_scans_out is not None and _pm_scans:
            period_scans_out[node] = _pm_scans[0].as_dict()

    scan_out: list[UnitScaleScan] = []
    text, unit_desc = enforce_unit_scale(
        text,
        kr_block,
        language=language,
        agent_name=node,
        usd_check=usd_check,
        currency_check=currency_check,
        soft_eok_block=soft_eok_block,
        supplied_kr_block=supplied_kr_block,
        banner=banner,
        inline_note=inline_note,
        scan_out=scan_out,
    )
    text, ratio_desc = enforce_ratio_labels(
        text,
        language=language,
        agent_name=node,
        banner=banner,
        inline_note=inline_note,
    )
    if scans_out is not None and scan_out:
        scans_out[node] = scan_out[0].as_dict()
    descriptions = [f"{node}: {d}" for d in (*unit_desc, *ratio_desc)]
    return text, descriptions
