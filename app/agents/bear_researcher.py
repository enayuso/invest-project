from __future__ import annotations

from app.agents.debater import make_speaker, render_reports
from app.llm.language import DEFAULT_LANGUAGE
from app.state import SpeakerFn

SYSTEM_PROMPT = (
    "You are the Bear Researcher. Argue the case AGAINST investing in this stock, "
    "grounded in the analyst reports provided. Be concise and substantive. If the "
    "reports mention Wall Street analyst consensus or institutional positioning, "
    "treat it as one data point to weigh, not as your argument itself — restating "
    "'analysts are bullish' is not a substantive case; explain the underlying "
    "fundamentals/technicals/news reasoning that would justify disagreeing with it.\n\n"
    "Draw on evidence across the different analyst reports (technical, "
    "fundamental, news/macro, sentiment) rather than anchoring your whole case on "
    "just one — a case built by triangulating multiple independent signals is "
    "stronger than one resting on a single report type. Before making your case, "
    "acknowledge the single strongest point against your own position found in "
    "the reports themselves, then explain why it doesn't change your conclusion "
    "— ignoring the best counter-evidence is evasive, not persuasive."
)


def build(
    reports: dict[str, str],
    *,
    language: str = DEFAULT_LANGUAGE,
    identity: str = "",
    # 2026-09-10 audit (ids 1373/1374/1375): the only prose-generating nodes
    # in the pipeline that never received the SESSION IN PROGRESS warning —
    # every other node already does (see pipeline.py's session_note threading).
    session_note: str = "",
) -> SpeakerFn:
    session_block = f"{session_note}\n\n" if session_note else ""
    context = f"{session_block}Analyst reports:\n{render_reports(reports, identity=identity)}"
    return make_speaker(SYSTEM_PROMPT, context, label="bear_researcher", language=language)
