"""Shared heuristics for spotting an LLM response that reads like unedited
working notes (leaked chain-of-thought) rather than a finished argument.

Extracted 2026-09-01 (root cause #9 of the 2026-09-01 real-account audit) so
`app.llm.client._strip_reasoning_leak` (tier 4, untagged detection) and
`app.agents.untrusted_content.redact_injection_flag` (tier C, the text a
model writes AFTER an INJECTION_SUSPECTED banner) score the same shape of
text against one tuned tuple instead of two copies that can silently drift
apart. This module has no imports of its own so either caller can depend on
it without pulling in the other's dependencies (`untrusted_content.py` is
otherwise a pure text module with no `settings`/network surface, and must
stay that way for its own test suite).

Calibrated against the 2026-09-01 id=1157 transcript (Samsung Electro-
Mechanics risk debate, ~150 leaked lines including this project's own prompt
text quoted back verbatim) — see `_strip_reasoning_leak`'s module comment in
app.llm.client for the full incident writeup. The threshold (2, not 1) is
deliberate: any single marker below appears in ordinary argument; several
together is the leaked-deliberation shape.
"""

from __future__ import annotations

import re

UNTAGGED_REASONING_MARKERS = (
    re.compile(r"\bthe prompt (?:says|asks|defines|mentions|instructs)\b", re.IGNORECASE),
    re.compile(r"\b(?:let'?s|let me) re-?read\b", re.IGNORECASE),
    re.compile(r"\b(?:my|the) role is (?:the|to be)\b", re.IGNORECASE),
    re.compile(r"\bre-?evaluation\s*:", re.IGNORECASE),
    re.compile(r"^\s*\**\s*(?:wait|actually,?\s*wait|hold on)\b", re.IGNORECASE | re.MULTILINE),
    re.compile(r"\bwriting (?:it )?now\b", re.IGNORECASE),
)
UNTAGGED_REASONING_MIN_MARKERS = 2
