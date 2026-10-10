"""Loads the analysis-desk debate script from package data.

The script text (`analysis_script.md`) must stay byte-identical to the
"# 대본 전문 (이하 그대로 사용)" section of the design doc it was copied
from -- wording drift here would silently undo design decisions made
against specific observed failure modes. The .md file carries a one-line
HTML-comment pointer to that design doc at its top; this loader strips
only that pointer (and the blank line(s) after it) before handing back
the script, so what callers get is exactly the copied section, nothing
added or removed.
"""

from __future__ import annotations

import re
from importlib import resources

_SCRIPT_FILENAME = "analysis_script.md"

# Matches one leading HTML comment (the design-doc pointer) plus any blank
# lines that follow it, anchored to the start of the file.
_LEADING_POINTER_RE = re.compile(r"\A<!--.*?-->\n+", re.DOTALL)


def load_analysis_script() -> str:
    """Return the debate script verbatim, minus the leading pointer comment."""
    raw = resources.files(__package__).joinpath(_SCRIPT_FILENAME).read_text(encoding="utf-8")
    script, count = _LEADING_POINTER_RE.subn("", raw, count=1)
    if count != 1:
        raise ValueError(
            f"{_SCRIPT_FILENAME}: expected a leading '<!-- ... -->' design-doc pointer "
            "comment followed by a blank line -- file layout changed unexpectedly."
        )
    return script
