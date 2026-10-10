"""Static well-formedness checks over every INSERT statement in app/.

These exist because a real, production-blocking bug shipped undetected:
`app.execution.safety_switch._daily_loss_baseline` listed six target columns
but supplied only five VALUES expressions, so the query could not even be
PREPAREd. Postgres rejected it at runtime, the caller's fail-closed handler
turned that into "BLOCKED: daily loss state unreadable", and every BUY on
every market was blocked — with the daily max-loss circuit breaker silently
enforcing nothing.

The existing unit tests could not catch it: they fake the asyncpg connection
and only string-match the query, so nothing ever parsed the SQL. A live-DB
test would catch it but can't run in CI (and the DB is frequently
unreachable from dev machines here). These checks are the cheap middle
ground — pure string analysis, no DB, no network — and they catch exactly
the arity class of mistake that slipped through.
SQL 문자열을 실제로 파싱하지 않는 기존 테스트가 놓친 컬럼/값 개수 불일치를 정적으로 검사.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_APP = Path(__file__).resolve().parents[1] / "app"

# INSERT INTO <table> (<columns>) VALUES (<expressions>)
# DEFAULT VALUES and INSERT ... SELECT have no parenthesised VALUES list and
# are skipped by this pattern, which is correct — they have no arity to check.
_INSERT = re.compile(
    r"INSERT\s+INTO\s+(?P<table>[a-zA-Z_][\w.]*)\s*\((?P<cols>[^)]*)\)\s*VALUES\s*\(",
    re.IGNORECASE,
)


def _split_top_level(text: str) -> list[str]:
    """Split on commas that are not nested inside parentheses — VALUES lists
    legitimately contain calls like `now()` and casts like `$2::interval`."""
    parts, depth, current = [], 0, []
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    parts.append("".join(current))
    return [p.strip() for p in parts if p.strip()]


def _values_body(source: str, open_paren_index: int) -> str | None:
    """The text inside the VALUES(...) list, given the index of its opening
    paren. Returns None if the parens never balance (malformed source)."""
    depth = 0
    for i in range(open_paren_index, len(source)):
        if source[i] == "(":
            depth += 1
        elif source[i] == ")":
            depth -= 1
            if depth == 0:
                return source[open_paren_index + 1 : i]
    return None


def _insert_statements() -> list[tuple[str, str, list[str], list[str]]]:
    """(file, table, columns, value_expressions) for every INSERT in app/."""
    found = []
    for path in sorted(_APP.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        for match in _INSERT.finditer(source):
            body = _values_body(source, match.end() - 1)
            if body is None:
                continue
            rel = path.relative_to(_APP.parent).as_posix()
            found.append((
                rel,
                match.group("table"),
                _split_top_level(match.group("cols")),
                _split_top_level(body),
            ))
    return found


def test_found_the_insert_statements():
    """Guards the guard: if the regex silently stops matching (a reformat, a
    move to a query builder), the arity test below would vacuously pass."""
    statements = _insert_statements()
    assert len(statements) >= 10, f"only found {len(statements)} INSERTs — has the pattern gone stale?"
    tables = {table for _, table, _, _ in statements}
    assert "daily_loss_state" in tables  # the statement this file exists for
    assert "trades" in tables


@pytest.mark.parametrize("stmt", _insert_statements(), ids=lambda s: f"{s[0]}::{s[1]}")
def test_insert_column_count_matches_value_count(stmt):
    """The exact defect that blocked every BUY: 6 target columns, 5 values."""
    rel, table, columns, values = stmt
    assert len(columns) == len(values), (
        f"{rel}: INSERT INTO {table} lists {len(columns)} columns "
        f"but supplies {len(values)} VALUES expressions.\n"
        f"  columns: {columns}\n"
        f"  values:  {values}"
    )


@pytest.mark.parametrize("stmt", _insert_statements(), ids=lambda s: f"{s[0]}::{s[1]}")
def test_insert_placeholders_are_contiguous_from_one(stmt):
    """A gap or duplicate in $1..$N across the whole statement usually means
    an argument was added to the column list but not threaded through — the
    same edit that produced the arity bug also left `SET x = $3` pointing at
    user_id instead of the date it meant."""
    rel, table, _, values = stmt
    used = {int(n) for n in re.findall(r"\$(\d+)", " ".join(values))}
    if not used:
        return  # all-literal VALUES (e.g. a seed row) — nothing to check
    assert used == set(range(1, max(used) + 1)), (
        f"{rel}: INSERT INTO {table} uses placeholders {sorted(used)} — "
        f"expected a contiguous $1..${max(used)}"
    )
