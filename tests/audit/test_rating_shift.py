"""app.audit.rating_shift — asyncpg pool/connection faked, no real Postgres.
Same fake-pool convention as tests/audit/test_guard_alpha.py."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from app.audit import rating_shift


class FakeConn:
    def __init__(self, fetch_result=None):
        self.fetch_result = fetch_result if fetch_result is not None else []
        self.queries: list[tuple] = []

    async def fetch(self, query, *args):
        self.queries.append((query, args))
        return self.fetch_result


class _AcquireCM:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


class FakePool:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        return _AcquireCM(self._conn)


async def _async_return(value):
    return value


def _row(ticker, created_at, transcript, trade_date=None):
    return {
        "ticker": ticker,
        "trade_date": trade_date or created_at.date(),
        "created_at": created_at,
        "full_transcript": transcript,
    }


BOUNDARY = datetime(2026, 9, 16, 4, 53, 16, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# parse_boundary
# ---------------------------------------------------------------------------

def test_parse_boundary_accepts_trailing_z_and_defaults_to_utc():
    dt = rating_shift.parse_boundary("2026-09-16T04:53:16Z")
    assert dt == BOUNDARY
    assert dt.tzinfo is not None


def test_parse_boundary_naive_string_defaults_to_utc():
    dt = rating_shift.parse_boundary("2026-09-16T04:53:16")
    assert dt.tzinfo == timezone.utc


def test_parse_boundary_preserves_explicit_offset():
    dt = rating_shift.parse_boundary("2026-09-16T13:53:16+09:00")
    assert dt == BOUNDARY


# ---------------------------------------------------------------------------
# extraction: research_plan as a rendered markdown STRING (the live-row
# shape, and the exact trap the task brief calls out).
# ---------------------------------------------------------------------------

def test_extract_recommendation_from_markdown_string():
    transcript = {"research_plan": "**Recommendation**: Hold\n\n**Rationale**: because reasons."}
    rec, parsed = rating_shift._extract_rm_recommendation(transcript)
    assert rec == "Hold"
    assert parsed is True


def test_extract_recommendation_from_dict():
    transcript = {"research_plan": {"recommendation": "Underweight", "rationale": "..."}}
    rec, parsed = rating_shift._extract_rm_recommendation(transcript)
    assert rec == "Underweight"
    assert parsed is True


def test_extract_recommendation_unparseable_is_unknown_not_dropped():
    # No **Recommendation**: marker in the string at all.
    transcript = {"research_plan": "just some prose with no marker"}
    rec, parsed = rating_shift._extract_rm_recommendation(transcript)
    assert rec is None
    assert parsed is False


def test_extract_recommendation_missing_research_plan_is_unknown():
    rec, parsed = rating_shift._extract_rm_recommendation({})
    assert rec is None
    assert parsed is False


def test_extract_recommendation_dict_missing_recommendation_key_is_unknown():
    transcript = {"research_plan": {"rationale": "no recommendation key here"}}
    rec, parsed = rating_shift._extract_rm_recommendation(transcript)
    assert rec is None
    assert parsed is False


def test_extract_recommendation_unexpected_type_is_unknown():
    transcript = {"research_plan": 12345}
    rec, parsed = rating_shift._extract_rm_recommendation(transcript)
    assert rec is None
    assert parsed is False


# ---------------------------------------------------------------------------
# full_transcript may arrive as a JSON string or an already-decoded dict —
# reused from guard_alpha._parse_transcript, exercised here end-to-end.
# ---------------------------------------------------------------------------

def test_build_cohort_handles_transcript_as_json_string():
    transcript_str = json.dumps({"research_plan": "**Recommendation**: Buy"})
    rows = [_row("A.KS", BOUNDARY, transcript_str)]
    cohort = rating_shift._build_cohort(rows, "before")
    assert cohort.n == 1
    assert cohort.counts == {"Buy": 1}
    assert cohort.n_unknown == 0


# ---------------------------------------------------------------------------
# unparsed rows are counted explicitly, never silently dropped from n.
# ---------------------------------------------------------------------------

def test_build_cohort_counts_unparsed_explicitly():
    rows = [
        _row("A.KS", BOUNDARY, {"research_plan": "**Recommendation**: Hold"}),
        _row("B.KS", BOUNDARY, {"research_plan": "no marker at all"}),
        _row("C.KS", BOUNDARY, None),
    ]
    cohort = rating_shift._build_cohort(rows, "before")
    assert cohort.n == 3
    assert cohort.counts == {"Hold": 1}
    assert cohort.n_unknown == 2


# ---------------------------------------------------------------------------
# before/after cohort split around the boundary.
# ---------------------------------------------------------------------------

async def test_compute_rating_shift_splits_cohorts_at_boundary(monkeypatch):
    rows = [
        _row("A.KS", datetime(2026, 9, 15, tzinfo=timezone.utc), {"research_plan": "**Recommendation**: Underweight"}),
        _row("B.KS", datetime(2026, 9, 15, 12, tzinfo=timezone.utc), {"research_plan": "**Recommendation**: Hold"}),
        # exactly-on-boundary row belongs to "after" (>=).
        _row("C.KS", BOUNDARY, {"research_plan": "**Recommendation**: Hold"}),
        _row("D.KS", datetime(2026, 9, 17, tzinfo=timezone.utc), {"research_plan": "**Recommendation**: Hold"}),
    ]
    conn = FakeConn(fetch_result=rows)
    monkeypatch.setattr(rating_shift, "get_pool", lambda: _async_return(FakePool(conn)))
    monkeypatch.setattr(rating_shift, "get_current_user_id", lambda: "default")

    report = await rating_shift.compute_rating_shift(BOUNDARY)

    assert report.before.n == 2
    assert report.before.counts == {"Underweight": 1, "Hold": 1}
    assert report.after.n == 2
    assert report.after.counts == {"Hold": 2}


async def test_compute_rating_shift_rejects_naive_boundary(monkeypatch):
    with pytest.raises(ValueError):
        await rating_shift.compute_rating_shift(datetime(2026, 9, 16, 4, 53, 16))


# ---------------------------------------------------------------------------
# market / scope / account-mode filters build the expected WHERE fragments,
# matching compute_guard_alpha's exact semantics.
# ---------------------------------------------------------------------------

async def test_compute_rating_shift_applies_market_scope_account_mode_filters(monkeypatch):
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(rating_shift, "get_pool", lambda: _async_return(FakePool(conn)))
    monkeypatch.setattr(rating_shift, "get_current_user_id", lambda: "default")

    await rating_shift.compute_rating_shift(
        BOUNDARY, market="domestic", scope="live", account_mode="real",
    )

    assert len(conn.queries) == 1
    query, params = conn.queries[0]
    assert "ticker LIKE '%.KS'" in query
    assert "is_backtest = false" in query
    assert "account_mode = " in query
    assert "user_id = " in query
    assert "real" in params
    assert "default" in params


async def test_compute_rating_shift_overseas_and_backtest_scope(monkeypatch):
    conn = FakeConn(fetch_result=[])
    monkeypatch.setattr(rating_shift, "get_pool", lambda: _async_return(FakePool(conn)))
    monkeypatch.setattr(rating_shift, "get_current_user_id", lambda: "default")

    await rating_shift.compute_rating_shift(BOUNDARY, market="overseas", scope="backtest")

    query, _params = conn.queries[0]
    assert "ticker NOT LIKE '%.KS'" in query
    assert "is_backtest = true" in query


def test_compute_rating_shift_validates_market_scope_account_mode():
    with pytest.raises(ValueError):
        import asyncio
        asyncio.run(rating_shift.compute_rating_shift(BOUNDARY, market="bogus"))


# ---------------------------------------------------------------------------
# renderer includes boundary, filters, counts, percentages and the
# explicit unparsed line.
# ---------------------------------------------------------------------------

def test_render_rating_shift_includes_counts_percentages_and_unparsed():
    before = rating_shift.CohortRatingCounts(
        label="before", n=4, counts={"Hold": 2, "Buy": 1}, n_unknown=1,
    )
    after = rating_shift.CohortRatingCounts(label="after", n=2, counts={"Hold": 2}, n_unknown=0)
    report = rating_shift.RatingShiftReport(
        boundary=BOUNDARY, market="domestic", scope="live", account_mode=None,
        before=before, after=after,
    )
    rendered = rating_shift.render_rating_shift(report)
    assert "boundary=" in rendered
    assert "market=domestic" in rendered
    assert "scope=live" in rendered
    assert "before boundary (n=4)" in rendered
    assert "after boundary (n=2)" in rendered
    assert "Hold" in rendered
    assert "50.0%" in rendered  # 2/4 Hold in "before"
    assert "(unparsed)" in rendered
    # No verdict language (the module deliberately disclaims a pass/fail
    # threshold in its trailing note, so that phrase itself is expected).
    for banned in ("overcorrection", "verdict:"):
        assert banned not in rendered.lower()


def test_render_rating_shift_empty_cohort():
    before = rating_shift.CohortRatingCounts(label="before", n=0)
    after = rating_shift.CohortRatingCounts(label="after", n=1, counts={"Hold": 1})
    report = rating_shift.RatingShiftReport(
        boundary=BOUNDARY, market=None, scope=None, account_mode=None, before=before, after=after,
    )
    rendered = rating_shift.render_rating_shift(report)
    assert "before boundary (n=0)" in rendered
    assert "(no rows)" in rendered
