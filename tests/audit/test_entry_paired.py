"""app.audit.entry_paired — asyncpg pool/connection faked, no real Postgres.
Same fake-pool convention as tests/audit/test_guard_alpha.py."""

from __future__ import annotations

from datetime import date

from app.audit import entry_paired


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


def _row(
    trade_date, rating, alpha_return, *,
    held_at_decision=None, held_at_decision_inferred=None, git_commit=None, created_at=None,
):
    return {
        "trade_date": trade_date, "ticker": "T", "rating": rating, "alpha_return": alpha_return,
        "held_at_decision": held_at_decision, "held_at_decision_inferred": held_at_decision_inferred,
        "git_commit": git_commit, "created_at": created_at,
    }


# ---------------------------------------------------------------------------
# _classify_hold: three-state held/flat/unknown classification.
# ---------------------------------------------------------------------------


def test_classify_hold_observed_column_wins():
    assert entry_paired._classify_hold(_row(date(2026, 9, 1), "Hold", 0.0, held_at_decision=True)) is False
    assert entry_paired._classify_hold(_row(date(2026, 9, 1), "Hold", 0.0, held_at_decision=False)) is True


def test_classify_hold_falls_back_to_inferred():
    assert entry_paired._classify_hold(
        _row(date(2026, 9, 1), "Hold", 0.0, held_at_decision_inferred=True)
    ) is False
    assert entry_paired._classify_hold(
        _row(date(2026, 9, 1), "Hold", 0.0, held_at_decision_inferred=False)
    ) is True


def test_classify_hold_neither_set_is_unknown():
    assert entry_paired._classify_hold(_row(date(2026, 9, 1), "Hold", 0.0)) is None


# ---------------------------------------------------------------------------
# Analysis 1: pairing logic.
# ---------------------------------------------------------------------------


def test_day_without_both_cohorts_is_dropped():
    rows = [
        # Day 1: Buy only — must not contribute a paired day.
        _row(date(2026, 9, 1), "Buy", 0.05),
        # Day 2: Hold(flat) only — must not contribute a paired day.
        _row(date(2026, 9, 2), "Hold", 0.03, held_at_decision=False),
        # Day 3: both present — must pair.
        _row(date(2026, 9, 3), "Buy", 0.02),
        _row(date(2026, 9, 3), "Hold", 0.01, held_at_decision=False),
    ]
    report = entry_paired._build_paired_report(
        rows, basis="variable", market=None, scope=None, account_mode=None,
    )
    assert report.n_paired_days == 1
    assert report.n_buy_only_days == 1
    assert report.n_hold_only_days == 1


def test_unknown_held_state_excluded_and_counted():
    rows = [
        _row(date(2026, 9, 1), "Buy", 0.05),
        # Neither held_at_decision nor inferred set — must be excluded, not guessed.
        _row(date(2026, 9, 1), "Hold", 0.01),
    ]
    report = entry_paired._build_paired_report(
        rows, basis="variable", market=None, scope=None, account_mode=None,
    )
    assert report.n_excluded_unknown_held == 1
    assert report.n_hold_flat_rows == 0
    assert report.n_paired_days == 0


def test_held_hold_row_not_counted_as_flat_or_excluded():
    rows = [
        _row(date(2026, 9, 1), "Buy", 0.05),
        # held_at_decision=True -> HELD, not flat — not part of this comparison,
        # and not "excluded" since its state IS known.
        _row(date(2026, 9, 1), "Hold", 0.01, held_at_decision=True),
    ]
    report = entry_paired._build_paired_report(
        rows, basis="variable", market=None, scope=None, account_mode=None,
    )
    assert report.n_excluded_unknown_held == 0
    assert report.n_hold_flat_rows == 0
    assert report.n_buy_rows == 1


def test_raw_alpha_used_not_sign_flipped():
    """A Hold(flat) row with POSITIVE alpha_return must contribute +alpha to
    its cohort mean, never -alpha — mixing in the 'signed alpha' convention
    (where Hold(flat) wins on alpha<0) would silently invert this cohort."""
    rows = [
        _row(date(2026, 9, 1), "Buy", 0.10),
        _row(date(2026, 9, 1), "Hold", 0.05, held_at_decision=False),
    ]
    report = entry_paired._build_paired_report(
        rows, basis="variable", market=None, scope=None, account_mode=None,
    )
    # delta = mean(buy) - mean(hold_flat) = 0.10 - 0.05 = 0.05, NOT 0.10 - (-0.05) = 0.15.
    assert abs(report.mean_daily_delta - 0.05) < 1e-9


def test_daily_delta_arithmetic_hand_computed():
    rows = [
        # Day 1: buy mean = (0.10+0.20)/2 = 0.15, hold_flat mean = 0.05 -> delta = 0.10
        _row(date(2026, 9, 1), "Buy", 0.10),
        _row(date(2026, 9, 1), "Buy", 0.20),
        _row(date(2026, 9, 1), "Hold", 0.05, held_at_decision=False),
        # Day 2: buy mean = 0.00, hold_flat mean = 0.02 -> delta = -0.02
        _row(date(2026, 9, 2), "Buy", 0.00),
        _row(date(2026, 9, 2), "Hold", 0.02, held_at_decision=False),
    ]
    report = entry_paired._build_paired_report(
        rows, basis="variable", market=None, scope=None, account_mode=None,
    )
    assert report.n_paired_days == 2
    expected = (0.10 + (-0.02)) / 2
    assert abs(report.mean_daily_delta - expected) < 1e-9
    assert report.n_buy_rows == 3
    assert report.n_hold_flat_rows == 2


def test_no_paired_days_yields_none_stats():
    rows = [_row(date(2026, 9, 1), "Buy", 0.05)]
    report = entry_paired._build_paired_report(
        rows, basis="variable", market=None, scope=None, account_mode=None,
    )
    assert report.n_paired_days == 0
    assert report.mean_daily_delta is None
    assert report.ci95 is None
    assert report.p_value is None
    rendered = entry_paired.render_entry_paired_alpha(report)
    assert "cannot be computed" in rendered


def test_render_includes_key_numbers():
    rows = [
        _row(date(2026, 9, 1), "Buy", 0.10),
        _row(date(2026, 9, 1), "Hold", 0.05, held_at_decision=False),
        _row(date(2026, 9, 2), "Buy", 0.02),
        _row(date(2026, 9, 2), "Hold", 0.01, held_at_decision=False),
        _row(date(2026, 9, 3), "Buy", -0.01),
        _row(date(2026, 9, 3), "Hold", 0.03, held_at_decision=False),
    ]
    report = entry_paired._build_paired_report(
        rows, basis="horizon", market="domestic", scope="live", account_mode="paper",
    )
    rendered = entry_paired.render_entry_paired_alpha(report)
    assert "n_buy_rows=3" in rendered
    assert "n_hold_flat_rows=3" in rendered
    assert "n_paired_days=3" in rendered
    assert "mean daily delta" in rendered
    assert "market=domestic" in rendered
    assert "basis=horizon" in rendered


# ---------------------------------------------------------------------------
# Analysis 3: code-version bucketing.
# ---------------------------------------------------------------------------


def test_null_git_commit_gets_own_bucket():
    rows = [
        _row(date(2026, 9, 1), "Buy", 0.05, git_commit=None),
        _row(date(2026, 9, 1), "Hold", 0.01, held_at_decision=False, git_commit=None),
        _row(date(2026, 9, 8), "Buy", 0.02, git_commit="abc1234"),
        _row(date(2026, 9, 8), "Hold", 0.01, held_at_decision=False, git_commit="abc1234"),
    ]
    report = entry_paired._build_version_report(
        rows, basis="variable", market=None, scope=None, account_mode=None,
    )
    labels = [b.label for b in report.buckets]
    assert entry_paired._NO_COMMIT_LABEL in labels
    # NULL-commit bucket must be first.
    assert labels[0] == entry_paired._NO_COMMIT_LABEL
    no_commit_bucket = next(b for b in report.buckets if b.label == entry_paired._NO_COMMIT_LABEL)
    assert no_commit_bucket.n_buy == 1
    assert no_commit_bucket.n_hold_flat == 1
    assert no_commit_bucket.commits == []


def test_same_iso_week_groups_together_with_commit_range():
    rows = [
        _row(date(2026, 9, 1), "Buy", 0.10, git_commit="commit1"),
        _row(date(2026, 9, 3), "Buy", 0.20, git_commit="commit2"),
        _row(date(2026, 9, 3), "Hold", 0.05, held_at_decision=False, git_commit="commit2"),
    ]
    report = entry_paired._build_version_report(
        rows, basis="variable", market=None, scope=None, account_mode=None,
    )
    assert len(report.buckets) == 1
    bucket = report.buckets[0]
    assert bucket.label == entry_paired._iso_week_label(date(2026, 9, 1))
    assert bucket.commits == ["commit1", "commit2"]
    assert bucket.n_buy == 2
    assert abs(bucket.mean_buy_alpha - 0.15) < 1e-9
    assert bucket.n_hold_flat == 1


def test_different_iso_weeks_produce_separate_buckets():
    rows = [
        _row(date(2026, 9, 1), "Buy", 0.10, git_commit="commit1"),  # week of Sep 1
        _row(date(2026, 9, 15), "Buy", 0.20, git_commit="commit2"),  # different week
    ]
    report = entry_paired._build_version_report(
        rows, basis="variable", market=None, scope=None, account_mode=None,
    )
    week_labels = {b.label for b in report.buckets}
    assert len(week_labels) == 2


def test_version_report_unknown_held_excluded_and_counted():
    rows = [
        _row(date(2026, 9, 1), "Buy", 0.05, git_commit="commit1"),
        _row(date(2026, 9, 1), "Hold", 0.01, git_commit="commit1"),  # neither held column set
    ]
    report = entry_paired._build_version_report(
        rows, basis="variable", market=None, scope=None, account_mode=None,
    )
    assert report.n_excluded_unknown_held == 1


def test_render_code_version_stratification_smoke():
    rows = [
        _row(date(2026, 9, 1), "Buy", 0.10, git_commit="commit1"),
        _row(date(2026, 9, 1), "Hold", 0.02, held_at_decision=False, git_commit="commit1"),
    ]
    report = entry_paired._build_version_report(
        rows, basis="variable", market=None, scope=None, account_mode=None,
    )
    rendered = entry_paired.render_code_version_stratification(report)
    assert "Code-version stratification" in rendered
    assert "commit1"[:8] in rendered


# ---------------------------------------------------------------------------
# end-to-end wiring: compute_entry_paired_alpha / compute_code_version_
# stratification issue one query each and build a report from the fake
# pool's fetch result.
# ---------------------------------------------------------------------------


async def test_compute_entry_paired_alpha_end_to_end(monkeypatch):
    rows = [
        _row(date(2026, 9, 1), "Buy", 0.05),
        _row(date(2026, 9, 1), "Hold", 0.01, held_at_decision=False),
    ]
    conn = FakeConn(fetch_result=rows)
    monkeypatch.setattr(entry_paired, "get_pool", lambda: _async_return(FakePool(conn)))
    monkeypatch.setattr(entry_paired, "get_current_user_id", lambda: "default")

    report = await entry_paired.compute_entry_paired_alpha(scope="live", account_mode="paper")

    assert report.n_buy_rows == 1
    assert report.n_hold_flat_rows == 1
    assert len(conn.queries) == 1
    query, params = conn.queries[0]
    assert "rating IN ('Buy', 'Hold')" in query
    assert "is_backtest = false" in query
    assert "paper" in params


async def test_compute_code_version_stratification_end_to_end(monkeypatch):
    rows = [
        _row(date(2026, 9, 1), "Buy", 0.05, git_commit="deadbeef"),
        _row(date(2026, 9, 1), "Hold", 0.01, held_at_decision=False, git_commit="deadbeef"),
    ]
    conn = FakeConn(fetch_result=rows)
    monkeypatch.setattr(entry_paired, "get_pool", lambda: _async_return(FakePool(conn)))
    monkeypatch.setattr(entry_paired, "get_current_user_id", lambda: "default")

    report = await entry_paired.compute_code_version_stratification(market="domestic")

    assert len(report.buckets) == 1
    assert report.buckets[0].commits == ["deadbeef"]
    query, params = conn.queries[0]
    assert "git_commit" in query
    assert "ticker LIKE '%.KS'" in query


def test_validate_filters_rejects_bad_values():
    import pytest

    with pytest.raises(ValueError):
        entry_paired._validate_filters("bogus", None, None, "variable")
    with pytest.raises(ValueError):
        entry_paired._validate_filters(None, "bogus", None, "variable")
    with pytest.raises(ValueError):
        entry_paired._validate_filters(None, None, "bogus", "variable")
    with pytest.raises(ValueError):
        entry_paired._validate_filters(None, None, None, "bogus")
