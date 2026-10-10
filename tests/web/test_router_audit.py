"""audit router — TestClient, no network. Every `app.audit.*` call is
monkeypatched; this only tests the HTTP shaping (validation, JSON field
names, null-safety, error mapping), not the underlying statistics (those
are covered in tests/audit/)."""

from __future__ import annotations

from datetime import date, datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.audit.stats import ConfidenceBucketStats, ConfidenceCalibration, DebateImpactStats, RatingStats
from app.audit.weekly import F7Cell, F7Panel, RatingRow, WeeklyReport
from app.web.routers import audit as audit_router


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(audit_router.router)
    return TestClient(app, base_url="http://127.0.0.1:8094")


@pytest.fixture(autouse=True)
def _fixed_account_mode(monkeypatch):
    monkeypatch.setattr(audit_router, "current_account_mode", lambda: "paper")


def _rating_stats(**overrides):
    defaults = dict(
        rating="Buy", n=10, win_rate=0.6, mean_alpha=0.01, mean_raw_return=0.02,
        win_condition="alpha > 0", unscorable_n=0, stdev_alpha=0.05,
    )
    defaults.update(overrides)
    return RatingStats(**defaults)


def _confidence_calibration():
    return ConfidenceCalibration(
        buckets=[
            ConfidenceBucketStats(
                label="[0.90, 1.00]", lower=0.90, size_multiplier=1.0, n=4,
                win_rate=0.5, mean_signed_alpha=0.0102, unscorable_n=0,
            ),
        ],
        n_with_confidence=4, n_without_confidence=6, signed_alpha_corr=0.142,
    )


def _debate_impact():
    return DebateImpactStats(n=91, changed=34, changed_pct=0.3736, breakdown={"Buy->Hold": 21})


# ---------------------------------------------------------------------------
# /api/audit/accuracy
# ---------------------------------------------------------------------------

async def _fake_hit_rate(**kwargs):
    return [_rating_stats(), _rating_stats(rating="Hold", n=90, win_rate=None, win_condition="unscored (needs held/flat)")]


async def _fake_confidence(**kwargs):
    return _confidence_calibration()


async def _fake_debate(**kwargs):
    return _debate_impact()


def test_accuracy_happy_path_shape(client, monkeypatch):
    monkeypatch.setattr(audit_router, "compute_hit_rate_by_rating", _fake_hit_rate)
    monkeypatch.setattr(audit_router, "compute_confidence_calibration", _fake_confidence)
    monkeypatch.setattr(audit_router, "compute_debate_impact", _fake_debate)

    resp = client.get("/api/audit/accuracy")
    assert resp.status_code == 200
    body = resp.json()

    assert body["filters"] == {
        "market": None, "scope": "live", "trigger": "auto", "basis": "variable", "account_mode": "paper",
    }
    assert body["thresholds"] == {"min_n_for_ratio": 10, "min_n_for_f7_verdict": 30}
    assert body["headline"] == {
        "directional_n": 10, "directional_win_rate": pytest.approx(0.6),
        "unscored_hold_n": 90, "unscorable_n": 0,
    }
    assert body["ratings"][0] == {
        "rating": "Buy", "n": 10, "win_rate": pytest.approx(0.6), "mean_alpha": pytest.approx(0.01),
        "mean_raw_return": pytest.approx(0.02), "win_condition": "alpha > 0",
        "unscorable_n": 0, "stdev_alpha": pytest.approx(0.05), "inferred_n": 0,
    }
    # Hold has no directional win_rate — must round-trip as JSON null, never 0.
    assert body["ratings"][1]["win_rate"] is None
    assert body["confidence"]["n_with_confidence"] == 4
    assert body["confidence"]["buckets"][0]["label"] == "[0.90, 1.00]"
    assert body["debate"]["n"] == 91
    assert body["debate"]["breakdown"] == {"Buy->Hold": 21}


def test_accuracy_passes_filters_through(client, monkeypatch):
    seen = {}

    async def _capture_hit_rate(**kwargs):
        seen.update(kwargs)
        return []

    monkeypatch.setattr(audit_router, "compute_hit_rate_by_rating", _capture_hit_rate)
    monkeypatch.setattr(audit_router, "compute_confidence_calibration", _fake_confidence)
    monkeypatch.setattr(audit_router, "compute_debate_impact", _fake_debate)

    resp = client.get("/api/audit/accuracy?market=domestic&scope=backtest&basis=horizon")
    assert resp.status_code == 200
    assert seen == {
        "market": "domestic", "scope": "backtest", "account_mode": "paper", "trigger": "auto", "basis": "horizon",
    }
    assert resp.json()["filters"] == {
        "market": "domestic", "scope": "backtest", "trigger": "auto", "basis": "horizon", "account_mode": "paper",
    }


def test_accuracy_trigger_all_pools_manual_and_auto(client, monkeypatch):
    seen = {}

    async def _capture_hit_rate(**kwargs):
        seen.update(kwargs)
        return []

    monkeypatch.setattr(audit_router, "compute_hit_rate_by_rating", _capture_hit_rate)
    monkeypatch.setattr(audit_router, "compute_confidence_calibration", _fake_confidence)
    monkeypatch.setattr(audit_router, "compute_debate_impact", _fake_debate)

    resp = client.get("/api/audit/accuracy?trigger=all")
    assert resp.status_code == 200
    # "all" means pooled -> None reaches the stats layer, but the raw query
    # value is echoed back in filters so the frontend can label it.
    assert seen["trigger"] is None
    assert resp.json()["filters"]["trigger"] == "all"


def test_accuracy_trigger_manual_passes_through(client, monkeypatch):
    seen = {}

    async def _capture_hit_rate(**kwargs):
        seen.update(kwargs)
        return []

    monkeypatch.setattr(audit_router, "compute_hit_rate_by_rating", _capture_hit_rate)
    monkeypatch.setattr(audit_router, "compute_confidence_calibration", _fake_confidence)
    monkeypatch.setattr(audit_router, "compute_debate_impact", _fake_debate)

    resp = client.get("/api/audit/accuracy?trigger=manual")
    assert resp.status_code == 200
    assert seen["trigger"] == "manual"
    assert resp.json()["filters"]["trigger"] == "manual"


def test_accuracy_rejects_bad_trigger(client):
    resp = client.get("/api/audit/accuracy?trigger=bogus")
    assert resp.status_code == 400


def test_accuracy_headline_win_rate_is_none_when_no_directional_rows(client, monkeypatch):
    async def _only_hold(**kwargs):
        return [_rating_stats(rating="Hold", n=5, win_rate=None, win_condition="unscored (needs held/flat)")]

    monkeypatch.setattr(audit_router, "compute_hit_rate_by_rating", _only_hold)
    monkeypatch.setattr(audit_router, "compute_confidence_calibration", _fake_confidence)
    monkeypatch.setattr(audit_router, "compute_debate_impact", _fake_debate)

    resp = client.get("/api/audit/accuracy")
    assert resp.status_code == 200
    assert resp.json()["headline"]["directional_win_rate"] is None


@pytest.mark.parametrize("query", ["market=mars", "scope=nope", "basis=nope"])
def test_accuracy_rejects_bad_query_params(client, query):
    resp = client.get(f"/api/audit/accuracy?{query}")
    assert resp.status_code == 400


def test_accuracy_returns_503_on_db_failure(client, monkeypatch):
    async def _boom(**kwargs):
        raise RuntimeError("pool exhausted")

    monkeypatch.setattr(audit_router, "compute_hit_rate_by_rating", _boom)
    monkeypatch.setattr(audit_router, "compute_confidence_calibration", _fake_confidence)
    monkeypatch.setattr(audit_router, "compute_debate_impact", _fake_debate)

    resp = client.get("/api/audit/accuracy")
    assert resp.status_code == 503


def test_accuracy_echoes_active_account_mode(client, monkeypatch):
    monkeypatch.setattr(audit_router, "current_account_mode", lambda: "real")
    monkeypatch.setattr(audit_router, "compute_hit_rate_by_rating", _fake_hit_rate)
    monkeypatch.setattr(audit_router, "compute_confidence_calibration", _fake_confidence)
    monkeypatch.setattr(audit_router, "compute_debate_impact", _fake_debate)

    resp = client.get("/api/audit/accuracy")
    assert resp.json()["filters"]["account_mode"] == "real"


# ---------------------------------------------------------------------------
# /api/audit/weekly
# ---------------------------------------------------------------------------

def _weekly_report(**overrides):
    defaults = dict(
        week_start=date(2026, 8, 17),
        week_end=date(2026, 8, 23),
        scope="live",
        account_mode="paper",
        trigger="auto",
        basis="variable",
        generated_at=datetime(2026, 8, 20, 18, 42, 11),
        this_week_n=6,
        this_week_unscorable_n=1,
        last_8_weeks_n=44,
        cumulative_n=91,
        ratings=[
            RatingRow(rating="Buy", this_week=None, last_8_weeks=_rating_stats(n=19, win_rate=0.3684),
                      cumulative=_rating_stats(n=27, win_rate=0.3704)),
        ],
        confidence=_confidence_calibration(),
        f7=F7Panel(cells=[F7Cell(row_label="Hold (held)", n=11, wins=6, win_rate=None)], total_overridden=18),
    )
    defaults.update(overrides)
    return WeeklyReport(**defaults)


def test_weekly_happy_path_shape(client, monkeypatch):
    async def _fake_weekly(**kwargs):
        return _weekly_report()

    monkeypatch.setattr(audit_router, "compute_weekly_report", _fake_weekly)

    resp = client.get("/api/audit/weekly")
    assert resp.status_code == 200
    body = resp.json()

    assert body["week_label"] == "2026-W34"
    assert body["week_start"] == "2026-08-17"
    assert body["week_end"] == "2026-08-23"
    assert body["filters"] == {"scope": "live", "trigger": "auto", "basis": "variable", "account_mode": "paper"}
    assert body["thresholds"] == {"min_n_for_ratio": 10, "min_n_for_f7_verdict": 30}
    assert body["totals"] == {
        "this_week_n": 6, "this_week_unscorable_n": 1, "last_8_weeks_n": 44, "cumulative_n": 91,
    }
    row = body["ratings"][0]
    assert row["rating"] == "Buy"
    # A window with no rows for this rating is null, not a zeroed object.
    assert row["this_week"] is None
    assert row["last_8_weeks"]["n"] == 19
    assert row["cumulative"]["win_rate"] == pytest.approx(0.3704)
    assert body["f7"]["total_overridden"] == 18
    assert body["f7"]["cells"][0]["win_rate"] is None  # below _MIN_N_FOR_F7_VERDICT


def test_weekly_passes_parsed_week_through(client, monkeypatch):
    seen = {}

    async def _capture_weekly(**kwargs):
        seen.update(kwargs)
        return _weekly_report()

    monkeypatch.setattr(audit_router, "compute_weekly_report", _capture_weekly)

    resp = client.get("/api/audit/weekly?week=2026-W34&scope=backtest&basis=horizon")
    assert resp.status_code == 200
    assert seen["week"] == date(2026, 8, 17)
    assert seen["scope"] == "backtest"
    assert seen["basis"] == "horizon"
    assert seen["account_mode"] == "paper"
    assert seen["trigger"] == "auto"


def test_weekly_trigger_all_pools_manual_and_auto(client, monkeypatch):
    seen = {}

    async def _capture_weekly(**kwargs):
        seen.update(kwargs)
        return _weekly_report()

    monkeypatch.setattr(audit_router, "compute_weekly_report", _capture_weekly)

    resp = client.get("/api/audit/weekly?trigger=all")
    assert resp.status_code == 200
    assert seen["trigger"] is None
    assert resp.json()["filters"]["trigger"] == "all"


def test_weekly_rejects_bad_trigger(client):
    resp = client.get("/api/audit/weekly?trigger=bogus")
    assert resp.status_code == 400


def test_weekly_rejects_malformed_week(client):
    resp = client.get("/api/audit/weekly?week=not-a-week")
    assert resp.status_code == 400


@pytest.mark.parametrize("query", ["scope=nope", "basis=nope"])
def test_weekly_rejects_bad_query_params(client, query):
    resp = client.get(f"/api/audit/weekly?{query}")
    assert resp.status_code == 400


def test_weekly_returns_503_on_db_failure(client, monkeypatch):
    async def _boom(**kwargs):
        raise RuntimeError("pool exhausted")

    monkeypatch.setattr(audit_router, "compute_weekly_report", _boom)

    resp = client.get("/api/audit/weekly")
    assert resp.status_code == 503
