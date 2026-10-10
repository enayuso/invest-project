"""The 16 sizing / portfolio / exposure rules that became per-user (2026-08-06).

These arrive from an HTTP body, and every one of them is a cap that decides how
much of somebody's money is at stake — so what matters is not that a good value
round-trips, but that a bad one cannot be stored.
HTTP 본문으로 들어오는 상한값들이므로, 정상값 왕복보다 '잘못된 값이 저장되지
않는지'가 핵심.
"""

from __future__ import annotations

import pytest

from app.execution import risk_rules
from app.execution.risk_rules import RULE_SPEC, _coerce


_NEW_KEYS = [
    "sizing_target_atr_pct", "sizing_min_vol_multiplier", "sizing_min_stop_atr_multiple",
    "sizing_max_stop_distance_pct", "sizing_max_risk_pct_of_equity",
    "portfolio_concentration_max_pct", "portfolio_correlation_bucket_threshold",
    "portfolio_correlation_bucket_max_pct", "portfolio_correlation_lookback_days",
    "portfolio_drawdown_breaker_pct", "portfolio_cash_floor_pct",
    "portfolio_cash_floor_pct_risk_off", "faber_sma_window_days",
    "max_open_positions_domestic", "max_open_positions_overseas", "kis_daily_max_loss_pct",
]

_FRACTIONS = [k for k in _NEW_KEYS if risk_rules._MAX_VALUE.get(k) == 1.0]
_INTEGERS = [k for k, t in RULE_SPEC.items() if t is int]


def test_every_new_key_is_in_the_spec():
    missing = [k for k in _NEW_KEYS if k not in RULE_SPEC]
    assert not missing, f"not registered as user-tunable: {missing}"


def test_every_new_key_has_a_settings_default_to_fall_back_to():
    """`rule()` falls back to getattr(settings, key) — a key with no matching
    field would raise AttributeError inside a live order gate."""
    from app.config import settings

    for key in _NEW_KEYS:
        assert hasattr(settings, key), f"{key} has no Settings default"


@pytest.mark.parametrize("key", _FRACTIONS)
def test_fraction_rules_reject_values_above_one(key):
    """0-1 fractions. Without a ceiling, sizing_max_risk_pct_of_equity=5 means
    'risk 500% of equity' and the cap stops being a cap."""
    with pytest.raises(ValueError):
        _coerce(key, 1.5)
    assert _coerce(key, 1.0) == 1.0
    assert _coerce(key, 0.25) == 0.25


@pytest.mark.parametrize("key", _NEW_KEYS)
def test_no_rule_accepts_a_negative(key):
    with pytest.raises(ValueError):
        _coerce(key, -0.01)


@pytest.mark.parametrize("key", _NEW_KEYS)
def test_no_rule_accepts_nan_or_infinity(key):
    """float() parses both from a JSON body, and either silently defeats every
    comparison a gate makes: `x > nan` is False, so a NaN cap blocks nothing."""
    for bad in ("nan", "inf", "-inf", float("nan"), float("inf")):
        with pytest.raises(ValueError):
            _coerce(key, bad)


@pytest.mark.parametrize("key", _INTEGERS)
def test_integer_rules_reject_fractional_values(key):
    with pytest.raises(ValueError):
        _coerce(key, 12.5)
    assert _coerce(key, 12) == 12
    assert isinstance(_coerce(key, 12.0), int)


@pytest.mark.parametrize("key", _INTEGERS)
def test_integer_rules_have_a_sanity_ceiling(key):
    with pytest.raises(ValueError):
        _coerce(key, 10_000_000)


def test_non_numeric_is_refused():
    with pytest.raises(ValueError):
        _coerce("portfolio_cash_floor_pct", "quite a lot")


def test_unknown_key_is_refused():
    with pytest.raises(ValueError):
        _coerce("portfolio_delete_everything", 1)


def test_consensus_override_thresholds_default_and_reject_out_of_range():
    """Desk-consensus override gate thresholds (2026-09-04, id=1212/1217) —
    user-overridable via the same path as hysteresis_min_confidence_to_flip,
    0-1 fractions."""
    from app.config import settings

    assert "consensus_override_min_confidence" in RULE_SPEC
    assert "consensus_override_min_confidence_bearish" in RULE_SPEC
    assert risk_rules.rule("consensus_override_min_confidence") == settings.consensus_override_min_confidence
    assert risk_rules.rule("consensus_override_min_confidence_bearish") == 0.85
    assert _coerce("consensus_override_min_confidence", 0.9) == 0.9
    with pytest.raises(ValueError):
        _coerce("consensus_override_min_confidence_bearish", 1.5)


def test_rule_is_per_user(monkeypatch):
    """Two users, one process, different caps — the whole point."""
    monkeypatch.setattr(risk_rules, "_overrides", {
        ("alice", "paper"): {"sizing_max_risk_pct_of_equity": 0.02},
        ("bob", "paper"): {"sizing_max_risk_pct_of_equity": 0.005},
    })
    assert risk_rules.rule("sizing_max_risk_pct_of_equity", "alice") == 0.02
    assert risk_rules.rule("sizing_max_risk_pct_of_equity", "bob") == 0.005
    # Someone with no override falls back to the deployment default.
    from app.config import settings

    assert risk_rules.rule("sizing_max_risk_pct_of_equity", "carol") == settings.sizing_max_risk_pct_of_equity


def test_current_reports_unconfigured_mode_scoped_values_instead_of_raising():
    """Regression, found live 2026-08-06: a brand-new user has no credentials,
    so their mode resolves to 'real'; the *_real money limits are unset on a
    deployment that hasn't configured them, and Settings._mode_scoped_money
    raises. That is right for an order gate and wrong for the settings screen —
    it 500'd before the user could configure anything.
    신규 사용자의 설정 화면이 500 나던 문제 — 표시 경로에서는 None으로 보고."""
    from app.config import account_mode_scope, settings

    assert settings.kis_max_order_krw_real is None, "precondition for this test"
    with account_mode_scope("real"):
        snapshot = risk_rules.current("brand-new-user")

    assert snapshot["values"]["kis_max_order_krw"] is None
    assert "kis_max_order_krw" in snapshot["unavailable"]
    # Non-mode-scoped rules still resolve normally alongside them.
    assert snapshot["values"]["portfolio_cash_floor_pct"] == settings.portfolio_cash_floor_pct


def test_the_gate_path_still_refuses_an_unconfigured_limit():
    """The other half: `rule()` must NOT be softened, or an order would be
    evaluated against a cap that does not exist."""
    import pytest as _pytest
    from app.config import account_mode_scope

    with account_mode_scope("real"):
        with _pytest.raises(RuntimeError):
            risk_rules.rule("kis_max_order_krw", "brand-new-user")


# ------------------------------------------------ operator-configured ceilings

def test_order_caps_are_unbounded_by_default(monkeypatch):
    """None = no ceiling, the pre-2026-08-06 behavior. Kept as the default so
    adding this could not silently start rejecting a value already in use.
    Checks the pydantic FIELD default directly (not the `settings` singleton)
    — the singleton loads this machine's real .env.local, which may set an
    actual ceiling for live trading, and that's not what this test is about.
    `_coerce` reads the LIVE settings singleton by design (an operator can
    reconfigure the ceiling at runtime) — monkeypatched to None here so the
    'unbounded' behavior this test checks doesn't depend on what any given
    machine's .env.local happens to set."""
    from app.config import Settings, settings

    assert Settings.model_fields["kis_max_order_krw_ceiling"].default is None
    monkeypatch.setattr(settings, "kis_max_order_krw_ceiling", None)
    assert _coerce("kis_max_order_krw", 10_000_000_000) == 10_000_000_000


def test_a_configured_ceiling_bounds_the_order_cap(monkeypatch):
    """The gap this closes: the order-value cap is the last backstop on order
    size, and one extra keystroke (55000000 for 5500000) used to sail through."""
    from app.config import settings

    monkeypatch.setattr(settings, "kis_max_order_krw_ceiling", 10_000_000.0)
    assert _coerce("kis_max_order_krw", 5_500_000) == 5_500_000
    assert _coerce("kis_max_order_krw", 10_000_000) == 10_000_000
    with pytest.raises(ValueError, match="between 0 and 10,000,000"):
        _coerce("kis_max_order_krw", 55_000_000)
    # The two numbers in the message must be distinguishable — `:g` rendered
    # both a 100,000,000 ceiling and a 100,000,001 input as "1e+08".
    monkeypatch.setattr(settings, "kis_max_order_krw_ceiling", 100_000_000.0)
    with pytest.raises(ValueError) as exc:
        _coerce("kis_max_order_krw", 100_000_001)
    assert "100,000,000" in str(exc.value) and "100,000,001" in str(exc.value)


def test_each_currency_has_its_own_ceiling(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "kis_max_order_krw_ceiling", 10_000_000.0)
    monkeypatch.setattr(settings, "kis_max_order_usd_ceiling", 50_000.0)
    with pytest.raises(ValueError):
        _coerce("kis_max_order_usd", 60_000)
    # The KRW ceiling must not leak across to the USD cap or vice versa.
    assert _coerce("kis_max_order_usd", 40_000) == 40_000


def test_approval_thresholds_need_no_ceiling_of_their_own(monkeypatch):
    """They are bounded transitively: _validate_cross_field rejects a
    threshold >= its cap, and the cap is ceilinged."""
    from app.config import settings

    monkeypatch.setattr(settings, "kis_max_order_krw_ceiling", 10_000_000.0)
    with pytest.raises(ValueError):
        risk_rules._validate_cross_field({
            "kis_approval_threshold_krw": _coerce("kis_approval_threshold_krw", 10_000_000),
            "kis_max_order_krw": _coerce("kis_max_order_krw", 10_000_000),
        })


def test_boot_refuses_a_ceiling_below_its_own_default():
    """Otherwise every user inherits a cap that _coerce would then refuse to
    save back — a state only an operator can cause and only they can fix."""
    from app.config import Settings

    with pytest.raises(ValueError, match="below kis_max_order_krw_paper"):
        Settings(kis_max_order_krw_paper=5_500_000.0, kis_max_order_krw_ceiling=1_000_000.0)


# ------------------------------------------------------- per-mode isolation

def test_a_users_paper_and_real_limits_are_independent(monkeypatch):
    """Regression: the override cache was keyed by user_id alone while the
    rows were mode-scoped, so caps tuned in 모의투자 came back as the same
    person's 실전 caps. Invisible while the process had one global mode.
    캐시가 모드를 구분하지 않아 모의투자 한도가 실전 한도로 새어나오던 문제."""
    from app.config import account_mode_scope

    monkeypatch.setattr(risk_rules, "_overrides", {
        ("alice", "paper"): {"kis_max_order_krw": 5_000_000.0},
        ("alice", "real"): {"kis_max_order_krw": 1_000_000.0},
    })
    with account_mode_scope("paper"):
        assert risk_rules.rule("kis_max_order_krw", "alice") == 5_000_000.0
    with account_mode_scope("real"):
        assert risk_rules.rule("kis_max_order_krw", "alice") == 1_000_000.0


def test_a_mode_with_no_override_falls_back_rather_than_borrowing_the_other(monkeypatch):
    """The dangerous half of the same bug: a user who tuned only paper must
    NOT inherit those numbers as their real-account limits."""
    from app.config import account_mode_scope, settings

    monkeypatch.setattr(risk_rules, "_overrides", {
        ("alice", "paper"): {"sizing_max_risk_pct_of_equity": 0.05},
    })
    with account_mode_scope("real"):
        assert risk_rules.rule("sizing_max_risk_pct_of_equity", "alice") == (
            settings.sizing_max_risk_pct_of_equity
        )
