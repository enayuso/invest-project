"""Runtime store for the user-settable risk rules (A-3).

The A-1 gates in `kis_bridge.py` read their thresholds through `rule(key)`
here instead of straight off `settings`, so a value the user changes in the
dashboard takes effect on the very next order without a restart. Resolution
order is: in-memory override (set from the dashboard, persisted to Postgres)
→ the `settings` default. The override cache is loaded once at app startup
(`load()`) and refreshed on every `save()`, so the gates stay synchronous
(no DB round-trip per order).

Only *risk* knobs are settable here. The master switch
(`KIS_EXECUTION_ENABLED`) and `KIS_EXECUTION_DRY_RUN` are deliberately NOT.

That was originally because the order endpoints were unauthenticated and
localhost-bound, so a "go live" toggle over the API would have let anything on
the machine flip the system into real submission. **That premise is gone** —
the API requires a session now (multi-user migration Phase 1) and the user has
decided these two should eventually become user-controllable. They stay
config-only until the prerequisites for that are in place: TLS with
`secure=True` session cookies, CSRF protection on state-changing routes, and
password re-entry as step-up for this specific toggle. Without those, a stolen
session would be enough to enable live order submission, not merely to place a
paper order.
원래 근거(무인증·로컬호스트)는 사라졌고 사용자도 전환에 동의했으나, TLS/CSRF/
재인증이 갖춰지기 전까지는 설정 전용 유지 — 세션 탈취만으로 실주문이 켜지면 안 됨.

The order-value caps below ARE user-settable, bounded by the operator-set
`kis_max_order_*_ceiling` (see `_settings_ceiling`): the user owns their risk
appetite, the deployment owns the outer bound.

Postgres is the same shared, quiet-degrade instance as the memory store: if
it is unreachable, `load()` leaves the cache empty (gates fall back to
`settings` defaults) and `save()` reports the failure to the caller.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import Any

from app.config import current_account_mode, settings
from app.execution import _schema_bootstrap
from app.user_context import get_current_user_id

logger = logging.getLogger(__name__)

# The settable rules and their python types — the single source of truth for
# what the API accepts and how each value is coerced. Defaults live on
# `settings`; this only records which keys are user-tunable and their type.
RULE_SPEC: dict[str, type] = {
    "kis_max_order_krw": float,
    "kis_max_order_usd": float,
    "kis_max_position_pct_enabled": bool,
    "kis_max_position_pct": float,
    "kis_approval_required_enabled": bool,
    "kis_approval_threshold_krw": float,
    "kis_approval_threshold_usd": float,
    # Position sizing (2026-08-06). These decide how much of one person's own
    # money a single decision puts at stake, so they belong to that person
    # rather than to the deployment — the reason they were global was that
    # there was only ever one person.
    # 한 사람의 자금이 얼마나 걸리는지를 정하는 값이므로 배포가 아니라 사용자 소유.
    "sizing_target_atr_pct": float,
    "sizing_min_vol_multiplier": float,
    "sizing_min_stop_atr_multiple": float,
    "sizing_max_stop_distance_pct": float,
    "sizing_max_risk_pct_of_equity": float,
    "trailing_stop_atr_multiplier": float,
    "trailing_stop_activation_pct": float,
    "trailing_stop_fallback_pct": float,
    "hysteresis_min_confidence_to_flip": float,
    "consensus_override_min_confidence": float,
    "consensus_override_min_confidence_bearish": float,
    # Portfolio-level caps. Same reasoning one level up: these look across a
    # whole book, and the book is per-user.
    "portfolio_concentration_max_pct": float,
    "portfolio_correlation_bucket_threshold": float,
    "portfolio_correlation_bucket_max_pct": float,
    "portfolio_correlation_lookback_days": int,
    "portfolio_drawdown_breaker_pct": float,
    "portfolio_cash_floor_pct": float,
    "portfolio_cash_floor_pct_risk_off": float,
    "faber_sma_window_days": int,
    # Exposure limits.
    "max_open_positions_domestic": int,
    "max_open_positions_overseas": int,
    "kis_daily_max_loss_pct": float,
}

# Per-key upper bounds. Without these a fraction field would happily accept
# 5.0 and mean "risk 500% of equity" — every one of these is a 0-1 fraction
# EXCEPT kis_max_position_pct (0-100, the documented outlier in app.config)
# and the day-count/position-count integers. A cap that can be set to nonsense
# is not a cap, and these arrive from an HTTP body.
# 비율 필드에 5.0 같은 값이 들어오면 상한이 상한 구실을 못 하므로 키별 상한 지정.
_MAX_VALUE: dict[str, float] = {
    "kis_max_position_pct": 100.0,
    "sizing_target_atr_pct": 1.0,
    "sizing_min_vol_multiplier": 1.0,
    "sizing_max_stop_distance_pct": 1.0,
    "sizing_max_risk_pct_of_equity": 1.0,
    "trailing_stop_activation_pct": 1.0,
    "trailing_stop_fallback_pct": 1.0,
    "hysteresis_min_confidence_to_flip": 1.0,
    "consensus_override_min_confidence": 1.0,
    "consensus_override_min_confidence_bearish": 1.0,
    # Not a fraction — sanity ceiling only (mirrors sizing_min_stop_atr_multiple).
    "trailing_stop_atr_multiplier": 20.0,
    "portfolio_concentration_max_pct": 1.0,
    "portfolio_correlation_bucket_threshold": 1.0,
    "portfolio_correlation_bucket_max_pct": 1.0,
    "portfolio_drawdown_breaker_pct": 1.0,
    "portfolio_cash_floor_pct": 1.0,
    "portfolio_cash_floor_pct_risk_off": 1.0,
    "kis_daily_max_loss_pct": 1.0,
    # Not fractions — sanity ceilings only, to catch a fat-fingered value
    # rather than to express a policy.
    "sizing_min_stop_atr_multiple": 20.0,
    "portfolio_correlation_lookback_days": 2000,
    "faber_sma_window_days": 2000,
    "max_open_positions_domestic": 500,
    "max_open_positions_overseas": 500,
}

_TABLE = """
CREATE TABLE IF NOT EXISTS risk_rules (
    key        TEXT NOT NULL,
    value      JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Paper/real account separation (2026-08-05 — see project memory:
-- user-account-mode-db-separation-design). Before this column, a relaxed
-- order cap / disabled approval-requirement set during paper testing would
-- silently carry over the moment kis_account_mode flips to 'real' — the
-- real account would inherit whatever risk limits were last convenient for
-- paper, with no re-validation. Same drop-PK/unique-index treatment as
-- portfolio_drawdown_state, since Postgres has no `ADD CONSTRAINT IF NOT
-- EXISTS` and this block re-runs on every process start.
ALTER TABLE risk_rules DROP CONSTRAINT IF EXISTS risk_rules_pkey;
ALTER TABLE risk_rules ADD COLUMN IF NOT EXISTS account_mode TEXT NOT NULL DEFAULT 'paper'
    CHECK (account_mode IN ('paper', 'real'));

-- Per-user separation (multi-user migration Phase 4, plan §2.2). Order value
-- caps, position-% caps and approval thresholds were shared, so one user
-- raising their own cap raised it for everyone — a risk limit that is not
-- yours to set is not a limit. Backfilled to 'default' like account_mode was
-- to 'paper'; the old two-column unique index is DROPped because it would now
-- collide one user's save with another's.
-- 리스크 한도가 공유되면 한 사용자가 올린 한도가 전원에게 적용됨.
ALTER TABLE risk_rules ADD COLUMN IF NOT EXISTS user_id TEXT NOT NULL DEFAULT 'default';
DROP INDEX IF EXISTS risk_rules_key_mode_uidx;
CREATE UNIQUE INDEX IF NOT EXISTS risk_rules_key_mode_user_uidx
    ON risk_rules (key, account_mode, user_id);
"""

# In-memory override cache: (user_id, account_mode) -> {rule key -> value}.
# A missing scope means "use the settings defaults". Preloaded for every scope
# rather than fetched per call because `rule()` is synchronous and called from
# inside kis_bridge's order gates — same reasoning as runtime_settings.
#
# Keyed by (user_id, account_mode), not user_id alone. The rows have always
# been mode-scoped, but this cache was not — so a user who tuned their caps in
# 모의투자 saw those same values as their 실전 caps, and `load()` only ever
# fetched ONE mode's rows, leaving the other mode silently on deployment
# defaults. Both were invisible while the process had a single global mode.
# 행은 모드별로 저장되는데 캐시는 사용자별이라 두 모드가 서로의 값을 보던 문제.
_overrides: dict[tuple[str, str], dict[str, Any]] = {}

# CREATE TABLE IF NOT EXISTS is idempotent but not free — every load()/save()
# call re-ran it against Postgres with no guard, unlike safety_switch.py's/
# portfolio_risk.py's already-established _schema_ready flag pattern.
# 매 호출마다 스키마 생성 쿼리를 반복 실행하지 않도록 다른 모듈과 동일한 이중 확인 잠금 적용.
_schema_ready = False
_schema_lock = asyncio.Lock()


async def _ensure_schema(conn) -> None:
    """Double-checked locking around the first-time DDL — shared with the
    other app.execution modules via _schema_bootstrap."""
    await _schema_bootstrap.ensure_schema(sys.modules[__name__], conn, _TABLE)


def rule(key: str, user_id: str | None = None) -> Any:
    """The effective value of a rule **for one user**: their override if set,
    otherwise the `settings` default. Synchronous — safe to call from a gate,
    which is exactly why the cache is preloaded rather than queried here."""
    who = user_id or get_current_user_id()
    mine = _overrides.get((who, current_account_mode()))
    if mine is not None and key in mine:
        return mine[key]
    return getattr(settings, key)


def current(user_id: str | None = None) -> dict[str, Any]:
    """The full resolved rule set (override-or-default per key) for the API /
    dashboard, plus which keys that user currently overrides.

    An unconfigured mode-scoped value is reported as None rather than raising.
    The mode-scoped money pairs (kis_max_order_*/kis_approval_threshold_*)
    have no `_real` value until an administrator sets one, and
    Settings._mode_scoped_money deliberately raises in that case — correct for
    an order gate, wrong here. Reading this endpoint is how someone SEES their
    limits; a brand-new user has no credentials, so their mode resolves to
    'real' (see app.account_mode), and propagating would 500 the settings
    screen for every new account before they can configure anything. Confirmed
    live, 2026-08-06.

    This is a display path only. `rule()` is untouched, so every gate still
    refuses rather than proceeding with a missing cap.
    표시용 경로에서만 None으로 완화 — 실제 게이트(rule)는 그대로 예외를 던짐."""
    who = user_id or get_current_user_id()
    values: dict[str, Any] = {}
    unavailable: list[str] = []
    for key in RULE_SPEC:
        try:
            values[key] = rule(key, who)
        except RuntimeError:
            values[key] = None
            unavailable.append(key)
    return {
        "values": values,
        "overridden": sorted(_overrides.get((who, current_account_mode()), {})),
        # So the UI can say "an administrator has not configured this" instead
        # of rendering an empty box that looks like a value the user cleared.
        "unavailable": sorted(unavailable),
    }


# Rules whose upper bound is set by the deployment rather than by the shape of
# the value. Only the two order-value CAPS: their approval thresholds are
# bounded transitively by _validate_cross_field (threshold < cap).
_CEILING_SETTING: dict[str, str] = {
    "kis_max_order_krw": "kis_max_order_krw_ceiling",
    "kis_max_order_usd": "kis_max_order_usd_ceiling",
}


def _settings_ceiling(key: str) -> float | None:
    """The operator-configured ceiling for `key`, or None if unbounded.

    Mode-agnostic on purpose: one ceiling per currency applies to paper and
    real alike. A separate paper ceiling would be extra configuration guarding
    simulated money, and app.config's boot check already keeps the ceiling at
    or above both mode defaults so neither can be left unsavable.
    통화별 상한 하나를 두 모드에 공통 적용 — 모의투자용 별도 상한은 가상 자금을
    지키자고 설정만 늘리는 셈."""
    name = _CEILING_SETTING.get(key)
    return getattr(settings, name) if name else None


def _coerce(key: str, raw: Any) -> Any:
    """Validate/convert one incoming value against RULE_SPEC. Raises
    ValueError on an unknown key or a value that won't convert / is invalid."""
    if key not in RULE_SPEC:
        raise ValueError(f"unknown risk rule {key!r}")
    expected = RULE_SPEC[key]
    if expected is bool:
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str):
            normalized = raw.strip().lower()
            if normalized in ("true", "1", "yes", "on"):
                return True
            if normalized in ("false", "0", "no", "off"):
                return False
        # Anything else (an unrecognized string, None, a list, ...) is
        # rejected rather than passed through python's own truthiness
        # (`bool(raw)`) — that used to silently accept e.g. "" -> False or
        # any typo'd non-empty string -> True for gates like
        # kis_approval_required_enabled, unlike runtime_settings._coerce's
        # sibling enum validation, which already rejects an unrecognized
        # value outright instead of guessing. A safety-gate toggle that
        # can't tell "off" from "disabled" (typo) is worse than one that
        # just refuses the save.
        raise ValueError(f"{key} expects a boolean, got {raw!r}")
    # numeric rules: reject non-numeric, negative, and out-of-range values
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"{key} expects a number, got {raw!r}")
    if value != value or value in (float("inf"), float("-inf")):
        # float() happily parses "nan" and "inf" from a JSON body. Either one
        # silently defeats every comparison a gate makes against it: `x > nan`
        # is False, so a NaN cap blocks nothing at all.
        # NaN/Inf는 모든 비교를 무력화해 상한이 사실상 사라지므로 거부.
        raise ValueError(f"{key} must be a finite number, got {raw!r}")
    if value < 0:
        raise ValueError(f"{key} must be non-negative, got {value}")
    ceiling = _MAX_VALUE.get(key)
    if ceiling is None:
        # Deployment-configured ceiling, read at validation time rather than
        # baked into _MAX_VALUE: it comes from the environment and an operator
        # can change it, unlike the structural 0-1 bounds above.
        # 배포 설정값이라 정적 표가 아니라 검증 시점에 읽음.
        ceiling = _settings_ceiling(key)
    if ceiling is not None and value > ceiling:
        # `:,.10g`, not `:g`: the latter collapses anything large into
        # scientific notation, so rejecting 100,000,001 against a
        # 100,000,000 ceiling rendered as "between 0 and 1e+08, got 1e+08" —
        # the same string twice, reading as if a value equal to the limit had
        # been refused. Observed live 2026-08-06.
        # :g는 큰 수를 지수표기로 뭉개서 한도와 입력값이 같은 문자열로 보임.
        raise ValueError(f"{key} must be between 0 and {ceiling:,.10g}, got {value:,.10g}")
    if expected is int:
        if value != int(value):
            raise ValueError(f"{key} expects a whole number, got {raw!r}")
        return int(value)
    return value


def _effective(key: str, coerced: dict[str, Any]) -> Any:
    """The value `key` would resolve to if `coerced` were applied — the
    incoming value if this save touches it, else whatever `rule()` already
    resolves to (override or settings default)."""
    return coerced[key] if key in coerced else rule(key)


def _validate_cross_field(coerced: dict[str, Any]) -> None:
    """The approval threshold only ever matters for orders in the range
    [threshold, cap] — `over_value_cap` (kis_bridge.py) hard-blocks anything
    strictly ABOVE the cap (`est_value > cap`, not `>=`; a value exactly at
    the cap still needs approval, it isn't auto-blocked), so an order needing
    approval never even reaches that check unless the threshold is strictly
    below the cap.
    승인 임계값은 [threshold, cap] 구간(양끝 포함)에서만 의미가 있음 — over_value_cap은
    캡을 "초과"할 때만(> cap) 강제 차단하고, 캡과 정확히 같은 값은 여전히 승인이 필요함.
    Without this, raising the approval threshold — the actual lever the
    agent-architecture-roadmap's gradual-automation plan turns — could
    silently do nothing if it's left at or above the cap. Only checked when
    this save touches either half of a pair, so saves that don't touch
    either don't re-validate an already-accepted combination."""
    for threshold_key, cap_key in (
        ("kis_approval_threshold_krw", "kis_max_order_krw"),
        ("kis_approval_threshold_usd", "kis_max_order_usd"),
    ):
        if threshold_key not in coerced and cap_key not in coerced:
            continue
        threshold = _effective(threshold_key, coerced)
        cap = _effective(cap_key, coerced)
        if threshold >= cap:
            raise ValueError(
                f"{threshold_key} ({threshold:,.2f}) must be less than {cap_key} ({cap:,.2f}) — "
                "otherwise every order that would need approval is already hard-blocked by the "
                "value cap first, making the approval threshold pointless"
            )


def _validate_cross_field_on_load(loaded: dict[str, Any]) -> None:
    """Same invariant as `_validate_cross_field`, but for `load()`'s
    "degrade quietly" contract: drop the offending override(s) and fall
    back to the settings default instead of raising. `save()` already
    blocks a bad combination from ever being written, but a row already in
    Postgres can still end up violating it later — e.g. the boot-config
    ceiling (env file) for the untouched half of the pair was lowered after
    the override was saved. Loading that row as-is would silently make the
    approval gate pointless (see `_validate_cross_field`'s docstring)."""
    for threshold_key, cap_key in (
        ("kis_approval_threshold_krw", "kis_max_order_krw"),
        ("kis_approval_threshold_usd", "kis_max_order_usd"),
    ):
        if threshold_key not in loaded and cap_key not in loaded:
            continue
        threshold = loaded.get(threshold_key, getattr(settings, threshold_key))
        cap = loaded.get(cap_key, getattr(settings, cap_key))
        if threshold >= cap:
            logger.warning(
                "risk_rules: stored %s (%.2f) >= %s (%.2f) violates the threshold<cap invariant — "
                "dropping both stored overrides and falling back to settings defaults",
                threshold_key,
                threshold,
                cap_key,
                cap,
            )
            loaded.pop(threshold_key, None)
            loaded.pop(cap_key, None)


async def load() -> None:
    """Populate the override cache from Postgres. Quietly no-ops (empty cache
    → settings defaults) if the DB is unconfigured or unreachable."""
    global _overrides
    try:
        from app.memory.db import get_pool

        pool = await get_pool()
        async with pool.acquire() as conn:
            await _ensure_schema(conn)
            # Every mode, not just the current one: with per-user account
            # modes, one process serves 실전 and 모의투자 users at the same
            # time, so a single-mode load leaves half of them on defaults.
            # 모드별로 사용자가 다르므로 전체를 불러와야 함.
            rows = await conn.fetch("SELECT key, value, user_id, account_mode FROM risk_rules")
    except Exception:  # noqa: BLE001 - DB optional, same convention as memory store
        logger.info("risk_rules: store unavailable, using settings defaults", exc_info=True)
        return
    loaded: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = row["key"]
        if key not in RULE_SPEC:  # a key removed from the spec since it was saved
            continue
        scope = (row["user_id"], row["account_mode"])
        try:
            loaded.setdefault(scope, {})[key] = _coerce(key, row["value"])
        except ValueError:
            logger.warning(
                "risk_rules: dropping invalid stored value for %s (user_id=%s, mode=%s)",
                key, row["user_id"], row["account_mode"],
            )
    # Per scope, not once over the whole (user, mode) -> {key: value} map —
    # _validate_cross_field_on_load expects a flat {key: value} dict for ONE
    # scope. Calling it on `loaded` itself (keyed by scope tuples) would make
    # every `threshold_key not in loaded` check trivially True, silently
    # turning this into a no-op across every scope.
    for scope_overrides in loaded.values():
        _validate_cross_field_on_load(scope_overrides)
    _overrides = loaded
    logger.info(
        "risk_rules: loaded overrides for %d (user, mode) scope(s): %s",
        len(_overrides), {f"{u}/{m}": sorted(v) for (u, m), v in _overrides.items()},
    )


async def save(
    values: dict[str, Any], user_id: str | None = None, account_mode: str | None = None
) -> dict[str, Any]:
    """Validate and persist rule overrides, then refresh the cache. Returns
    the new resolved rule set. Raises ValueError on a bad key/value, or
    RuntimeError if the store is unreachable (nothing is changed in that case
    so the dashboard and the gates never diverge).

    `account_mode` defaults to the request's own mode. Onboarding passes it
    explicitly: a user who registers BOTH credential sets resolves to 모의투자
    (the safer default), so a plain save there would write their 실전 order
    limits into the paper scope and leave the real caps at 0 — blocking every
    real order they later place.
    온보딩은 모드를 명시함 — 두 계좌를 다 등록하면 활성 모드가 모의투자가 되어
    실전 한도가 엉뚱한 스코프에 저장되기 때문."""
    who = user_id or get_current_user_id()
    mode = account_mode or current_account_mode()
    coerced = {key: _coerce(key, raw) for key, raw in values.items()}
    _validate_cross_field(coerced)
    try:
        from app.memory.db import get_pool

        pool = await get_pool()
        async with pool.acquire() as conn:
            await _ensure_schema(conn)
            async with conn.transaction():
                for key, value in coerced.items():
                    # JSON-encode the scalar and cast to jsonb — one code path
                    # for both bool and float, no per-type SQL branching.
                    await conn.execute(
                        "INSERT INTO risk_rules (key, value, account_mode, user_id, updated_at) "
                        "VALUES ($1, $2::jsonb, $3, $4, now()) "
                        "ON CONFLICT (key, account_mode, user_id) DO UPDATE SET "
                        "value = EXCLUDED.value, updated_at = now()",
                        key, json.dumps(value), mode, who,
                    )
    except Exception as exc:  # noqa: BLE001
        logger.warning("risk_rules: save failed", exc_info=True)
        raise RuntimeError(f"risk-rule store unavailable: {exc}") from exc
    _overrides.setdefault((who, mode), {}).update(coerced)
    logger.info("risk_rules: saved %s for user_id=%s mode=%s", coerced, who, mode)
    return current(who)
