"""Runtime store for the dashboard-toggleable AI Profile / Trading Mode axes.

Same resolution order and persistence shape as `app/execution/risk_rules.py`
(in-memory override, set from the dashboard and persisted to Postgres, checked
before falling back to the `settings` default) — kept as a separate module
rather than folded into risk_rules.py because that module's documented
responsibility is risk *knobs* specifically; ai_profile/execution_mode are a
different axis (AI persona, approval routing), not a risk limit.
risk_rules.py와 동일한 오버라이드 패턴이지만, 리스크 수치가 아닌 AI 성향/승인모드 축이라
SRP상 별도 모듈로 분리.

`kis_account_mode` (paper/real) is deliberately absent from RUNTIME_SPEC and
never touched here — Trading Environment stays config-file-only, per
docs/safety-and-risk.md 5.1 (real-account switching must always be a
deliberate server-side edit, never reachable from the unauthenticated
localhost dashboard API).
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from app.config import AIProfile, ExecutionMode, current_account_mode, settings
from app.user_context import get_current_user_id

logger = logging.getLogger(__name__)

# key -> Enum class. Values are stored as the enum's own `.value` string and
# reconstructed via `RUNTIME_SPEC[key](raw)` (both are `str, Enum`).
RUNTIME_SPEC: dict[str, type] = {
    "ai_profile": AIProfile,
    "execution_mode": ExecutionMode,
}

_TABLE = """
CREATE TABLE IF NOT EXISTS runtime_settings (
    key        TEXT NOT NULL,
    value      JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Paper/real account separation (2026-08-05 — see project memory:
-- user-account-mode-db-separation-design). Before this column, an
-- execution_mode=FULL_AUTO override saved while paper-testing would
-- silently survive a switch to kis_account_mode='real' with zero
-- re-validation (load() only restores from Postgres, it never re-runs
-- _validate_real_account_guard) — the real account would inherit
-- whatever automation level was last set for paper with no confirmation.
-- Same drop-PK/unique-index treatment as portfolio_drawdown_state, since
-- Postgres has no `ADD CONSTRAINT IF NOT EXISTS` and this block re-runs on
-- every process start.
ALTER TABLE runtime_settings DROP CONSTRAINT IF EXISTS runtime_settings_pkey;
ALTER TABLE runtime_settings ADD COLUMN IF NOT EXISTS account_mode TEXT NOT NULL DEFAULT 'paper'
    CHECK (account_mode IN ('paper', 'real'));

-- Per-user separation (multi-user migration Phase 4). THE highest-severity
-- item in that plan: until this column existed, execution_mode was one value
-- for the whole process, so one person selecting FULL_AUTO put every other
-- user's account into autonomous trading. Per-user credentials without this
-- is strictly more dangerous than the old single-user setup, because the
-- orders it fires land in other people's real accounts.
-- 이 컬럼 이전에는 execution_mode가 프로세스 전역이라, 한 사용자가 FULL_AUTO를
-- 켜면 다른 모든 사용자의 계좌가 자동매매로 전환됐음.
--
-- Backfilled to 'default' exactly as account_mode was backfilled to 'paper':
-- every pre-existing row belongs to the single user who created it, and
-- 'default' is the id every historical row in this system already carries
-- (see app.memory.db's users seed row).
ALTER TABLE runtime_settings ADD COLUMN IF NOT EXISTS user_id TEXT NOT NULL DEFAULT 'default';

-- Superseded by the three-column index below. Dropped explicitly because the
-- old two-column uniqueness is actively WRONG now — it would let one user's
-- save collide with another's on the same (key, account_mode).
-- 예전 2컬럼 유니크는 사용자 간 저장이 서로 충돌하므로 반드시 제거.
DROP INDEX IF EXISTS runtime_settings_key_mode_uidx;
CREATE UNIQUE INDEX IF NOT EXISTS runtime_settings_key_mode_user_uidx
    ON runtime_settings (key, account_mode, user_id);
"""

# user_id -> {key -> enum member}. A missing user, or a missing key for a
# present scope, means "use the settings default".
#
# Keyed by (user_id, account_mode), NOT user alone. execution_mode is exactly
# the kind of value that must not cross the paper/real boundary: a FULL_AUTO
# override set while paper-testing would otherwise answer a real-money read
# for the same person. Same bug that was found and fixed in risk_rules'
# cache — keyed by user but not by mode, so paper limits leaked into real.
# 사용자만으로 키를 잡으면 모의투자에서 켠 FULL_AUTO가 실전 조회에 응답하게 됨
# (risk_rules에서 실제로 발생했던 버그와 동일).
#
# Loaded eagerly for EVERY scope rather than fetched per call, because `rule()`
# is synchronous and called from deep inside sync code paths (kis_bridge's
# gates, the pipeline) that cannot await a query.
# rule()이 동기 함수라 호출 시점에 조회할 수 없어, 전체를 미리 적재하는 구조.
_overrides: dict[tuple[str, str], dict[str, Any]] = {}

_schema_ready = False
_schema_lock = asyncio.Lock()


async def _ensure_schema(conn) -> None:
    global _schema_ready
    if _schema_ready:
        return
    async with _schema_lock:
        if not _schema_ready:
            await conn.execute(_TABLE)
            _schema_ready = True


def rule(key: str, user_id: str | None = None, account_mode: str | None = None) -> Any:
    """The effective value of a runtime setting **for one user**: that user's
    override if they set one, otherwise the `settings` default.

    Still synchronous, deliberately — every existing call site
    (kis_bridge's gates, ledger.rebalance_to_target, the pipeline,
    auto_trade_loop) is sync or deep inside one, and making this a coroutine
    would ripple through the entire execution path for no benefit. That is
    why `_overrides` is preloaded rather than queried here.

    `user_id` defaults to the current request/loop identity, so no call site
    had to change. The background loops bind theirs with `user_scope`
    (Phase 1), which is what makes an unattended FULL_AUTO cycle run as the
    right person.
    인자 없이 호출하면 현재 사용자·모드 기준 — 기존 호출부를 바꾸지 않기 위한 설계."""
    scope = (user_id or get_current_user_id(), account_mode or current_account_mode())
    mine = _overrides.get(scope)
    if mine is not None and key in mine:
        return mine[key]
    return getattr(settings, key)


def current(user_id: str | None = None, account_mode: str | None = None) -> dict[str, Any]:
    who = user_id or get_current_user_id()
    mode = account_mode or current_account_mode()
    return {
        "values": {key: rule(key, who, mode).value for key in RUNTIME_SPEC},
        "overridden": sorted(_overrides.get((who, mode), {})),
        # Read-only — Trading Environment is never part of RUNTIME_SPEC.
        "kis_account_mode": mode,
    }


def _coerce(key: str, raw: Any) -> Any:
    """Validate/convert one incoming value against RUNTIME_SPEC. Raises
    ValueError on an unknown key or a value that isn't a valid enum member."""
    if key not in RUNTIME_SPEC:
        raise ValueError(f"unknown runtime setting {key!r}")
    enum_cls = RUNTIME_SPEC[key]
    if isinstance(raw, enum_cls):
        return raw
    try:
        return enum_cls(raw)
    except ValueError:
        allowed = ", ".join(m.value for m in enum_cls)
        raise ValueError(f"{key} expects one of [{allowed}], got {raw!r}")


def _validate_real_account_guard(coerced: dict[str, Any], mode: str) -> None:
    """Same fail-safe condition as app.config.Settings's boot-time
    model_validator, replicated here because a runtime override can flip
    ai_profile without a restart — the boot-time check alone can't catch it.
    kis_account_mode itself is never touched by this module, so this is the
    only place that needs to re-check the combination.
    부팅시 검증과 동일한 조건을 런타임 오버라이드에도 복제 — 재부팅 없이 바뀌므로 필요.

    `mode` is taken explicitly rather than read from `current_account_mode()`
    here — `save()` may be asked to write on behalf of a mode other than the
    request's own (see its own docstring), and `load()` walks every (user,
    mode) scope at once with no request bound at all, so the ambient
    ContextVar is the wrong thing to check in either case; only the mode
    actually being written/loaded is."""
    if "ai_profile" not in coerced:
        return
    profile = coerced["ai_profile"]
    if profile is AIProfile.PRODUCTION:
        return
    if mode == "real" and not settings.ai_profile_live_ack:
        raise ValueError(
            f"ai_profile={profile.value} with kis_account_mode='real' requires "
            "ai_profile_live_ack=True (set in the config file) — refusing to switch "
            "to a non-production AI profile against the real account without an "
            "explicit acknowledgement."
        )


async def load() -> None:
    """Populate the override cache from Postgres for **every** user. Quietly
    no-ops (empty cache → settings defaults) if the DB is unconfigured or
    unreachable.

    Loads every (user, mode) scope, not just the serviced one, because
    `rule()` is sync and has no way to fetch a missing scope's row at call
    time — a scope absent from this cache silently falls back to the settings
    default, which for `execution_mode` means MANUAL. That fallback is the safe
    direction (it withholds automation rather than granting it), but it is
    still wrong, so the cache has to be complete.

    It used to filter `WHERE account_mode = $1` on the mode that happened to be
    current at startup. With per-user account modes there IS no such mode at
    startup — no user is bound, so it resolved to the deployment default and
    every user on the other mode silently lost their overrides.
    예전엔 부팅 시점 모드로 필터링했는데, 사용자별 모드에서는 부팅 시점에 바인딩된
    사용자가 없어 배포 기본값으로 해석됨 — 다른 모드 사용자는 오버라이드를 잃었음.
    캐시에 없는 스코프는 기본값(MANUAL)으로 떨어짐 — 안전한 방향이지만 부정확함."""
    global _overrides
    try:
        from app.memory.db import get_pool

        pool = await get_pool()
        async with pool.acquire() as conn:
            await _ensure_schema(conn)
            rows = await conn.fetch(
                "SELECT key, value, user_id, account_mode FROM runtime_settings"
            )
    except Exception:  # noqa: BLE001 - DB optional, same convention as risk_rules
        logger.info("runtime_settings: store unavailable, using settings defaults", exc_info=True)
        return
    loaded: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = row["key"]
        if key not in RUNTIME_SPEC:
            continue
        scope = (row["user_id"], row["account_mode"])
        try:
            # asyncpg returns jsonb columns as raw JSON text (no codec
            # registered), e.g. '"test_aggressive"' with the quotes still in
            # it — json.loads first to get the plain string back, unlike
            # risk_rules.py's float/bool values whose JSON text happens to
            # already equal their coerced form. Skipped here previously,
            # every enum override silently failed to survive a restart.
            # asyncpg는 jsonb를 원시 JSON 텍스트로 반환 — 문자열 값은 따옴표까지
            # 포함되므로 json.loads로 한 번 풀어줘야 함(재시작 후 복원 안 되던 버그 수정).
            loaded.setdefault(scope, {})[key] = _coerce(key, json.loads(row["value"]))
        except ValueError:
            logger.warning(
                "runtime_settings: dropping invalid stored value for %s (user_id=%s, mode=%s)",
                key, row["user_id"], row["account_mode"],
            )
    # Re-check the real-account/ai_profile guard on load, not just save() —
    # without this, an override saved while ai_profile_live_ack=True (real
    # account, non-production profile) survives a restart even after the
    # operator flips ai_profile_live_ack back to False specifically to
    # re-arm this safety check: save()'s own guard only ever ran once, at
    # the moment it was originally set, and load() blindly restored
    # whatever was last persisted with no re-validation. Dropped (not
    # raised) rather than failing the whole load — same "log and fall back
    # to the config default" convention as an unparseable stored value
    # just above. Checked per scope (the scope tuple's own mode), not once
    # over the whole map — see _validate_real_account_guard's own docstring.
    for (who, mode), scope_overrides in loaded.items():
        if "ai_profile" not in scope_overrides:
            continue
        try:
            _validate_real_account_guard(scope_overrides, mode)
        except ValueError as exc:
            logger.warning(
                "runtime_settings: dropping stored ai_profile override for user_id=%s mode=%s — %s",
                who, mode, exc,
            )
            del scope_overrides["ai_profile"]
    _overrides = loaded
    logger.info(
        "runtime_settings: loaded overrides for %d (user, mode) scope(s): %s",
        len(_overrides), {f"{u}/{m}": sorted(v) for (u, m), v in _overrides.items()},
    )


async def save(
    values: dict[str, Any], user_id: str | None = None, account_mode: str | None = None
) -> dict[str, Any]:
    """Validate and persist runtime-setting overrides, then refresh the
    cache. Raises ValueError on a bad key/value or the real-account safety
    guard, RuntimeError if the store is unreachable (nothing changes in that
    case so the dashboard and the running gates never diverge).

    `account_mode` defaults to the request's own mode, matching
    `risk_rules.save`. Passing it explicitly is for callers that write on
    someone's behalf outside their request context."""
    who = user_id or get_current_user_id()
    mode = account_mode or current_account_mode()
    coerced = {key: _coerce(key, raw) for key, raw in values.items()}
    _validate_real_account_guard(coerced, mode)
    try:
        from app.memory.db import get_pool

        pool = await get_pool()
        async with pool.acquire() as conn:
            await _ensure_schema(conn)
            async with conn.transaction():
                for key, value in coerced.items():
                    await conn.execute(
                        "INSERT INTO runtime_settings (key, value, account_mode, user_id, updated_at) "
                        "VALUES ($1, $2::jsonb, $3, $4, now()) "
                        "ON CONFLICT (key, account_mode, user_id) DO UPDATE SET "
                        "value = EXCLUDED.value, updated_at = now()",
                        key, json.dumps(value.value), mode, who,
                    )
    except Exception as exc:  # noqa: BLE001
        logger.warning("runtime_settings: save failed", exc_info=True)
        raise RuntimeError(f"runtime-settings store unavailable: {exc}") from exc
    _overrides.setdefault((who, mode), {}).update(coerced)
    logger.info(
        "runtime_settings: saved %s for user_id=%s mode=%s",
        {k: v.value for k, v in coerced.items()}, who, mode,
    )
    return current(who, mode)
