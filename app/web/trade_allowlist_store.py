"""Persists `settings.trade_allowlist_domestic`/`_overseas`/`settings.watchlist`/
`settings.trade_allowlist_auto_added` across restarts.

All three are otherwise pure in-memory pydantic Settings fields (env-var
backed, mutated in place by app.web.state.add_ticker_to_trade_allowlist/
add_ticker_to_watchlist and the dashboard's remove endpoints) — a server
restart silently reset them to the `.env` defaults, dropping every ticker
added during the running session (AI auto-add-on-BUY included) and, worse,
making a real KIS holding that fell off the list invisible to /api/portfolio
(see app.web.routers.portfolio's own docstring). Same load-at-startup/save-on-
change shape as `app.execution.runtime_settings`, kept as its own module for
the same SRP reason that one gives for staying separate from risk_rules.py —
this is a different axis (which tickers exist) from AI persona/execution
mode.

재시작해도 트레이드얼라우리스트(국내/해외)/관심종목이 유지되도록 Postgres에 저장.

Paper/real account separation (2026-08-04 — see project memory:
user-account-mode-db-separation-design): every persisted key is namespaced
with the CURRENT `current_account_mode()` (`_scoped`) before it ever
touches Postgres. Before this, trade_allowlist was one shared list
regardless of account mode — a ticker the AI auto-added to the allowlist
while paper-testing would carry straight over the moment kis_account_mode
flips to 'real', making it immediately tradable for real money without any
human ever reviewing it for that purpose.

`load()` also migrates bare (pre-scoping) `_KEYS` rows the first time it
runs after this change — every row this project has ever written used the
unscoped name, so without this migration, restarting under the unchanged
'paper' mode would find nothing under the new `_scoped` names and silently
reset trade_allowlist/watchlist/auto-added to the `.env` defaults, exactly
as if a person had never used the dashboard. Reissues each bare row under
save() (which writes it back under the scoped name), then deletes the old
bare row — one migration pass per key.

`load()` also migrates the OLD scoped key from before the `favorites` ->
`watchlist` rename (2026-08-05) — same one-time reissue-then-delete shape,
just old-scoped-name -> new-scoped-name instead of unscoped -> scoped.
"""

from __future__ import annotations

import asyncio
import json
import logging

from app.config import account_mode_scope, current_account_mode, normalize_scope, settings
from app.user_context import get_current_user_id, user_scope

logger = logging.getLogger(__name__)

_TABLE = """
ALTER TABLE IF EXISTS watchlist_state RENAME TO trade_allowlist_state;

CREATE TABLE IF NOT EXISTS trade_allowlist_state (
    key        TEXT PRIMARY KEY,
    tickers    JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

# key -> the settings attribute it restores/persists.
_KEYS = ("trade_allowlist_domestic", "trade_allowlist_overseas", "watchlist", "trade_allowlist_auto_added")

_schema_ready = False
_schema_lock = asyncio.Lock()


def _scoped(key: str, user_id: str | None = None, account_mode: str | None = None) -> str:
    """Namespaces a persistence key with the CURRENT user AND account_mode, so
    neither two people nor paper-and-real ever read or write each other's
    trade_allowlist_state row.

    user_id joined the prefix in the multi-user migration Phase 4 (plan §2.2).
    One shared allowlist across users is not a cosmetic problem: this is the
    list `kis_bridge` gates on to decide what may be traded at all, so with it
    shared, adding a ticker to your own allowlist silently authorizes it for
    everyone else's automated trading too.
    이 목록이 kis_bridge의 거래 허용 판단 기준이라, 공유되면 내가 추가한 종목이
    다른 사용자의 자동매매에도 허용됨.

    Order is user-then-mode so a `LIKE 'user:%'` prefix scan can find one
    person's rows across both modes — the shape Phase 6 will want when it has
    to load per-user state on login.

    Both arguments default to the current context, so every existing call site
    is unchanged. They exist because `load()` needs to name its scope
    explicitly — see that function.
    """
    return f"{_prefixed_scope(user_id, account_mode)}{key}"


def _prefixed_scope(user_id: str | None = None, account_mode: str | None = None) -> str:
    who = user_id or get_current_user_id()
    mode = account_mode or current_account_mode()
    return f"{who}:{mode}:"


def _mode_scoped_legacy(key: str, account_mode: str | None = None) -> str:
    """The pre-Phase-4 key shape: account_mode only, no user. Read once by
    `load()` to migrate, never written."""
    return f"{account_mode or current_account_mode()}:{key}"


async def _ensure_schema(conn) -> None:
    global _schema_ready
    if _schema_ready:
        return
    async with _schema_lock:
        if not _schema_ready:
            await conn.execute(_TABLE)
            _schema_ready = True


async def load(user_id: str | None = None, account_mode: str | None = None) -> None:
    """Overwrite `settings.trade_allowlist_domestic`/`_overseas`/
    `settings.watchlist`/`settings.trade_allowlist_auto_added` with
    whatever was last persisted for ONE (user, mode) scope — call once at
    startup, BEFORE anything reads any of them (history seeding, the realtime
    feed's initial KRX codes, ...). Quietly no-ops (keeps the
    `.env`/legacy-migration defaults) if the store is
    unconfigured/unreachable/empty, same degrade convention as
    `runtime_settings.load`.

    **One scope, not all of them — and that is a limitation, not a design.**
    Unlike `risk_rules.load` / `runtime_settings.load`, which were fixed to
    load every `(user, mode)` scope into a keyed cache, this function's
    destination is the process-global `settings` lists. There is exactly one of
    each, so there is nowhere to put a second user's allowlist. Making the
    in-memory copy per-user is the remaining Phase 5 work
    (docs/multi-user-credentials-plan.md); until then one scope is the most
    this can honestly restore.
    다른 두 스토어와 달리 목적지가 전역 settings 리스트 하나뿐이라, 여러 스코프를
    담을 곳이 없음 — 인메모리 사본의 사용자별 분리가 남은 Phase 5 작업.

    What DID change (2026-08-06): the scope is now an explicit argument rather
    than whatever `get_current_user_id()`/`current_account_mode()` happened to
    resolve to. At startup no user is bound and no mode is chosen, so the old
    implicit form silently resolved to the deployment default — a real-mode
    user's allowlist was never restored, and nothing said so. Callers that pass
    nothing get the same behavior as before; the difference is that a caller
    which knows whose state it wants can now say.
    스코프를 인자로 명시 — 부팅 시점에는 바인딩된 사용자·모드가 없어 배포 기본값으로
    조용히 해석됐음.
    """
    who = user_id or get_current_user_id()
    mode = account_mode or current_account_mode()
    try:
        from app.memory.db import get_pool

        pool = await get_pool()
        async with pool.acquire() as conn:
            await _ensure_schema(conn)
            rows = await conn.fetch(
                "SELECT key, tickers FROM trade_allowlist_state WHERE key = ANY($1)",
                # Scoped keys (the normal case going forward), PLUS the bare,
                # pre-account_mode-scoping names (needed for the one-time
                # migration below), PLUS the OLD scoped "favorites" key from
                # before the favorites -> watchlist rename.
                [
                    *(_scoped(k, who, mode) for k in _KEYS),
                    *_KEYS,
                    _scoped("favorites", who, mode),
                    # Pre-Phase-4 (account_mode-only) names, for the one-time
                    # user-scoping migration below.
                    *(_mode_scoped_legacy(k, mode) for k in _KEYS),
                    _mode_scoped_legacy("favorites", mode),
                ],
            )
    except Exception:  # noqa: BLE001 - DB optional, same convention as runtime_settings
        logger.info("trade_allowlist_store: store unavailable, using .env defaults", exc_info=True)
        return
    by_key = {row["key"]: json.loads(row["tickers"]) for row in rows}
    restored = []

    # One-time migration (multi-user migration Phase 4): a key scoped by
    # account_mode but not by user (e.g. "paper:watchlist") predates per-user
    # scoping. Every such row was written when this process served exactly one
    # person, so it belongs to 'default' — the same id every other historical
    # row in this system carries, and the same reasoning that backfilled
    # account_mode to 'paper'.
    #
    # Gated on the current identity actually BEING 'default', for the same
    # reason the bare-key migration below is gated on paper mode: reissuing
    # one-user-era data under whichever user happens to log in first would
    # hand them someone else's allowlist. A new user finds nothing and starts
    # empty, which is correct.
    # 1인 시절 데이터를 '먼저 로그인한 사람'에게 넘겨주지 않도록 default일 때만 이관.
    migrated_mode_keys = []
    if who == "default":
        for key in (*_KEYS, "favorites"):
            legacy_key = _mode_scoped_legacy(key, mode)
            scoped_key = _scoped(key, who, mode)
            if legacy_key not in by_key:
                continue
            if scoped_key in by_key:
                # Both shapes present. This is genuinely ambiguous and must
                # NOT be resolved by guessing:
                #   - preferring the legacy row would clobber whatever has
                #     been saved under the new key since the upgrade;
                #   - preferring the new row silently discards the legacy
                #     data, which is how this was found in the first place
                #     (a stray "default:paper:trade_allowlist_auto_added"
                #     row stranded an 11-ticker legacy row);
                #   - merging is the worst of the three, because a ticker the
                #     user deliberately REMOVED from the allowlist would come
                #     back — and this list is a trading permission.
                # So: keep both, change nothing, and say so on every boot
                # until a human resolves it. Recurring noise is the point;
                # this should not be ignorable.
                # 두 형태가 공존하면 무엇이 옳은지 알 수 없으므로 추측하지 않음 —
                # 특히 병합은 사용자가 일부러 뺀 종목을 되살려 거래 권한을 주게 됨.
                logger.warning(
                    "trade_allowlist_store: BOTH %r (%d tickers) and %r (%d tickers) exist. "
                    "Not migrating automatically — the correct answer is ambiguous and this list "
                    "grants trading permission. Delete whichever row is wrong, by hand.",
                    legacy_key, len(by_key[legacy_key]), scoped_key, len(by_key[scoped_key]),
                )
                continue
            by_key[scoped_key] = by_key[legacy_key]
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO trade_allowlist_state (key, tickers, updated_at)
                    VALUES ($1, $2::jsonb, now())
                    ON CONFLICT (key) DO UPDATE SET tickers = EXCLUDED.tickers, updated_at = now()
                    """,
                    scoped_key, json.dumps(by_key[scoped_key]),
                )
                await conn.execute("DELETE FROM trade_allowlist_state WHERE key = $1", legacy_key)
            migrated_mode_keys.append(legacy_key)
        if migrated_mode_keys:
            logger.info(
                "trade_allowlist_store: migrated pre-user-scoping key(s) %s to the 'default' user",
                sorted(migrated_mode_keys),
            )

    # One-time migration (favorites -> watchlist rename, 2026-08-05): the OLD
    # scoped key predates this rename — reissue its value under the new
    # scoped key once, then delete the old row. Same "reissue under save(),
    # then delete the old row" shape as the bare-key migration below.
    old_favorites_scoped = _scoped("favorites", who, mode)
    new_watchlist_scoped = _scoped("watchlist", who, mode)
    if old_favorites_scoped in by_key and new_watchlist_scoped not in by_key:
        by_key[new_watchlist_scoped] = by_key[old_favorites_scoped]
        await save("watchlist", by_key[new_watchlist_scoped], who, mode)
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM trade_allowlist_state WHERE key = $1", old_favorites_scoped)
        logger.info(
            "trade_allowlist_store: migrated %s -> %s", old_favorites_scoped, new_watchlist_scoped,
        )

    # One-time migration (2026-08-04): a bare key (e.g. "trade_allowlist_domestic")
    # predates account_mode scoping entirely — every row this project has
    # EVER written used these exact bare names, and every one of them is
    # 'paper' data (this project has never run in 'real' mode). Gated on
    # kis_account_mode == 'paper' specifically — reissuing it under whatever
    # mode happens to be running would otherwise let a bare row (always
    # paper's data) leak straight into 'real' the first time this runs under
    # that mode, exactly the cross-mode leak this whole migration exists to
    # prevent. A first-ever 'real' run finds nothing under either name,
    # which is correct: start empty, never silently inherit paper's list.
    # Without this migration (paper-mode case), restarting would otherwise
    # make every trade_allowlist/watchlist/auto-added entry accumulated so
    # far invisible (the old bare row sits unmatched by the new scoped
    # lookup), resetting them to the `.env` defaults.
    migrated_bare_keys = []
    if mode == "paper":
        for key in _KEYS:
            scoped_key = _scoped(key, who, mode)
            if scoped_key in by_key or key not in by_key:
                continue
            by_key[scoped_key] = by_key[key]
            await save(key, by_key[scoped_key], who, mode)
            async with pool.acquire() as conn:
                await conn.execute("DELETE FROM trade_allowlist_state WHERE key = $1", key)
            migrated_bare_keys.append(key)
        if migrated_bare_keys:
            logger.info(
                "trade_allowlist_store: migrated pre-account_mode-scoping key(s) %s into the 'paper' namespace",
                sorted(migrated_bare_keys),
            )

    # Assign into the (who, mode) scope specifically, not "the current
    # context" — load() may be restoring a scope other than the caller's, and
    # `settings.<list> = ...` resolves the scope from the ContextVars.
    # settings 프로퍼티는 컨텍스트에서 스코프를 정하므로, 복원 대상 스코프를
    # 직접 지정해야 함.
    with user_scope(who), account_mode_scope(mode):
        for key in _KEYS:
            scoped_key = _scoped(key, who, mode)
            if scoped_key not in by_key:
                continue
            if key == "trade_allowlist_domestic":
                settings.trade_allowlist_domestic = by_key[scoped_key]
            elif key == "trade_allowlist_overseas":
                settings.trade_allowlist_overseas = by_key[scoped_key]
            elif key == "watchlist":
                settings.watchlist = by_key[scoped_key]
            elif key == "trade_allowlist_auto_added":
                settings.trade_allowlist_auto_added = by_key[scoped_key]
            restored.append(key)
    # The two cross-list invariants used to be config-construction validators.
    # This is where stored data now enters, so this is where they run.
    normalize_scope(who, mode)
    if restored:
        logger.info(
            "trade_allowlist_store: restored %s for %s/%s from a previous session",
            sorted(restored), who, mode,
        )


# Scopes whose lists have been restored from Postgres in this process. The
# in-memory lists are per (user, mode) now, so a scope nobody has loaded looks
# identical to a scope with an empty allowlist — and for the background loops
# those two mean opposite things: "not restored yet" vs "this person permits
# nothing". Tracking it explicitly is what lets `ensure_loaded` be idempotent
# without re-running load()'s migrations on every loop pass.
# 미복원 스코프와 '빈 허용목록'은 메모리상 구분이 안 되므로 명시적으로 추적함.
_loaded_scopes: set[tuple[str, str]] = set()


async def ensure_loaded(user_id: str, account_mode: str) -> None:
    """`load()` this scope unless it has already been loaded in this process.

    Phase 6's precondition. The background loops bind each serviced scope and
    then read `settings.trade_allowlist`, which resolves per (user, mode) — but
    a scope only has content after an explicit `load()`. Widening the serviced
    set without this gives every user beyond the first an EMPTY allowlist, so
    their stop-losses silently never run. That is a missed protective action,
    not a safe degradation, which is why this is called before the tickers are
    read rather than at some convenient startup moment.
    스코프는 명시적 load() 후에만 내용이 있음 — 없으면 허용목록이 비어 손절 감시가
    조용히 멈추므로, 종목을 읽기 전에 반드시 호출.

    Idempotent per process. Not a cache with a TTL: every mutation goes through
    `save()`, which writes both Postgres and the in-memory scope, so a loaded
    scope cannot drift from the table by anything this process does. A change
    made by ANOTHER process would not be picked up — acceptable while
    deployment is one container, and the thing to revisit if that changes.
    다른 프로세스의 변경은 반영되지 않음 — 단일 컨테이너 배포에서는 문제 없음."""
    scope = (user_id, account_mode)
    if scope in _loaded_scopes:
        return
    await load(user_id=user_id, account_mode=account_mode)
    _loaded_scopes.add(scope)


def forget_loaded_scopes() -> None:
    """Drop the loaded-scope bookkeeping. For tests, and for any future path
    that needs a scope re-read from Postgres."""
    _loaded_scopes.clear()


async def save(
    key: str, tickers: list[str], user_id: str | None = None, account_mode: str | None = None
) -> None:
    """Persist one list (`key` is one of `_KEYS`) — best-effort: logs and
    returns on failure rather than raising, so a DB hiccup never undoes an
    add/remove that already succeeded in memory (same degrade-quietly
    convention as every other execution-layer writer). Stored under
    `_scoped(key)` (current user + account_mode by default), never the bare
    `key`. The explicit scope arguments exist so `load()`'s migrations write
    back under the same scope they read from — without them, loading a scope
    other than the current one would migrate rows INTO the current one.
    명시적 스코프 인자는 load()의 마이그레이션이 읽은 곳에 그대로 쓰도록 하기 위함."""
    try:
        from app.memory.db import get_pool

        pool = await get_pool()
        async with pool.acquire() as conn:
            await _ensure_schema(conn)
            await conn.execute(
                """
                INSERT INTO trade_allowlist_state (key, tickers, updated_at)
                VALUES ($1, $2::jsonb, now())
                ON CONFLICT (key) DO UPDATE SET tickers = EXCLUDED.tickers, updated_at = now()
                """,
                _scoped(key, user_id, account_mode), json.dumps(tickers),
            )
    except Exception:  # noqa: BLE001
        logger.warning("trade_allowlist_store: save failed for %s", key, exc_info=True)
