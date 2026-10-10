"""Phase 0-1 helper for the `trading` Postgres role password rotation.

Context: project memory db-password-rotation-design-2026-09-10.md and
db-exposed-weak-credentials.md. `git log --all -S "trading:trading"` finds
the plaintext password in 5 already-pushed commits, so every repo reader
already knows it. History rewrite is pointless (already propagated); the
only real remediation is changing the password. This script covers the
one step inside that plan that benefits from automation - running the
`ALTER ROLE` and *immediately* proving the new password works on a fresh
connection, with an automatic rollback if it does not.

This script does NOT:
  - generate the password (a human runs `openssl rand -base64 96 | tr -dc
    'A-Za-z0-9' | head -c 40` and stores it in a password manager FIRST),
  - touch `.env.local` / `.env.dev` / Jenkins (all human-only, see runbook),
  - rewrite git history.

It only ever sees the password the operator types into a getpass prompt;
nothing is read from argv or written to disk/logs.

Usage (run in the same environment as the app - repo root, `uv run`):

    # 1. no-risk rehearsal: re-sets the password to its CURRENT value, so a
    #    lack of ALTER ROLE privilege surfaces without changing anything.
    uv run python -m scripts.rotate_db_password --rehearsal

    # 2. the real rotation, during the maintenance window (Phase 2):
    uv run python -m scripts.rotate_db_password

Both modes first print `pg_stat_activity` (so the operator can abort if an
unknown client_addr is connected) and `SHOW password_encryption`, then
require an explicit `yes`.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import re
import sys
from urllib.parse import urlsplit, urlunsplit

import asyncpg

from app.config import settings

# base62 only, exactly 40 chars - matches the generator in the runbook.
# The design deliberately bans punctuation so no DSN parser / dotenv
# interpolation / `docker run --env-file` / Jenkins Secret layer can
# mangle it (db-password-rotation-design-2026-09-10.md, decision 1).
_PASSWORD_RE = re.compile(r"^[A-Za-z0-9]{40}$")

_CONNECT_TIMEOUT = 10.0


def _split_dsn(dsn: str) -> tuple[str, str, str]:
    """Return (role, current_password, dsn_without_credentials_for_display)."""
    parts = urlsplit(dsn)
    if not parts.username or parts.password is None:
        sys.exit(
            "DATABASE_URL has no user:password - this script rotates the "
            "password embedded in that DSN and needs both present."
        )
    host = parts.hostname or "localhost"
    port = f":{parts.port}" if parts.port else ""
    safe_netloc = f"{parts.username}:***@{host}{port}"
    safe = urlunsplit((parts.scheme, safe_netloc, parts.path, parts.query, parts.fragment))
    return parts.username, parts.password, safe


def _dsn_with_password(dsn: str, password: str) -> str:
    parts = urlsplit(dsn)
    host = parts.hostname or "localhost"
    port = f":{parts.port}" if parts.port else ""
    netloc = f"{parts.username}:{password}@{host}{port}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


async def _print_preflight(conn: asyncpg.Connection, role: str) -> None:
    version = await conn.fetchval("SHOW server_version")
    encryption = await conn.fetchval("SHOW password_encryption")
    print(f"\nconnected  : server_version={version}  password_encryption={encryption}")
    if encryption != "scram-sha-256":
        print(
            "  [warn]  password_encryption is not scram-sha-256 - the new hash will "
            "use this weaker scheme. Consider fixing postgresql.conf first."
        )

    rows = await conn.fetch(
        """
        SELECT datname, usename, client_addr, state,
               to_char(backend_start, 'YYYY-MM-DD HH24:MI') AS since
        FROM pg_stat_activity
        WHERE backend_type = 'client backend' AND pid <> pg_backend_pid()
        ORDER BY client_addr, backend_start
        """
    )
    print(f"\nother client backends currently connected ({len(rows)}):")
    if not rows:
        print("  (none)")
    for r in rows:
        print(
            f"  {str(r['client_addr'] or 'local'):>15}  db={r['datname']:<16} "
            f"user={r['usename']:<10} state={r['state'] or '?':<10} since={r['since']}"
        )
    print(
        "\n  -> If any client_addr above is unexpected, abort now (answer 'no') "
        "and investigate before rotating.\n"
    )


def _prompt_new_password(rehearsal: bool, current: str) -> str:
    if rehearsal:
        print("REHEARSAL MODE: the password will be re-set to its current value (no effective change).")
        return current

    while True:
        first = getpass.getpass("new password (40 base62 chars, from your password manager): ")
        if not _PASSWORD_RE.match(first):
            print("  [x] must match ^[A-Za-z0-9]{40}$ - regenerate with the runbook's openssl command.")
            continue
        second = getpass.getpass("re-enter to confirm: ")
        if first != second:
            print("  [x] entries differ, try again.")
            continue
        if first == current:
            print("  [x] that is the CURRENT password. Use --rehearsal if that is what you want.")
            continue
        return first


async def _rotate(rehearsal: bool) -> int:
    role, current_password, safe_dsn = _split_dsn(settings.database_url)
    print(f"target role: {role!r}   dsn: {safe_dsn}")

    try:
        conn = await asyncpg.connect(settings.database_url, timeout=_CONNECT_TIMEOUT)
    except (OSError, asyncpg.PostgresError, asyncio.TimeoutError) as exc:
        print(f"could not connect with the current DATABASE_URL: {exc!r}")
        return 1

    try:
        await _print_preflight(conn, role)
        new_password = _prompt_new_password(rehearsal, current_password)

        action = (
            f"re-set {role!r} password to its current value (rehearsal)"
            if rehearsal
            else f"change {role!r} password to the new 40-char value"
        )
        if input(f"Proceed to {action}? type 'yes': ").strip() != "yes":
            print("aborted, nothing changed.")
            return 1

        # Build the statement server-side so quoting is Postgres' problem,
        # not ours (base62 needs no escaping, but this stays correct if the
        # charset rule ever loosens).
        stmt = await conn.fetchval(
            "SELECT format('ALTER ROLE %I WITH PASSWORD %L', $1::text, $2::text)",
            role,
            new_password,
        )
        await conn.execute(stmt)
        print("ALTER ROLE executed.")
    except Exception as exc:  # noqa: BLE001 - report and fall through to close
        print(f"failed before/at ALTER ROLE (no change committed if it printed nothing): {exc!r}")
        await conn.close()
        return 1

    # This connection stays authenticated even after its own role's
    # password changes (Postgres never drops live connections on ALTER
    # ROLE) - so it is our rollback handle until we confirm the new one.
    try:
        verify_dsn = _dsn_with_password(settings.database_url, new_password)
        verify_conn = await asyncpg.connect(verify_dsn, timeout=_CONNECT_TIMEOUT)
        who = await verify_conn.fetchval("SELECT current_user")
        await verify_conn.close()
        print(f"[ok] verified: a fresh connection with the new password authenticated as {who!r}.")
        if rehearsal:
            print("\nREHEARSAL OK - you have ALTER ROLE privilege and the round-trip works.")
        else:
            print(
                "\nROTATION OK. Remaining Phase 2 steps are human-only (see "
                "docs/db-password-rotation-runbook.md):\n"
                "  - Jenkins credential `invest-env-dev` - re-upload, then run a build\n"
                "  - .env.local / .env.dev - edit by hand\n"
                "  - home laptop .env.local copy\n"
                "  - watch the next 30-min cycle for `password authentication failed`"
            )
        return 0
    except (OSError, asyncpg.PostgresError, asyncio.TimeoutError) as exc:
        print(f"\n[FAIL] NEW PASSWORD FAILED TO AUTHENTICATE: {exc!r}")
        print("rolling back to the previous password on the still-open connection...")
        try:
            rollback_stmt = await conn.fetchval(
                "SELECT format('ALTER ROLE %I WITH PASSWORD %L', $1::text, $2::text)",
                role,
                current_password,
            )
            await conn.execute(rollback_stmt)
            print("[ok] rolled back - the previous password is active again. Investigate before retrying.")
        except Exception as exc2:  # noqa: BLE001
            print(
                f"[FATAL] ROLLBACK ALSO FAILED: {exc2!r}\n"
                f"    Keep THIS terminal open. Use a superuser to run:\n"
                f"    ALTER ROLE {role} WITH PASSWORD '<previous value>';"
            )
        return 1
    finally:
        await conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--rehearsal",
        action="store_true",
        help="re-set the password to its current value to prove ALTER ROLE privilege without changing anything",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_rotate(args.rehearsal)))


if __name__ == "__main__":
    main()
