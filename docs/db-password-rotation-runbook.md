# `trading` Postgres role - password rotation runbook

Status: **executed 2026-09-10.** The `trading` role password was rotated to
a 40-char base62 value; the deploy container, local `.env.local` /
`.env.dev`, the Jenkins `invest-env-dev` credential, and the home-laptop
copy were all updated and verified (`get_pool()` OK from both the server
container and locally). Kept for the record and for any future rotation.
Phase 4 status (checked 2026-10-02 against the code): `scratchpad/` is in
`.gitignore` and the `app/config.py` default no longer carries credentials -
done. Still open (re-checked 2026-10-02): 8 untracked, git-ignored
`scratchpad/*.py` files keep the old (dead, already-rotated) DSN hardcoded -
local scratch only, none is tracked; and the runbook is not yet linked from
`docs/architecture.md` / `docs/safety-and-risk.md`.

## Why

`git log --all -S "trading:trading"` finds the plaintext password in 5
already-pushed commits (`052c299`, `7788578`, `7e13fa7`, `7ae9a22`,
`c094aa2`) on `enayuso/invest-project`. Every repo reader
already has it. The DB is also reachable from outside the office network
(ipTIME port-forward `45434 -> 5434`). This DB is the trading system's
control plane - `risk_rules`, `execution_safety_state` (kill switch),
`trade_allowlist_state`, `trades` (stop-loss levels the auto-exit logic
follows). History rewrite is pointless (already propagated); rotating the
password is the only remediation.

Follow-up (NOT in this scope): restricting the router forward to specific
source IPs, or `pg_hba.conf` CIDR limits, or SSH-tunnel-only access.

## Where the credential lives (all must end up on the new value)

| # | Location | Who updates it |
|---|----------|----------------|
| 1 | Postgres server, `trading` role in `pg_authid` - the source of truth | human (`scripts/rotate_db_password.py`) |
| 2 | Jenkins credential `invest-env-dev` (Secret file) - the deploy source of truth | human, Jenkins UI |
| 3 | `.env.dev:74` (local copy, git-untracked) | human, editor |
| 4 | `.env.local:43` (local dev / backtest) | human, editor |
| 5 | (done 2026-10-02: now without credentials) `app/config.py:359` default `postgresql://trading:trading@localhost:5434/...` | Sonnet, Phase 4 (remove creds from default) |
| 6 | `scratchpad/*.py` hardcoded DSNs (~8 files, not gitignored) | Sonnet, Phase 4 |
| 7 | git history (5 commits) - irreversible, rotation *is* the response | - |
| 8 | running container `Config.Env` (`docker inspect` shows it in plaintext) | auto-replaced on redeploy |
| 9 | home-laptop `.env.local` copy (see running-backtest-from-home.md) | human - **if forgotten, that machine silently stops working** |

## Design decisions (from db-password-rotation-design-2026-09-10.md)

1. **Charset: base62 (alphanumeric only), 40 chars.** No punctuation, so
   no DSN parser / python-dotenv interpolation / `docker run --env-file` /
   Jenkins Secret parser can mangle it. 40 base62 chars ~= 238 bits.
   Generate:
   ```
   openssl rand -base64 96 | tr -dc 'A-Za-z0-9' | head -c 40; echo
   ```
   (Use a cryptographic source like this, not a shell `$RANDOM`.) **Never paste
   the password into chat or a commit.** The operator generates and stores
   it in a password manager first.

2. **No downtime, because** Postgres does not drop existing connections on
   `ALTER ROLE ... PASSWORD` - only new connections need the new secret.
   `app/config.py:437` `database_idle_session_timeout_seconds = 300.0`
   means the server reclaims idle pooled connections after 5 min, so the
   effective grace period to get the app onto the new password is ~5 min.

3. **Order: change the DB first, then recreate the container within 5 min.**
   "DB first -> instant failure" is not how it works (connections survive);
   "app first" is genuinely risky (someone hits a Jenkins build in the gap
   and the new-secret container starts before the DB knows it). So: rotate
   the DB, then trigger the Jenkins build immediately.

4. **Window: pick a time when both KRX and US regular sessions are closed**
   (`app/live/market_hours.py`: KRX 09:00-15:30 KST; US regular
   ~22:30/23:30-05:00/06:00 KST). **Recommended: Saturday morning KST.**
   If the app is on the old password when the server reclaims idle
   connections, every `get_pool()` caller catches the error and degrades
   (`app/memory/db.py:526-529`) - but "degraded" means *risk rules not
   loaded / ledger not written while decisions still run*, which must never
   happen mid-session. The time window is the real safety mechanism.

5. **Rollback is easy** because the old password (`trading`) is known. The
   absolute rules: (a) store the new password in the password manager
   *before* running `ALTER ROLE`; (b) verify a fresh connection with the
   new password *in the same run*. `scripts/rotate_db_password.py` does (b)
   automatically and rolls back on the still-open connection if it fails.
   The one unrecoverable scenario is running `ALTER ROLE` to a value you
   did not save AND losing the session - then neither old nor new works and
   you need a superuser.

6. **Alternatives (not recommended, for reference):** two-role overlap
   (schema-ownership split risk); skip `docker build` and recreate the
   container from `docker inspect` env (only as a fallback if the build
   exceeds 5 min - and never with local `.env.dev`, whose KIS real-account
   keys are intentionally blank).

## Phase 0 - preparation (no risk)

- [ ] **0-1 (done)** `scripts/rotate_db_password.py` exists.
- [ ] **0-2 (done)** this runbook exists.
- [ ] **0-3** Human: audit every connection source. Connect to the DB and
      run:
      ```sql
      SELECT client_addr, usename, datname, count(*), min(backend_start)
      FROM pg_stat_activity
      WHERE backend_type = 'client backend'
      GROUP BY 1,2,3 ORDER BY 1;
      ```
      If any `client_addr` is unrecognized, **stop** and investigate.
      (`scripts/rotate_db_password.py` also prints this at the top of every
      run.)
- [ ] **0-4** Human: confirm the `trading` role can change its own
      password. Run the rehearsal (re-sets the password to its current
      value, so nothing actually changes):
      ```
      uv run python -m scripts.rotate_db_password --rehearsal
      ```
      Expect `REHEARSAL OK`. If it reports a privilege error, arrange a
      superuser for Phase 2.
- [x] **0-5** Superuser recovery path (confirmed 2026-09-10): `trading` is
      itself a superuser (`rolsuper = t`), and the local socket inside the
      `trading-postgres` container uses `trust` auth, so from an SSH session
      on the DB host the password can always be reset regardless of its
      current value:
      ```
      docker exec -it trading-postgres psql -U trading -d trading_memory \
        -c "ALTER ROLE trading WITH PASSWORD '<value>'"
      ```
      (DB name is `trading_memory`, not `trading` - psql defaults dbname to
      the username and errors "database \"trading\" does not exist"
      otherwise.) No separate human or `postgres` role is needed. There is
      no `postgres` role on this instance.
- [ ] **0-6** Human: generate the 40-char password (command above), store
      in the password manager. Do not put it anywhere else yet.
- [ ] **0-7** Human: pick the window (Saturday AM KST recommended),
      announce a 30-min "no Jenkins builds" freeze to the team.

## Phase 1 - T-30 min

- [ ] Human: run one Jenkins `invest-env-dev` build now to warm the Docker
      layer cache, and time it. If the build takes > 3 min, plan to use the
      fallback in decision 6 (recreate container from `docker inspect`
      env), not a fresh build, during the window.
- [ ] Human: confirm the "no builds" freeze is understood by anyone with
      Jenkins access.

## Phase 2 - inside the window (target: 5 min, all human)

1. [ ] Jenkins UI: update credential `invest-env-dev` to the new
       `DATABASE_URL`. **Do not start a build yet.**
2. [ ] Run the rotation:
       ```
       uv run python -m scripts.rotate_db_password
       ```
       Paste the new password at the getpass prompt (twice), review
       `pg_stat_activity`, type `yes`. The script runs `ALTER ROLE` and
       immediately verifies a fresh connection.
       - On `ROTATION OK` -> continue.
       - On failure the script auto-rolls-back; **stop**, investigate,
         restore the Jenkins credential to the old value.
3. [ ] Jenkins UI: start the `invest-env-dev` build. Watch it to green.
4. [ ] Edit `.env.local` and `.env.dev` by hand - replace the password in
       `DATABASE_URL` on line 43 / 74. (Sonnet is never given the value.)
5. [ ] Confirm the new container is `Up` and its logs show no
       `password authentication failed`.

## Phase 3 - verification

- [ ] `docker ps` - new container Up, old one gone.
- [ ] Container logs: no `password authentication failed`, no repeated
      `get_pool` degrade warnings.
- [ ] `GET /api/settings/runtime` returns 200 with real values.
- [ ] Local: `uv run python -c "import asyncio; from app.memory.db import get_pool; asyncio.run(get_pool())"` succeeds.
- [ ] Wait for the next 30-min cycle; confirm a fresh row lands in
      `memory_entries`.
- [ ] Update the home-laptop `.env.local` copy (item #9) before its next
      use.

## Phase 4 - cleanup (Sonnet, local, separate change)

- [x] `.gitignore`: add `scratchpad/`. (done; verified 2026-10-02)
- [ ] The ~8 `scratchpad/*.py` files: replace hardcoded DSNs with
      `os.environ["DATABASE_URL"]`.
- [x] `app/config.py:359`: drop credentials from the default (done; verified 2026-10-02, now `app/config.py:478`)
      (`postgresql://localhost:5434/trading_memory`). Fail-fast on a
      missing `DATABASE_URL` is a separate scope.
- [ ] Link this runbook from `docs/architecture.md` / `docs/safety-and-risk.md`.

## Rollback quick reference

> **Corrected 2026-10-02.** An earlier version of this section told the
> operator to roll back by setting the role password back to the **old,
> already-exposed value** (it is in 5 pushed commits, see "Why"). That
> instruction was removed: **never restore the exposed password** - doing so
> would undo the only remediation this runbook exists for.

- **During a rotation**: `scripts/rotate_db_password.py` rolls back on its own
  if verification of the new password fails. That automatic step is the only
  sanctioned "rollback".
- **After a rotation (e.g. the new value was lost, or a consumer still holds a
  stale value)**: do not go backwards. Generate a fresh strong value and run a
  new rotation with the same procedure (Phase 0-3), updating every location in
  the table above. The Jenkins credential `invest-env-dev` is the deploy source
  of truth.
- Emergency access from the DB host itself (SSH session, socket `trust`) is
  still possible for that new rotation; it does not require any old value.
- Keep real passwords out of this file, tickets, chat and the `handoff/`
  folder.
