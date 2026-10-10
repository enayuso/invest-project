# 멀티유저 자격증명 이전 계획 (Multi-User Credential Migration Plan)

> ## ⚠️ 2026-08-07 되돌림 — `docs/single-user-rollback-guide.md` 참조
>
> 이 문서가 설계한 멀티유저 구조(로그인/세션, 사용자별 암호화 자격증명 DB,
> `app/auth/`, `app/credentials/{store,crypto}.py`)는 **2026-08-07
> 단일계좌 전환으로 대부분 되돌려졌다.** 실제로는 계좌 하나만
> 단독으로 쓰기로 결정했기 때문이다. 구체적으로:
>
> - **삭제됨**: `app/auth/` 패키지 전체(로그인/세션/비밀번호), `app/web/routers/auth.py`,
>   `app/credentials/store.py`/`crypto.py`(암호화 DB 저장소), `sessions`/
>   `user_broker_credentials` 테이블, 온보딩 로그인·키입력 화면
>   (`onboarding.js`의 `renderAuth`/`renderCredentials`).
> - **되돌아감**: KIS 자격증명은 다시 `app/config.py`의 env 필드
>   (`kis_app_key_real` 등)이다 — 단, 이 문서가 없애려던 "프로세스 하나가
>   시크릿을 프로세스 환경에 평문으로 들고 있는" 형태로 그냥 복귀한 게 아니라,
>   시크릿 수명주기를 Jenkins Credentials가 배포 시점에 관리하는 형태다.
> - **유지됨(중요)**: `users` 테이블의 `user_id` 컬럼·FK와 모든 SELECT
>   필터링(§2.1 Tier A)은 **그대로 남아 있다.** 이건 "멀티유저 기능"이
>   아니라 행 소유자 태그였고, 값이 `'default'` 하나뿐이어도 동작에 차이가
>   없어서 걷어낼 이유가 없었다. `app.user_context.serviced_scopes()`는
>   DB JOIN 대신 `Settings.configured_kis_modes()` 기반으로 단일 스코프만
>   반환하도록 바뀌었지만, 반환 타입(`list[ServicedScope]`)은 그대로라 이
>   문서가 설계한 하위 호출부(poller/stop_loss_loop/auto_trade_loop 등)는
>   손대지 않았다.
> - **paper/real 두 모드**는 여전히 함께 쓴다(실전 + 모의 테스트) —
>   §5의 계좌모드 축은 이 문서가 설계한 대로 남아 있고, `PUT
>   /api/account-mode`도 유지된다. 다만 인증이 없어졌으므로 전환 이력을
>   `account_mode_switches` 감사로그로 남기는 조항이 새로 추가됐다
>   (`docs/safety-and-risk.md` §5.1도 함께 갱신됨).
>
> **이 문서는 삭제하지 않고 남겨둔다** — 왜 애초에 이런 구조(사용자별 스코프,
> Tier A/B 구분, 세션 미들웨어 설계)가 만들어졌는지의 유일한 기록이기
> 때문이다. 아래 본문을 "지금의 아키텍처"로 읽지 말 것 — Phase 0-7이 실제로
> 구현했던 것과, 그걸 2026-08-07에 왜/무엇을 되돌렸는지는 반드시
> `docs/single-user-rollback-guide.md`와 함께 읽을 것.

**Status:** Phases 0-6 **DONE** except ~24 deployment-wide settings (§7.7) and
per-user 체결통보 (§6.6). Phase 7: documentation items DONE; the network-posture
item is deliberately NOT done (no TLS/CSRF yet) and `status-and-decisions.md`
§2's "personal tool" premise is an open product/legal decision, flagged in place.

> **Read §6.5 before starting anything.** It records what landed on 2026-08-06
> after Phase 4 closed, which does not map cleanly onto the original Phase 5
> description below: account mode became per-user (54 call sites), the
> per-user risk-rule set grew 7 → 23, and user-raisable order caps gained
> absolute ceilings. The Phase 5 text below was written before any of that and
> still lists `kis_account_mode` as pending.
>
> **What still blocks a real multi-user deployment:**
>
> 1. ~~The in-memory allowlist is still one copy.~~ **FIXED 2026-08-06.** The
>    four ticker lists are per-(user, mode) properties over
>    `config._scoped_lists`. See §6.5.
> 2. ~~Two stores still load one mode at startup.~~ **FIXED 2026-08-06.**
>    `runtime_settings` now keys its cache `(user_id, account_mode)` and loads
>    every scope, matching `risk_rules` — the per-mode keying also closes the
>    same leak found there, where a paper override answered a real read.
>    `trade_allowlist_store.load()` now takes an explicit `(user_id,
>    account_mode)` instead of deriving it from an unbound boot context. It
>    still restores only ONE scope, because its destination is the global
>    `settings` lists; that is item 1, not this one.
> 3. ~~Background loops are still global.~~ **FIXED 2026-08-06.** Both loops,
>    the poller and the realtime feed now resolve their work from
>    `user_context.serviced_scopes()` — users with stored credentials, each in
>    their own resolved mode — and call `trade_allowlist_store.ensure_loaded`
>    before reading any allowlist. See §6.6.
>
> 4. **41 tests depend on the dev Postgres being reachable.** Not a migration
>    item, but it is the "~40 phantom failures" this document's handoff notes
>    kept referring to, and the diagnosis there was wrong: it is not shared
>    state between tests, it is DB reachability. Verified by running the same
>    commit with the DB up (1582 pass) and down (41 fail) — the failing set is
>    identical before and after Phase 6. Anyone debugging a red suite should
>    check the DB first.
>    '유령 실패'의 원인은 테스트 간 상태 공유가 아니라 DB 접속 가능성이었음.
>
> 5. **`get_broker()` breaks its own contract when the credential store is
>    unreachable.** It is documented to return `None` so callers "degrade
>    quietly", but `credentials_for` raises on a store outage (deliberately —
>    auth must not degrade), and that propagates through `broker_for_user` into
>    callers that only wanted to know whether the market is open
>    (`kr_holidays.is_market_open_on` → HTTP 500). This is the mechanism behind
>    item 4. Not fixed here: the fix changes order-path semantics (no broker
>    means orders do not place) and deserves a deliberate decision rather than
>    being bundled into Phase 6.
>    자격증명 저장소 장애가 '조용히 저하'되지 않고 500으로 전파됨 — 주문 경로
>    의미가 바뀌는 수정이라 별도 판단이 필요.
> 4. **Not verified live.** The settings-tab guard and the onboarding →
>    settings flow have only been exercised against fakes and a local
>    container; getting past onboarding for real needs live KIS credentials.

**Scope owner:** project maintainer
**Handoff note:** this document is deliberately self-contained. It is written to be
picked up by a session with no prior context: every claim below cites the file and
line it was verified against, so nothing has to be taken on trust or re-derived from
a chat log.

한 프로세스가 여러 사용자를 동시에 서비스하도록 바꾸고, KIS 자격증명을 `.env.local`에서
빼내 사용자 계정(이메일)에 연결된 암호화된 Postgres 저장소로 옮기는 작업 계획.

---

## 1. What is being asked

Today every credential lives in `.env.local` and the process serves exactly one
person. The target state:

1. A user registers an account with **email + password**, stored in Postgres.
2. After logging in, that user **manually enters their own KIS API credentials**
   through the dashboard.
3. Those credentials are stored in Postgres, **encrypted**, linked to their email
   account.
4. Only once the required credentials are present can that user actually use the
   service. Until then the dashboard shows an onboarding state and trading routes
   refuse.
5. **One process serves many concurrent users** — same model a typical multi-tenant SaaS uses.

### 1.1 The exact fields moving out of `.env.local`

Confirmed with the user, 2026-08-06. These seven and only these seven:

| Env var | `Settings` field |
|---|---|
| `KIS_PAPER_APP_KEY` | `kis_paper_app_key` |
| `KIS_PAPER_APP_SECRET` | `kis_paper_app_secret` |
| `KIS_PAPER_ACCOUNT_8` | `kis_paper_account_8` |
| `KIS_PAPER_HTS_ID` | `kis_paper_hts_id` |
| `KIS_REAL_APP_KEY` | `kis_real_app_key` |
| `KIS_REAL_APP_SECRET` | `kis_real_app_secret` |
| `KIS_REAL_ACCOUNT_8` | `kis_real_account_8` |

### 1.2 What explicitly stays in `.env.local` for now

Everything else, including `DATABASE_URL`, `FRED_API_KEY`, `LLM_BASE_URL`, and all
caps/limits. The user noted some of the limits (`KIS_MAX_ORDER_KRW_PAPER`,
`KIS_MAX_ORDER_USD_PAPER`, …) may become user-configurable later, but that is **not**
part of this work.

`DATABASE_URL` cannot move into Postgres (chicken-and-egg) and must stay in the
environment permanently. `CREDENTIAL_ENCRYPTION_KEY` (introduced in Phase 2) must
also stay in the environment for the same reason.

### 1.3 Decisions already settled

Do not re-litigate these; they were decided by the user on 2026-08-06.

| Question | Decision |
|---|---|
| Process model | **One process, many concurrent users.** Not process-per-person. |
| Login mechanism | **Email + password stored in Postgres.** Explicitly "good enough for now, optimize later." Not OAuth, not magic-link. |
| Credential storage | **Encrypted at rest with a key held in the environment.** Not plaintext. |

---

## 2. Why this is bigger than seven fields

This is the single most important thing for the next session to understand before
touching anything.

The persistence layer splits into two tiers. The seven credential fields are the
smaller half of the job; the larger half is that **most trading state is not
user-scoped at all**, so isolating credentials alone produces isolated keys driving
*shared* trading state — strictly more dangerous than the current single-user setup.

### 2.1 Tier A — already user-scoped (reads AND writes filter on `user_id`)

The "stage 4 multi-instance seam" actually completed these. No work needed:

- `trades` — `app/execution/ledger.py:191, 221, 253, 850, 858`
- `memory_entries` — `app/memory/store.py:110, 273`
- `alerts` — `app/live/alerts.py:136, 156`

Note: the comment at `app/memory/db.py:73-78` claims "every query still reads/writes
globally, unfiltered by this column." **That comment is stale** for these three
tables. Verified by reading the queries.

### 2.2 Tier B — NOT user-scoped (scoped only by `account_mode`)

Zero `get_current_user_id()` references in any of these. Each row is shared by every
user in the process:

| Module | Concrete failure with 2 concurrent users |
|---|---|
| `app/execution/safety_switch.py` | One kill switch + daily-loss breaker per `(market, account_mode)`. User A's kill switch blocks user B's orders; A can also switch B's back off. |
| `app/execution/runtime_settings.py` | `execution_mode` / `ai_profile` are global. **User A selecting FULL_AUTO puts user B's account into autonomous trading.** Highest-severity item in this table. |
| `app/execution/risk_rules.py` | Order value caps, position-% caps, approval thresholds all shared. |
| `app/web/trade_allowlist_store.py` | One shared allowlist. This is the list `kis_bridge` gates on to decide what may be traded at all. |
| `app/execution/portfolio_risk.py:123` | Drawdown high-water mark shared; the `market_lock` Postgres advisory lock is keyed `(market, account_mode)`, so it serializes *every* user's order commits against each other. |
| `app/execution/pending_orders.py` | Approval queue shared — user A can approve/reject user B's pending order. |
| `app/audit/stats.py` | Win-rate / alpha reporting blends all users into one number. |

### 2.3 Cross-user data leaks in the web layer — FIXED in Phase 2, and it was bigger than this

This section named the two cache keys and missed the larger one: the broker
itself. `app.web.state._kis_broker` was a process-global built once at boot and
read from ~40 call sites across six routers, so with two users logged in every
account-backed response — and `POST /api/order` — used whichever account the
process started with. Both that and the two keys below are fixed; the detail is
in Phase 2's notes. Kept as written for the record.


`app/web/state.py` has an async TTL cache (`cached()`, line 77) whose keys are bare
literals. Two of them hold per-account data:

- `cached("account", 35, …)` — `app/web/state.py:542` — KIS balances and positions
- `cached("executions", 35, …)` — `app/web/state.py:129` — KIS fills

With multiple users these serve one user's account data to another. `cached("index",
…)` (`app/web/routers/market_data.py:217`) is public market data and is fine shared.

---

## 3. Current architecture (cold-start summary)

Full detail is in `docs/architecture.md`. The minimum needed for this work:

- **Process shape.** `app.daemon` is a thin uvicorn launcher for `app.web.app`. That
  FastAPI app *is* the long-running job; its `lifespan`
  (`app/web/state.py:753`) starts the minute poller, the stop-loss loop, the opt-in
  auto-trade loop, and the KIS WebSocket `RealtimeFeed`.
- **Config.** `Settings` (`app/config.py`) is a pydantic-settings class reading
  `(".env.local", ".env.kis.local")`, instantiated once at import as a module-level
  singleton (`settings = Settings()`, line 856) and imported by **42 modules**.
  Several of its fields are *mutated at runtime* (see §6, Phase 5).
- **Identity.** `get_current_user_id()` (`app/user_context.py:49`) just returns
  `settings.user_id`, a per-process env value. Its own module docstring states it is
  explicitly not an access control, and anticipates being swapped for real auth:
  *"adopting real auth later means changing this one function's body, not every call
  site."* `identity_claims` is forensic-only.
- **Auth.** There is none. All **54 routes** across 10 routers are unauthenticated.
  `dashboard_host` defaults to `127.0.0.1` *specifically because* `POST /api/order`
  places real orders with no auth (`app/config.py:345-349`).
- **Decision pipeline.** `run_pipeline()` (`app/orchestrator/pipeline.py:479`), 7
  stages, ending in `rebalance_to_target()` which branches on `execution_mode` and,
  in FULL_AUTO, goes through the single choke point `app/execution/kis_bridge.py`
  where every safety gate lives.

---

## 4. Where the credentials are consumed

The credential surface is small and contained, which is the good news.

**Superseded by Phase 2 — kept because the reasoning below still holds.** As
built, the consumers are:

| Consumer | Location | Notes |
|---|---|---|
| `broker_for_user()` | `app/brokers/kis/broker.py` | The only place credentials are resolved into a `KISBroker`. Async. `is_real` comes from the loaded credentials, never from `settings.kis_account_mode`. Returns `None` when that user has not configured that mode; every caller degrades quietly. |
| `state.kis_broker()` | `app/web/state.py` | What all ~40 web-layer call sites use. Thin wrapper over `get_broker()`; exists because the routers used to read a process-global broker. |
| `RealtimeFeed` construction | `app/web/state.py` (lifespan) | Still **bypasses the broker factory** — it needs raw key/secret/hts_id, not a broker — so it does its own `credentials_for()` call. Easy to miss; it is the one place that must be updated by hand whenever credential handling changes. |

The first two sit behind `get_broker(name, *, user_id=None, account_mode=None)`
(`app/brokers/__init__.py`), which caches one instance per
`(name, user_id, account_mode)`.

### 4.1 The throttle invariant (important, and not a blocker)

`app/brokers/__init__.py:25-36` documents why the broker instance is process-global:
each instance owns its own request throttle, so creating fresh instances per call
silently defeated it and burst past KIS's per-second limit (`EGW00201`).

**This survives multi-tenancy, and in fact becomes more correct.** KIS rate-limits
per app key. Different users have different app keys, therefore different rate-limit
buckets. Keying the instance cache on `(name, user_id, account_mode)` is the right
granularity — not a regression. Do not treat this comment as an obstacle.

### 4.2 The token cache is a single shared file

`token_cache_file()` = `_token_cache_dir() / ".kis_token_cache.json"`
(`app/brokers/kis/const.py`) — by default `_token_cache_dir()` is the `app/`
package dir (not the repo root), overridable via the `KIS_TOKEN_CACHE_DIR` env
var (a persistent Docker volume in deploy — see `pipeline.sh`). Historically
this was one path, no user scoping. Two users in one process would overwrite each other's
access tokens. KIS token issuance is rate-limited to **1 per minute per app key**, so
thrashing here is expensive, not merely wasteful.

Fix: derive the path per credential set from a hash of the app key.
`KISAuth.__init__` (`app/brokers/kis/auth.py:27`) and `KISHttpClient` already accept a
`token_cache_path` parameter, so the seam exists. Hashing the *app key* rather than
the `user_id` means paper and real sets separate automatically, and preserves the
cross-process sharing the file exists for (daemon + CLI runs reuse one token).

---

## 5. Security state of the repo right now

**Partly addressed by Phase 0. The outstanding item is a real one.**

- `.env.local` was tracked in git — **fixed** (untracked, `.gitignore` `.env.*`
  re-activated, `.env.example` committed with values stripped).
- `HEAD:.env.local` contains a live `FRED_API_KEY` and a `DATABASE_URL` with embedded
  credentials. A non-empty `FRED_API_KEY` appears in **8 separate commits**.
  **Still true.** Untracking the file does not remove it from history, and the
  user deferred both rotations (still in development, to be rotated later) and
  skipped the history rewrite. **These two credentials remain live in a pushed
  branch's history** — that is a conscious, recorded decision, not an oversight,
  but it does need to happen before this is exposed to anyone.
- The remote is `https://github.com/enayuso/invest-project.git`.
  Current branch at time of writing: `feature/multi-user`.
- The KIS fields were empty in `.env.local` and expected in `.env.kis.local`
  (gitignored, absent from this checkout). **Moot as of Phase 2** — those
  settings no longer exist, so there is nothing for either file to hold.
- New as of Phase 2: `CREDENTIAL_ENCRYPTION_KEY` is now a secret of the same
  class as `DATABASE_URL`, and it must never be committed. `.gitignore`'s
  `.env.*` rule covers it; `.env.example` carries the name with an empty value.

---

## 6. Implementation phases

Phases 0 and 1 are prerequisites for everything. Phases 4 and 5 are **not optional
polish** — see §2.

### Phase 0 — credential hygiene — DONE 2026-08-06 (with deferrals)

1. ~~Rotate the FRED API key.~~ **Deferred by the user** — still in development,
   will be rotated later. The live key remains in git history.
2. ~~Rotate the Postgres password.~~ **Deferred by the user**, same reason.
3. `.gitignore` `.env.*` re-activated. DONE.
4. `git rm --cached .env.local .env.dev .env.alice`. DONE, committed, tree clean.
5. `.env.example` — DONE, but it did **not** initially take effect, and the reason is
   worth knowing because it can recur: `.gitignore` had a stray bare `.env.example`
   line *after* the `!.env.example` negation, and git applies the last matching rule,
   so the file stayed ignored and untracked. Worse, the file was a copy of
   `.env.local` **with live values still in it** — a real `FRED_API_KEY`, the
   credentialed `DATABASE_URL`, the internal `LLM_BASE_URL`. The stray ignore line was
   the only thing preventing those from being re-committed. Both fixed: values
   stripped (tuning knobs and the public KIS base URLs kept), stray line removed.
   **If you regenerate `.env.example` from `.env.local`, strip the values again.**
6. History rewrite vs. rotation-only. **Skipped by the user** — not a concern for now.

### Phase 1 — identity becomes request-scoped — DONE 2026-08-06

1. **DONE.** `users` gained nullable `email` + `password_hash`
   (`app/auth/store.py`). `user_id TEXT` stayed the primary key and a UUID is
   generated per registration, so no FK churn. Nullable rather than `NOT NULL`
   as originally written: the legacy `'default'` seed row that every
   pre-existing row FKs to has neither, and must survive. A NULL
   `password_hash` simply cannot be logged into. Uniqueness is a partial
   `UNIQUE INDEX ON (lower(email)) WHERE email IS NOT NULL` instead of `CITEXT`
   — same effect, one less extension a fresh Postgres has to have installed.
2. **DONE, but not argon2 or bcrypt.** `app/auth/passwords.py` uses stdlib
   `hashlib.scrypt`: same memory-hard class of defense, zero new dependency,
   and no compiled wheel — which matters because this must build on x86_64 and
   ARM64 both. Cost parameters are stored inside each hash string, so raising
   them later re-hashes people lazily on next login instead of locking them out.
3. **DONE.** `get_current_user_id()` reads a ContextVar with a
   `settings.user_id` fallback (`app/user_context.py`). The fallback keeps CLI,
   backtest and the test suite unchanged, exactly as planned.
4. **DONE, and this one was initially missed — check it if anything here looks
   suspect.** The sessions table, `bind_user_identity` middleware and
   `require_user` dependency all existed and were correct, but **no route
   referenced `require_user`**, so all 53 non-auth endpoints — `POST /api/order`
   among them — were still completely open. It is now applied once at the
   `include_router` wiring point in `app/web/app.py`, so a route added later is
   authenticated by default rather than by remembering.
   `tests/auth/test_route_protection.py` proves it behaviorally (real requests,
   not router internals) for every route, and will fail if a new unprotected
   one appears. Note the real count is **57 routes = 53 protected + 4 auth**,
   not the 54 stated elsewhere in this document.
5. **DONE, also initially missed.** `user_scope` existed with zero call sites.
   `stop_loss_loop` and `auto_trade_loop` now iterate
   `app.user_context.background_user_ids()` and run each pass inside
   `user_scope`. `background_user_ids()` returns exactly one identity today; it
   is async purely so Phase 6 changes its body and nothing else.
6. **DONE** — superseded by `app.auth.store.register()`, which owns account
   creation now. `register_user()` remains for the display-name upsert.

Also added, not in the original plan: `settings.auth_required` defaults to
False (the login UI does not exist until Phase 3, so defaulting it True would
brick the dashboard mid-migration), with a boot validator that **refuses to
start** if `dashboard_host` is non-loopback while `auth_required` is False.
That is what keeps the convenient default from ever reaching a
network-reachable deployment.

### Phase 2 — the encrypted credential store — DONE 2026-08-06

Delivered as written, with four deviations and two discoveries worth reading
before starting Phase 3.

**Deviations from the plan as specified:**

- **§1.1 undercounted the fields.** It lists seven. `KIS_REAL_HTS_ID`,
  `KIS_PAPER_ACCOUNT_PROD` and `KIS_REAL_ACCOUNT_PROD` also had to go: the
  table stores one `hts_id` and one `account_prod` *per row*, so leaving env
  copies behind would have meant every user sharing one person's HTS ID.
  **Ten** env vars were removed, not seven.
- **`get_broker()` is now async.** Credentials come from Postgres, so
  resolving one is I/O. Every call site was already inside a coroutine, so
  this cost no restructuring — but it does mean `await get_broker()`.
- **Encryption is bound to its row.** The plan said "AES-GCM"; the
  implementation also passes `(user_id, account_mode, field)` as GCM
  additional-authenticated-data (`app/credentials/crypto.py::_aad`). Without
  it, anyone who can UPDATE the table — a compromised app, a careless
  migration — could copy Alice's `app_key_enc` into Bob's row and have the
  service place Bob's orders against Alice's account, **without ever needing
  the master key**. It also makes a paper↔real swap within one user's own rows
  fail, which otherwise routes a believed-paper order to real money.
- **CLI scripts got `--user-email` now, not "later"** (§7.1). They had no
  identity to run as the moment the env credentials went away.
  `scripts/_cli_user.py` is the shared resolver.

**Discoveries — things the plan did not know about:**

- **§2.3 understated the web-layer leak by a lot.** It names two cache keys.
  The real problem was `app.web.state._kis_broker`, a process-global broker
  built once at boot and read from ~40 call sites across six routers. With two
  users logged in, *every* balance, position, valuation and ranking response
  would have come from whichever account the process started with — and
  `POST /api/order` would have placed the second user's orders into the first
  user's account. It is gone, replaced by `state.kis_broker()`, which resolves
  per request identity.
- **Namespacing the cache keys silently broke cache invalidation.**
  `invalidate_cache()` matches on key *prefix*, and the three
  `invalidate_cache("account", "executions")` calls in `orders.py` stopped
  matching once the keys became `"<user_id>:account"`. Left alone, the
  dashboard would have shown stale balances for a full 35s TTL after every
  order, with nothing failing. Call sites now pass `state.user_key(...)`.

**What landed:**

1. New table `user_broker_credentials`, exactly the shape below
   (`app/credentials/store.py`).

   ```
   user_broker_credentials (
       user_id       TEXT NOT NULL REFERENCES users(user_id),
       account_mode  TEXT NOT NULL CHECK (account_mode IN ('paper','real')),
       app_key_enc   BYTEA NOT NULL,
       app_secret_enc BYTEA NOT NULL,
       account_8_enc BYTEA NOT NULL,
       account_prod  TEXT NOT NULL DEFAULT '01',
       hts_id_enc    BYTEA,
       updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
       PRIMARY KEY (user_id, account_mode)
   )
   ```

   `account_prod` is not sensitive and stays plaintext. `hts_id_enc` is nullable —
   `None` disables only the 체결통보 WebSocket subscription (`app/config.py:432-434`).

2. AES-256-GCM via `pycryptodome` (`app/credentials/crypto.py`), master key in
   `CREDENTIAL_ENCRYPTION_KEY`, plus the row binding described above. Blobs are
   `version || nonce || tag || ciphertext`; the leading version byte is what
   makes key rotation survivable instead of a flag day. A key that is set but
   does not decode to exactly 32 bytes **refuses to boot** rather than failing
   at somebody's first login after a deploy. A short key is never stretched or
   hashed up to length.
3. **DONE** — ten fields deleted from `Settings`, `credentials_for()` added.
   No env fallback, deliberately.
4. **DONE** — `broker_for_user(user_id, account_mode)`, `_broker_instances`
   keyed `(name, user_id, account_mode)`. Two additions beyond the plan: an
   `asyncio.Lock` around construction (without it, concurrent cache misses
   build two instances with independent throttles — the exact EGW00201 burst
   the cache exists to prevent, visible only under concurrency), and
   `invalidate_broker_cache(user_id, account_mode=None)`, which the
   credential-save path must call — a cached instance otherwise keeps using
   the old app key until process restart, which for someone rotating a leaked
   key is the opposite of what they asked for.
5. **DONE** — `token_cache_path_for(app_key)` in `app/brokers/kis/const.py`,
   hashed on the app key so paper/real separate automatically and the
   cross-process sharing the file exists for is preserved. See §4.2.
6. **DONE** — via `state.user_key()`; see the invalidation gotcha above.
7. **DONE** — the `RealtimeFeed` site now does its own explicit
   `credentials_for()` lookup. It is bound to the single serviced identity and
   logs a warning if `background_user_ids()` ever returns more, because one app
   key means one WebSocket session (KIS caps registrations at 41 per session),
   so N users genuinely need N feeds. That is Phase 6, and it must not be faked
   with a shared connection.

**Test coverage added:** `tests/credentials/test_crypto.py` (round-trip, the
row-binding attack in all three directions, tampering, wrong key, version byte,
unusable keys), `tests/credentials/test_broker_registry.py` (per-user caching,
the concurrent-construction invariant, eviction/close, token cache paths,
`is_real` sourced from credentials rather than global settings, secret-free
`repr`), `tests/auth/test_route_protection.py`, and
`tests/auth/test_background_loop_identity.py`.

### Phase 3 — onboarding gate — DONE 2026-08-06

The credential entry path that Phase 2 was missing now exists, plus the login
screen Phase 1 deferred.

1. **DONE.** `require_credentials` (`app/auth/dependencies.py`) returns **412**
   for a logged-in user with no credentials for the active account mode, and
   **401** takes precedence for anyone without a session — different screens,
   and a 412 to an anonymous caller would confirm the route exists. It fails
   **closed**: a credential-store error is a 412, never a pass, matching every
   other order gate. It reads `configured_modes`, which touches no ciphertext,
   so the gate keeps telling the truth even when the master key is wrong.

   Applied **per route, not per router** (`_needs_kis` in `orders.py` and
   `portfolio.py`, 10 routes). `/api/exec-mode`, `/api/settings/*`, the
   safety-switch routes and the pending-order *list* endpoints are deliberately
   left open: they are configuration and ledger reads the onboarding screen
   itself needs, and gating the whole router would leave the setup page unable
   to draw itself.

2. **DONE.** `app/credentials/validation.py` issues a real KIS token before
   anything is stored. The issued token is **deliberately kept**, written to
   `token_cache_path_for(app_key)` — the same file the broker will read — so
   validation warms the cache instead of burning the 1/min allowance and
   leaving the next real call to trip the limit. It uses `_get_token()`
   (cache-first), so re-saving unchanged credentials inside the token lifetime
   issues nothing at all. The UI disables its submit button during the round
   trip, since a double submit is the easy way to hit that limit.

   Note what this does **not** prove: token issuance does not involve the
   account number, so a wrong `account_8` still passes and surfaces on the
   first balance call.

3. **DONE.** `lifespan` no longer builds the feed. `state.ensure_market_feed()`
   is lazy, lock-guarded (two concurrent first-requests would otherwise open
   two WebSocket sessions on one app key and silently halve the subscription
   budget), and called from `require_credentials` — the earliest point a
   request is known to belong to someone with working credentials. `lifespan`
   still calls it once as an optimistic attempt, which succeeds only when the
   serviced user already has credentials from a previous session. The broker
   was already lazy as of Phase 2.

   `state.stop_market_feed()` exists because a running socket is authenticated
   with the old app key and **cannot be re-pointed in place** — the credential
   save path stops and restarts it, and the delete path stops it.

**Also landed (not in the original plan):**

- `app/web/routers/credentials.py` — `GET /api/credentials`,
  `PUT /api/credentials/{mode}`, `DELETE /api/credentials/{mode}`. **No
  response ever returns a stored secret**, masked or otherwise: a store that
  can read values back to the browser is one XSS away from handing them over,
  and changing a key means typing the new one. `GET` reports only *which modes
  are configured*, and reports `encryption_configured` separately so the UI can
  distinguish "you haven't set up KIS" from "this deployment has no master key,
  so nobody can" — the second is not something the user can fix by re-typing.
- `app/web/static/js/onboarding.js` + styles in `base.css` — a full-page gate
  that resolves before `buildCards()` runs. Login/register, then the credential
  form. It must run first: booting the dashboard underneath fires ~15 polls
  that all 401/412, and a half-drawn dashboard reads as a broken account rather
  than an unfinished setup. Resolves immediately and renders nothing when the
  visitor is already set up, which is what keeps the `auth_required=false`
  mid-migration state working.

**Still open, and it is the reason `auth_required` still defaults to False:**
flipping it to True is now safe from a UI standpoint (the login screen exists),
but that is a deployment decision, not a code one — see Phase 7, which owns the
move off `127.0.0.1` and needs TLS first. The boot validator already refuses to
start a non-loopback bind with auth off, so the two cannot get out of step.

### Phase 4 — close the Tier B gaps — DONE 2026-08-06

All seven, in the suggested severity order. Every one backfills existing rows
to `'default'`, matching how `account_mode` was backfilled to `'paper'`.
Verified live: the whole DDL set applies to real Postgres, is idempotent
across two passes, and left every existing row intact (39 pending_orders, 2
each in runtime_settings / daily_loss_state / portfolio_drawdown_state, all
stamped `default`; 125 trades / 357 memory_entries / 1280 alerts untouched).

| Store | What was shared | How it is scoped now |
|---|---|---|
| `runtime_settings` | `execution_mode`, `ai_profile` — one FULL_AUTO switch for everyone | `_overrides` keyed `user_id -> {key: value}`; unique index `(key, account_mode, user_id)` |
| `safety_switch` | Kill switch was literally one row (`id = 1`); one daily-loss baseline | Unique index on `user_id`; `daily_loss_state` on `(market, account_mode, user_id)` |
| `trade_allowlist_store` | The list `kis_bridge` gates trading on | Key prefix `"<user>:<mode>:<key>"` |
| `pending_orders` | Approval queue — A could approve B's order into B's account | `user_id` column + filter on every read **and on the resolve UPDATE** |
| `risk_rules` | Order/position caps, approval thresholds | Same shape as `runtime_settings` |
| `portfolio_risk` | Drawdown high-water mark; `market_lock` serialized all users | `user_id` on the HWM row; lock key is now `crc32(market:mode:user)` |
| `audit/stats` | Win-rate / alpha blended every user into one number | `user_id = $N` on all three queries (the column already existed) |

**Three things worth knowing before touching this again:**

- **`rule()` in `runtime_settings` and `risk_rules` stays SYNCHRONOUS.** Every
  call site is a gate deep inside sync code (`kis_bridge`, `ledger`, the
  pipeline). That is why both modules preload *every* user's overrides in
  `load()` rather than fetching on demand — a user missing from the cache
  silently falls back to the settings default. For `execution_mode` that
  default is MANUAL, so the failure direction is safe, but it is still wrong.
  **Phase 6 must re-run `load()` when a user logs in.**
- **The old two-column unique indexes are explicitly `DROP INDEX`ed**, not
  left alongside the new three-column ones. Leaving them would make one user's
  save collide with another's row — the original bug, wearing a new column.
- **`portfolio_risk`'s advisory lock key is a hash.** A collision makes two
  unrelated users serialize against each other: a slowdown, never a
  correctness failure. Acceptable here precisely because the lock decides
  ordering, not access — the same shortcut would not be acceptable for
  anything that decides who may do what.

**Deliberately NOT done here (it is Phase 5):** `trade_allowlist_store.load()`
still overwrites the global `settings.trade_allowlist_*` / `watchlist` lists.
The persistence is per-user; the in-memory copy is not.

**Migration hazard found and fixed (2026-08-06), worth knowing for Phases 5-6.**
The key-prefix migration originally skipped whenever a row already existed
under the new shape. That is correct for idempotency and wrong in one case: if
*any* row exists under the new key — even a spurious one — the legacy row is
stranded, never migrated and never deleted, and its contents silently vanish
on the next boot. It was found because exactly that happened on the dev DB (a
stray `default:paper:trade_allowlist_auto_added` holding one ticker stranded
the real 11-ticker legacy row).

It now **warns on every boot and changes nothing** when both shapes are
present. Not resolved automatically, because all three automatic answers are
wrong: preferring the legacy row clobbers post-upgrade saves, preferring the
new row is the silent data loss above, and **merging is the worst — a ticker
the user deliberately removed from the allowlist would come back, and this
list grants trading permission.** Any Phase 5/6 migration that re-keys an
existing table should copy this stance rather than guessing.

**`TRADE_ALLOWLIST_DOMESTIC` / `_OVERSEAS` env vars removed (2026-08-06).**
Only ever a first-boot seed: `load()` overwrites them from Postgres on every
startup, and tickers accumulate via the dashboard and the AI BUY hook. Verified
live — with both env vars gone, a boot restored 20 domestic + 11 overseas + 33
watchlist entries from the store. The list itself and the `kis_bridge` gate are
untouched. Note for anyone tempted to remove more: the allowlist is *newer*
than KIS candidate discovery (gate 2026-07-23, discovery 2026-07-20) and exists
precisely because discovery sweeps the whole market.

### Phase 5 — per-user runtime settings

**Read §6.5 first.** Roughly half of this section landed on 2026-08-06 in a
shape the text below does not describe — `kis_account_mode` in particular is
listed here as pending and is done. What remains of this section is the
allowlist/watchlist in-memory copy and the ~24 unmigrated settings.

The largest single refactor. These `Settings` fields are mutated at runtime and must
become per-user rather than global:

`trade_allowlist_domestic`, `trade_allowlist_overseas`, `watchlist`,
`trade_allowlist_auto_added`, `execution_mode`, `ai_profile`, `kis_account_mode`, and
every `risk_rules` override.

Mutation sites to rework: `add_ticker_to_trade_allowlist` /
`remove_ticker_from_trade_allowlist` / `add_ticker_to_watchlist` /
`mark_trade_allowlist_auto_added` (`app/web/state.py:281-431`), and the
load-at-startup restores in `trade_allowlist_store.load()` /
`runtime_settings.load()` / `risk_rules.load()`, all of which currently *overwrite the
global `settings` object*.

Note `kis_account_mode` is per-user in a multi-tenant world, which interacts with the
boot-time validators `_block_non_production_profile_on_real_account` and
`_require_real_money_limits_on_real_account` (`app/config.py:809-854`). Those run once
at boot against one global config; they need per-user equivalents evaluated at
credential-save time or at login.

### 6.5 What landed 2026-08-06 after Phase 4 (read this before Phase 5/6)

Phase 4 closed the Tier B stores. The work below happened afterwards, driven by
a product decision rather than the original plan, so it does not map onto the
Phase 5 text above. Recording it here because a session that reads only the
numbered phases will get the current state wrong.

**Product decision (2026-08-06): every user may use both 실전 and 모의투자.**
Real credentials are required at onboarding, paper is optional, and which of
the two is active is a per-user preference. That is a change of shape, not just
scope — the deployment no longer *has* an account mode.

#### Account mode became per-user

- New `app/account_mode.py` holds the policy (`allowed_modes_for`,
  `resolve_preference`, `resolve_account_mode`). The ContextVar itself lives in
  `app/config.py` (`current_account_mode`, `account_mode_scope`) purely to break
  an import cycle — `Settings` needs to read it.
- **54 call sites** moved off `settings.kis_account_mode`. Verify with:
  `grep -rn "settings\.kis_account_mode" --include='*.py' app | grep -v "^app/config.py"`
  — that must return **nothing**. A hit means a code path went back to the
  deployment-wide mode and will file one user's state under another's scope.
- `resolve_preference` honours a stated preference **only while credentials
  back it**. Someone who preferred paper and then deleted their paper
  credentials must not stay scoped to a mode whose broker cannot be built —
  their ledger rows, allowlist and kill-switch state would be filed under it.
- With nothing configured, it resolves to **`real`**, not the safer-looking
  `paper`. Nothing is reachable in that state either way (`require_credentials`
  refuses every gated route), so the only thing the choice affects is what the
  refusal *says* — and "no real credentials configured" is the actual next
  step, where "no paper credentials" would send someone to open a 모의투자
  account they never needed.

#### Per-user risk rules: 7 → 23

`RULE_SPEC` now covers sizing, portfolio caps and exposure limits, each with
bounds / NaN / integer validation. Assert the count directly rather than
eyeballing the dict:
`python -c "from app.execution.risk_rules import RULE_SPEC; print(len(RULE_SPEC))"`

#### Order-cap ceilings — the reason a user may raise their own caps at all

`KIS_MAX_ORDER_KRW_CEILING=100000000` / `..._USD_CEILING=70000` bound how far a
user may raise their *own* order-value cap. The four `*_REAL` limits default to
**0**, which is fail-closed: a real-money order cap of zero blocks everything
until someone sets it deliberately. Without the ceilings, making caps
user-editable would have meant a user could raise their own cap to anything,
which is not a cap.

#### Bugs found and fixed while doing the above

Worth knowing because each is a class of thing likely to recur:

- The settings screen **500'd for every new user** — a per-user store queried
  before the user had any rows.
- `risk_rules._overrides` **leaked paper limits into real**: the cache was
  keyed by user but not by mode, so a paper override answered a real-mode read.
  This is why `risk_rules.load()` now loads every `(user_id, account_mode)`
  scope, and why `runtime_settings`/`trade_allowlist_store` still needing that
  fix is listed as a blocker at the top of this document.
- A validation error rendered a raw float as `1e+08`, i.e. the message told the
  user their limit was too high without saying what the limit was.
- **Test pollution produced ~40 phantom failures** — tests reaching shared
  state (and, separately, the real dev Postgres). If a run shows failures that
  do not reproduce in isolation, suspect this before the code.

#### The two `load()` fixes (2026-08-06, after the above)

`risk_rules.load()` had already been fixed to load every scope. The other two
had not, and they were not the same bug:

- **`runtime_settings`** was the same bug. Its cache was keyed by user alone,
  so a paper `execution_mode` override answered a real-mode read — FULL_AUTO
  set while paper-testing could drive a real account. And `load()` filtered
  `WHERE account_mode = $1` on the mode current at startup, which with per-user
  modes is nobody's mode: no user is bound, so it resolved to the deployment
  default and every user on the other mode lost their overrides. Now keyed
  `(user_id, account_mode)` and loads every scope. `rule()`, `current()` and
  `save()` all take an optional `account_mode` defaulting to the request's own.

- **`trade_allowlist_store`** could not be fixed the same way, and pretending
  otherwise would have been the wrong move: `load()` writes into the
  process-global `settings.trade_allowlist_*` / `watchlist`, of which there is
  exactly one set. There is nowhere to put a second scope. What was actually
  wrong is that the scope was *implicit* — derived from
  `get_current_user_id()` / `current_account_mode()` at a moment when neither
  is bound. It now takes explicit `user_id` / `account_mode` arguments
  (defaulting to current, so all 8 call sites are unchanged), and `lifespan`
  names the scope it means. `save()` gained the same arguments because
  `load()`'s three migrations must write back to the scope they read from — a
  half-threaded scope would read one scope and migrate into another, which is
  worse than the original bug.

Making that in-memory copy per-user remains the top blocker in the header.

#### Per-user in-memory ticker lists (2026-08-06) — Phase 5's last piece

`trade_allowlist_domestic`/`_overseas`/`watchlist`/`trade_allowlist_auto_added`
stopped being pydantic fields on `Settings` and became per-(user, mode)
properties over `config._scoped_lists`. `trade_allowlist_store` being per-user
in Postgres had achieved nothing on its own: every gate reads the in-memory
copy, and there was one.

**Why properties instead of accessor functions.** 72 reads, 13 writes and 156
test sites — 241 in all. Converting the fields means every one of them keeps
working and becomes per-scope automatically; the alternative was 241 manual
edits where a single missed *read* is a silent cross-user leak on the list that
grants trading permission. The pattern was not invented for this: the
`kis_max_order_*` money limits are already scope-backed properties on
`Settings`, and pydantic v2 honours property setters, so `monkeypatch.setattr`
and `settings.x = [...]` both route through them unchanged. Total fallout: 8
failing tests, all of them asserting the old shape.

**Behaviour change worth knowing: a fresh scope is EMPTY.** The old
`_default_trade_allowlist_when_unset` validator seeded
`trade_allowlist_overseas` with `["AAPL"]` when both lists were empty, so a
first run did not show an empty dashboard. Carried into per-scope defaults that
becomes "every new user is born with AAPL on their trading allowlist" — a
ticker they never chose, tradable on their own real credentials, through the
gate `kis_bridge` checks. The seed is gone. Phase 3's onboarding flow already
gives first-run an explicit path, so an empty list afterwards reads as "add
your first ticker" on its own.
신규 스코프는 빈 목록 — 예전 기본값을 유지하면 사용자가 고르지도 않은 종목에
매매 권한이 생김.

**The three `model_validator`s became `config.normalize_scope(user, mode)`.**
They enforced `trade_allowlist ⊆ watchlist` and `auto_added ⊆ trade_allowlist`
at config-construction time, which cannot work when a scope is populated long
after `Settings` exists and there is no one object whose construction covers
every scope. `trade_allowlist_store.load()` calls it for the scope it just
restored — where stored data now enters.

**`load()` assigns inside `user_scope`/`account_mode_scope`.** It may be
restoring a scope other than the caller's, and `settings.<list> = ...` resolves
the scope from the ContextVars, so without that the rows would land in the
caller's scope instead. Same class of mistake as the half-threaded scope fixed
earlier in the same function.

**Tests: an autouse `conftest.py` fixture clears `_scoped_lists` between every
case.** The old global fields leaked between tests too, so this is not a new
hazard — but it is newly fixable in one place, and the handoff recorded ~40
phantom failures traced to shared state.

**One trap this cost 7 minutes of wall clock to find**, worth repeating because
Phase 6 will meet it again: `tests/auth/test_background_loop_identity.py`
seeded one global allowlist and waited on `while len(seen) < 2: await
sleep(0)`. Once the loops read each user's own (now empty) list, the loop
produced nothing and the unbounded wait hung the entire suite rather than
failing. Those waits are now bounded by `_wait_for()`, which fails in
milliseconds with a message naming the likely cause.

#### State of the tree

1593 tests pass with the dev Postgres reachable (41 of them require it — see the
header). Local deploy verified end-to-end through `./pipeline.sh dev`
on Colima; Server 4 is the same architecture (arm64), so the locally built
image is the real artifact rather than an approximation.

**Operational, not code:** `.env.dev` changed and must be re-uploaded to the
Jenkins `invest-env-dev` secret-file credential. Deploying without that step
runs the new image against the old configuration — which now means a missing
`CREDENTIAL_ENCRYPTION_KEY` and the old order-cap values.

### 6.6 Phase 6 as built (2026-08-06)

The three bullets below were the plan. Two landed as written; the feed landed
differently and for a better reason than the plan knew.

**`background_user_ids()` → `serviced_scopes()`.** Returns
`ServicedScope(user_id, account_mode)` pairs, not bare ids: every store the
loops touch is keyed by both, so a loop binding only the identity writes into
whatever mode the process defaults to. Membership is **users with stored KIS
credentials** whose account is active — credentials are the real precondition,
since without them `broker_for_user` returns None and a pass would only log
failures. Exactly one scope per user (their resolved preference), because
running someone's automation in a mode they are not currently in would trade
the wrong account.

A store failure returns `[]` rather than falling back to `[settings.user_id]`.
That fallback would run the legacy identity's automation using whatever
allowlist happened to be in memory, on a process that cannot confirm who its
users are. Idling one interval misses a stop-loss check, a real cost; guessing
whose account to trade is worse.

**`ensure_loaded` before any allowlist read.** The lists are per (user, mode)
and empty until loaded, so this is load-bearing, not defensive: skip it and the
loops iterate zero tickers and check nobody's stop-loss, silently, because an
empty allowlist is indistinguishable from "this user permits nothing".
Idempotent per process so the loops can call it every pass without re-running
`load()`'s three migrations. `lifespan` loads every serviced scope at startup;
the loops' calls cover users who register credentials mid-session.

**Per-scope failure isolation.** Both loops wrap each scope in its own `try`, on
top of the existing per-ticker one. A broker outage for user A is not a reason
to stop checking user B's stop-losses.

**The poller stays ONE shared loop**, over the deduplicated union of every
scope's allowlist — as the plan said, because a price is public data and
polling per user would refetch the same ticker N times against one shared
per-key rate limit. It does have to run *bound to somebody*: `poll_once`
resolves a broker from the current identity, so `poll_plan()` returns the scope
to poll as, alongside the tickers. That one user's app key carries the whole
union's polling — fewer requests than per-user polling, but neither free nor
fair; if it bites, rotate the polling scope between passes rather than polling
per user.

**The feed did NOT become N feeds, and should not.** The plan said "each user
needs their own RealtimeFeed... N users = N WebSocket connections". That is
true only for 체결통보 (fill notifications), which is keyed by HTS ID and only
reports the authenticating account's fills. Everything the *order gates*
consume from this feed — trading halt, VI, circuit breaker, ticks, orderbook —
is **public per-ticker data**, so one connection legitimately serves everybody
and spends the 41-registration cap once on the union instead of N times on
overlapping lists.
게이트가 쓰는 데이터는 전부 공개 시세라 연결 하나로 충분하며, 41개 등록 한도를
사용자마다 중복 소모하지 않아도 됨.

What that required was fixing `_prioritized_krx_codes` to subscribe the
**union** across scopes. Built from one scope, every other user's tickers had no
real-time coverage, and `kis_cb_fail_open_when_unknown` defaults to False — so
their domestic orders would all have failed closed on "circuit-breaker status
unknown". Safe, and a total trading outage for them. Held positions still get
first claim on the 41 slots, now collected per scope since whose position it is
decides which account snapshot reports it.

**Still outstanding: per-user 체결통보.** On the single feed only the first
serviced user gets live fill events; everyone else's fills surface through the
existing REST reconciliation path. A display/notification gap, not a gate, which
is why it does not block anything.

**Also per-scope now:** `auto_trade_loop._status` and the SEMI_AUTO
first-cycle flag. Both were single module slots, so every user's dashboard
showed whichever cycle ran last, and the second serviced user never got its one
SEMI_AUTO run. `_status` is reached through a small proxy that resolves the
scope on attribute access — same reasoning as the Settings ticker-list
properties: keep the ~22 call sites, move the resolution underneath them.

**Test-suite consequence.** `conftest.py` gained an autouse fixture defaulting
every test to one serviced scope and treating scopes as already loaded.
Unstubbed, Phase 6 would have given the whole suite a new Postgres dependency,
and the tests that assert on their own fixture tickers would instead see
whatever is stored in the dev DB. Tests that mean to exercise the real
functions reach them through `conftest.REAL`.

### Phase 6 — loops, feeds, caches

**Read §6.6 first** — this is the original plan, kept for its reasoning. The
feed bullet in particular is wrong as stated: N feeds are needed only for
체결통보, not for the order gates, which consume public per-ticker data.

- `stop_loss_loop` and `auto_trade_loop` become per-user tasks, started when a user
  logs in / has credentials and cancelled when they do not.
- The minute poller (`app/live/poller.py:115`) can stay a single shared loop over the
  deduplicated union of all users' allowlists — it is public market data and never
  calls an LLM or places an order.
- Each user needs their **own** `RealtimeFeed`: their own app key, and KIS caps
  real-time registrations at 41 per session (`app/web/state.py:714-750` documents the
  cap and its held-position prioritization). N users = N WebSocket connections.

### Phase 7 — network posture and documentation — 2/3/4 DONE 2026-08-06, 1 deliberately NOT

1. **NOT DONE, and must not be.** `dashboard_host` stays `127.0.0.1`. The
   precondition the plan named is met — auth covers 57 of 61 routes, the other
   four being the auth endpoints themselves — but the prerequisites it listed
   are not: no TLS, no reverse proxy, session cookie is `secure=False`
   (`app/web/routers/auth.py`), no CSRF protection. Moving off loopback today
   would ship session cookies over plain HTTP. The boot validator already
   refuses a non-loopback bind with `AUTH_REQUIRED=false`, so the two cannot
   drift apart.
   루프백을 벗어나려면 TLS·secure 쿠키·CSRF가 먼저 필요하며 아직 없음.

2. **DONE.** Both `docs/safety-and-risk.md` sections rewritten. The important
   part was not deleting the old reasoning but replacing its *basis*: the
   config-only execution switches stay config-only, and the conclusion is
   unchanged, but "because the order API is unauthenticated" is no longer true
   and "because there is no trustworthy transport for a session yet" is. §5.1's
   real-money procedure was the sharper problem — it asserted as project law
   that account mode can only change by editing a config file, which the code
   contradicts, so anyone reading it in good faith concluded the code was in
   violation. It now documents the four layers that actually gate real money.

3. **DONE.** `docs/architecture.md` §1's three false claims corrected
   (single-user, unauthenticated `/api/order`, `user_id` as a placeholder), with
   the network-posture caveat from item 1 stated alongside so the correction
   cannot be read as "we are ready to expose this".

4. **DONE for the roadmap row.** 단계 4 went from "부분 완료 (2개 항목 완료, 2개
   보류)" to complete-with-remainder, pointing here as the source of truth.

   **NOT done, and not mine to do: `docs/status-and-decisions.md` §2's core
   premise.** It records "본인 계좌 전용 개인 도구" and the legal analysis behind
   it (투자일임업 exposure, KIS keys being personal-use, self-host-not-service as
   the only distribution direction). Everything Phases 0-6 built is the trigger
   that section names for revisiting itself. A prominent unresolved-decision
   note now sits at the head of §2 listing what has to be answered, and
   `architecture.md`'s summary line points at it rather than silently dropping
   the word "개인용". Rewriting the premise is a product and legal judgement.

**Two claims were wrong in the first draft of these edits and were caught by
verifying them against the code rather than re-reading the prose** — worth
repeating as a method, not just an anecdote. The draft said the four `*_REAL`
order limits "default to 0"; the code default is `None`, which raises
`RuntimeError` from `_mode_scoped_money` (0 is what `.env.dev` sets). Both are
fail-closed but by different mechanisms, and the doc now says which. It also
cited `PUT /api/credentials/preference`, which does not exist — the route is
`PUT /api/account-mode`.

---

## 7. Open items and risks

1. **CLI entry points assume one implicit user.** **Partly closed.** The three
   broker-using scripts (`backfill_minute_candles`, `scan_minute_data_gaps`,
   `check_daily_minute_chart_tr`) now take `--user-email` via
   `scripts/_cli_user.py`. Still open: `app.main`, `app.backtest`, `app.memory`,
   `app.execution` and `app.web`'s admin reseed resolve identity from
   `settings.user_id` and keep working only through the ContextVar fallback.
   They do not touch broker credentials, so nothing is broken today — but
   anything that starts placing orders from those paths needs the same flag.
   Note `app.auth.store.account_for_email()` is authentication-free by
   construction and exists only for these local CLI paths; it must never be
   called from a request.
2. **`.env.kis.local` disappears as a concept.** Still open — `app/config.py`'s
   `model_config` `env_file` tuple still lists it and the scripts' help text
   may still mention it. Harmless (the file is absent and the credentials it
   held no longer exist as settings) but now dead, so clean it up.
3. **Test suite.** Verified, not assumed: **1582 pass** as of 2026-08-06.
   `tests/test_multi_instance_seam.py`'s `get_current_user_id() == "default"`
   assertion still holds via the ContextVar fallback.
   Three test-facing conventions to know about: `tests/conftest.py` provides
   `broker_stub()` (because `state._kis_broker`, which tests assigned directly,
   became the coroutine `state.kis_broker()`) and `bypass_kis_gate()` (router
   unit tests must not also assert that a database is reachable);
   `tests/dataflows/test_kis_provider.py` has a local `_returns()` for the same
   reason as `broker_stub`.
   **Tests can reach the real dev Postgres.** This caused ~40 phantom failures
   once and wrote a stray row into `trade_allowlist_state` that stranded a
   legacy row. If failures do not reproduce in isolation, suspect this first.
   Not fixed.
4. **Key rotation UX.** Not designed yet: what happens when a user replaces their KIS
   credentials while positions are open and a `RealtimeFeed` is running.
5. **Encryption key rotation.** Not designed yet. Re-encrypting all rows needs a
   documented procedure before this goes anywhere near production.
6. **`kis_execution_enabled` / `kis_execution_dry_run` are not user-controllable,
   deliberately.** Approved in principle 2026-08-06, gated on prerequisites that
   do not exist yet: TLS with `secure=True` session cookies, CSRF protection,
   and step-up re-authentication. These two switches are the master on/off for
   real orders, so exposing them over a plain-HTTP session would make a stolen
   or CSRF'd session sufficient to start live trading. Not built.
7. **~24 settings remain deployment-wide** (automation cadence, screening
   counts, alerting, analysis depth). The automation ones cannot move to
   per-user until the background loops are per-user (Phase 6) — a per-user
   `auto_trade_interval_minutes` means nothing while one shared loop drives
   everybody.
8. **`docs/safety-and-risk.md` §5.1 now actively contradicts the code.** It
   forbids exposing account-mode selection through the HTTP API, on the stated
   premise that the dashboard is unauthenticated, localhost-only and
   single-user. All three premises are gone, and the API now does expose a
   per-user mode preference. `app/account_mode.py`'s header explains why that
   is safe on the new basis (a preference only ever moves the person who set
   it; real mode is reachable only by entering real credentials). The document
   has not been updated to say so, so anyone reading it in good faith will
   conclude the code is in violation. Same for
   `docs/status-and-decisions.md` §2 (§8 below) and `docs/architecture.md` §1.
   This is Phase 7 item 2-4, and it is now the largest documentation debt.

---

## 8. Documented project premise this contradicts

Raised with the user on 2026-08-06. Recording it here so it is a conscious decision
rather than a silent contradiction — it does not change the plan above.

`docs/status-and-decisions.md` §2 records as a core premise that this is a
personal-account-only tool, that commercialization is out of roadmap, and
specifically:

- Applying "완전자동" (execution without human confirmation) to **another person's**
  account may constitute 투자일임업 under the Capital Markets Act — FSC authorization,
  minimum capital, and qualified personnel required.
- Providing recommendations alone may trigger 투자자문업/유사투자자문업 registration.
- KIS Open API keys are **issued for personal use**, so productizing third-party
  account access may violate the API terms.
- The intended distribution model, if any, was limited to "software people self-host
  with their own keys," never a hosted service.
- That document names the move to handling other people's accounts as the explicit
  trigger to expand the documentation system, and calls for separate legal advice
  before any billing.

Hosting multiple users' KIS credentials in one service is that trigger. Someone
should update §2 deliberately, and the legal question is a real one to route to
whoever owns it — not something this plan resolves.

---

## 9. How the findings above were verified

So a later session can re-check rather than trust:

```bash
# Which env keys exist, values never printed
grep -oE '^[A-Za-z_][A-Za-z0-9_]*=' .env.local | tr -d '='

# Post-Phase-2: this must return NOTHING outside app/config.py's own comments.
# A hit means a credential setting came back from the dead.
grep -rnE 'settings\.kis_(paper|real)_(app_key|app_secret|account_8|account_prod|hts_id)' \
  --include='*.py' app scripts tests

# Post-2026-08-06: this must return NOTHING. A hit means a code path went back
# to the deployment-wide account mode and will file one user's state (ledger,
# allowlist, kill switch) under whatever mode the process defaults to.
grep -rn 'settings\.kis_account_mode' --include='*.py' app | grep -v '^app/config.py'

# The four ticker lists must NOT be pydantic fields again — as properties they
# are per (user, mode); as fields they would be one shared list, and that list
# is the kis_bridge trading gate. Expect all four to print False.
python -c 'from app.config import Settings; print({k: k in Settings.model_fields for k in ("trade_allowlist_domestic","trade_allowlist_overseas","watchlist","trade_allowlist_auto_added")})'

# Per-user risk rules: expect 23 (was 7 before 2026-08-06).
python -c 'from app.execution.risk_rules import RULE_SPEC; print(len(RULE_SPEC))'

# Fixed 2026-08-06: neither of these may filter load() by account_mode again.
# A hit means a real-mode user's overrides silently vanish on restart.
grep -n 'WHERE account_mode' app/execution/runtime_settings.py app/execution/risk_rules.py

# Post-Phase-2: no route may be unauthenticated except the four auth ones.
# tests/auth/test_route_protection.py asserts this, but to check by hand:
grep -rn "include_router" app/web/app.py

# Every consumer of a still-global credential-adjacent setting
grep -rnE 'settings\.(fred_api_key|database_url|llm_base_url|kis_account_mode|user_id|credential_encryption_key)' --include='*.py' app scripts tests

# Tier A vs Tier B: which persistence modules scope by user
for f in app/memory/store.py app/execution/pending_orders.py app/execution/safety_switch.py app/execution/portfolio_risk.py app/live/alerts.py app/execution/risk_rules.py app/audit/stats.py; do echo "--- $f"; grep -c get_current_user_id $f; done

# The 54 unauthenticated routes
grep -rnE '@router\.(get|post|put|delete)' --include='*.py' app/web/routers
```
