# 단일계좌 전환 가이드 (로그인 제거 · Jenkins 시크릿 주입)

> 작성일: 2026-08-07
> 대상 브랜치: `main` (기준 커밋 `330dee7`)
> 선행 문서: [`docs/multi-user-credentials-plan.md`](multi-user-credentials-plan.md) — **이 문서는 그 계획의 되돌리기(rollback) 계획서다.** 왜 그렇게 만들었는지는 그쪽에 다 적혀 있으니, 각 단계에서 "이걸 지워도 되나?"가 헷갈리면 반드시 원문의 해당 Phase를 먼저 읽을 것.

---

## 0. 이 문서가 전제하는 결정 (2026-08-07 확정)

| # | 항목 | 결정 |
|---|---|---|
| 1 | KIS 앱키/시크릿/계좌번호 | **Jenkins Credentials → 환경변수 주입.** `app/credentials/`(암호화 DB 테이블)와 화면 입력(온보딩) 폐지, `app/config.py` 설정 필드로 복귀 |
| 2 | 대시보드 접근 통제 | **`dashboard_host=127.0.0.1` 유지 + SSH 터널.** 앱 자체 인증 없음. 네트워크 자세는 지금과 동일하게 유지 |
| 3 | 멀티유저 코드 | **단일 사용자로 고정.** 로그인·세션·온보딩 UI/라우터는 삭제, DB의 `user_id` 컬럼·FK·필터링은 **그대로 둔다** |
| 4 | 계좌 모드 | **paper / real 둘 다 유지.** 본인 실전계좌 + 모의계좌 테스트 병행 |

**결정 3에 대한 근거(질문에 대한 직접적인 답):**

"단일 사용자로 만들 것인가, 기능만 숨길 것인가"는 사실 **두 개의 다른 축**이고, 축마다 답이 다르다.

- **인증 축(로그인/세션/회원가입)** — **실제로 제거한다.** 숨기기만 하면 `auth_required=false` + 세션 미들웨어 + `require_user`의 대체계정 폴백이 그대로 살아 있는 상태가 된다. 이건 "꺼져 있는 인증"이 아니라 "언제든 켜질 수 있고, 켜지면 아무도 못 들어오는 코드"다. 단일 사용자 전용이라 켤 일이 없다면 이 코드는 유지비만 남는다.
- **데이터 스코프 축(`user_id` 컬럼, FK, SELECT 필터)** — **건드리지 않는다.** 이건 "멀티유저 기능"이 아니라 **행 소유자 태그**다. 값이 항상 하나여도 동작은 완전히 동일하고, 제거하려면 `trades`/`memory_entries`/`alerts`/`pending_orders`/`identity_claims` 마이그레이션 + 20여 개 모듈 수정 + `tests/` 대량 수정이 필요하다. 2026-08-07~08 `app/execution`·`app/brokers`·`app/live` 감사에서 잡은 P0 버그들이 대부분 이 스코프 경계에 걸려 있었으므로, 지금 걷어내면 그 수정들을 다시 흔든다. **이득 없음 / 위험 큼**.

즉 **"로그인은 삭제, user_id는 값을 하나로 고정"** 이 이 문서의 방침이다.

---

## 1. 현재 구조 요약 (무엇이 멀티유저인가)

```
요청 → bind_user_identity (미들웨어)
         ├─ 세션 쿠키 → app/auth/store.resolve_session() → Account
         ├─ set_current_user_id(account.user_id)        ← ContextVar
         └─ set_current_account_mode(resolve_account_mode(...))  ← ContextVar
      → 라우터 (Depends(require_user) / Depends(require_credentials))
      → get_broker() → broker_for_user(user_id, mode)
                     → credentials_for(user_id, mode)  → user_broker_credentials 테이블 복호화
```

| 계층 | 파일 | 역할 |
|---|---|---|
| 인증 | `app/auth/{store,dependencies,passwords}.py` | 이메일/비밀번호, `sessions` 테이블, 세션쿠키, `require_user`/`require_credentials` |
| 인증 API | `app/web/routers/auth.py` | `/api/auth/{register,login,logout,me}` |
| 자격증명 | `app/credentials/{store,crypto,validation}.py` | `user_broker_credentials` 테이블 AES-256 암호화, 저장 전 KIS 실토큰 검증 |
| 자격증명 API | `app/web/routers/credentials.py` | `GET/PUT/DELETE /api/credentials`, `PUT /api/account-mode` |
| 신원 seam | `app/user_context.py` | `get_current_user_id()`, `user_scope()`, `serviced_scopes()`, `register_user()`, `record_session_start()` |
| 모드 정책 | `app/account_mode.py` | `resolve_preference()`, `resolve_account_mode()`, `allowed_modes_for()` |
| 프론트 게이트 | `app/web/static/js/onboarding.js` | 로그인/회원가입 화면 + KIS 자격증명 입력 화면 (`ensureReady()`) |
| 배경 루프 | `app/web/state.py` | `serviced_scopes()`를 순회하며 `user_scope()`+`account_mode_scope()` 안에서 실행 |

**핵심 의존 고리 (여기가 이번 작업의 진짜 난이도다):**

`serviced_scopes()`는 `user_broker_credentials JOIN users WHERE u.active` 쿼리 결과다.
→ 자격증명 테이블을 없애면 **이 함수가 빈 목록을 반환하고, 그 순간 poller / stop_loss_loop / auto_trade_loop / 실시간 피드 / 부팅 시 allowlist 복원이 전부 조용히 아무것도 안 한다.**
→ `app/user_context.py`의 문서에 적힌 대로 "빈 목록 = 한 주기 쉼"이 의도된 동작이라 **에러도 안 난다.** 매매가 멈춘 걸 로그로만 알 수 있다.

**⚠️ Phase 2를 하면 Phase 4를 반드시 같은 커밋에서 해야 한다.** 나눠서 커밋하면 그 사이 커밋은 "부팅은 되는데 자동매매가 전혀 안 도는" 상태가 된다.

---

## 2. 목표 구조

```
부팅 → Settings (Jenkins가 주입한 env) 검증
         └─ KIS_APP_KEY_REAL / _PAPER 등이 있는 모드만 "설정된 모드"
요청 → bind_account_mode (미들웨어, 축소판)
         └─ set_current_account_mode(현재 선택된 모드)   ← ContextVar 유지
      → 라우터 (Depends(require_kis) 만 남음)
      → get_broker() → broker_for_user(SINGLE_USER_ID, mode)
                     → settings.kis_credentials(mode)   ← env에서 구성
```

- `user_id`는 언제나 `settings.user_id` 한 값 (기본 `"default"`, 배포 시 예: `owner`).
- `account_mode`는 **여전히 요청/컨텍스트 스코프**로 남긴다 (paper/real 둘 다 쓰기로 했으므로).
- 로그인·세션·회원가입·자격증명 입력 화면 없음.

---

## 3. Jenkins 시크릿 설계 (Phase 1)

### 3.1 Jenkins가 주입해야 할 값

기존 `.env.kis.local` → 현재 `.env.local`/`.env.dev`에 흩어져 있던 것 중 **비밀값**은 다음이 전부다.

| Jenkins Credential ID (제안) | 주입될 env 키 | 비고 |
|---|---|---|
| `invest-kis-app-key-real` | `KIS_APP_KEY_REAL` | 본인 실전계좌 |
| `invest-kis-app-secret-real` | `KIS_APP_SECRET_REAL` | |
| `invest-kis-account-8-real` | `KIS_ACCOUNT_8_REAL` | 계좌번호 앞 8자리 |
| `invest-kis-hts-id-real` | `KIS_HTS_ID_REAL` | 선택 — 없으면 체결통보 웹소켓만 비활성 |
| `invest-kis-app-key-paper` | `KIS_APP_KEY_PAPER` | 모의계좌 |
| `invest-kis-app-secret-paper` | `KIS_APP_SECRET_PAPER` | |
| `invest-kis-account-8-paper` | `KIS_ACCOUNT_8_PAPER` | |
| `invest-kis-hts-id-paper` | `KIS_HTS_ID_PAPER` | 선택 |
| `invest-database-url` | `DATABASE_URL` | 계정/비밀번호 포함 |
| `invest-fred-api-key` | `FRED_API_KEY` | 선택 |

**모의계좌 키도 Jenkins에 넣는다** (2026-08-07 결정, §7-3). 보안 가치는 실전보다 낮지만 `.env.base`(git 추적 대상)에 두면 저장소에 키가 남고, 실전/모의를 다른 방식으로 관리하면 "이 키는 어디 있더라"가 매번 생긴다. 관리 창구를 하나로 유지하는 편이 실수 여지가 적다.

**비밀 아닌 값** (`KIS_ACCOUNT_PROD_*`=`01`, `KIS_*_BASE_URL`, 리스크 한도, 루프 주기 등)은 Jenkins Credentials가 아니라 `.env.dev`(git 미추적) 또는 Jenkins 파이프라인의 평문 환경변수로 둔다. 시크릿 스토어에 넣을수록 회전 비용만 늘어난다.

**폐기되는 값:** `CREDENTIAL_ENCRYPTION_KEY`, `AUTH_REQUIRED`, `SESSION_COOKIE_NAME`, `SESSION_TTL_HOURS`.

### 3.2 Jenkins 파이프라인 형태 (예시)

`pipeline.sh`가 `--env-file "$ENV_FILE"`을 쓰므로, Jenkins에서 시크릿을 **파일로 조립해서 넘기는** 방식이 현재 스크립트와 가장 잘 맞는다.

```groovy
// Jenkinsfile (신규)
pipeline {
  agent any
  stages {
    stage('Compose env') {
      steps {
        withCredentials([
          string(credentialsId: 'invest-kis-app-key-real',    variable: 'KIS_APP_KEY_REAL'),
          string(credentialsId: 'invest-kis-app-secret-real', variable: 'KIS_APP_SECRET_REAL'),
          string(credentialsId: 'invest-kis-account-8-real',  variable: 'KIS_ACCOUNT_8_REAL'),
          string(credentialsId: 'invest-kis-app-key-paper',   variable: 'KIS_APP_KEY_PAPER'),
          string(credentialsId: 'invest-kis-app-secret-paper',variable: 'KIS_APP_SECRET_PAPER'),
          string(credentialsId: 'invest-kis-account-8-paper', variable: 'KIS_ACCOUNT_8_PAPER'),
          string(credentialsId: 'invest-database-url',        variable: 'DATABASE_URL'),
        ]) {
          // .env.base = git에 커밋된 비(非)비밀 설정 템플릿
          sh '''
            set -eu
            umask 077
            cp .env.base .env.dev
            {
              echo "KIS_APP_KEY_REAL=${KIS_APP_KEY_REAL}"
              echo "KIS_APP_SECRET_REAL=${KIS_APP_SECRET_REAL}"
              echo "KIS_ACCOUNT_8_REAL=${KIS_ACCOUNT_8_REAL}"
              echo "KIS_APP_KEY_PAPER=${KIS_APP_KEY_PAPER}"
              echo "KIS_APP_SECRET_PAPER=${KIS_APP_SECRET_PAPER}"
              echo "KIS_ACCOUNT_8_PAPER=${KIS_ACCOUNT_8_PAPER}"
              echo "DATABASE_URL=${DATABASE_URL}"
            } >> .env.dev
          '''
        }
      }
    }
    stage('Deploy') {
      steps { sh './pipeline.sh dev' }
    }
  }
  post {
    always { sh 'rm -f .env.dev' }   // 워크스페이스에 시크릿 잔존 금지
  }
}
```

**주의 3가지:**
1. `umask 077` + `post { always { rm -f } }` — 워크스페이스에 평문 env가 남으면 Jenkins에 접근 가능한 모든 사람이 운용 계좌 키를 읽는다.
2. `sh` 스텝 안에서 `echo "$KIS_APP_SECRET_REAL"` 같은 걸 **로그로 출력하지 말 것.** Jenkins의 마스킹은 `withCredentials` 범위 안에서만 동작하고, 완벽하지 않다.
3. `.env.base`(비밀 없는 템플릿)를 새로 만들고 `.gitignore`에 `.env.dev`, `.env.local`이 이미 있는지 확인할 것.

**Jenkins agent는 배포 서버와 같은 호스트다** (2026-08-07 확정, §7-5). 따라서 `pipeline.sh`의 `docker run --network host`가 그대로 동작하고 SSH/registry push 같은 원격 배포 단계는 필요 없다. 대신 두 가지가 전제된다: Jenkins 실행 계정이 그 서버의 Docker 소켓에 접근할 수 있어야 하고, **워크스페이스가 곧 운영 서버의 디스크이므로 위 1번(umask)과 `rm -f .env.dev`가 선택이 아니라 필수다.**

### 3.3 이 방식이 DB 암호화 대비 잃는 것 (명시적으로 기록)

- 시크릿이 **컨테이너 프로세스 환경에 평문으로 존재**한다. `docker inspect`, `/proc/<pid>/environ`로 읽힌다. DB 암호화 방식에서는 마스터키가 있어야만 읽혔다.
- 키 회전이 **재배포**를 요구한다 (기존엔 화면에서 교체 + `invalidate_broker_cache()`로 무중단).
- 대신 얻는 것: `CREDENTIAL_ENCRYPTION_KEY` 분실 = 전체 자격증명 영구소실이라는 단일 실패점이 사라지고, 시크릿 수명주기가 Jenkins 한 곳으로 모인다.

단일 계좌·SSH 터널 접근이라는 전제에서는 합리적인 교환이다. 다만 **서버 OS 계정 접근 = 운용 계좌 키 노출**이 되었다는 점은 인지하고 있어야 한다.

---

## 4. 작업 순서

각 Phase 끝에 `uv run pytest tests/ -x -q`와 실제 부팅 확인을 넣는다.
**Phase 2와 Phase 4는 반드시 한 커밋에 묶는다** (§1의 경고 참조).

### Phase 1 — 설정 필드 복귀 (`app/config.py`)

1. **KIS 자격증명 필드 추가.** `credential_encryption_key` 자리에 넣는다.

```python
    # KIS 자격증명 — 2026-08-07 단일 계좌 전환으로 env 복귀.
    # 멀티유저 시절에는 user_broker_credentials 테이블에 사용자별 암호화 저장이었음
    # (docs/multi-user-credentials-plan.md Phase 2). 단일 계좌만 쓰기로
    # 결정되면서 "누구의 키인가"라는 질문 자체가 사라졌고, 시크릿 수명주기는
    # Jenkins Credentials가 담당한다 — docs/single-user-rollback-guide.md §3.
    # None = 그 모드 미설정. paper/real 각각 독립적으로 설정 가능해야 한다
    # (실전만 등록하고 모의는 안 쓰는 상태가 정상 상태이므로).
    kis_app_key_real: str | None = None
    kis_app_secret_real: str | None = None
    kis_account_8_real: str | None = None
    kis_account_prod_real: str = "01"
    kis_hts_id_real: str | None = None          # 없으면 체결통보 웹소켓 구독만 비활성

    kis_app_key_paper: str | None = None
    kis_app_secret_paper: str | None = None
    kis_account_8_paper: str | None = None
    kis_account_prod_paper: str = "01"
    kis_hts_id_paper: str | None = None
```

2. **모드별 자격증명 조회 헬퍼**를 `Settings`에 추가. 반환 타입은 기존 `KISCredentials`를 재사용한다 (`__repr__`이 시크릿을 가리는 그 클래스 — **절대 dict나 tuple로 바꾸지 말 것.** 예외/로그 경유 유출을 막는 유일한 장치다). `app/credentials/store.py`에서 `KISCredentials` 정의만 `app/domain/models.py`로 옮기고 나머지는 삭제한다.

```python
    def configured_kis_modes(self) -> set[str]:
        """자격증명이 실제로 채워진 모드 집합. 기존 credentials.store.configured_modes()의
        대체 — 복호화 없이 '설정 여부'만 답하던 성질을 그대로 유지한다."""
        return {
            mode for mode in ("paper", "real")
            if all(getattr(self, f"kis_{f}_{mode}")
                   for f in ("app_key", "app_secret", "account_8"))
        }

    def kis_credentials(self, account_mode: str) -> KISCredentials | None:
        """해당 모드의 자격증명, 미설정이면 None.

        None은 '이 배포에 그 모드가 설정되지 않았다'만 의미한다. 다른 모드 값으로
        폴백하지 않는다 — 모의계좌 키로 실전 주문이 나가거나 그 반대가 되는 것이
        이 프로젝트가 가장 피해야 할 사고다.
        모드가 다르면 절대 대체하지 않음."""
        if account_mode not in ("paper", "real"):
            raise ValueError(f"unknown account mode {account_mode!r}")
        if account_mode not in self.configured_kis_modes():
            return None
        return KISCredentials(
            user_id=self.user_id,
            account_mode=account_mode,
            app_key=getattr(self, f"kis_app_key_{account_mode}"),
            app_secret=getattr(self, f"kis_app_secret_{account_mode}"),
            account_8=getattr(self, f"kis_account_8_{account_mode}"),
            account_prod=getattr(self, f"kis_account_prod_{account_mode}"),
            hts_id=getattr(self, f"kis_hts_id_{account_mode}") or None,
        )
```

3. **`credential_encryption_key` 필드와 `_validate_credential_encryption_key` 검증기 삭제** (`app/config.py:545`, `:1033`).

4. **`auth_required` / `session_cookie_name` / `session_ttl_hours` 필드 삭제** (`:451~455`).

5. **`_require_auth_when_network_reachable` 검증기(`:1087`)를 약화시키지 말고 강화한다.** `auth_required`가 사라지면 이 검증기는 참조 대상을 잃는데, **그냥 지우면 안 된다.** 지금 이 앱을 지키는 건 loopback 바인딩 하나뿐이므로:

```python
    @model_validator(mode="after")
    def _refuse_non_loopback_bind(self) -> "Settings":
        # 앱 자체 인증이 없다(2026-08-07 단일계좌 전환으로 로그인 제거).
        # POST /api/order는 실제 운용 계좌에 주문을 넣는다. 따라서 루프백 외의
        # 주소로 바인딩하는 순간 내부망 누구나 계좌의 돈을 움직일 수 있게 된다.
        # 접근은 SSH 터널로만 — 이 검증기가 그 전제를 코드로 강제하는 유일한 지점이다.
        # 이걸 완화하려면 먼저 앞단(nginx Basic Auth/VPN)을 두고, 그 사실을 여기에 적을 것.
        loopback = {"127.0.0.1", "localhost", "::1"}
        if self.dashboard_host not in loopback:
            raise ValueError(
                f"dashboard_host={self.dashboard_host!r} is not loopback. This deployment has "
                "no application-level authentication; order placement must stay unreachable "
                "from the network. See docs/single-user-rollback-guide.md §0."
            )
        return self
```

6. `_require_real_money_limits_on_real_account`(`:1124`)는 **그대로 둔다.** 실전 모드에서 `kis_max_order_krw_real` 등이 없으면 부팅 거부하는 가드로, 이번 변경과 무관하게 계속 필요하다. Jenkins env에 `KIS_MAX_ORDER_KRW_REAL`, `KIS_MAX_ORDER_USD_REAL`, `KIS_APPROVAL_THRESHOLD_KRW_REAL`, `KIS_APPROVAL_THRESHOLD_USD_REAL` 4개를 **반드시 추가**해야 한다 (현재 `.env.local`/`.env.dev`에 `_PAPER`만 있다 — 실전 전환 시 부팅 실패한다).

### Phase 2 — 자격증명 저장소 폐지 (`app/credentials/`)

| 대상 | 처리 |
|---|---|
| `app/credentials/crypto.py` | **삭제** |
| `app/credentials/store.py` | **삭제.** 단 `KISCredentials` dataclass는 `app/domain/models.py`로 이동 |
| `app/credentials/validation.py` | **보존 + 용도 변경** (아래 참조) |
| `app/web/routers/credentials.py` | **부분 삭제** (아래 참조) |
| `tests/credentials/test_crypto.py`, `test_credentials_router.py` | 삭제 |
| `tests/credentials/test_broker_registry.py` | 유지하되 fixture를 settings 기반으로 수정 |

**`validation.py`는 살려서 부팅 시 자격증명 점검에 쓴다.** 지금은 "저장 전에 KIS에 실제 토큰을 발급해 값이 맞는지 확인"하는 용도인데, env 주입 방식에서는 오타 난 키가 조용히 들어와 **장 시작 후 첫 주문에서야 발각**된다. `app/web/state.py`의 `lifespan`에서 기동 시 1회 호출하고 실패하면 크게 경고 로그를 남기도록 바꾼다 (부팅 자체를 막지는 말 것 — KIS 점검 서버가 죽었다고 대시보드까지 못 뜨면 안 된다).

**호출부 수정 (5곳):**

```python
# app/brokers/kis/broker.py:131
-    creds = await credentials_for(user_id, mode)
+    creds = settings.kis_credentials(mode)      # async 아님 — 아래 주의 참조
```

> ⚠️ `broker_for_user`는 `async def`로 남겨둘 것. 동기로 바꾸면 `app/brokers/__init__.py:102`의 `await _BROKER_FACTORIES[name](...)`부터 `get_broker()` 호출부 전부(수십 곳)가 연쇄로 바뀐다. **함수 본체만 바꾸고 시그니처는 유지한다.**

```python
# app/account_mode.py:89-91
-        from app.credentials.store import configured_modes
-        configured = await configured_modes(user_id)
+        configured = settings.configured_kis_modes()
```

```python
# app/web/state.py:167
-            feed_credentials = await credentials_for(serviced[0], current_account_mode())
+            feed_credentials = settings.kis_credentials(current_account_mode())
```

`app/auth/dependencies.py:156`, `app/web/routers/credentials.py`의 호출부는 Phase 3에서 파일째 사라지므로 별도 수정 불필요.

**남길 라우트 / 지울 라우트 (`app/web/routers/credentials.py`):**

| 라우트 | 처리 | 이유 |
|---|---|---|
| `GET /api/credentials` | **축소 후 유지** | 프론트가 "실전/모의 중 뭐가 설정됐나"를 알아야 모드 토글을 그린다. `configured`/`active_mode`/`preferred`만 반환하고 `encryption_configured`는 제거 |
| `PUT /api/credentials/{mode}` | **삭제** | env로만 들어온다 |
| `DELETE /api/credentials/{mode}` | **삭제** | 위와 동일 |
| `PUT /api/account-mode` | **유지 + 감사로그 추가** | paper/real 둘 다 쓰기로 했으므로 필수. 저장처는 Phase 3에서 조정, 전환 이력 기록은 Phase 3.4 |

### Phase 3 — 로그인/세션 제거 (`app/auth/`)

| 대상 | 처리 |
|---|---|
| `app/auth/store.py` | **대부분 삭제.** `preferred_account_mode` 읽기/쓰기 2개 함수만 `app/account_mode.py`로 이동 (아래 참조) |
| `app/auth/passwords.py` | **삭제** |
| `app/auth/dependencies.py` | **재작성** (아래) |
| `app/web/routers/auth.py` | **삭제** |
| `scripts/_cli_user.py` | **삭제.** 이걸 쓰는 스크립트 3개는 유지하되 `--user-email` 인자만 제거 (아래 3.5) |
| `tests/auth/*` | `test_passwords.py`/`test_dependencies.py`/`test_onboarding_gate.py`/`test_route_protection.py` 삭제, `test_account_mode.py`/`test_background_loop_identity.py`/`test_user_context_scoping.py`는 단일 사용자 기준으로 재작성 |

**3.1 `preferred_account_mode`는 살린다.** `users` 테이블의 이 컬럼은 "지금 실전인가 모의인가"를 재시작 후에도 기억하는 유일한 장치다. `app/auth/store.py`의 `set_preferred_account_mode()`와 `resolve_session()`의 SELECT 중 이 컬럼 읽기만 떼어내 `app/account_mode.py`에 옮긴다:

```python
# app/account_mode.py 에 추가
async def stored_preference() -> str | None:
    """이 배포가 마지막으로 선택한 계좌 모드. users 테이블의 단일 행에서 읽는다.
    로그인이 사라져도 이 컬럼은 남는다 — '재시작하면 실전으로 되돌아가는' 사고를
    막는 것이 목적이지, 사용자 구분이 목적이 아니었기 때문."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT preferred_account_mode FROM users WHERE user_id = $1", settings.user_id
        )
    return row["preferred_account_mode"] if row else None


async def set_stored_preference(account_mode: str) -> None:
    if not is_valid_mode(account_mode):
        raise ValueError(f"unknown account mode {account_mode!r}")
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE users SET preferred_account_mode = $1 WHERE user_id = $2",
            account_mode, settings.user_id,
        )
```

`resolve_preference()`(자격증명 없는 모드는 선호로 지정돼 있어도 활성화 안 함)는 **그대로 둔다.** 모의계좌 키를 Jenkins에서 뺐는데 선호가 `paper`로 남아 있는 상황이 실제로 생길 수 있고, 그때 이 함수가 유일한 방어다.

**3.2 `dependencies.py` 재작성.** `bind_user_identity` 미들웨어를 `bind_account_mode`로 축소한다. `user_id` 바인딩은 불필요하다 — `get_current_user_id()`가 ContextVar 미설정 시 `settings.user_id`로 폴백하므로 아무것도 안 하는 게 정답이다.

```python
# app/auth/ → app/web/account_mode_middleware.py 로 이동 권장 (auth 패키지 자체를 없애기 위해)
_cached_mode: str | None = None   # 모드는 프로세스당 하나뿐 — 요청마다 DB를 치지 않는다


async def bind_account_mode(request: Request, call_next):
    """요청 동안 활성 계좌 모드를 ContextVar에 바인딩.

    user_id 바인딩이 사라졌어도 이 미들웨어는 남아야 한다: account_mode는 영속 키의
    스코프(ledger 행, allowlist, 킬스위치 상태의 paper:/real: 접두사)이고, 이걸
    바인딩하지 않으면 모드를 토글해도 데이터가 프로세스 기본 모드 아래 쌓인다.
    모드는 데이터의 소속을 결정하므로 요청 진입점에서 반드시 정해져야 함."""
    global _cached_mode
    if _cached_mode is None:
        _cached_mode = await resolve_account_mode(settings.user_id, await stored_preference())
    token = set_current_account_mode(_cached_mode)
    try:
        return await call_next(request)
    finally:
        reset_current_account_mode(token)
```

> `PUT /api/account-mode` 핸들러에서 `set_stored_preference()` 호출 직후 `_cached_mode = None`으로 무효화 + `invalidate_broker_cache()` + `stop_market_feed()`를 반드시 호출할 것. 지금은 세션 재해석이 그 역할을 대신하고 있었다.

**3.3 `require_user` / `require_credentials` 정리.**

- `require_user` → **완전 삭제.** `app/web/app.py:120`의 `_authenticated = [Depends(require_user)]`도 제거하고, 각 라우터 등록에서 그 dependency 리스트를 뺀다.
- `require_credentials` → **`require_kis`로 축소해서 유지.** 이 함수는 인증만 하는 게 아니라 **두 가지 일**을 더 한다. 지우면 조용히 깨진다:
  1. 자격증명 없는 모드에서 412를 내려 "설정 안 됨"과 "보유종목 없음"을 구분 → 단일 계좌에서도 유효 (모의 키를 안 넣고 모의 모드로 전환한 경우)
  2. **`ensure_market_feed()` 최초 호출 지점** (`app/auth/dependencies.py:179`) — 실시간 시세/서킷브레이커 피드가 여기서 시작된다. 없어지면 `kis_cb_fail_open_when_unknown=False`(현 설정) 때문에 **모든 국내 주문이 서킷브레이커 상태 불명으로 차단**된다.

```python
async def require_kis() -> None:
    """KIS 계좌가 실제로 필요한 라우트용. 현재 모드에 자격증명이 없으면 412.

    ensure_market_feed() 호출은 그대로 유지 — 실시간 피드가 시작되지 않으면
    kis_market_status가 서킷브레이커 상태를 '불명'으로 보고하고, 그 상태는
    fail-closed라 모든 국내 주문이 막힌다. 인증과 무관하게 살아 있어야 하는 로직.
    이 호출을 지우면 주문이 전부 차단됨 — 인증 코드로 오인해서 제거하지 말 것."""
    mode = current_account_mode()
    if mode not in settings.configured_kis_modes():
        raise HTTPException(
            status.HTTP_412_PRECONDITION_FAILED,
            f"이 배포에 {mode} KIS 자격증명이 설정되지 않았습니다 — 서버 환경변수를 확인하세요",
        )
    from app.web.state import ensure_market_feed
    await ensure_market_feed()
```

`app/web/routers/orders.py:34`, `portfolio.py:32`의 `_needs_kis = [Depends(require_credentials)]`를 `[Depends(require_kis)]`로 바꾼다. **이 두 줄은 반드시 남긴다.**

**3.4 모드 전환 감사로그** (2026-08-07 결정, §7-1)

화면 토글을 유지하기로 했으므로, 인증 없는 대시보드에 접근한 누구나 실전↔모의를 바꿀 수 있다. 이건 `docs/safety-and-risk.md` §5.1이 원래 금지했던 형태이고, 그 규칙을 완화하는 대가로 **전환 이력을 남긴다.**

`identity_claims`와 같은 성격의 테이블을 하나 더 두는 것이 가장 단순하다 — `app/user_context.py`의 `record_session_start()`가 이미 같은 패턴(DDL + 베스트에포트 INSERT + 실패 시 로그만)을 갖고 있으므로 그대로 따라간다.

```sql
CREATE TABLE IF NOT EXISTS account_mode_switches (
    id           SERIAL PRIMARY KEY,
    user_id      TEXT NOT NULL REFERENCES users(user_id),
    from_mode    TEXT,                    -- NULL = 최초 선택
    to_mode      TEXT NOT NULL,
    hostname     TEXT NOT NULL,           -- 요청을 받은 프로세스
    client_host  TEXT,                    -- request.client.host (SSH 터널이면 127.0.0.1)
    switched_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

**설계 시 지켜야 할 것 3가지:**

1. **베스트에포트여야 한다.** 기록 실패가 전환 자체를 막으면 안 된다 — 로그 남기고 계속 진행 (`record_session_start()`와 동일한 규약).
2. **`set_stored_preference()` 안이 아니라 라우터에서 기록한다.** 전환 *요청*의 맥락(client_host)은 스토어 계층이 알 수 없고, 알아야 할 이유도 없다.
3. **`client_host`로 사람을 특정할 수 없다는 걸 문서에 적어둔다.** SSH 터널을 거치면 전부 `127.0.0.1`로 보인다. 실제 추적은 서버의 SSH 접속 로그와 시각을 대조해야 하며, 이 테이블은 그 대조를 위한 **앵커**이지 그 자체로 범인을 지목하지 못한다. `identity_claims`가 사칭을 "막지 못하고 사후 추적만 가능하게" 하는 것과 정확히 같은 한계다.

**3.5 CLI 스크립트 3개** (2026-08-07 결정, §7-4)

`scripts/backfill_minute_candles.py`, `check_daily_minute_chart_tr.py`, `scan_minute_data_gaps.py`는 유지하고 `--user-email` 인자만 뺀다. 각 스크립트에서:

- `add_user_argument(parser)` 호출 제거
- `broker_for_email(args.user_email)` → `get_broker()` (인자 없이 호출하면 `settings.user_id` + 현재 모드로 해결된다 — `app/brokers/__init__.py`의 seam이 그대로 살아 있으므로 추가 배선 불필요)
- `_cli_user.py` 삭제

### Phase 4 — 단일 사용자 고정 (`app/user_context.py`) ★ Phase 2와 같은 커밋

```python
async def serviced_scopes() -> list[ServicedScope]:
    """배경 루프가 돌 스코프. 단일 계좌 배포에서는 항상 정확히 하나다.

    이 함수는 원래 user_broker_credentials JOIN users 쿼리였다(멀티유저 Phase 6).
    자격증명이 env로 옮겨가면서 그 쿼리는 성립하지 않고, 그대로 두면 빈 목록을
    반환해 poller/stop_loss/auto_trade/실시간피드가 전부 조용히 멈춘다 —
    에러 없이. 그래서 반환 형태(list[ServicedScope])는 유지한 채 내용만 바꾼다.
    호출부(state.py의 루프 3곳, lifespan)는 손대지 않아도 된다.
    반환 타입을 바꾸지 말 것 — 호출부가 이 모양에 의존함.

    자격증명이 없으면 빈 목록을 반환하는 성질은 유지한다. 그래야 KIS 키를 아직
    안 넣은 배포에서 루프가 실패 로그만 찍으며 도는 대신 조용히 쉰다.
    """
    from app.account_mode import resolve_account_mode, stored_preference

    configured = settings.configured_kis_modes()
    if not configured:
        logger.warning("KIS 자격증명이 설정되지 않아 배경 루프가 이번 주기를 건너뜁니다")
        return []
    mode = await resolve_account_mode(settings.user_id, await stored_preference())
    return [ServicedScope(settings.user_id, mode)]
```

**나머지는 그대로 둔다:**
- `get_current_user_id()` / `user_scope()` — 배경 루프가 `user_scope()` 안에서 도는 구조는 유지. 값이 하나여도 동작 동일하고, 지우면 `state.py` 루프 3곳을 건드려야 한다.
- `register_user()` — `lifespan`에서 1회 호출해 `users` 행을 보장. **오히려 이제 필수다** (FK 대상이므로). `lifespan` 맨 앞에 `await register_user(settings.user_id, settings.user_id)` 추가.
- `record_session_start()` / `identity_claims` — **유지 권장.** 서버에서 여러 사람이 같은 OS 계정으로 접속하는 구조(메모리: 사용자 사칭 위험)는 그대로이므로, 사후 추적용 감사 로그는 오히려 더 중요해졌다.

### Phase 5 — 프론트 정리

**`app/web/static/js/onboarding.js`:**
- `renderAuth()` 전체 삭제 (로그인/회원가입 폼)
- `renderCredentials()` 전체 삭제 (KIS 키 입력 폼) — 단 **"실전 주문 한도(KRW/USD) 입력" 섹션의 문제의식은 살릴 것.** 실전 한도 미설정 시 모든 주문이 차단되는데 화면엔 아무 설명이 없다. 이제 Jenkins env로 들어오므로, 대신 `GET /api/credentials`가 `real_limits_configured: false`를 반환하면 대시보드 상단에 경고 배너를 띄우도록 옮긴다.
- `ensureReady()`는 **껍데기만 남긴다**: `GET /api/credentials`를 호출해 `ready`가 false면 "서버에 KIS 자격증명이 설정되지 않았습니다 — 관리자에게 문의하세요" 안내만 표시. 로그인 루프(`for(;;)`)와 401 처리는 삭제.
- `main.js`의 `await ensureReady()` 호출은 유지 (첫 렌더에서 412 폭탄이 터지는 걸 막는 역할은 그대로 유효).
- `base.css`의 `.onboard-*` 클래스 중 로그인 폼 전용은 정리, 안내 카드용은 유지.

**`index.html`** — 로그인 관련 마크업 없음(전부 JS 동적 생성). 수정 불필요.

### Phase 6 — DB 정리

**드롭할 테이블 (Phase 1~5 배포가 안정된 뒤, 별도 커밋):**
```sql
DROP TABLE IF EXISTS sessions;
DROP TABLE IF EXISTS user_broker_credentials;
```
`user_broker_credentials`는 `CREDENTIAL_ENCRYPTION_KEY`가 사라지면 어차피 복호화 불가능한 죽은 데이터가 된다. **드롭 전에 반드시 백업**하고, Jenkins 주입 방식으로 실제 주문이 정상 체결되는 걸 확인한 뒤에 지운다.

**절대 건드리지 않을 것:**
- `users` 테이블 (모든 FK의 대상, `preferred_account_mode` 컬럼 사용 중)
- `trades` / `memory_entries` / `alerts` / `pending_orders`의 `user_id` 컬럼과 FK
- `identity_claims`
- `users.email` / `users.password_hash` 컬럼은 nullable이므로 남겨둬도 무해하다. 굳이 `DROP COLUMN`할 필요 없음.

### Phase 7 — 문서/테스트

- `docs/multi-user-credentials-plan.md` 최상단에 **"2026-08-07 단일계좌 전환으로 되돌림 — `single-user-rollback-guide.md` 참조"** 배너 추가. 삭제하지 말 것 (왜 그런 코드가 있었는지의 유일한 기록).
- `docs/safety-and-risk.md` §5.1 (계좌모드 HTTP 노출 금지 규칙) — `app/account_mode.py` 주석이 지적한 대로 갱신 필요. 지금은 "무인증 + HTTP로 모드 전환 가능" 상태로 되돌아가므로, §5.1의 원래 우려가 **다시 유효해진다.** `PUT /api/account-mode`를 유지하려면 그 사실을 문서화하거나, 실전↔모의 전환을 env(`KIS_ACCOUNT_MODE`) 전용으로 되돌리는 것도 고려할 것 (→ §7 남은 질문).
- `.env.example` 갱신: `CREDENTIAL_ENCRYPTION_KEY`/`AUTH_REQUIRED`/`SESSION_*` 제거, `KIS_APP_KEY_REAL` 등 추가.
- `app/web/app.py` 모듈 docstring의 "multi-user migration" 문단 갱신.
- `PROJECT_OVERVIEW.md` / `README.md` 확인.

---

## 5. 절대 깨뜨리면 안 되는 것 (작업 중 체크)

이번 변경에서 **조용히 깨지는** 것들이다. 에러가 안 나므로 테스트로만 잡힌다.

| # | 불변식 | 깨졌을 때의 증상 |
|---|---|---|
| 1 | `serviced_scopes()`가 비어 있지 않다 | 자동매매·손절·폴러·실시간피드 전부 무동작. **에러 로그 없음** |
| 2 | `ensure_market_feed()`가 어디선가 호출된다 | 서킷브레이커 상태 불명 → 국내 주문 전량 BLOCKED |
| 3 | `account_mode` ContextVar가 요청마다 바인딩된다 | ledger/allowlist/킬스위치가 잘못된 모드 접두사로 저장 |
| 4 | `KISCredentials.__repr__`가 시크릿을 가린다 | 예외 스택트레이스/로그에 운용 계좌 앱시크릿 유출 |
| 5 | paper 키로 real 주문이 나가지 않는다 | **계좌의 돈으로 의도치 않은 주문.** `kis_credentials()`가 모드 간 폴백을 하지 않아야 함 |
| 6 | `dashboard_host`가 루프백이다 | 내부망 누구나 운용 계좌로 주문 가능 |
| 7 | 실전 모드 부팅 시 `kis_max_order_*_real` 4종이 설정돼 있다 | 부팅 거부(의도된 동작) 또는 한도 없는 주문 |
| 8 | `users` 행이 존재한다 (`register_user`) | 모든 FK 삽입 실패 → 거래 기록 자체가 안 남음 |

---

## 6. 검증 체크리스트

각 Phase 후:

```bash
uv run pytest tests/ -x -q
```

> 참고: `pytest tests/`(기본 순서) 전체 실행 시 **기존에 45개가 실패**하는 테스트 격리 버그가 있다(2026-08-08 기록, 이 작업과 무관). 이번 변경의 영향을 보려면 폴더 단위로 나눠 돌릴 것:
> ```bash
> uv run pytest tests/execution tests/live tests/web tests/brokers -q
> ```

전체 완료 후 수동 검증:

```bash
# 1) 부팅 — 자격증명 점검 로그가 real/paper 각각 뜨는지
./pipeline.sh dev && docker logs -f invest-project-dev

# 2) 루프백 가드 — 이건 반드시 실패해야 정상
DASHBOARD_HOST=0.0.0.0 uv run python -c "from app.config import settings"

# 3) 시크릿이 로그/응답에 안 나오는지
docker logs invest-project-dev 2>&1 | grep -iE "app_secret|appsecret|PS[A-Za-z0-9]{20}"

# 4) 인증 라우트가 정말 사라졌는지 (전부 404여야 함)
curl -s -o /dev/null -w '%{http_code}\n' localhost:8094/api/auth/me
curl -s -o /dev/null -w '%{http_code}\n' -X POST localhost:8094/api/auth/login

# 5) 배경 루프가 실제로 도는지 — 이게 §5 #1의 유일한 검증 수단
docker logs invest-project-dev 2>&1 | grep -i "serviced scopes"

# 6) 실시간 피드 기동 확인 (§5 #2)
docker logs invest-project-dev 2>&1 | grep -i "real-time feed started"
```

브라우저:
- 대시보드가 로그인 화면 없이 바로 뜬다
- 보유종목/잔고가 본인 실전계좌 값과 일치한다
- 실전↔모의 토글이 동작하고, **토글 후 보유종목이 실제로 바뀐다** (브로커 캐시 무효화 확인 — §Phase 3.2의 주의사항)
- 주문 티켓에서 소액 매수 1건이 정상 처리된다 (**반드시 모의계좌에서 먼저**)

---

## 7. 확정된 결정과 남은 질문

### 7.1 확정 (2026-08-07)

| # | 질문 | 결정 | 반영 위치 |
|---|---|---|---|
| 1 | 실전↔모의 전환을 화면에 남길 것인가 | **화면 토글 유지 + 전환 이력을 감사로그로 기록** | Phase 2 표, Phase 3.4 |
| 2 | `USER_ID` 값 | **`default` 그대로 유지** — DB 확인 결과 전부 `default` 하나뿐, 마이그레이션 불필요 | Phase 4, §7.2 |
| 3 | 모의계좌 키도 Jenkins에 넣을 것인가 | **넣는다** — 관리 창구를 하나로 | §3.1 |
| 4 | `--user-email` 쓰는 CLI 스크립트 3개 | **유지, 인자만 제거** | Phase 3.5 |
| 5 | Jenkins agent = 배포 서버인가 | **같은 호스트** — `pipeline.sh`의 `docker run --network host`가 그대로 동작, 추가 설계 불필요 | §3.2 |

**모든 항목이 확정되었으므로, 이 가이드의 Phase 1~7은 그대로 실행 가능한 상태다.** §7.2는 확인 결과만 기록으로 남긴다(아래).

**1번에 딸린 후속 작업:** `docs/safety-and-risk.md` §5.1을 갱신해야 한다. 원문은 "계좌모드 선택을 HTTP API로 노출하지 말 것"인데, 그 규칙이 쓰인 전제(무인증·단일 사용자·localhost)가 이번 전환으로 **다시 성립한다.** 규칙을 완화하는 게 아니라 **"완화하되 감사로그로 상쇄한다"** 는 판단이었음을 그쪽에 명시해야, 나중에 이 문서를 못 본 사람이 §5.1만 읽고 되돌리지 않는다.

**5번에 딸린 확인:** Jenkins agent가 배포 서버 위에서 돌므로, 그 서버의 Docker 소켓 접근 권한을 Jenkins 실행 계정이 갖고 있어야 한다. 또한 §3.2의 `post { always { rm -f .env.dev } }`가 **더 중요해진다** — 워크스페이스가 곧 운영 서버의 디스크이기 때문이다.

### 7.2 `USER_ID` 확인 결과 (2026-08-07)

`trades`/`users`/`memory_entries` 전부 `user_id='default'` 하나뿐 — 멀티유저 전환 이후에도 실제로 다른 사용자가 온보딩한 적이 없었다는 뜻이다. 따라서:

- `.env.local`/`.env.dev`/`.env.base`의 `USER_ID`는 **`default`로 고정.** 값을 바꾸지 않는다.
- Phase 4의 `serviced_scopes()`와 `register_user()`는 항상 `settings.user_id`(= `"default"`)로 동작하므로 별도 수정 없이 그대로 맞는다.
- §7.2 표의 "UUID 통합" 시나리오는 **해당 없음** — 데이터 마이그레이션 단계가 이번 작업 범위에서 완전히 빠진다. 그만큼 Phase 6(DB 정리) 이전까지는 git revert만으로 전량 복구 가능하다는 §9의 전제도 그대로 유지된다.

---

## 8. 전환 후 개발 환경 (로컬)

배포는 Jenkins가 시크릿을 주입하지만, **로컬 개발은 지금과 동일하게 `.env.local` + `uv run`** 이다. `app/config.py`의 `env_file=".env.local"`은 이번 변경에서 건드리지 않는다.

### 8.1 `.env.local` 변화

| | 전환 전 | 전환 후 |
|---|---|---|
| 제거 | — | `CREDENTIAL_ENCRYPTION_KEY`, `AUTH_REQUIRED`, `SESSION_COOKIE_NAME`, `SESSION_TTL_HOURS` |
| 추가 | — | `KIS_APP_KEY_PAPER`, `KIS_APP_SECRET_PAPER`, `KIS_ACCOUNT_8_PAPER`, (`KIS_HTS_ID_PAPER`) |
| 유지 | `DATABASE_URL`, `LLM_BASE_URL`, 리스크 한도, 루프 주기 등 | 동일 |

**첫 실행 절차가 사라진다.** 지금은 새 체크아웃에서 서버를 띄우면 회원가입 → 로그인 → KIS 키 입력 → 주문한도 입력을 거쳐야 대시보드가 보이지만, 전환 후에는 `.env.local`만 채우면 바로 뜬다. 반대로 **설정이 잘못됐을 때의 안내가 화면 폼이 아니라 부팅 로그로 옮겨간다** — §8.7의 로그 확인이 그만큼 중요해진다.

### 8.2 로컬에는 모의계좌 키만 둔다

이번 구조 변경의 핵심 이득 중 하나가 "실전 키를 아는 곳이 Jenkins 하나"라는 점이다. 개발용 `.env.local`에 실전 키를 복사하는 순간 그 이득이 사라진다.

- `.env.local`에는 `KIS_*_PAPER`만, `KIS_ACCOUNT_MODE=paper`
- `KIS_MAX_ORDER_KRW_REAL` 등 실전 한도 4종은 **로컬에 불필요** — `_require_real_money_limits_on_real_account` 검증기는 `kis_account_mode == "real"`일 때만 걸린다
- 실전 경로 자체를 디버깅해야 하는 예외적 경우에만 값을 넣고, 작업이 끝나면 즉시 지운다

### 8.3 실행

```bash
uv run python -m app.daemon        # → http://127.0.0.1:8094, 로그인 없이 바로 대시보드
```

일상 개발은 `uv run`, 배포 검증만 `./pipeline.sh dev`로 나눈다. 로컬에서 `pipeline.sh`를 돌리면 Jenkins가 주입하는 시크릿이 없으므로 `.env.dev`를 손으로 채워야 한다.

### 8.4 공유 DB에서의 데이터 격리 — `USER_ID`

**`user_id` 스코프를 남기기로 한 결정(§0)의 실질적 이득이 여기서 나온다.**

여러 명이 같은 dev Postgres를 쓸 때, 각자 `.env.local`의 `USER_ID`를 다르게 두면(예: `dev-alice`) 거래 원장·판단 메모리·알림·승인 대기열·매매허용목록이 전부 그 값으로 스코프되어 서로를 오염시키지 않는다. 배포용 값(예: `owner`)과도 자동으로 분리된다.

- `users` 테이블에 해당 행이 있어야 FK가 성립한다 → Phase 4에서 `lifespan`에 추가하는 `register_user()`가 이걸 보장한다. **이 호출이 빠지면 거래 기록 삽입이 전부 실패한다**(§5 불변식 #8).
- 반대로 운영 데이터를 보면서 개발하려면 `USER_ID`를 배포값과 같게 둔다. 이때는 로컬에서 만든 판단·주문 기록이 운영 이력에 섞이므로 §8.5의 스위치를 반드시 잠근다.

### 8.5 로컬에서 잠가둘 스위치

로그인이 사라져도 주문 게이트는 전혀 완화되지 않는다. 다만 개발 중 실수로 주문이 나가는 경로를 막는 건 여전히 사람 몫이다.

| 설정 | 로컬 권장 | 의미 |
|---|---|---|
| `KIS_ACCOUNT_MODE` | `paper` | 모의계좌 |
| `KIS_EXECUTION_ENABLED` | `false` | 주문 전송 마스터 스위치 |
| `KIS_EXECUTION_DRY_RUN` | `true` | 켜져 있어도 실제 전송 안 함 |
| `EXECUTION_MODE` | `manual` | AI가 주문 시도 자체를 안 함 |
| `AUTO_TRADE_ENABLED` | `false` | LLM 토큰을 매 주기 태우는 루프 정지 |

4개는 서로 독립이라 하나만 풀어도 나머지가 계속 막아준다. 주문 파이프라인을 실제로 태워봐야 할 때만 위에서부터 한 칸씩 푼다.

### 8.6 테스트

```bash
uv run pytest tests/execution tests/live tests/web tests/brokers -q
```

- `pytest tests/` 전체를 기본 순서로 돌리면 **기존 격리 버그로 45개가 실패**한다(2026-08-08 기록, 이번 변경과 무관). 폴더 단위로 나눠 도는 편이 신호가 깨끗하다.
- `tests/auth/`·`tests/credentials/`는 삭제 대상과 재작성 대상이 섞여 있다 — Phase 3·7의 표를 기준으로 판단한다.
- 과거에 **테스트가 실제 운영 DB를 오염시킨 사건**이 있었다. `DATABASE_URL`이 운영을 가리키는 상태로 테스트를 돌리지 않는지 매번 확인한다.

### 8.7 프론트 개발

`onboarding.js`가 대폭 줄지만 **`main.js`의 `await ensureReady()` 호출은 남긴다** — 첫 렌더에서 폴링 요청 15개가 한꺼번에 412로 터지는 걸 막는 역할은 그대로 유효하다. 이 호출까지 지우면 "KIS 키 미설정" 상태가 "대시보드가 고장난 것"처럼 보인다.

정적 파일에 `Cache-Control: no-cache` 미들웨어가 이미 걸려 있지만, 수정이 화면에 안 보이면 코드 버그로 단정하기 전에 서버 응답부터 확인한다.

```bash
curl -s http://127.0.0.1:8094/js/onboarding.js | head -20
```

### 8.8 커밋/배포 전 최종 확인

**이 명령은 실패해야 정상이다.** 통과하면 배포 후 내부망 누구나 운용 계좌로 주문할 수 있는 상태가 된다.

```bash
DASHBOARD_HOST=0.0.0.0 uv run python -c "from app.config import settings"
```

부팅 후에는 아래 세 줄이 로그에 있는지 본다. `serviced scopes`가 없으면 자동매매가 **에러 없이** 멈춰 있는 상태다(§5 불변식 #1).

```bash
uv run python -m app.daemon 2>&1 | grep -iE "serviced scopes|real-time feed started|credential"
```

### 8.9 하지 말 것

- `.env.local` / `.env.dev` 커밋 (둘 다 이미 `.gitignore` 대상 — 확인만)
- 실전 키를 개발 PC·채팅·이슈에 붙여넣기
- `.env.base`(커밋 대상 템플릿)에 시크릿 넣기
- 디버깅한다고 `KISCredentials`를 dict/tuple로 풀어 로그에 찍기 — 시크릿을 가리는 `__repr__`이 유일한 방어다(§5 불변식 #4)
- 부팅 로그의 `serviced scopes` 확인 없이 "동작한다"고 판단하기

---

## 9. 롤백

Phase 6(테이블 드롭) 전까지는 **git revert 한 번으로 완전 복구된다.** `sessions`/`user_broker_credentials` 테이블과 `CREDENTIAL_ENCRYPTION_KEY`만 그대로 보존하면 된다.

Phase 6 이후로는 되돌리려면 각 사용자가 KIS 자격증명을 재입력해야 한다. **Phase 6은 최소 2주 안정 운영 후에 실행할 것.**
