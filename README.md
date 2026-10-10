# invest-project — AI 멀티에이전트 자동매매 시스템

> English: [README_en.md](README_en.md)

LLM 에이전트 여러 개가 증권사 트레이딩 데스크처럼 **분석 → 찬반토론 → 종합 → 트레이더 제안 → 리스크토론 → 최종결정**
순서로 종목별 투자등급을 낸다. 그 등급을 **코드**가 사이징·손절·리스크 게이트를 거쳐 한국투자증권(KIS) Open API
계좌에 주문으로 반영하는 개인용 자동매매 시스템이다.

투자 자체와, LLM 에이전트를 돈이 오가는 시스템에 안전하게 붙이는 방법(검증 가드, 결정적 리스크 게이트, 사후 측정)을
직접 이해하기 위해 만든 프로젝트다. 코드·테스트·문서 대부분은 **Claude Code**와 함께 작성했다.

- **단일계좌 도구**다. 계좌 하나를 서비스한다. 로그인이 없고, 접근 통제는 네트워크 위치(루프백 + SSH 터널)로 한다.
- **롱온리 · 무레버리지 · 무공매도**. 스윙(며칠 단위) 판단용이다.
- 국내(KRX)와 해외(미국)를 지원한다. 두 시장은 후보 발굴·현금·계좌·로그 등 **모든 층에서 분리**돼 있다.
- LLM은 자체 호스팅 vLLM(Qwen 계열)을 기본으로 쓰고(Anthropic 등 다른 provider도 지원), DB는 Postgres + pgvector를 쓴다.
- **기본값은 전부 모의투자(paper)·수동 모드다.** 실계좌·자동주문은 사람이 여러 스위치를 명시적으로 열어야 한다(§3).

> ⚠️ 이 저장소는 학습·연구용이다. 투자 조언이 아니며, 실계좌 사용에 따른 손익 책임은 사용자에게 있다.

---

## 목차

1. [판단 한 건이 만들어지는 과정](#1-판단-한-건이-만들어지는-과정)
2. [자동매매 사이클과 상시 루프](#2-자동매매-사이클과-상시-루프)
3. [안전장치](#3-안전장치)
4. [디렉터리 구조](#4-디렉터리-구조)
5. [설치와 실행](#5-설치와-실행)
6. [핵심 설정](#6-핵심-설정)
7. [배포](#7-배포)
8. [측정·감사 도구](#8-측정감사-도구)
9. [개발 규칙과 함정](#9-개발-규칙과-함정)
10. [관련 문서](#10-관련-문서)

---

## 1. 판단 한 건이 만들어지는 과정

진입점은 `app/orchestrator/pipeline.py`의 `run_pipeline()`이다.

```
 ① Analysts 4명 (동시 실행)       Market(가격·기술지표·52주 요약·외국인/기관 수급) / Fundamentals / News+Macro / Sentiment
        │                          ※ Sentiment는 국내(.KS) 종목에서 영구 스킵(고정 중립 리포트)
        ▼
 ② 출력 검증 가드                  app/agents/*_guard.py, identity_gate.py — 단위배율(억원↔billion)·비율라벨·
        │                          기술지표 날조·거래정지 서술·세션 종가 오인·가격제한폭·종목 동일성 등
        ▼
 ③ Bull ↔ Bear 토론                최대 2라운드 (MAX_DEBATE_ROUNDS)
        ▼
 ④ Research Manager                등급 + 목표비중 제안
        ▼
 ⑤ Trader                          방향 · 진입 · 손절
        ▼
 ⑥ Risk 토론                       Aggressive / Conservative / Neutral, 최대 2라운드 (MAX_RISK_ROUNDS)
        ▼
 ⑦ Portfolio Manager               최종 등급(Buy/Overweight/Hold/Underweight/Sell) + 손절가
        │                          과거 유사 판단 기억(pgvector)은 PM에게만 준다
        ▼
 ⑧ ledger.rebalance_to_target()    등급 → 목표배분 (Buy 100% · Overweight 75% · Hold 유지 · Underweight 25% · Sell 0%)
        │                          ATR 다운스케일 · 1% 리스크 룰 · 포트폴리오 캡(집중/상관/낙폭) · 손절 최소거리 보정
        ▼
 ⑨ execution_mode 분기             MANUAL: 원장만 기록 / SEMI_AUTO: 승인 대기열 / FULL_AUTO: 바로 주문
        ▼
 ⑩ kis_bridge.mirror_trade_to_kis()   ★ 실주문 단일 관문 (§3)
```

- 모든 판단은 `memory_entries` 테이블에 **전체 원문(`full_transcript`)과 함께** 저장된다. 나중에 결과
  (벤치마크 대비 알파)가 자동으로 확정된다(`app/memory/auto_resolve.py`). 원문이 남아 있어서 가드 로직을 고쳤을 때
  LLM 호출 없이 과거 판단에 소급 재채점할 수 있다.
- 판단 1건에 LLM 호출은 약 12회, 토큰은 10만~20만 개다. 시간은 LLM 부하에 따라 수 분~수십 분 걸린다.
  LLM 백엔드가 하나라서 종목은 **순차 처리**한다.
- 분석가 1명이 실패해도 나머지 리포트로 진행한다(실패 자리는 플레이스홀더로 채운다).

---

## 2. 자동매매 사이클과 상시 루프

### 자동매매 사이클 — `app/live/auto_trade_loop.py`

`AUTO_TRADE_ENABLED=true`일 때만 돈다. 주기는 `AUTO_TRADE_INTERVAL_MINUTES`이고, 시장별로 장중에만 돈다.
`MANUAL` 모드에서는 사이클 자체를 건너뛴다.

1. **발굴** — KIS 순위 API로 후보를 모은다(`app/live/universe.py`). 국내는 거래대금 상위 · 시가총액 상위 ·
   등락률 하위 · 이격도 하위 4개 축이고, 해외는 거래대금·시가총액 축이다. 레버리지/인버스, 현금성·채권 ETF,
   단기과열·정리매매 지정, 저유동성 종목은 제외한다.
2. **자격 필터** — 재분석 락(`rebalance_interval_days`, 기본 5일), 시장별 보유 상한(기본 15), 계좌 가격상한.
   ⚠️ 자격 필터는 **반드시 랭킹 앞에** 둔다. 랭킹 뒤에 두면 상위 종목이 매번 락에 걸려 후보가 전멸한다.
3. **모멘텀 랭킹**(LLM 없음) — `app/chat/screening.py`. 변동성 조정과 상관 중복제거를 거쳐 상위 N개
   (`AUTO_TRADE_MAX_TICKERS_*`)를 뽑는다. ※ 폴더 이름이 `chat`이지만 실매매 후보선정 엔진이다.
4. **강제 포함** — 손절선 근접 보유종목, 현금이 생겨 재검토가 필요한 종목은 랭킹·예산과 무관하게 넣는다.
5. 후보마다 `run_pipeline()`을 순차 실행하고 주문한다.
6. **스크리닝 스냅샷 기록** — 매 사이클 후보 순위를 저장하고, 나중에 선행수익률로 채점한다(관측 전용).
   라이브 순위 예측력(IC) 측정의 원천이다.

### 상시 루프 — `app/web/state.py`의 `lifespan()`이 기동

| 루프 | 역할 |
|---|---|
| `run_poller_loop` | 시세 폴링(60초). LLM을 쓰지 않는다 |
| `stop_loss_loop` | 규칙 기반 손절·트레일링 스탑. **실행 모드와 무관하게 항상** 실행된다 |
| `fill_monitor_loop` | 주문 접수와 실제 체결을 구분해 체결을 확인한다 |
| `exit_monitor_loop` | 청산 감시 |
| `trade_reconciliation_loop` | 10분 주기로 원장과 KIS 잔고를 대조한다. **탐지·알림만 하고 자동정정은 하지 않는다(의도적)** |
| `RealtimeFeed` | KIS WebSocket — 시장운영정보(서킷브레이커/VI), 체결, 호가. 합산 41건 한도 |
| `auto_trade_loop` | 위 자동매매 사이클 |

---

## 3. 안전장치

### 실주문은 한 곳만 통과한다 — `app/execution/kis_bridge.py`

`KISBroker.submit_order`는 **일부러 가드가 없다.** 모든 안전장치는 `kis_bridge.py`에서만 적용된다.
진입점은 둘뿐이다. `mirror_trade_to_kis`(AI 자동 경로)와 `place_manual_order`(대시보드 수동 주문)이다.
**다른 모듈이 브로커 주문 함수를 직접 호출하면 안 된다.**

주요 게이트: 킬스위치 → 일일 최대손실 서킷브레이커 → 마스터 스위치(`KIS_EXECUTION_ENABLED`) →
레버리지/인버스 차단 → 주문금액 상한 → 승인 임계값 → 시장 서킷브레이커 → VI → 현금충분성 →
포지션비중 캡 → dry-run(`KIS_EXECUTION_DRY_RUN`).

- **fail-closed와 fail-open은 게이트마다 다르다.** 시장 전체 서킷브레이커·킬스위치·일일손실은 상태를 모르면
  **차단**한다. 개별종목 정지·VI·현금 조회는 실패하면 **통과**한다. 이 경우 KIS 자체가 최종적으로 거부하는 안전망이 있다.
- 킬스위치와 일일손실 상태는 Postgres에 영속된다. 재시작해도 풀리지 않는다.
  ⚠️ 킬스위치는 **매도·손절까지 막는다.** 포지션을 보유 중일 때 켜지 마라.
- 리스크 규칙(금액 상한·비중 캡·승인 임계값)은 대시보드 `설정`에서 바꿀 수 있다. 변경은 재시작 없이 다음 주문부터 적용된다(`app/execution/risk_rules.py`).

### 기본값은 전부 안전한 쪽이다

| 설정 | 코드 기본값 |
|---|---|
| `KIS_ACCOUNT_MODE` | `paper`(모의). 실전은 사람이 명시적으로 켠다 |
| `EXECUTION_MODE` | `manual` |
| `KIS_EXECUTION_ENABLED` / `KIS_EXECUTION_DRY_RUN` | `false` / `true` |
| `AUTO_TRADE_ENABLED` | `false` |
| 대시보드 바인딩 | `127.0.0.1`. 루프백이 아니면 **부팅을 거부**한다(끄는 스위치 없음) |

- 실계좌는 `AI_PROFILE=production`이 아니면 **기본적으로 부팅이 거부된다.** 단 `AI_PROFILE_LIVE_ACK=true`를 명시하면 비프로덕션 프로필도 부팅된다(`app/config.py`의 `_block_non_production_profile_on_real_account`). 이 예외는 시스템 기능이며 활성화를 권장하는 것이 아니다.
- 실계좌에서 잔고 조회에 실패하면 가상 시드로 폴백하지 않고 **멈춘다**(fail-closed).
- `POST /api/order`, `PUT /api/account-mode`에는 **앱 인증이 없다.** 127.0.0.1 바인딩 + SSH 터널이 유일한 보호이고,
  이건 단일계좌 도구로서 의도된 설계다. 계좌모드 전환은 `account_mode_switches` 테이블에 기록된다.

---

## 4. 디렉터리 구조

```
app/
├── daemon.py           상시 진입점: 대시보드 + 모든 백그라운드 루프
├── main.py             1회성 단일 종목 판단 CLI
├── config.py           전체 설정(pydantic Settings)과 부팅 검증 규칙
├── agents/             LLM 에이전트(분석가·리서처·RM·Trader·리스크·PM) + 출력 검증 가드(*_guard.py, identity_gate.py)
├── orchestrator/       pipeline.py(run_pipeline), debate.py
├── live/               auto_trade_loop, universe(발굴), poller, fill_monitor, exit_monitor, alerts, 장시간·휴장일
├── chat/               대시보드 채팅 라우터 + screening.py(실매매 후보 모멘텀 랭킹 엔진)
├── execution/          ledger(원장·리밸런스), kis_bridge(실주문 관문), sizing, risk_rules, safety_switch,
│                       portfolio_risk, trailing_stop, pending_orders, reconciliation, costs
├── brokers/            KIS SDK 격리층(kis/domestic, kis/overseas, WebSocket). 다른 레이어는 벤더를 직접 import하지 않는다
├── domain/             순수 규칙(레버리지·상품범위·체결가능성) + 포트(Protocol) 정의
├── dataflows/          데이터 제공자 라우터 — 국내 가격은 KIS 우선·yfinance 폴백, FRED(매크로), Reddit
├── memory/             판단 기억(pgvector 저장·유사검색), 결과 자동확정, 반성문
├── llm/                LLM 클라이언트(provider: vllm / gateway / anthropic), 동시성 제한, 메트릭
├── web/                FastAPI 대시보드(routers/, state.py, static/ — 빌드도구 없는 HTML+JS)
├── audit/              사후 측정 CLI (§8)
├── backtest/           백테스트 하네스, 포트폴리오·병렬 실행, screening_sweep
├── experiments/        페어드 A/B 실험 하네스
└── credentials/        KIS 자격증명 검증(부팅 시 토큰 발급으로 1회 확인)

tests/        app/와 같은 구조의 pytest 스위트(약 5,000개)
scripts/      일회성 운영·데이터 도구(백필, 뷰어빌리티 스윕 등)
docs/         설계 문서 (§10)
pipeline.sh   빌드·배포 스크립트 (§7)
```

**의존 방향**: Strategy → Application → Domain Ports ← Broker Adapter → Broker API.
`domain/`은 brokers·DB·HTTP를 모른다. 브로커 원시 JSON 키는 어댑터 밖으로 나가지 않는다.

---

## 5. 설치와 실행

### 요구사항

- Python **3.12**(고정, `requires-python = "==3.12.*"`) · [uv](https://docs.astral.sh/uv/)
- Postgres 16 + pgvector 인스턴스(판단 기억·원장·설정 저장). **`DATABASE_URL`이 비어 있으면 부팅이 거부된다**
- OpenAI 호환 LLM 엔드포인트(vLLM 등)
- (선택) KIS Open API 앱키. 없으면 KIS 관련 기능만 조용히 빠지고 나머지는 동작한다

### 설치

```bash
uv sync
cp .env.example .env.local      # 값 채우기 — .env.example이 형식의 정본(각 키 설명 포함)
```

- `.env.local`: 로컬 `uv run`용 · `.env.dev`: `pipeline.sh dev`용. **둘 다 gitignore 대상이다.**
- 로컬 개발에는 **모의계좌 키만** 둔다. 실전 키는 개발 PC에 두지 않는다. 배포 시에는 CI(Jenkins)가 주입한다.
- macOS 기준이다(`brew install uv`, Postgres+pgvector는 `docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=... pgvector/pgvector:pg16`). `uv run` 대신 `.venv/bin/python`을 직접 지정해도 된다. 시스템 Python에는 의존성이 없다.

### 실행

```bash
# 대시보드 + 백그라운드 루프 (http://127.0.0.1:8094)
uv run python -m app.daemon

# 단일 종목 1회 판단
uv run python -m app.main --ticker 005930.KS --date 2026-09-18

# 백테스트 (과거 날짜마다 전체 파이프라인 실행 → 느림, --cadence-days로 표본 수 조절)
uv run python -m app.backtest --ticker AAPL --start 2026-05-01 --end 2026-07-01 --cadence-days 7 --benchmark SPY
uv run python -m app.backtest --random --cadence-days 7                 # 무작위 종목·기간
uv run python -m app.backtest.portfolio_harness --start ... --end ... --cadence-days 7
uv run python -m app.backtest.parallel --tickers AAPL,MSFT --start ... --end ... --cadence-days 30

# 판단 결과 확정(라이브) / 손절 점검
uv run python -m app.memory resolve --ticker AAPL
uv run python -m app.execution check-stops --ticker AAPL
```

⚠️ 로컬 데몬을 **2개 이상 동시에 띄우지 마라.** 같은 계좌·원장에 대해 루프가 중복으로 돈다.

### 테스트

저장소 폴더에서 프로젝트 가상환경 Python으로 실행한다. 전체 회귀의 기본 명령은 **외부망을 차단하는 단일 프로세스 도구**다(약 4분, 병렬 `-n` 불가).

```bash
uv run python scripts/run_protected_regression.py
uv run python scripts/run_protected_regression.py -- tests/execution -q
```

기존 전체 병렬 pytest 안내는 위 명령으로 대체했다(2026-10-06). 차단은 해당 Python 프로세스의 DNS·소켓 연결에 적용되고, 자식 프로세스 생성은 기록만 한다. 모든 자식 프로그램의 외부 접근이나 DB 읽기 전용 세션을 보장하는 도구는 아니다.

`tests/conftest.py`의 autouse 픽스처 3개는 serviced scope를 1개로 고정하고 시장별 티커 목록·계좌 잔고 캐시를 테스트마다 비운다. **DB·외부망 접근을 전부 막는 장치가 아니다**(2026-10-02 정정: 예전 문장 "실DB 접근을 막는다"는 과장이었고, 일부 시험이 실제로 운영 DB·Yahoo에 접속한 사례가 있었다). 새 시험은 합성 데이터와 fake만 쓰고, 우회하지 마라.
`asyncio_mode = "auto"`라서 async 테스트에 데코레이터가 필요 없다.

---

## 6. 핵심 설정

전체 목록과 설명은 `.env.example`, 기본값과 검증 규칙은 `app/config.py`에 있다.

| 분류 | 키 | 비고 |
|---|---|---|
| LLM | `LLM_PROVIDER`, `LLM_BASE_URL`, `LLM_MODEL`, `LLM_MAX_CONCURRENT_REQUESTS` | 서버 안정성 이슈로 배포 동시성을 낮게 유지 중이다(`.env.example` 주석 참고) |
| 판단 | `MAX_DEBATE_ROUNDS`, `MAX_RISK_ROUNDS`, `AI_PROFILE` | 실계좌는 기본적으로 `production`만 허용된다(`AI_PROFILE_LIVE_ACK=true`를 명시하면 예외) |
| DB | `DATABASE_URL` | 필수 |
| 계좌 | `KIS_ACCOUNT_MODE`, `KIS_APP_KEY_*`, `KIS_APP_SECRET_*`, `KIS_ACCOUNT_8_*`, `KIS_ACCOUNT_PROD_*`, `KIS_HTS_ID_*` | paper/real 키는 독립이다. 가진 것만 채운다 |
| 실행 | `EXECUTION_MODE`(manual/semi_auto/full_auto), `KIS_EXECUTION_ENABLED`, `KIS_EXECUTION_DRY_RUN` | 세 겹이 모두 열려야 실주문이 나간다 |
| 리스크 | `KIS_DAILY_MAX_LOSS_PCT`, `KIS_MAX_ORDER_*`, `KIS_MAX_POSITION_PCT*`, `KIS_APPROVAL_*`, `KIS_CB_FAIL_OPEN_WHEN_UNKNOWN` | 대시보드에서 런타임 변경 가능한 항목은 DB 값이 우선한다 |
| 자동매매 | `AUTO_TRADE_ENABLED`, `AUTO_TRADE_INTERVAL_MINUTES`, `AUTO_TRADE_MAX_TICKERS_DOMESTIC/OVERSEAS`, `AUTO_TRADE_SCAN_COUNT_*` | 해외를 끄려면 `..._OVERSEAS=0` |

⚠️ 함정 두 가지:
- **활성 계좌모드는 env만으로 정해지지 않는다.** 자격증명 유무와 저장된 선호(`PUT /api/account-mode`)로 결정된다.
- **`execution_mode` 등 일부 설정은 DB(`runtime_settings`)에 저장된 값이 env보다 우선한다.** 스코프(paper/real)별로 따로 저장된다.
- 매매 허용 목록(trade allowlist)은 env가 아니라 **DB**(`trade_allowlist_state`)에 있다. 대시보드에서 관리한다.

---

## 7. 배포

```
로컬 커밋 → GitHub push → Jenkins 빌드(수동) → 서버에서 pipeline.sh dev 실행 → 컨테이너 교체
```

- Jenkins Job이 GitHub 브랜치를 체크아웃하고, Secret file(`invest-env-dev`)을 `.env.dev`로 복사한 뒤 `./pipeline.sh dev`를 실행한다.
  Jenkinsfile은 저장소에 없고 Job 설정에 인라인으로 있다.
- `pipeline.sh`는 이미지를 빌드하고(`GIT_COMMIT`을 빌드 인자로 굽는다 — 판단마다 코드 버전이 기록된다),
  `docker stop && rm && run`으로 컨테이너를 교체한다. 옵션은 `--network host`와 `--restart unless-stopped`이다.
  - `--network host`를 쓰는 이유: 대시보드가 127.0.0.1에만 바인딩되기 때문이다. `-p` 포트 게시로는 컨테이너 루프백에 닿지 않는다.
  - 유지되는 볼륨은 **KIS 토큰 캐시 하나뿐**이다. 그래서 🔴 **배포하면 컨테이너 로그가 전부 사라진다.** 사고 조사 중에는 배포하지 마라.
- 1회성 명령도 같은 이미지로 돌릴 수 있다: `./pipeline.sh dev main --ticker AAPL --date 2026-07-08`,
  `./pipeline.sh dev backtest ...`, `./pipeline.sh dev memory resolve --ticker AAPL`.
- 대시보드 접속: 서버에 SSH 터널을 연다(`ssh -L 8094:127.0.0.1:8094 <서버>`) → `http://127.0.0.1:8094`.

---

## 8. 측정·감사 도구

"잘 되고 있나"는 감이 아니라 이 도구들로 잰다. 공통 옵션은
`--market {domestic,overseas} --scope {live,backtest} --account-mode {paper,real}`이다.

| 명령 | 무엇을 재나 |
|---|---|
| `python -m app.audit report` | 등급별 승률·알파(케이스당 최신 실행만 집계) |
| `python -m app.audit history --ticker X --date Y` | 한 케이스의 모든 재실행 이력 |
| `python -m app.audit experiments` | 백테스트 실행 비교(커밋·라벨·지표) |
| `python -m app.audit screening-ic` | 라이브 스크리닝 순위의 예측력(IC). 축별·체류도 코호트 포함 |
| `python -m app.audit entry-paired` / `entry-version-strat` | 진입 판단의 동일일자 짝 비교 / 코드 버전별 층화 |
| `python -m app.audit rating-shift --boundary <ISO시각>` | 특정 시점 전후 등급 분포 변화 |
| `python -m app.audit guard-alpha` | 가드 위반 × 실현 알파 상관. ⚠️ 실계좌에 날짜 옵션을 주면 0행이 된다 |
| `python -m app.audit slippage` / `reconcile` / `weekly` | 슬리피지 / 원장↔KIS 대조 / 주간 리포트 |
| `python -m app.audit rating-action-gap` / `confidence` / `debate-impact` / `screen-validation` / `checkpoints` | 등급-행동 불일치 / 신뢰도 보정 / 토론 영향 / 사전스크린 검증 / 체크포인트 |
| `python -m app.backtest.screening_sweep` | **LLM 없이** 과거 스크리닝 순위를 재계산해 바로 채점한다. 장 마감(15:30) 이후에 돌린다 |
| `python -m app.experiments.<하네스>` | 프롬프트·설정 변경의 페어드 A/B |

- 수익률 두 가지를 구분한다. `excess_return`은 스크리닝, 같은 날 후보 평균 대비다. `alpha_return`은 판단 채점, 지수(KOSPI/SPY) 대비다.
- 소표본으로 결론 내지 않는다. 페어·유효 날짜 20개 미만의 결과는 잠정치로만 본다.

---

## 9. 개발 규칙과 함정

| 규칙 / 함정 | 이유 |
|---|---|
| 스크립트에서 `run_pipeline()`을 **직접 호출하지 않는다.** 측정은 `app.backtest`/`app.experiments` CLI로 한다 | 직접 부르면 실원장(`trades`)에 `is_backtest=false`로 기록돼 유령 보유가 생긴다 |
| 실험 하네스를 돌린 뒤에는 `is_backtest` 태깅을 확인한다 | 격리 결함으로 실험 판단이 라이브로 기록된 적이 있다 |
| 브로커 주문 함수는 `kis_bridge.py` 밖에서 부르지 않는다 | 안전 게이트가 그곳에만 있다 |
| `datetime.now()`를 직접 쓰지 않는다. `app.clock`을 거친다 | 라이브·백테스트가 같은 코드 경로를 쓰기 위해서 |
| 국내·해외 로직을 한 풀로 합치지 않는다 | 통화·시장시간·API가 다르다 |
| 로그 문구를 바꾸면 그 문구로 grep하는 운영 도구도 같이 고친다 | 안 고치면 조용히 0건이 되고, 그 0건은 거짓이다 |
| 프론트 수정이 안 보이면 정적파일 캐시부터 의심한다 | 코드 문제가 아닌 경우가 많았다 |
| 코드 주석은 영어로 쓰고 옆에 한글 요약 1줄을 단다. 커밋 메시지는 영어, 한 줄 요약 + 짧은 본문 | 저장소 관례 |

---

## 10. 관련 문서

| 문서 | 내용 | 비고 |
|---|---|---|
| `docs/architecture.md` | 레이어별 구조, 핵심 설계 결정, 실행 모드 3종의 차이 | 구조 설명은 유효하다. 일부 수치는 7~8월 기준 |
| `docs/safety-and-risk.md` | 안전장치·리스크 게이트 상세 | |
| `docs/development-conventions.md` | 개발 규칙 | |
| `docs/single-user-rollback-guide.md` | 멀티유저 → 단일계좌 전환(2026-08-07)의 이유와 범위 | 로그인을 다시 넣자는 논의 전에 읽는다 |
| `docs/status-and-decisions.md` | 결정 기록(재논의 방지용) | ⚠️ 7월 기준이라 "FULL_AUTO 미사용" 등 현재와 다른 서술이 있다 |
| `docs/db-password-rotation-runbook.md` | DB 비밀번호 교체 절차 | |

> 코드 주석에 나오는 `handoff/...` 경로는 저장소 밖에 따로 보관한 개인 운영 노트(진행 상황, 조사 기록)를 가리킨다. 저장소에는 포함하지 않았다.
