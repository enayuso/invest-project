# 아키텍처 개요

## 1. 시스템 한 줄 요약 + 전제

멀티 에이전트(LLM) 파이프라인이 종목별 투자 판단(Buy/Overweight/Hold/Underweight/Sell)을 내리고, 그 결과를 코드로 검증된 포지션 사이징·스탑로스 로직을 거쳐 paper ledger에 기록한 뒤, `execution_mode` 설정에 따라 KIS(한국투자증권) 계좌에 그대로/승인 후/아예 안 반영되는 방식으로 미러링하는 트레이딩 시스템이다 (`app/orchestrator/pipeline.py`, `app/execution/ledger.py`, `app/execution/kis_bridge.py`).

> **2026-08-07 되돌림.** 아래 "멀티유저" 전제(2026-08-06)는 그 사이 단일 사용자 배포
> 결정으로 대부분 되돌려졌다 — 이 서비스는 단일 계좌 하나만 단독으로
> 쓴다. 무엇을 왜 되돌렸는지는 `docs/single-user-rollback-guide.md`가
> 정본이고, 아래 문단은 갱신했다. 로그인/세션/사용자별 암호화 자격증명 저장이
> 있었다는 역사적 맥락은 `docs/multi-user-credentials-plan.md`에 남아 있다.

전제 (2026-08-07 갱신):
- **한 프로세스가 운용 계좌 하나만 서비스한다.** 로그인이 없다 — `app/auth/` 패키지
  전체가 삭제됐고, API 라우트에 세션 요구가 없다. `user_id`는 여전히 모든 관련
  테이블(`trades`/`memory_entries`/`alerts`/`runtime_settings`/`risk_rules`/
  `pending_orders`/`trade_allowlist_state` 등)의 FK/필터 키로 남아 있지만 값은
  항상 `settings.user_id`(기본 `'default'`) 하나뿐이다 — 이건 멀티유저 기능이
  아니라 행 소유자 태그였고, 걷어내는 것보다 남겨두는 쪽이 위험이 적어 그대로
  뒀다(가이드 §0 참조).
  - KIS 자격증명은 다시 환경변수다(`app.config.Settings.kis_credentials`) — 단,
    프로세스 환경에 직접 손으로 넣는 게 아니라 Jenkins Credentials가 배포 시점에
    조립해 주입한다. 사용자별 암호화 저장소(`user_broker_credentials`,
    `CREDENTIAL_ENCRYPTION_KEY`)는 폐지됐다.
  - 매매허용목록·관심종목·리스크 한도·실행 모드는 여전히 (user, mode) 스코프
    구조를 그대로 쓰지만 user 축의 값이 하나뿐이라 실질적으로 mode(paper/real)
    축만 의미가 있다. 백그라운드 루프는 `app/user_context.py::serviced_scopes`가
    항상 정확히 하나의 스코프(`ServicedScope(settings.user_id, 활성모드)`)를
    반환하는 것을 전제로 돈다 — 자격증명 미설정이면 빈 목록(루프 idle, 에러 아님).
- **접근 통제는 인증이 아니라 네트워크 위치다.** `dashboard_host`는 여전히
  `127.0.0.1`이고, 애플리케이션 수준 인증 자체가 없다 — TLS·세션쿠키·CSRF와
  무관하게, 대시보드에 닿을 수 있는 사람은(SSH 터널 보유자는) `POST /api/order`를
  포함해 전부 쓸 수 있다. `app/config.py`의 `_refuse_non_loopback_bind` 검증기가
  루프백이 아닌 주소로의 부팅을 무조건 거부한다(꺼지는 스위치 없음). 이 자세는
  "고칠 것"이 아니라 단일계좌 배포로서 의도적으로 선택된 것이다.
- **기본값은 항상 paper(모의투자).** 여전히 유효하되, 실전 진입 경로는 다시
  "배포 설정(Jenkins env)을 사람이 편집해야 한다" 형태다 — 실전에 도달하려면 네
  겹을 통과해야 한다 — `docs/safety-and-risk.md` §5.1 참조. 실전↔모의 전환
  자체(`PUT /api/account-mode`)는 무인증으로 남았고, 대신 모든 전환이
  `account_mode_switches`에 감사로그로 기록된다(같은 §5.1).
- **실주문 경로는 기본 꺼짐.** `kis_execution_enabled: bool = False`, `kis_execution_dry_run: bool = True`, `execution_mode: ExecutionMode = ExecutionMode.MANUAL` — 세 겹 모두 안전한 쪽이 기본값.
- **롱온리, 무레버리지, 무공매도.** `app/execution/sizing.py` docstring: "Long-only: Sell means 0% allocation, not a short position. No margin/shorting in this project's scope."

## 2. 전체 데이터/판단 흐름도

```
[후보 발굴 단계]                     [다중 에이전트 파이프라인]                    [execution_mode 분기]         [KIS]
                                    (app/orchestrator/pipeline.py)

trade allowlist (사용자별, settings.trade_allowlist) ┐
                                 │
KRX/US 거래량 상위 스캔         ├─▶ Market / Fundamentals /            ┐
(app.live.universe.krx_top_n /  │   News+Macro / Sentiment 4개 분석    │
 overseas_top_n)                │   Analyst 동시 실행 (asyncio.gather) │
                                 │            │                        │
momentum 랭킹으로 top-N 선별    │            ▼                        │
(app.chat.screening             │   Bull vs Bear 순차 디베이트          │
 .top_candidates, LLM-free)      │   (app.orchestrator.debate)          │
                                 │            │                        │
보유 종목 중 스탑로스 근접      │            ▼                        │
종목은 랭킹과 무관하게 강제 포함│   Research Manager 종합              │  rebalance_to_target()
(_held_needing_review)          │            │                        │  (app/execution/ledger.py)
                                 │            ▼                        │
                                 │   Trader 초안 제시                  │       │
                                 │            │                        │       ├─ MANUAL:
                                 │            ▼                        │       │   ledger만 기록, KIS 미전송
                                 │   Risk 디베이트                      │       │
                                 │   (Aggressive/Conservative/Neutral)  │       ├─ SEMI_AUTO:
                                 │            │                        │       │   pending_orders에 대기,
                                 │            ▼                        │       │   사람 승인/거절 필요
                                 │   Portfolio Manager 최종 결정         │       │
                                 │   (rating + stop_loss)               │       └─ FULL_AUTO:
                                 └───────────────────────────────────────┘           mirror_trade_to_kis()
                                                                                       → kis_bridge.py의
                                                                                         choke point 통과
                                                                                       → KIS 페이퍼/실전 서버
```

한 티커에 대해 파이프라인은 순차적으로 실행되며(`app/live/auto_trade_loop.py`: "run through ... one at a time (sequential — no concurrency"), 국내(KRX)/해외(US) 후보군은 절대 하나로 합치지 않고 완전히 분리된 풀로 스코어링한다(`_domestic_candidates`/`_overseas_candidates`).

## 3. 레이어별 설명

### web (`app/web/`)
FastAPI 대시보드. `app/web/state.py`의 `lifespan()`이 프로세스 시작 시 백그라운드 태스크를 건다:
- `run_poller_loop()` — 실시간 시세 폴링(60초, `poll_interval_seconds`)
- `stop_loss_loop()` — 저빈도(기본 1시간) 룰 기반 스탑로스 체크, 전체 trade allowlist 순회
- `auto_trade_loop()` — `settings.auto_trade_enabled`가 True일 때만 생성되는 opt-in 태스크
- `RealtimeFeed` (KIS WebSocket) — KIS 인증정보 + KRX trade allowlist 코드가 있을 때만 시작, 매매 자체는 하지 않고 시장운영정보(CB/VI)·체결틱·호가만 유지

의존 방향: web은 execution/live/dataflows/brokers를 호출하지만 orchestrator/agents는 chat 경유로만 접근한다.

### chat (`app/chat/router.py`)
자연어 질문 하나를 5개 intent 중 하나로 분류하는 라우터. `quick_factual`(단순 시세 조회) / `news_query` / `screening`(trade allowlist 비교 랭킹, LLM-free) / `general`(잡담·계좌 잔고 질문) / `deep_dive`(전체 분석 요청)로 나뉘며, **오직 `deep_dive`만 `run_pipeline()`을 호출**한다. 나머지 4개는 데이터 도구 결과를 그대로 반환하거나(스냅샷 포맷팅만) 가벼운 LLM 호출 하나로 끝낸다 — "quick questions stay quick"이 module docstring의 명시적 설계 목표.

### orchestrator / agents (`app/orchestrator/pipeline.py`, `app/agents/`)
`run_pipeline()`이 전체 체인을 조립: 4개 Analyst(market/fundamentals/news_macro/sentiment) 동시 실행 → Bull/Bear 리서치 디베이트(`app/orchestrator/debate.py`, `max_debate_rounds=2`) → Research Manager 종합 → Trader 제안 → Risk 디베이트(Aggressive/Conservative/Neutral, `max_risk_rounds=2`) → Portfolio Manager 최종 결정. 결정 전에 포지션 컨텍스트, ATR 변동성 컨텍스트, 1% 리스크캡 컨텍스트, 포트폴리오 레벨 캡 컨텍스트를 Portfolio Manager에게 미리 알려줘서 "코드가 실제로 적용할 사이징/캡을 서사가 반박하지 않도록" 한다(`_atr_context`/`_risk_cap_context`/`_portfolio_risk_context` docstring).

`app/agents/` 디렉터리 구성: `market_analyst.py`/`fundamentals_analyst.py`/`news_macro_analyst.py`/`sentiment_analyst.py`(4개 분석가), `bull_researcher.py`/`bear_researcher.py`(리서치 디베이트 화자), `research_manager.py`, `trader.py`, `risk_debators.py`(3개 리스크 화자 빌더), `portfolio_manager.py`, 그리고 공용 인프라인 `tool_loop.py`(도구 호출 루프, `as_of` 바인딩)와 `debater.py`(디베이트 실행기), `untrusted_content.py`(외부 데이터 프롬프트 인젝션 경고 래퍼). Market Analyst는 `get_stock_data`/`get_indicators`/`get_verified_snapshot`(yfinance) 외에 `get_investor_flow`/`get_kr_market_regime`(`app.dataflows.kis_provider`)를 도구로 갖는다 — 한국 시장의 외국인/기관 수급 신호. Memory/reflection은 Phase 3 설계에 따라 **Portfolio Manager 노드에만** 연결되고 다른 판단 노드로는 방송되지 않는다 — "broadcasting past_context to every judgment node was tried and deliberately not adopted"(`pipeline.py` 모듈 docstring).

4개 Analyst 중 하나가 LLM 연결 문제로 실패해도 `asyncio.gather(..., return_exceptions=True)`로 나머지 3개의 리포트는 살아남고, 실패한 자리는 플레이스홀더 문자열로 대체된다(`run_pipeline`). 정상 4개 리포트만 15분 TTL로 캐시되어(`_reports_cache`), 같은 티커/날짜/언어에 대한 반복 질문이 분석가 재실행 비용을 물지 않는다.

### dataflows (`app/dataflows/`)
`app/dataflows/router.py`의 `VendorRouter`가 카테고리별 provider 체인을 순서대로 시도한다. "core" 카테고리(가격/펀더멘털)는 전부 실패하면 예외를 던지고, "optional"(매크로 등)은 조용히 플레이스홀더로 대체된다. `app/dataflows/kis_provider.py`가 KRX 티커(`.KS`)의 price 카테고리를 KIS-first로 재등록하고 yfinance는 폴백으로 남긴다 — "yfinance closes for KRX names were observed off by ~2%"가 근거. 그 외 provider 모듈: `yfinance/`(하위 패키지 — `market_data.py`/`research.py`/`shared.py`, 가격·펀더멘털·기술지표의 기본 소스), `reddit_provider.py`/`stocktwits_provider.py`(Sentiment Analyst용, 라이브 스트림 전용·과거 조회 불가), `fred_provider.py`(매크로 지표, `fred_api_key` 미설정 시 optional 카테고리로 조용히 비활성화).

### live (보조: `app.live.market_hours`/`app.live.universe`/`app.live.poller`/`app.live.store`)
`market_hours.py`가 국내/해외 개장 여부를 판단해 `auto_trade_loop`의 시장별 스킵을 가능케 하고, `universe.py`의 `krx_top_n`/`overseas_top_n`이 거래량 랭킹으로 후보 풀을 뽑는다. `poller.py`는 LLM을 전혀 호출하지 않는 별도 캐던스의 시세 폴러이고, `store.py`는 그 틱을 인메모리에 유지하며 대시보드/알림에 서빙한다.

### execution (`app/execution/`)
- `sizing.py`: rating → target allocation 매핑(Buy 100%/Overweight 75%/Hold 변경없음/Underweight 25%/Sell 0%), ATR%가 목표치를 넘으면 다운스케일만(업스케일 없음).
- `ledger.py`: `rebalance_to_target()`이 target allocation을 실제 매수/매도 주식수로 변환, `resolve_stop_loss()`로 스탑로스를 ATR 기반 최소거리로 보정, "1% rule"로 신규 익스포저의 달러 리스크를 캡, 포트폴리오 레벨 캡(concentration/correlation/drawdown)까지 적용한 뒤 `execution_mode`에 따라 분기. 매매 체결가에는 `app/execution/costs.py`의 `transaction_cost_bps_for`가 계산한 시장별 비용을 적용한다(2026-09-08 수수료·세금·슬리피지 감사 이후): 국내(`.KS`)는 수수료 `transaction_cost_bps`(코드 기본 5bp) 양쪽 + **매도에만 증권거래세 `sell_tax_bps_domestic`(코드 기본 20bp, 2026년 코스피 0.05%+농특세 0.15%·코스닥 0.20%)**, 해외는 수수료 `transaction_cost_bps_overseas`(코드 기본 25bp, 추정치·실제 계약 요율 미검증)를 양쪽에 적용한다. 알려진 공백: 국내 ETF는 증권거래세가 면제인데 상품 유형 감지가 없어 ETF 매도는 과다 계산된다(보수적 방향). 수치는 코드 기본값이고 서버 배포값은 별도다. (2026-10-02 정정: 이 줄은 예전에 "대칭 5bp·매도세 미모델링 v1 단순화"라고 적혀 있었다.)
- `kis_bridge.py`: KIS로 나가는 모든 실주문의 단일 관문(choke point). `KIS_EXECUTION_ENABLED`(마스터 스위치)와 dry-run이 각각 별도로 꺼져 있어야만 실제 API 호출까지 도달하며, 미러링은 "fire-and-report"라 실패해도 파이프라인을 죽이지 않고 disposition 문자열만 반환한다.

### brokers (`app/brokers/`)
`KISBroker`(`app/brokers/kis/broker.py`)는 `KISHttpClient` 위에 domestic/overseas 각각의 account/order/quote/ranking/finance/info mixin ~35개 메서드를 조합한 클래스. `app/brokers/__init__.py`의 `get_broker(name)` 레지스트리를 통해서만 조회되며(`_BROKER_FACTORIES = {"kis": broker_from_settings}`), 신규 브로커 추가는 이 딕셔너리에 한 줄 추가로 끝나도록 설계됐다. `app/brokers/kis/domestic/websocket.py`의 `RealtimeFeed`가 유일한 push 채널(H0STMKO0 시장운영정보/H0STCNT0 체결가/H0STASP0 호가).

### domain (`app/domain/ports/`)
`BrokerPort`(`broker_port.py`)와 `MarketDataPort`(`market_data_port.py`)는 `Protocol`(구조적 타이핑) — 상속 없이 메서드 시그니처만 맞으면 충족된다. `MarketDataPort`에는 KIS 전용 부가데이터(펀더멘털/뉴스/수급)를 일부러 포함하지 않는다 — "KIS value-add, not something every future broker is expected to offer."

### live (`app/live/`)
`auto_trade_loop.py`: 30분(`auto_trade_interval_minutes`) 주기 POC 루프. `MANUAL` 모드면 사이클 자체를 스킵(비용 방지, "no AI involvement"라고 확인됨). 그 외 모드에서는 국내/해외 각각의 trade allowlist+스크리닝 후보를 **자격 필터(`_eligible_universe` — rebalance_interval_days 재분석 스로틀 + max_open_positions 상한)로 먼저 좁힌 뒤** 모멘텀 랭킹(`top_candidates`, LLM-free, 예산의 `auto_trade_candidate_overfetch_multiplier`배만큼 오버페치)으로 top-N까지 좁히고, 여기에 스탑로스 근접 보유종목(`_held_needing_review`, 가격 1회 조회 + DB 1회 조회, LLM 없음)을 강제로(랭킹/자격필터/예산 전부 면제) 추가한 뒤 `run_pipeline()`을 순차 호출한다. **자격 필터는 반드시 랭킹보다 먼저 적용되어야 한다** — 랭킹 뒤에 적용하면(2026-08-26 실제 발생) 매 사이클 거의 같은 순서로 나오는 상위 N개가 스로틀에 걸려 전부 탈락하고, 5위 이하 종목은 영원히 슬롯을 못 받는 자기영속적 아사 상태가 된다.

### memory (`app/memory/db.py`, `app/memory/store.py`)
Postgres+pgvector 커넥션 풀. `database_pool_min_size=2`/`max_size=5`로, "개인용 도구가 팀 공용 dev Postgres를 나눠 쓰는 상황"에 맞춘 작은 사이즈(asyncpg 기본값 10/10과 대비). 모든 커넥션에 서버 강제 타임아웃(`idle_in_transaction_session_timeout`, `statement_timeout`, `idle_session_timeout`)을 적용 — 2026-07-20 인시던트(강제 종료된 클라이언트가 트랜잭션을 2시간 잡고 있어 96개 세션이 밀린 사고)의 실제 수정. `get_pool()`은 더블체크 락 + 실패 후 `database_failure_backoff_seconds`(15초) 동안 재시도 자체를 안 함(fail-fast).

`app/memory/store.py`는 2단계(two-phase) 메모리 스토어 — 결정 시점에 `store_pending()`으로 임시 저장하고, 실현 수익률이 확정된 뒤 별도로 resolve된다. TradingAgents 논문의 append-only 로그 구조를 따르되, recency 기반이 아니라 pgvector 코사인 유사도로 "오늘과 비슷한 상황"을 검색한다는 점이 다르다(`get_similar_context`). `memory_entries.flagged_bad` 컬럼은 과거 티커 identity 환각 버그로 잘못 기록된 교훈이 임베딩되어 이후 유사 검색에서 계속 "검증된 사실"처럼 재등장하는 것을 막기 위한 삭제 없는 배제 플래그다.

### 리스크 인프라 (`app/execution/portfolio_risk.py`, `app/execution/safety_switch.py`, `app/execution/pending_orders.py`)
- `portfolio_risk.py`: 종목별 1% 룰(`ledger.py`)이 보지 못하는 "trade allowlist 전체가 동시에 Buy로 몰리거나 서로 고상관인" 상황을 잡는 3개의 포트폴리오 레벨 캡 — concentration, correlation-bucket, drawdown breaker. 셋 다 "줄이기만 하고, 절대 완화하거나 기존 보유분을 강제 청산하지 않는다"는 동일 철학. 하이워터마크(`portfolio_risk_state`)는 라이브 전용이며 백테스트는 별도의 로컬 인메모리 하이워터마크를 쓴다 — `trades` 테이블에 run-id/환경 구분 컬럼이 없어서, 공유 테이블을 쓰면 백테스트가 오늘의 라이브 자산 상태를 상속해 엉뚱한 이유로 breaker가 발동할 수 있기 때문.
- `safety_switch.py`: 킬스위치 + 일일 최대손실 서킷브레이커. kis_bridge 체인의 첫 번째 게이트이며 국내/해외, 매수/매도(킬스위치는 양쪽 다, 일일손실은 매수만) 모두에 적용. Postgres에 영속화되어 프로세스 재시작에도 유지된다 — "재시작하면 조용히 풀리는 비상정지는 비상정지가 아니다."
- `pending_orders.py`: SEMI_AUTO 모드와 FULL_AUTO의 승인 임계값 게이트가 보류시키는 주문의 영속 큐. 승인 시점에 `mirror_trade_to_kis`의 전체 게이트 체인(현금/VI/서킷브레이커/캡 등)을 처음부터 다시 통과시킨다 — 사람이 확인하는 사이 시장이 움직일 수 있으므로 생성 시점 상태를 신뢰하지 않는다.

## 4. 핵심 설계 결정들

**Single choke point (`kis_bridge.py`).** `KISBroker.submit_order`/`submit_overseas_order`는 의도적으로 가드가 없고, 모든 안전장치(킬스위치, 일일손실 서킷브레이커, 마스터 스위치, 레버리지/인버스 차단, 주문금액 캡, 승인 임계값, 시장 서킷브레이커, VI, 현금충분성, 포지션비중 캡, dry-run)는 이 파일에서만 적용된다. 두 개의 진입점(`mirror_trade_to_kis` 자동 경로, `place_manual_order` 수동 경로)이 이 게이트들을 공유한다 — "no other module may call them for trading."

**KIS-first / yfinance-fallback.** KRX 종가가 yfinance에서 실거래소 대비 ~2% 어긋난다는 관측 때문에, `.KS` 티커의 price 카테고리는 KIS를 우선 시도하고 실패하거나 비KRX 티커면 yfinance로 폴백한다(`app/dataflows/kis_provider.py`). `VendorRouter`의 provider 체인 방식이 이를 가능케 함.

**Port(Protocol) 추상화.** `BrokerPort`/`MarketDataPort`는 상속이 아닌 구조적 타이핑이라, `KISBroker`는 이 Protocol들을 import조차 하지 않고도 계약을 만족한다. domain 레이어는 brokers를 모른다(의존 방향이 brokers→domain이 아니라 "타입 계약만 domain에 정의, 구현은 brokers").

**순차 파이프라인 실행 이유.** `auto_trade_loop.py`가 후보들을 병렬이 아니라 "one at a time"으로 처리하는 이유는 명시적으로 "the LLM backend is a single instance anyway, so there's little to gain from parallelizing"이며, 백테스트 하네스의 기존 방식과도 일치시킨 것.

**Sentiment `.KS` 스킵.** `sentiment_analyst.py`는 `.KS` 티커를 백테스트뿐 아니라 라이브에서도 영구적으로 스킵하고 고정된 중립 리포트를 반환한다. 이유: Reddit/StockTwits 프로바이더가 KRX 코드를 해석하지 못해(r/stocks, r/investing 검색만, 종목명 폴백 없음) 사실상 매번 빈/플레이스홀더 결과를 내면서도 풀 LLM 호출 비용을 지불했었고, 한국시장 특유의 수급 신호는 이미 다른 곳(market_analyst의 외국인/기관 순매수, risk debate의 `get_kr_risk_signals` 공매도/프로그램매매)에서 실제 구조화 데이터로 커버되고 있기 때문(2026-07-20 평가, agent-architecture-roadmap 메모리 참조).

**사이징과 스탑로스를 서로 묶기(1% rule).** `sizing.py`의 ATR 기반 다운스케일과 `resolve_stop_loss()`의 ATR 기반 스탑로스 최소거리는, `rebalance_to_target()`의 1% 룰(신규 익스포저의 달러 리스크 = shares × (price - stop) ≤ `sizing_max_risk_pct_of_equity`)로 처음 서로 묶인다 — 그 전에는 변동성이 높아 스탑 거리가 멀어져도 등급의 전체 배분이 그만큼 더 큰 리스크를 질 수 있었다. 스탑을 못 구하면(값 누락 + ATR 데이터도 없음) 익스포저 증가 자체를 차단한다("아무 보호막 없는 포지션을 늘리지 않는다").

**Fail-open vs fail-closed을 게이트마다 다르게 가져가는 원칙.** kis_bridge.py의 게이트들은 하나의 규칙이 아니라 게이트별로 의도적으로 다르게 설계되어 있다 — 시장 전체 서킷브레이커(gate 7)와 킬스위치/일일손실은 상태를 못 읽으면 **차단**(fail-closed, `kis_cb_fail_open_when_unknown`/`kis_safety_fail_open_when_unknown` 모두 기본 `False`)하는 반면, 개별종목 거래정지 체크나 VI 체크, 현금충분성 체크는 조회 자체가 실패하면 **통과**(best-effort fail-open, 실제 정지 종목이면 KIS 자체 주문 API가 어차피 거부한다는 안전망이 있기 때문)한다. "시장 전체 이슈가 개별 종목 이슈보다 심각하다"는 판단이 근거.

**통화별 분리 현금 풀.** `ledger.py::starting_cash_for()`는 `settings.paper_starting_cash`(라이브 폴백) / `settings.backtest_starting_cash`(백테스트, `is_backtest_context()`로 분기) USD 값을 티커의 표시 통화로 환전해 사용한다 — 이게 없으면 USD 기준 현금과 KRW 기준 가격을 직접 나눠서 매수 수량을 정하다가 모든 KRW 종목의 Buy 신호가 0주로 내림되는 버그가 났었다("100,000 / ~250,000 < 1"). 티커별로 독립된 통화 환산 현금 풀을 쓰되, 이는 execution 레이어 전체의 통화 처리를 건드려야 하는 "진짜 통합 자금 풀"과는 다른 절충안이라고 `portfolio_risk.py` docstring이 명시한다.

## 5. 실행 모드 3종의 정확한 동작 차이

`ExecutionMode`(`app/config.py`)는 **AI 파이프라인의 자동매매 경로 하나만** 통제한다 — 대시보드에서 사람이 직접 누르는 수동 주문(`place_manual_order`)이나 스탑로스 강제청산(`check_and_apply_stop_loss`)에는 전혀 영향을 주지 않는다(스탑로스는 "리스크 감소" 조치이지 "AI 매매 결정"이 아니므로 항상 실행).

`rebalance_to_target()`(`app/execution/ledger.py`) 안에서 세 모드가 분기하는 지점은 동일 — **ledger에 trade row를 기록하는 것까지는 세 모드 모두 동일하게 수행**된다(paper 포트폴리오 자체 부기이므로 KIS 미러링 여부와 무관). 그 다음:

- **MANUAL**: `disposition = "skipped: MANUAL execution mode..."` — KIS로 아무것도 전송하지 않는다. `auto_trade_loop`도 이 모드에서는 사이클의 분석 자체를 건너뛴다(비용 방지, LLM 호출 없음).
- **SEMI_AUTO**: `app.execution.pending_orders.create_pending_order()`를 호출해 주문을 대기열에 넣고 `disposition = "PENDING_USER_APPROVAL: ..."`을 반환 — 사람이 명시적으로 승인/거절해야 KIS로 나간다.
- **FULL_AUTO**: `mirror_trade_to_kis()`를 직접 호출 — `kis_bridge.py`의 모든 게이트(킬스위치부터 dry-run까지)를 통과해야 실제(또는 paper) 주문이 나간다. `kis_execution_enabled=False`나 `kis_execution_dry_run=True`가 기본이므로, FULL_AUTO를 켜도 이 두 스위치가 별도로 꺼져 있지 않으면 실제로는 아무 주문도 나가지 않는다.

세 모드 모두 리스크 게이트(1% 룰, 포트폴리오 캡, 스탑로스 보정)는 ledger 단계에서 이미 동일하게 적용된 뒤이므로, `execution_mode`가 어떤 값이든 저장되는 목표 수량 자체는 달라지지 않고 "그 수량을 KIS로 보내느냐/누구 승인을 거치느냐"만 달라진다.

정리하면:

| 항목 | MANUAL | SEMI_AUTO | FULL_AUTO |
|---|---|---|---|
| ledger에 trade row 기록 | O | O | O |
| `auto_trade_loop`이 이 모드에서 분석 자체를 실행하는가 | X (사이클 스킵) | O | O |
| KIS로 실제 전송 시도 | X | X (승인 전) | O (게이트 통과 시) |
| 승인 필요 여부 | 해당 없음 (애초에 시도 안 함) | 항상 필요 (`create_pending_order`) | 승인 임계값 이상일 때만(opt-in) |
| 대시보드 수동 주문(`place_manual_order`) 영향 | 없음 | 없음 | 없음 |
| 스탑로스 강제청산(`check_and_apply_stop_loss`) 영향 | 없음 (항상 실행) | 없음 (항상 실행) | 없음 (항상 실행) |

세 모드 어느 것도 `kis_execution_enabled`/`kis_execution_dry_run`/각종 위험 게이트를 우회하지 못한다 — `execution_mode`는 오직 "AI가 만든 제안이 그 게이트들 앞까지 도달하느냐"만 결정한다.
