# 안전/리스크 원칙 (Safety & Risk Principles)

> **2026-10-02 정정 안내(먼저 읽을 것)**: 이 문서에는 2026-08-06(멀티유저·로그인) 시점의 설명과
> 2026-08-07 이후(단일 계좌·**로그인 없음**)의 설명이 섞여 있다. **현재 애플리케이션 수준 인증은
> 없다** — `POST /api/order`를 포함한 모든 라우트가 무인증이고, 접근 통제는 `dashboard_host`를
> 루프백에 묶는 것(`app.config._refuse_non_loopback_bind`)과 SSH 터널뿐이다(`app/web/app.py` 모듈
> 설명). 세션·라우트 보호를 현재 보호 장치로 설명하는 절은 과거 기록으로 읽는다.
> 주문·승인 한도의 **숫자**(기본값·상한)는 이 문서가 아니라 `app/config.py` 정의를 정본으로 보고,
> 코드 기본값과 실제 배포값(Jenkins env)은 다를 수 있으니 따로 확인한다. 이 안내는 문서 정리만 했고
> 인증·매매 동작은 바꾸지 않았다.

본 문서는 이 프로젝트(개인용 AI 주식매매 시스템, KIS 모의투자 기본)의 실제 코드에 구현된
안전장치만을 정리한다. 여기 없는 게이트/정책은 존재하지 않는다 — 코드가 바뀌면 이 문서도
같이 갱신되어야 한다. 모든 인용 경로는 `invest-project/` 기준 상대경로다.

## 1. 안전 철학

### 1.1 단일 관문 원칙 (Single Choke Point)

`app/execution/kis_bridge.py`의 모듈 독스트링이 명시하듯, KIS로 나가는 모든 실주문은 이
파일 하나를 반드시 통과한다. `KISBroker.submit_order`/`submit_overseas_order`는 의도적으로
가드가 없는(guard-free) 저수준 함수이며, 트레이딩 목적으로는 이 두 진입점 외에 어떤
모듈도 브로커를 직접 호출하지 않는다:

- `mirror_trade_to_kis` — AI 파이프라인이 방금 ledger에 기록한 거래를 자동으로 미러링
- `place_manual_order` — 대시보드 주문 티켓에서 사용자가 직접 실행

두 경로는 거의 동일한 게이트 체인을 공유하며, 차이는 3절에 정리한다.

### 1.2 Fail-safe 기본값

`app/config.py`의 주석이 명시하는 철학: "설정 하나 빠져도 안전한 쪽으로."

- 실전 계좌 전환(`kis_account_mode: str = "paper"`)은 사람이 의도적으로 바꾸지 않는 한
  절대 `"real"`이 되지 않는다 (`app/config.py:213` 부근 주석 — "이건 편의를 위해 함부로
  바꾸거나 기본값을 real로 둘 토글이 아니다").
- 실주문은 마스터 스위치(`kis_execution_enabled: bool = False`)와 dry-run
  (`kis_execution_dry_run: bool = True`) 두 겹으로 막혀 있다. 신규 체크아웃에서 실제
  (모의)주문 하나가 나가려면 최소 두 번의 의도적 설정 변경이 필요하다
  (`app/config.py:234-243`). 과거 이 중 하나가 `True`로 잘못 설정되어 있던 이력이 주석에
  남아 있고("계약을 어기는 값이었다"), 지금은 두 겹 모두 안전 쪽 기본값으로 고정됨.
- `execution_mode`의 기본값은 `MANUAL` — AI가 애초에 주문 시도 자체를 하지 않는 모드
  (`app/config.py:251-254`).
- `kis_block_leveraged_inverse: bool = True` — 레버리지/인버스 상품 배제도 기본 ON.
- `dashboard_host: str = "127.0.0.1"` — 대시보드의 `POST /api/order` 등 주문 API는
  인증이 없다(single-user v1). `0.0.0.0`으로 바꾸면 실주문 엔드포인트가 네트워크 전체에
  노출되므로, 반드시 localhost 바인딩을 유지해야 한다 (`app/config.py:167-171`).

### 1.3 알 수 없으면 차단한다 (fail-closed vs best-effort fail-open)

이 프로젝트는 두 가지 실패 처리 방향을 의도적으로 구분한다:

- **Fail-closed** — 상태를 모르면 차단. 시장 전체에 영향을 주는 심각한 게이트(킬스위치,
  일일 손실 서킷브레이커, 시장 전체 서킷브레이커)에 적용. "이 게이트가 유일한 방어선"인
  경우.
- **Best-effort fail-open** — 체크 자체가 에러나면 통과시키되, 뒤에 KIS 자체의 주문
  거부라는 최종 방어선이 있는 경우에만 허용(개별 종목 매매정지, VI, 현금 충분성 체크).
  "이 조회가 실패해도 KIS 서버가 어차피 걸러준다"는 확신이 있을 때만 이 방향을 쓴다.

이 구분은 `kis_bridge.py` 모듈 독스트링과 각 게이트 함수의 독스트링에 개별적으로 명시되어
있으며, 아래 2절에서 게이트별로 방향을 표기한다.

## 2. 주문 경로와 게이트 체인

두 진입점(`mirror_trade_to_kis`, `place_manual_order`)이 호출 순서대로 통과해야 하는
게이트. `[공통]` `[자동전용]` `[수동전용]`으로 표기.

1. **티커 라우팅** `[공통]` — KRX 6자리 코드(`005930.KS` → `005930`, 시장가) 또는 US
   심볼(`AAPL`, NASDAQ 지정가)만 라우팅 가능. 어느 쪽도 아니면 스킵.
   (`kis_bridge.route_ticker`)

2. **킬스위치 + 일일 최대손실 서킷브레이커** `[fail-closed]` — 모든 게이트 중 가장
   먼저 체크. 킬스위치는 계좌 전체 공용(국내+해외, 매수+매도 모두 차단) — 사람이 누르는
   수동 비상정지라 계좌 잔고 파생 지표가 아니므로 시장별로 나누지 않음. 일일손실
   브레이커는 매수만 차단(매도는 항상 통과)하며, **국내/해외 각각 독립적으로 추적**
   (2026-07-21 개정 — 계좌 잔고 파생 지표는 국내/해외를 절대 합치지 않는다는 4.6절
   원칙과 맞춰 미국장 손실이 KRX 매수까지 막던 문제를 수정). Postgres에 영속화되어
   프로세스 재시작에도 유지됨 (`app/execution/safety_switch.py`). 실제 KIS 잔고 조회
   자체가 실패(`AccountUnavailable`)하면 **무조건 차단**(설정과 무관), 그 아래
   Postgres 저장 자체가 실패하면 `kis_safety_fail_open_when_unknown`(기본 `False`)
   정책에 따라 기본은 차단 — 서로 다른 두 실패 모드를 구분해서 처리함.
   (`safety_switch.safety_gate_block`)

3. **마스터 스위치** `[공통]` — `KIS_EXECUTION_ENABLED`(기본 off). off면 무조건 skip.

4. **레버리지/인버스 + 미지원 상품유형 필터** `[공통]` — `KIS_BLOCK_LEVERAGED_INVERSE`(기본 on).
   종목코드가 아닌 상품명 텍스트로 판정(`app/domain/leverage_filter.is_leveraged_or_inverse`,
   키워드: 레버리지/인버스/선물/leveraged/inverse/ultra/bull/bear/2x/3x/futures)에 더해
   `quote_type`이 일반 주식/ETF가 아니면 차단(`is_unsupported_product_type`, 예: MUTUALFUND).
   여기에 더해 **범위 외 상품**(`app/domain/product_filter.out_of_scope_reason`)도
   상품명으로 차단 — 현금성/채권/채권혼합형 펀드(CD금리·KOFR·국고채·종합채권 등)와
   커버드콜·옵션인컴형 펀드(커버드콜·covered call·buy-write·premium income). 분석할
   회사가 없고(실적·뉴스 없음), 커버드콜은 옵션 델타/세타 수익 구조라 모멘텀·밸류에이션·
   ATR 분석이 오히려 오도함(2026-09-01 481050.KS, 2026-09-02 498400.KS 사고).
   이 부분은 `KIS_BLOCK_CASH_BOND_FUNDS`(기본 on) 플래그로 게이트되며 **매수만** 차단
   (매도는 통과 — 보유분 청산 불가 방지). universe·trade_allowlist_add는 항상 제외.
   이름 조회 자체가 실패하면 best-effort로 통과(정보 부재는 판단 불가와 동일하게 취급).
   (`kis_bridge.unsupported_product_block`)

5. **주문 금액 한도** `[공통]` — `KIS_MAX_ORDER_KRW` / `KIS_MAX_ORDER_USD` — **모드별 값**이다(`app/config.py`의 `*_paper`·`*_real`). 코드 기본값은 모의 `kis_max_order_krw_paper`=5,500,000원 / `kis_max_order_usd_paper`=40,000달러이고, **실전 `*_real`은 기본값이 없다(None)** — 설정하지 않으면 실전 주문을 평가하지 못하고 부팅 검증이 막는다(§5.1). 배포 실제 값은 Jenkins env 정본(2026-10-02 정정: 예전에 "기본 5,000,000원/3,500달러"로 적혀 있었다). 엄격히 `>` 비교(캡과 정확히 같은 금액은 여기서
   차단되지 않고 승인 대상으로 넘어감 — 의도된 경계). (`kis_bridge.over_value_cap`)

6. **승인 임계값(pending_orders)** `[공통, opt-in]` — `KIS_APPROVAL_*`
   (`kis_approval_required_enabled` 기본 off). 임계값 이상 주문은 명시 승인 없이 통과 불가.
   자동 경로는 절대 승인을 자체 부여하지 않고 `pending_orders`에 적재, 수동 경로는 대시보드
   확인에서 승인 여부를 받아옴. (`kis_bridge.requires_approval`, 3절 참조)

7. **시장 전체 서킷브레이커** `[국내전용, fail-closed]` — `app.live.kis_market_status`의
   실시간 WebSocket 피드(`RealtimeFeed`) 상태를 무료로(네트워크 호출 없이) 읽는 게이트.
   KIS는 "서킷브레이커 활성 여부"를 조회하는 REST API를 제공하지 않으므로 실시간 피드 상태로만
   판단 가능. `True`(활성)면 차단. `None`(피드가 실제로 미연결이거나, 이 종목이 구독 목록에
   없거나, 코드 매핑이 미설정된 경우)도 `kis_cb_fail_open_when_unknown`(기본 `False`) 정책에
   따라 **차단** — 개별종목 체크의 best-effort 방향과 다르게, 시장 전체 정지가 개별 종목보다
   더 심각하기 때문. KOSDAQ은 커버 안 됨(관심종목이 전부 KOSPI). (`kis_bridge._circuit_breaker_block`)
   **"연결됨+구독중인데 이벤트가 아직 없는 경우"는 unknown이 아니라 정상(False)으로 처리됨**
   (2026-07-21 수정, `RealtimeFeed.connected` 플래그 추가) — H0STMKO0은 실제 이벤트(정지/VI/CB)가
   있을 때만 프레임을 보내므로, 조용한 날은 하루 종일 프레임이 안 오는 게 정상 상태이지 "짧은
   시작 구간"이 아니라는 걸 실사용 중 확인함(이전엔 이 케이스도 차단으로 처리해 정상적인 매도까지
   막았었음). 진짜 unknown(피드 미연결/미구독)은 여전히 차단 유지.

8. **개별종목 매매정지/관리종목 체크** `[국내전용, best-effort]` — 주문 직전 REST 1회 호출
   (`KISBroker.fetch_stock_status`, 주식기본조회). 체크 자체가 에러나면 통과(진짜 정지 종목은
   KIS 자체 주문 API가 어차피 거부하므로, 이건 더 빠르고 명확한 에러를 주기 위한 조기 안전망일
   뿐 유일한 방어선이 아님). 모의투자 서버가 이 TR을 지원 안 하는 경우(`EGW02006`)는 별도
   식별되어 "모의투자 미지원으로 생략됨" 안내가 최종 결과에 남음. US 티커는 대응 체크 없음.
   (`kis_bridge._stock_halt_block`)

9. **VI(변동성완화장치) 정지 체크** `[국내전용, best-effort]` — `KISBroker.is_vi_triggered`
   (변동성완화장치 현황 TR). 매수/매도 모두 적용(VI는 매칭 자체를 정지시킴). 이 TR은 현재
   정지된 코드 리스트를 직접 반환하므로 서킷브레이커 게이트처럼 별도 매핑 확인이 필요 없고,
   모의투자에서도 정상 동작 확인됨. (`kis_bridge._vi_block`)

10. **현금 충분성 체크** `[공통, 매수만, best-effort]` — 국내는 매수가능조회
    (`fetch_buyable`), 해외는 잔고조회 TR의 주문가능 USD. 조회 결과 통화가 매칭 안 되면
    (해외) 확정된 0원이 아니어도 "불명확"으로 차단(방향은 fail-closed와 동일하되 문구만
    다르게 표현). (`kis_bridge.insufficient_cash_block`)

11. **포지션 비중 상한** `[국내 매수만, opt-in]` — `KIS_MAX_POSITION_PCT`(기본 off, 20%).
    분모는 총평가금액(현금 포함 계좌 전체), 해외 포지션은 FX 정규화 미구현으로 이 게이트
    대상 아님(문서화된 TODO). 계좌 평가금액을 못 구하면(0) 판단 불가로 스킵.
    (`kis_bridge.over_position_pct`)

12. **Dry-run** `[공통]` — `KIS_EXECUTION_DRY_RUN`(기본 on). 단, 두 진입점에서 위치가
    다르다: 자동 경로(`mirror_trade_to_kis`)는 dry-run이 7~11번(브로커 조회 기반 게이트)
    **이전에** 끊어 불필요한 KIS 호출 자체를 생략하고, 수동 경로(`place_manual_order`)는
    dry-run이어도 7~11번을 전부 실제로 검증한 뒤 전송 직전에만 정지한다 — 수동 주문의
    dry-run 결과가 "검증된 것처럼 보이지만 실제론 브로커 게이트를 건너뛴" 상태가 되지 않도록
    의도적으로 고친 이력이 코드 주석에 남아 있다.

### 수동 전용 게이트

- **포트폴리오 레벨 캡** (`portfolio_caps_block`, 매수만) — 집중도/상관관계 버킷/드로우다운
  브레이커(`app/execution/portfolio_risk.py`). 자동 경로는 이미 `rebalance_to_target` 안의
  `apply_live_portfolio_caps`에서 주문 수량이 사전에 축소되어 나오므로 이 게이트가 필요 없지만,
  수동 주문 티켓은 사용자가 임의 수량을 넣을 수 있어 이 검증이 없으면 캡을 무제한 초과할 수
  있음. 조용히 축소하지 않고 그대로 차단(수동 주문은 의도적 행위이므로 사용자가 직접 작은
  수량을 선택해야 함).
  **2026-07-21 개정**: 이 세 캡의 계산 기준이 이 앱 내부의 시뮬레이션 장부(`ledger.py`의
  `trades` 테이블, 종목당 가상 시작자금)에서 **실제 연결된 KIS 계좌 잔고**
  (`real_account_snapshot`, `app/brokers/kis/*/account.py`의 `fetch_balance`/
  `fetch_overseas_balance`)로 바뀌었고, 국내/해외를 하나의 USD 합산치로 보던 것도
  **시장별로 완전히 분리**됐다(고점 기록도 `portfolio_drawdown_state`에 시장별로 별도 저장).
  실제 KIS 잔고 조회 자체가 실패하면(`AccountUnavailable` — 브로커 미연결/네트워크 오류)
  **무조건 fail-closed**로 노출 증가 거래를 차단한다 — 이전엔 이 조회가 실패하면 조용히
  캡을 건너뛰던(fail-open) 것과 반대 방향으로 바뀐 것.
- **매도 수량 상한** (`domestic_sell_over_held`, 국내전용) — 보유 수량 초과 매도 차단.
  전용 TR(매도가능수량조회, `fetch_sellable`)이 모의서버에서 검증되지 않아, 대신 잔고조회로
  대체 산출(당일 미결제 매도분 차감은 안 됨 — 다소 거칠지만 "보유 이상 매도"라는 핵심 실수는
  막음).

### 자동 전용 (mirror-only) 게이트

- **오늘 거래만** — 백테스트가 과거 이력을 재생하면서 라이브 계좌로 주문이 새어나가면 안 됨
  (`trade_date != get_clock().today()`이면 skip).
- **관심종목 소속 여부** — AI는 관심종목에 있는 종목만 거래 가능.

### 동시성 보호

`kis_bridge._order_lock`(프로세스 전역 `asyncio.Lock`) — 현금/포지션/캡 등 잔고 기반
게이트는 실제 주문 제출과 함께 이 락으로 직렬화된다. 수동 주문과 자동 미러링이 거의
동시에 들어오는 경우 같은 잔고 스냅샷을 각각 통과해 이중 제출되는 레이스를 막기 위한
의도적으로 거친 해법(개인용/저빈도 시스템이므로 세밀한 종목별 락 대신 전체 직렬화 선택).

## 3. 실행 모드와 승인 흐름

`app/config.py`의 `ExecutionMode` — AI 파이프라인의 자동매매 경로(`ledger.rebalance_to_target`
→ `kis_bridge.mirror_trade_to_kis`)에만 적용되며, 위 2절의 어떤 리스크 게이트도 우회하지
않는다. 세 값:

- **`MANUAL`(기본값)** — AI는 목표 배분만 계산하고 주문 시도 자체를 하지 않음.
- **`SEMI_AUTO`** — AI의 모든 제안이 예외 없이 `app/execution/pending_orders.py`에
  적재되어 사람의 명시적 승인/거절을 기다림.
- **`FULL_AUTO`** — AI가 직접 제출(오늘날의 기존 기본 경로), 단 여전히 모든 리스크
  게이트 적용.

`FULL_AUTO`에서도 6번 게이트(승인 임계값)에 걸리면 개별 주문은 `pending_orders`로
넘어간다 — `mirror_trade_to_kis`는 `approved=False`가 기본이며, 자동 경로는 절대
`approved=True`를 스스로 설정하지 않는다.

`pending_orders.approve_pending_order`는 승인 시점에 `mirror_trade_to_kis(approved=True)`를
다시 호출해 **그 시점 기준으로 전체 게이트 체인을 처음부터 다시** 통과시킨다(생성 당시
스냅샷을 신뢰하지 않음 — 사람이 검토하는 동안 시장이 움직일 수 있으므로). `approved=True`는
2번 게이트(승인 임계값)만 우회하고, 현금/VI/서킷브레이커/캡 등 나머지는 그대로 다시 검사됨.

### AI Profile — 판단 성향 (`app/config.py`의 `AIProfile`)

`ExecutionMode`(누가 승인하는가)·`kis_account_mode`(어디로 나가는가)와 별개인 세 번째 축.
Trader/Portfolio Manager 두 결정 노드(`app/agents/trader.py`, `app/agents/portfolio_manager.py`)의
system prompt에 성향 suffix를 덧붙이는 것 외에는 아무것도 바꾸지 않는다 — 애널리스트, 디베이트,
2절의 리스크 게이트 체인, 브로커 선택, 승인 흐름은 이 값과 완전히 무관하다.

- **`PRODUCTION`(기본값)** — 보수적, HOLD 활용, 오늘까지의 기존 프롬프트와 완전히 동일(suffix
  없음).
- **`TEST_AGGRESSIVE`** — 주문 파이프라인 자체를 검증할 목적으로 HOLD을 최소화하고 BUY/SELL을
  적극 생성하도록 지시. 투자 판단 품질이 아니라 주문 이벤트 다양성이 목표.
- **`DEBUG`** — 기존 `reasoning`/`investment_thesis` 텍스트 필드에 판단 근거를 최대한 상세히
  쓰도록 지시(스키마 변경 없음).

**안전장치** — `ai_profile != PRODUCTION`이면서 `kis_account_mode == "real"`인 조합은
`ai_profile_live_ack: bool = False`(기본값)가 `True`로 명시적으로 바뀌지 않는 한 `Settings()`
생성 자체가 `ValueError`로 실패해 앱이 부팅되지 않는다(`app/config.py`의
`_block_non_production_profile_on_real_account` model_validator). 테스트용으로 일부러
신호를 과다 생성하는 프로필이 사람의 명시적 확인 없이 실전 계좌의 주문 경로에 닿는 사고를
막기 위함 — 5.1절의 "실전 전환은 항상 opt-in" 원칙을 이 축에도 그대로 적용한 것.

백테스트(`app/backtest/harness.py`)는 `settings.ai_profile`과 무관하게 항상 `PRODUCTION`을
명시적으로 고정 전달한다 — 개발 환경이 `TEST_AGGRESSIVE`로 설정돼 있어도 백테스트 결과의
결정성/비교가능성이 깨지지 않도록.

### 대시보드 튜닝 가능 vs 설정 전용 (왜?) — 2026-08-06 근거 교체

`app/execution/risk_rules.py`: **리스크 수치는** 사용자별 런타임 override 가능하고
**실행 on/off 스위치는 설정 파일에서만** 바꿀 수 있다. 결론은 그대로지만 **이유가
완전히 바뀌었으므로** 예전 근거를 그대로 인용하지 말 것.

- **예전 근거(이제 사실이 아님)**: "대시보드의 주문 API는 무인증(로컬호스트 전용)이므로,
  '실전 전환' 토글을 노출하면 로컬호스트의 무엇이든 시스템을 실주문 모드로 바꿀 수 있다."
  멀티유저 이전 Phase 1/3 이후 `/api/order`를 포함한 61개 라우트 중 57개가 세션을
  요구한다(`app/web/app.py`의 `include_router(..., dependencies=_authenticated)`,
  `tests/auth/test_route_protection.py`가 라우트별로 검증). 무인증 전제는 사라졌다.

- **2026-08-06 근거(2026-08-07 로그인 제거로 다시 바뀜 — 아래 "2026-08-07 이후" 참조)**: 실행 스위치는 여전히 API로 노출하지 않는다. 이유는 인증의 유무가 아니라
  **세션을 신뢰할 만한 전송 계층이 아직 없다**는 것이다 — TLS 미도입, 따라서 세션 쿠키가
  `secure=False`(`app/web/routers/auth.py`), CSRF 방어 없음, 재인증(step-up) 없음.
  이 상태에서 실주문 마스터 스위치를 API로 노출하면 탈취되거나 CSRF된 세션 하나로 실거래를
  시작할 수 있다. 사용자 승인은 2026-08-06에 받았으나 위 세 가지가 선행 조건이며 아직
  구현되지 않았다(`docs/multi-user-credentials-plan.md` §7.6).
  즉 지금 이 항목을 지키는 것은 "무인증이라서"가 아니라 "인증을 실어 보낼 안전한 채널이
  없어서"다.

- **2026-08-07 이후(현재) 근거**: 로그인 자체가 제거돼(`docs/single-user-rollback-guide.md`,
  `app/web/app.py` 모듈 설명) 위 "57개 라우트가 세션을 요구" 설명과
  `tests/auth/test_route_protection.py`·`app/web/routers/auth.py`는 더 이상 존재하지 않는다
  (2026-10-02 확인). 지금은 **무인증**이고 서버 접근(SSH 터널)이 곧 API 권한이다. 그래서 실행
  on/off 스위치를 설정 파일(배포 env)에서만 바꾸게 하는 원칙은 "무인증 API로 실주문 모드를 켤 수
  없게 한다"는 원래 이유로 돌아왔다. 실전↔모의 계좌 모드 전환의 현재 상태는 §5.1을 본다.

- **대시보드에서 바꿀 수 있는 것 (`RULE_SPEC`, 사용자별·모드별로 분리 저장)**: 7개에서
  **23개**로 확장됨(2026-08-06). 주문/승인 한도(`kis_max_order_krw/usd`,
  `kis_approval_*`), 포지션·집중도·상관·드로다운 한도(`kis_max_position_pct*`,
  `portfolio_*`), 사이징 파라미터(`sizing_*`), 보유 종목 수 상한
  (`max_open_positions_*`), 일일 손실 한도(`kis_daily_max_loss_pct`),
  `faber_sma_window_days`. 정확한 목록은 코드가 단일 소스오브트루스다:
  `python -c 'from app.execution.risk_rules import RULE_SPEC; print(sorted(RULE_SPEC))'`

- **사용자가 자기 한도를 올릴 수 있게 된 대신 절대 상한이 생겼다**:
  `KIS_MAX_ORDER_KRW_CEILING` / `KIS_MAX_ORDER_USD_CEILING`은 설정 전용이며(2026-10-02 확인: 코드 기본값은 `None`=상한 미설정, 예전에 적힌 1억·7만달러는 코드 기본값이 아니므로 배포값은 Jenkins env로 확인),
  사용자가 자기 주문 한도를 이 값 위로 올릴 수 없다. 상한 없는 "사용자가 바꿀 수 있는
  한도"는 한도가 아니기 때문.

- **설정 전용(대시보드 API로 절대 불가)**: `KIS_EXECUTION_ENABLED`,
  `KIS_EXECUTION_DRY_RUN`, `KIS_BLOCK_LEVERAGED_INVERSE`,
  `KIS_MAX_ORDER_*_CEILING`, `AI_PROFILE_LIVE_ACK`, `AUTH_REQUIRED`.
  `KIS_ACCOUNT_MODE`는 배포 기본값으로만 남았고 실제 활성 모드는 사용자별이다 — §5.1 참조.
- 교차 검증(`risk_rules._validate_cross_field`): 승인 임계값이 주문 한도 캡 이상으로
  설정되면 저장이 거부된다 — 그렇지 않으면 "승인이 필요한 주문"이라는 개념 자체가 캡에 의해
  선점되어 무의미해지기 때문(`over_value_cap`은 캡 초과분을 `>`로 강제 차단하므로, 캡과
  같거나 큰 임계값은 죽은 설정이 됨).

## 4. 포지션/리스크 사이징 규칙

### 4.1 등급별 목표 배분 (`app/execution/sizing.py`)

Trader의 자유서술(`position_sizing` 프로세)이 아니라 코드로 강제된 등급→배분 매핑:
BUY 100% / OVERWEIGHT 75% / HOLD 변경없음 / UNDERWEIGHT 25% / SELL 0%. Long-only —
SELL은 공매도가 아니라 배분 0%를 의미하며, 이 프로젝트 범위에는 마진/공매도가 없다.
ATR%가 목표치(`sizing_target_atr_pct`, 기본 2%)를 넘으면 배분을 축소만 하고(하향 조정
전용), 절대 기본 배분 이상으로 확대(레버리지화)하지 않는다.

**2026-07-22 개정 — Flat(미보유) 종목은 Buy/Hold만 가능**: 실거래 테스트 중 Flat 종목이
"Underweight(비중 축소)"로 판정되면서 매수 주문이 나가는 버그가 발견됐다. 원인은 Portfolio
Manager 프롬프트가 이 등급→배분 수치를 LLM에게 전혀 알려주지 않아서, LLM은 "Underweight"를
자연어로 "비중을 0에 가깝게"라고 서술했지만 코드는 그걸 25% 목표로 실행했기 때문(서술과
실행의 모순). 수정 두 가지:
1. 프롬프트(`app/agents/portfolio_manager.py`의 `SYSTEM_PROMPT`)에 위 배분 수치를 명시.
2. 더 근본적으로, `app.state.FlatPortfolioDecision`(2값 enum `FlatPortfolioRating`: Buy/Hold만)을
   추가해 **Flat 종목은 애초에 Overweight/Underweight/Sell을 출력할 수 없도록 스키마 자체로
   강제**한다 — `portfolio_manager.decide(..., held=False)`가 이 스키마를 쓰고, 결과를 기존
   `PortfolioDecision`으로 정규화해 반환하므로 `sizing.py`/`ledger.py` 등 하류 코드는 전혀
   안 바뀐다. `held`는 `app/orchestrator/pipeline.py`의 `_position_context`가 실제 보유수량을
   보고 결정한다.

**같은 시점에 함께 고친 관련 문제 — 스크리닝 종목의 관심종목 미등록**: `auto_trade_loop`의
후보 선정은 관심종목 밖의 모멘텀 스크리닝 종목도 포함하는데, 실행 게이트
(`kis_bridge.route_ticker`쪽 trade allowlist 체크)는 관심종목에만 있는지 확인해 없으면 무조건
차단 — 전체 파이프라인을 다 태워 BUY 결정까지 내고도 순전히 관심종목 미등록으로 막히는
낭비가 있었다. `run_pipeline`이 이제 **실제 BUY 결정이 나왔을 때만**(Hold로 끝난 스크리닝
종목까지 관심종목을 불리지 않도록), 라이브 경로에서만(백테스트 제외)
`app.web.state.add_ticker_to_trade_allowlist`를 best-effort로 호출해 그 종목을 관심종목에
추가한 뒤 주문을 진행한다.

### 4.2 1% 리스크 규칙 (`app/execution/ledger.py::rebalance_to_target`)

`settings.sizing_max_risk_pct_of_equity`(기본 1%) — 이 거래가 **추가하는** 손실 위험만
제한하며, 결과 포지션 전체의 위험을 제한하지 않는다. 이전 결정에서 이미 보유 중인 물량을
나중에 더 넓은 스탑으로 강제 청산시키지 않기 위함. 노출을 늘리는 경우에만 적용되고, 줄이거나
청산하는 거래는 제한 없음. 계산: `risk_per_share = price - resolved_stop_loss`, 이번
추가분의 `incremental_shares × risk_per_share`가 `equity × 1%`를 넘으면 추가 수량을
축소.

### 4.3 ATR 기반 스탑로스 강제 (`ledger.resolve_stop_loss`)

Portfolio Manager(LLM)가 제시한 스탑로스 값을 코드로 검증/보정한다:

- 포지션이 없어지면(`shares_after <= 0`) 스탑 불필요.
- 값이 없거나, 0 이하거나, 현재가 이상(long-only 북에서 무효)이면 ATR 기반 기본값으로
  대체 — 단 ATR 데이터도 없으면 값을 추측해서 만들어내지 않고 미설정으로 둠("추측보다는
  성능 저하" 원칙).
- ATR 데이터가 있으면 `sizing_min_stop_atr_multiple`(기본 2.5배, Turtle Trading/Chandelier
  Exit 근거) × 현재 ATR% 거리 이하로는 절대 좁혀지지 않도록 하한선(floor)만 적용 — 너무
  타이트한 스탑이 통상적 변동성에 휘둘려 튕겨나가는 것을 방지. 극단적 변동성에서도 스탑이
  현재가에 닿거나 넘지 않도록 최대 하한 거리는 가격의 95%로 클램프됨.
- 스탑 방향은 항상 "넓히기만" — LLM이 이미 ATR 하한보다 넓은 스탑을 줬다면 그대로 유지.

### 4.4 스탑 미해결 시 노출 확대 금지 (no-stop-no-add 정책)

노출을 늘리려는데(`target_shares > position.shares`) 해결 가능한 스탑로스가 없으면
(LLM 값도 없고 ATR 데이터도 없음), 노출 확대 자체를 차단하고 기존 보유 수량에 머문다 —
방어 근거 없는 무제한 포지션을 만들지 않기 위함. 보유/축소는 이 조건과 무관하게 항상
허용됨.

### 4.5 레버리지/인버스 상품 배제 (`app/domain/leverage_filter.py`)

Semi-auto/full-auto 모드는 사람의 건별 판단을 제거하므로, 마진콜/청산 위험과 감가/변동성
특성을 가진 선물·레버리지·인버스 상품은 이 시스템의 사이징/스탑로스 로직이 애초에 다루도록
설계되지 않았다는 이유로 축소가 아니라 **전면 배제**된다. 종목코드가 아니라 상품 표시명으로
판정 — KRX 레버리지/인버스 ETF(예: KODEX 레버리지=122630.KS)는 일반 종목과 형식상 구분이
안 되고, 미국 레버리지 ETF 티커(TQQQ, SOXL 등)도 마찬가지이기 때문. 스크리닝 유니버스,
관심종목 추가, KIS 주문 관문 세 지점 모두 이 동일한 순수 함수를 호출한다(Domain-First
원칙 — 각 계층이 자체 키워드 목록을 재구현하지 않음).

### 4.6 계좌 잔고 파생 지표는 국내/해외를 절대 합치지 않는다 (2026-07-21)

이 프로젝트는 국내(KRX)/해외(US) 계좌를 실제 KIS 계좌 구조대로 서로 완전히 무관한 별개
계좌로 취급한다는 원칙을 따른다 — 모의투자/실전투자가 서로 다른 API 키·서버로 물리적으로
분리된 것과 비슷한 정신이지만, 국내/해외는 KIS 인증정보 자체는 하나를 공유하고(같은
증권계좌 아래 국내/해외 하위계좌 구조라 KIS가 애초에 그렇게 되어 있음) 대신 **로직/데이터
레벨에서** 이 분리를 지켜야 한다.

**분리 기준**: "계좌 잔고에서 파생되는 지표"는 국내/해외 각각 독립으로 관리하고, "AI가
판단하는 과정 자체"와 "운영 정책"은 국내/해외 공용으로 둔다.

- **계좌 연관 (시장별 분리)**: 포지션/현금/평가금액, 드로우다운·집중도·상관관계 캡
  (`app/execution/portfolio_risk.py`), 일일 최대손실 브레이커(`app/execution/safety_switch.py`),
  실제 KIS 잔고 조회. 판정은 `portfolio_risk.ticker_market(ticker)`(`.KS` 접미사) 공용 함수
  하나로 통일 — 이 로직을 여러 곳에서 각자 재구현하지 않는다.
- **AI 판단 과정/운영 정책 (공용 유지)**: 멀티에이전트 파이프라인 자체(애널리스트→디베이트→
  트레이더→포트폴리오매니저), `execution_mode`(수동/반자동/완전자동 — "누가 승인하나"는
  계좌가 아니라 운영 정책의 문제), `ai_profile`(AI 성향).
- **예외**: 킬스위치는 계좌 잔고 파생 지표가 아니라 사람이 누르는 수동 비상정지라서 국내/해외
  분리 대상이 아니다 — 계좌 전체를 대상으로 하는 단일 전역 스위치로 유지.

### 4.7 여러 제약이 동시에 걸릴 때: 항상 더 엄격한 쪽이 이긴다 (병합 원칙)

여러 리스크 제약(1% 룰, 포트폴리오 캡, 주문 한도, 승인 임계값, 포지션 비중 등)이 동시에
적용 대상일 때, 이 시스템은 어느 하나가 다른 것보다 완화되는 방향으로 작용하는 걸 절대
허용하지 않는다 — 결과는 언제나 "가장 낮은 허용치"만 살아남는다. 지금 코드는 이 원칙을
두 가지 다른 메커니즘으로 구현하고 있으며, 새 캡/게이트를 추가할 때도 이 둘 중 어느
쪽에 속하는지 먼저 판단해야 한다.

**(a) 수량 캡 — 순차 클램프 (`app/execution/ledger.py::rebalance_to_target`)**: 1%
리스크 룰(4.2절) → 포트폴리오 캡(집중도 → 상관관계 버킷 → 드로우다운 브레이커, 순서대로
`apply_portfolio_caps`) 순으로 적용되며, 각 단계는 **이전 단계가 넘겨준 `target_shares`를
입력으로 받아 그대로 두거나 더 줄이기만** 한다(`app/execution/portfolio_risk.py:338`
독스트링: "Only ever reduces target_shares below what the per-ticker 1% risk cap already
decided"). 여러 캡에 대해 개별적으로 `min()`을 취해 하나로 합치는 것과 수학적으로 동일한
결과이며, 실제로는 하나의 통합 병합 함수가 아니라 순차 실행으로 구현되어 있다. 새 숫자
캡을 추가할 때는 반드시 이 체인 어딘가에 끼워 넣고 "이전 결과 이하로만" 만들도록 작성할
것 — 절대 이전 단계보다 큰 값으로 되돌리지 않는다.

**(b) 통과/차단 게이트 — 첫 거부가 곧 최종 거부 (`app/execution/kis_bridge.py`)**: 2절의
게이트 체인은 숫자를 줄이는 게 아니라 순서대로 검사하다가 하나라도 막히면(FAIL) 그 즉시
전체 주문을 차단하고 이후 게이트는 평가하지 않는다 — "여러 결과를 합치는" 병합이 아니라
"먼저 막힌 것이 곧 최종 결정"인 거부권(veto) 방식이다. 승인 임계값(6번 게이트)이 주문
한도 캡(5번 게이트)보다 크거나 같으면 저장 자체를 거부하는 교차 검증
(`risk_rules._validate_cross_field`, 2절 "대시보드 튜닝 가능 vs 설정 전용")도 이
순서상의 관계가 무의미해지는 걸 막기 위한 것 — 캡이 먼저 막아버리면 그보다 높은
승인 임계값은 죽은 설정이 되기 때문이다.

**아직 없는 것**: 하한값 제약(예: "최소 신뢰도 X% 이상"류, 여러 개면 가장 높은 하한이
이겨야 함)이나 허용/금지 집합(예: "거래 가능 시장 목록", 여러 정책의 교집합/합집합)
형태의 캡은 이 코드베이스에 아직 존재하지 않는다 — 지금 있는 모든 제약은 전부 "상한을
줄이기만 하는" (a) 유형이거나 (b) 유형이라, 위 두 규칙만으로 충분히 커버된다. 나중에
하한값/집합형 제약이 실제로 필요해지면 그때 이 절에 규칙을 추가할 것(지금 선제적으로
만들지 않음).

## 5. 운영 안전 규칙

### 5.1 실전 전환 절차 (project law)

- 실전(`real`) 모드는 **절대 기본값이 될 수 없다** — 항상 명시적 opt-in만 허용된다.
  이 원칙은 그대로이지만 **집행 방식이 두 번 바뀌었다: 2026-08-06(사용자별 모드) →
  2026-08-07(단일 계좌로 되돌림, 로그인 제거)**. 아래는 2026-08-07 이후
  기준이다 — 자세한 배경과 결정 과정은
  `docs/single-user-rollback-guide.md` 참조.

  - **2026-08-06(이제 사실이 아님)**: 계좌 모드가 사용자별이고, 자격증명은 사용자가
    대시보드 화면에서 직접 입력해 암호화 DB(`user_broker_credentials`)에 저장했다.
  - **현재**: 이 배포는 본인 실전계좌 하나만 다룬다. 계좌 모드는 여전히 요청/컨텍스트
    스코프의 값이지만(`app/account_mode.py`, `app/config.py::current_account_mode`) —
    paper/real을 함께 쓰기로 한 결정 때문에 남겨둔 것 — "누구의 모드인가"라는 질문은
    사라졌다. 자격증명(`KIS_APP_KEY_REAL` 등)은 **Jenkins Credentials가 배포 시점에
    주입하는 환경변수**이며, 브라우저로 입력하거나 DB에 저장하는 경로 자체가 없다
    (`app.config.Settings.kis_credentials`).
  - **더 중요한 변화: 로그인이 없다.** `PUT /api/account-mode`(실전↔모의 전환)에
    세션 확인이 없다 — 이 서버에 접근할 수 있는 사람은(= SSH 터널을 가진 사람은) 누구나
    호출할 수 있다. 이것은 §5.1이 원래 막으려던 바로 그 상황(무인증 상태에서 API로
    실전 전환 가능)이 **의도적으로, 알고 되살아난 것**이다 — 아래 opt-in 층 4가지가
    "아무나 실전 자격증명 없이 실전에 진입"은 계속 막고, 대신 "이미 자격증명이 있는
    상태에서 모드만 전환하는 행위"는 감사로그(`account_mode_switches`,
    `app/account_mode.py::record_mode_switch`)로 사후 추적 가능하게 하는 것으로
    상쇄한다. 이 로그는 예방이 아니라 사후 대조용이라는 한계가 있다 —
    `client_host`는 SSH 터널을 거치면 전부 `127.0.0.1`로 보이므로, 실제 행위자
    특정은 서버 SSH 접속 로그와 시각을 대조해야 한다.

  **opt-in은 무엇이 보장하는가** — 다음 네 겹이다:
  1. **실전 자격증명이 Jenkins Credentials를 통해 env로 존재해야 한다.**
     `KIS_APP_KEY_REAL`/`KIS_APP_SECRET_REAL`/`KIS_ACCOUNT_8_REAL`이 모두 채워져
     있지 않으면 `Settings.configured_kis_modes()`에 `"real"`이 잡히지 않고,
     `resolve_preference`는 실전으로 해석하지 않는다 — 배포 기본값이나 남의 키로
     실전에 들어갈 경로가 없다. 이 값을 넣을 수 있는 사람은 Jenkins Credentials
     관리 권한을 가진 운영자뿐이다 — **opt-in의 실질적 주체가 "그 계정을 등록한
     사용자 본인"에서 "배포를 구성하는 운영자"로 옮겨갔다.**
  2. **부팅 시 KIS에 실제 토큰 발급으로 검증한다**(`app/credentials/validation.py`,
     호출 지점은 `app/web/state.py::_verify_configured_kis_credentials`). 예전처럼
     "저장 전 검증"이 아니라 "이미 배포된 값의 부팅 시 확인"으로 바뀌었다 — 검증
     실패는 부팅을 막지 않고 경고 로그만 남긴다(대시보드 자체는 KIS 문제와 무관하게
     떠야 하므로).
  3. **실전 주문 한도 4개는 설정되지 않으면 실전 주문을 평가조차 하지 못한다** —
     `KIS_MAX_ORDER_KRW_REAL`, `KIS_MAX_ORDER_USD_REAL`,
     `KIS_APPROVAL_THRESHOLD_*_REAL`. 코드 기본값은 `None`이고, 실전 모드에서 이 값을
     읽으면 `Settings._mode_scoped_money`가 `RuntimeError`를 던진다(`assert`가 아니라
     실제 예외 — `python -O`에서 제거되지 않도록). 즉 값을 의도적으로 넣기 전까지
     fail-closed다. 이 넷이 없는 상태(예: 실전 키는 있지만 한도 미설정)에서는 대시보드
     상단에 비차단 경고 배너가 뜬다(`app/web/static/js/onboarding.js::
     maybeShowRealLimitsBanner`).
  4. **실행 마스터 스위치는 여전히 설정 전용이다** — `KIS_EXECUTION_ENABLED`,
     `KIS_EXECUTION_DRY_RUN`(위 "대시보드 튜닝 가능 vs 설정 전용" 절 참조).

  즉 모드 선택이 무인증 API로 노출된 것은 §5.1 정신의 완화이지 무효화가 아니다 —
  실전에 도달하려면 여전히 위 네 겹을 모두 통과해야 하고(그중 1·2·3은 이제
  Jenkins/배포 단계에서만 통과 가능), 전환 행위 자체는 감사 대상이다. "코드가
  §5.1을 그냥 위반한다"고 단순화하지 말 것 — 무엇이 트레이드오프됐고 무엇으로
  상쇄했는지가 핵심이다.
- 실주문 전송(즉 `KIS_EXECUTION_DRY_RUN=false`로 dry-run 해제)은 **매번** 사용자의
  명시적 확인을 요구하며, 한 번의 승인이 영구 승인으로 취급되지 않는다. 승인 임계값
  (`kis_approval_*`) 로직도 마찬가지로 개별 주문 단위 승인이지, 정책 전환이 아니다.
- 실주문 재개는 2026-07-16 사용자가 명시적으로 보류(hold)한 상태다 — 이후 사용자의
  명시적 지시 없이 재개해서는 안 된다.
- 실전 전환에 필요한 최소 변경(2026-08-07 갱신): (1) 운영자가 Jenkins Credentials에
  `KIS_APP_KEY_REAL`/`KIS_APP_SECRET_REAL`/`KIS_ACCOUNT_8_REAL`을 등록하고 재배포,
  (2) `KIS_MAX_ORDER_*_REAL` / `KIS_APPROVAL_THRESHOLD_*_REAL`을 0이 아닌 실제 계좌
  규모에 맞는 값으로 설정, (3) `kis_execution_enabled=true` 및
  `kis_execution_dry_run=false`. 어느 하나만으로는 실주문이 나가지 않으며, 셋 다
  대시보드가 아니라 배포 설정(Jenkins/env) 편집이다 — 즉 실전 진입 경로가 다시
  "설정 파일을 사람이 편집해야 한다"는 형태로 돌아갔다(단, DB 대신 Jenkins가
  그 사람의 자리를 대신한다).

### 5.2 DB 장애 시 동작 (`app/memory/db.py`, 공유 Postgres 저하 대응)

2026-07-20 실제 장애(하드 킬된 클라이언트가 커넥션을 붙든 채 방치되어 다른 모든 스키마
부트스트랩을 무기한 블록시킨 사고)를 계기로 강화된 방어 로직:

- **서버 강제 세션 타임아웃** — 모든 커넥션(부트스트랩용 단발 커넥션 포함)에
  `idle_in_transaction_session_timeout`, `statement_timeout`, `idle_session_timeout`을
  서버 설정으로 강제 적용. 클라이언트가 하드킬되면 정리 코드를 실행할 수 없으므로, 좀비
  세션을 걷어내는 책임을 서버 쪽으로 옮김.
- **이중 타임아웃** — `asyncpg.connect`/`create_pool`의 `timeout`/`command_timeout`만으로는
  "TCP는 받아주지만 이후 전혀 응답 없음" 상황을 못 잡는 경우가 실제로 있었으므로,
  `get_pool()` 바깥에 `asyncio.wait_for`로 하드 데드라인을 한 번 더 씌움(내부 타임아웃 대비
  +1초 여유).
- **Fail-fast 백오프** — 최근 연결 실패 후 `database_failure_backoff_seconds` 동안은 재연결
  시도 자체를 생략하고 즉시 실패 반환. 장애 중 모든 DB 접근 호출이 각자 풀 커넥션 타임아웃을
  처음부터 다 기다리는 대신, 이미 죽은 걸 아는 상태에서는 곧바로 저하 경로로 빠짐. 모든
  호출자는 이 예외를 이미 잡아서 기본값으로 저하되도록 되어 있으므로(`ledger.get_position`,
  `risk_rules.load` 등), 더 빨리 실패시키는 것이 안전성을 해치지 않고 응답성만 개선함.
- **더블체크 락 스키마 부트스트랩** — `safety_switch.py`, `pending_orders.py`,
  `risk_rules.py` 모두 동일한 `_schema_ready` 플래그 + `asyncio.Lock` 패턴을 공유해,
  동시 다중 호출이 중복으로 `CREATE TABLE`을 실행하거나 풀을 여러 개 만들지 않도록 함.
- **조용한 저하(quiet degrade)** — 리스크 룰 스토어가 응답 없으면 `load()`는 캐시를
  비워두고(=설정 기본값 사용) `save()`는 `RuntimeError`로 실패를 알려 대시보드와 실제
  게이트가 서로 다른 값을 보는 상황(divergence)을 만들지 않는다. 킬스위치/일일손실
  상태 조회 실패는 반대로 **fail-closed**(위 1.3절) — 저하 대상이 안전 게이트 자체이므로
  "모르면 계속 진행"이 아니라 "모르면 막는다"가 기본.
