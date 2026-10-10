# 단타(Scalping) 매매 설계 문서

> **상태(2026-10-02 문서 정리 메모)**: **보류 중인 설계 문서**(폐기·완료 표시가 아니다). 단타(스캘핑) 백테스트 인프라 일부(`app/backtest/scalping_harness.py`, `fill_simulation.py`, `minute_candles.py`, `scripts/*minute*`)는 이 문서를 근거로 코드가 존재하지만, 운영 시스템은 스윙 전략이며 단타는 "살아있는 신호 없음"으로 보류돼 있다(`handoff/knowledge/open-issues.md` D절). 코드 주석이 이 경로를 인용하므로 파일은 이동하지 않았다.

본 문서는 기존 스윙/일단위 멀티에이전트 파이프라인(`docs/architecture.md`)과 별개로,
같은 저장소에 단타(scalping) 매매 모듈을 추가하기 위한 설계다. 2026-07-27 논의 기준으로
작성되었으며, 반응속도 목표치 등 일부 항목은 아직 확정되지 않은 채로 열어둔다 — 이 문서는
"결정된 것"과 "아직 정할 것"을 구분해서 기록하는 것을 목적으로 한다.

## 1. 배경 + 결정된 전제

기존 `app/live/auto_trade_loop.py`는 30분(`auto_trade_interval_minutes`) 주기로 후보를 추린 뒤,
종목당 ~20회 LLM 호출이 드는 풀 파이프라인(`app/orchestrator/pipeline.py`, Bull/Bear 디베이트 →
Research Manager → Trader → Risk 디베이트 → Portfolio Manager)을 그대로 태운다. `app/config.py:232`
주석도 "이 시스템은 multi-day/week 의사결정 주기용"이라고 명시한다 — 즉 지금 파이프라인은
스윙/일단위 전용으로 설계됐고, 단타는 이 설계 범위 밖의 신규 확장이다.

**결정된 것:**
- 단타를 별도 프로젝트로 분리하지 않고, 같은 저장소에 새 모듈로 추가한다(기존 브로커/RiskManager/
  웹소켓 자산 재사용).
- 기존 멀티에이전트 파이프라인은 제거하지 않는다. 역할을 "Deep Analysis"(느림, 심층분석)로
  재정의하고, 단타는 이 결과를 사전에 캐싱된 형태로 참조하는 "Trading Decision" 계층을 별도로 둔다.
- RiskManager(안전장치)는 단타 전용으로 새로 만들지 않는다 — 기존 `kis_bridge.py` 관문을
  그대로 통과시켜 재사용한다.

**아직 정해지지 않은 것 (설계하면서 확정 예정):**
- 목표 반응속도 (초 단위인지 분 단위인지에 따라 Entry Engine의 데이터 소스/로직이 크게 달라짐)
- Entry Engine의 실제 진입/청산 로직(어떤 시그널로 진입할지)
- KIS TPS(초당 호출 제한) 실측치 — 실전 약 20건/초, 모의 약 2건/초는 추정치이며 검증 필요

## 2. 전체 흐름 (Deep Analysis / Trading Decision 분리)

```
                    [Deep Analysis — 기존 자산, 역할만 재정의]
trade allowlist/스캔  ──▶  app/orchestrator/pipeline.py (변경 없음)
                     Bull/Bear 디베이트 → Research Manager → Trader
                     → Risk 디베이트 → Portfolio Manager
                            │
                            ▼
                     app/memory/store.py (Knowledge Base, 기존 자산)
                     get_recent_entries(ticker, limit=1) 로 최신 rating 조회

                    [Trading Decision — 신규, 단타 전용]
실시간 웹소켓 피드 ──▶  Scanner(모멘텀/변동성 후보 필터, 기존 screening.py 패턴 재사용)
(get_active_feed)          │
                            ▼
                     Knowledge Base 조회 (Deep Analysis 결과 참조, LLM 재호출 없음)
                            │
                            ▼
                     Entry Engine (신규 — 규칙 기반, LLM 미개입)
                            │
                            ▼
                     kis_bridge.mirror_trade_to_kis / place_manual_order (기존 관문, 변경 없음)
                            │
                            ▼
                     KIS 모의/실전 서버
```

핵심 원칙: **Entry Engine은 LLM을 실시간 경로에 두지 않는다.** Deep Analysis가 미리 만들어 둔
점수/등급을 참조만 하고, 실제 진입/청산 판단은 규칙 기반 로직(Rule Engine)으로 수행한다. 이는
다른 AI가 제안한 "Deep Analysis는 생각, Trading Decision은 빠른 실행" 원칙과 일치하며, 기존
코드가 이미 증명한 "quick questions stay quick"(`app/chat/router.py` 설계 목표, `docs/architecture.md`
62행)과 같은 방향이다.

## 3. 재사용 vs 신규 — 모듈별 정리

| 구성요소 | 재사용/신규 | 근거 |
|---|---|---|
| 실시간 시세(호가/체결가/시장상태) | 재사용 | `app/live/kis_market_status.py`의 `get_active_feed()` — 새 웹소켓 불필요 |
| 브로커 클라이언트 | 재사용 | `app.brokers.get_broker("kis")` 프로세스 전역 캐시 인스턴스 |
| RiskManager(안전장치 전체) | 재사용 | `app/execution/kis_bridge.py`의 `place_manual_order`/`mirror_trade_to_kis` — 킬스위치/일일손실/서킷브레이커/비중캡 전부 자동 적용 |
| Knowledge Base 조회 | 재사용(제한적) | `app/memory/store.py`의 `get_recent_entries(ticker, limit=1)` — 단, 전용 "Confidence 수치 조회" 함수는 없어 신규 헬퍼 필요 |
| Scanner(후보 필터링) | 재사용 패턴 참고, 신규 구현 | `app/chat/screening.py`의 `top_candidates`/`rank` 패턴을 단타용 변동성/거래량 급증 기준으로 재구성 |
| Entry Engine | 신규 | 규칙 기반 실시간 진입/청산 로직 — 기존에 없음 |
| 매매 루프 | 신규 | `app/live/auto_trade_loop.py` 옆에 `app/live/scalping_loop.py` 형태로 병렬 추가, 같은 `get_broker("kis")` 재사용 |
| 체결강도 계산 | 신규 | KIS에 전용 API 없음 — 호가/체결 데이터로 직접 계산 필요 |

## 4. Knowledge Base 연동 방식

Deep Analysis 결과를 단타에서 재사용하는 흐름:

1. Deep Analysis(`run_pipeline`)가 평소처럼 실행되어 `app/memory/store.py`에 결정을 기록한다
   (기존 동작 그대로, 변경 없음).
2. Entry Engine은 진입 후보 종목에 대해 `get_recent_entries(ticker, limit=1)`로 최신
   `DecisionEntry`(rating, decision_text, created_at 등)를 가볍게 조회한다 — 임베딩 계산이
   필요한 `get_similar_context`보다 훨씬 저비용.
3. **미해결**: `DecisionEntry`에는 confidence 수치 컬럼이 없고 문자열 rating만 있다. 단타
   Entry Engine이 "AI Score 91/100" 같은 정량 점수를 참조하려면 이 스키마를 확장하거나,
   당분간은 rating(Buy/Overweight/Hold/Underweight/Sell)만으로 방향성 필터(예: Sell/Underweight
   종목은 단타 매수 후보에서 배제)로 사용하는 축소된 범위로 시작하는 것이 현실적이다.

## 5. RiskManager 재사용 — 새로 만들 것이 없다

`docs/safety-and-risk.md`에 정리된 단일 관문 원칙(`kis_bridge.py`)은 단타 주문에도 예외 없이
그대로 적용된다:

- 킬스위치, 일일 최대손실 서킷브레이커, 마스터 스위치, 레버리지/인버스 필터, 주문금액 캡,
  승인 임계값, 시장 서킷브레이커, VI, 현금충분성, 포지션비중 캡, dry-run — 전부 기존 게이트
  그대로 통과.
- 단타 전용으로 새로 만들 리스크 로직은 없다. 다만 단타는 회전율이 높아 **승인 임계값
  (`kis_approval_*`) 게이트가 매 주문마다 걸리면 사실상 실행이 불가능**해질 수 있으므로,
  단타 대상 종목/계좌에 대한 임계값 설정을 별도로 검토할 필요는 있다(게이트 자체를 우회하는
  것이 아니라, 기존 대시보드 튜닝 기능(`app/execution/risk_rules.py`)으로 값만 조정하는 방식).
- 주문 정정/취소(`VTTC0013U`)가 "가능수량 조회 → 정정취소" 2회 왕복을 요구하는 점은 RiskManager
  범위가 아니라 Entry Engine의 지연시간 예산에 포함되어야 할 제약이다.

## 6. 기술적 제약 (2026-07-27 조사 결과 요약)

자세한 내용은 [[scalping-feasibility-notes]] 메모리 참조. 요약:

- 실시간 호가(H0STASP0)·체결가(H0STCNT0)·시장상태(H0STMKO0)는 모의계좌 웹소켓에서 실수신
  검증 완료 — 단타에 필요한 핵심 데이터는 이미 확보됨.
- 모의계좌에서 막히는 건 내 주문 체결통보(H0STCNI9)뿐 — ACK만 오고 체결 프레임 미수신
  (모의투자 플랫폼 자체 제약). 시장 데이터와 무관하므로 REST 잔고/미체결 조회로 대체 가능.
- 체결강도 전용 API 없음 — 호가/체결 데이터로 직접 계산 필요.
- 분봉조회(FHKST03010200)는 1회 호출 최대 30건, 당일만 — 초단타 시 REST 폴링 주기가 병목
  가능성.
- 정정/취소는 모의 지원되나 2회 왕복 권장 — 지연 누적 요인.
- TPS 실측치 미확인.

## 7. 미해결 항목 (다음 단계에서 확정 필요)

1. **목표 반응속도** — 2026-07-27 결정: 미리 값을 정하지 않고, **백테스트로 다양한 반응속도의
   수익성을 학습한 뒤 결정**하는 방식을 택함. 단, 이를 위해서는 아래 선행 과제가 먼저 필요하다는
   것이 조사로 확인됨:
   - `app/backtest/harness.py`는 현재 **완전히 일봉(daily) 전용**이다. `select_trading_dates`가
     `date` 객체 단위로만 동작하고 시/분/초 개념이 코드에 없다.
   - 분봉/틱 데이터 조회 경로가 없다 — KIS는 분봉 TR 연동 자체가 안 되어 있고, yfinance의
     `get_intraday_series`(1분봉)는 라이브 폴러 전용이며 과거 구간 조회는 불가(yfinance 분봉은
     최근 ~7일 제한이라 장기 백테스트 부적합).
   - 슬리피지/체결지연 시뮬레이션 모델이 없다 — "N초 늦게 진입하면 얼마나 불리해지는지"를
     계산할 방법이 없어, 반응속도 비교 실험 자체가 성립하지 않는다.
   - 실험추적/병렬실행 골격(`app/backtest/experiments.py`, `app/backtest/parallel.py`)은
     granularity와 무관하게 재사용 가능해 보임(`cadence_days` → `cadence_seconds`류 확장 정도).

   **결론: 반응속도를 백테스트로 학습하려면, 분봉/틱 데이터 조회 + 슬리피지/체결지연 모델
   + 시간 단위 시뮬레이션 루프를 먼저 구축해야 한다.** 구체 계획은
   `docs/scalping-backtest-infra-plan.md` 참조 — 과거 분봉 조회 전용 TR(`FHKST03010230`,
   최대 1년치 보관)이 추가로 확인되어 "분봉 데이터가 아예 없다"던 판단은 "있지만 모의계좌
   지원 여부 실측 검증 + 수집/캐시 과정이 필요하다"로 수정됨.
2. **Entry Engine의 실제 진입/청산 시그널** — 이 문서는 "LLM 미개입, 규칙 기반"이라는 방향만
   정했고, 구체적 룰(어떤 조건에서 진입/청산할지)은 아직 없다.
3. **TPS 실측** — KIS 개발자센터 공지 기준 추정치(실전 ~20건/초, 모의 ~2건/초)를 실제 호출로
   검증 필요. 단타는 스윙보다 API 호출 밀도가 훨씬 높아, 이 값이 Entry Engine의 동시 종목 수
   상한을 결정한다.
4. **Confidence 수치 스키마** — `app/memory/store.py`의 `DecisionEntry`에 정량 점수 필드가
   없음. 확장할지, rating 문자열만으로 축소된 범위로 시작할지 결정 필요.
5. **체결강도 계산 방식** — 어떤 필드 조합으로 근사할지 구체 로직 미정.
6. **승인 임계값 설정** — 단타 특성(고회전)에 맞는 별도 값을 둘지, 기존 값을 그대로 쓸지.
