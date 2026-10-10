# KIS 조건부 자동매매 기능 개발 지시 프롬프트 (for Claude Code)

당신은 시니어 Python 백엔드/AI 시스템 개발자입니다. `invest-project` 리포지토리(한국투자증권 KIS 모의투자 연동 + 멀티에이전트 트레이딩 파이프라인 + 실시간 대시보드가 **이미 구축된** 프로젝트)를 이어받아, **사용자 설정형 조건부 자동매매(conditional auto-trading) 기능**을 설계·구현합니다.

## Objective

기존 구조를 재사용하여, 사용자가 **리스크 규칙을 UI/설정으로 켜고 끄고 조정**할 수 있고, **조건 충족 시 자동으로 매매를 집행**하는 기능을 설계부터 구현까지 한 번에 만듭니다. **새 아키텍처를 새로 만들지 말고 기존 모듈을 확장**합니다.

## Context (현재 실제 상태 — 반드시 재사용)

- **LLM**: Anthropic이 아니라 **vLLM(Qwen)** 을 OpenAI 호환 엔드포인트로 사용 (`app/llm/client.py`, `settings.llm_base_url`). 역할별 reasoning 토글 방식.
- **이미 구현되어 있음 (재구현 금지, 그대로 확장)**:
  - `app/brokers/kis.py` — KIS 모의 주문 **전 기능**: 매수/매도(`submit_order`/`submit_overseas_order`), **취소·정정**(`cancel_order`/`revise_order`), 잔고·매수가능·체결내역·시세·일/주/월/년봉·분봉·지수·업종·수급·재무·투자의견·공매도·뉴스·배당. `app/brokers/kis_ws.py` — 실시간 웹소켓(장운영/CB·호가·체결가·**체결통보** H0STCNI9, AES 복호).
  - `app/execution/kis_bridge.py` — **주문 안전 게이트 단일 초크포인트**: 라우팅·마스터스위치(`KIS_EXECUTION_ENABLED`)·주문금액 상한(`KIS_MAX_ORDER_KRW/USD`)·dry-run(`KIS_EXECUTION_DRY_RUN`)·매도수량 상한. `mirror_trade_to_kis`(자동경로)·`place_manual_order`(수동). 원장 기록은 호출자 책임(`app/execution/ledger.py`).
  - `app/agents/*` + `app/dataflows/*` — 애널리스트(market/fundamentals/news/sentiment)·불베어 토론·research manager·trader·리스크 토론·portfolio manager. `app/orchestrator/pipeline.py`가 전체 파이프라인.
  - `app/chat/router.py` — 대화형 Q&A(시세·뉴스·스크리닝·심층분석·계좌질문).
  - `app/web/app.py` + `static/index.html` — FastAPI 대시보드(주문창·순위·캔들·호가창·한국데이터·AI분석 버튼·계좌요약·토글) + TTL 캐시.
  - `app/daemon.py` — 대시보드 앱 실행 겸 백그라운드 루프(minute poller + stop-loss loop). 파이프라인은 여기서 자동 실행 안 함(현재는 채팅 deep_dive·`python -m app.main`으로만).
- **설정**: pydantic settings, `.env.local`(깃 추적) + `.env.kis.local`(모의키, gitignore). 모의투자 계좌. dry-run 기본.
- **리스크 규칙**(현재 코드 상수/설정, **사용자별 on/off·조정 기능은 미완성** ← 이번 작업 핵심): 종목당 최대 주문금액, 매도수량 상한. 확장 대상: 종목당 최대 포지션 비중(%), 승인 임계금액.

## 개발 참고 우선순위 (중요)

- **KIS API 관련 구현(신규 TR 연동, 기존 TR 확장, 요청/응답 파싱, 인증, 웹소켓 등)은 반드시 `open-trading-api`(로컬 경로: `D:\Investment_Project\open-trading-api`, 특히 `examples_llm/`·`examples_user/` 하위 도메인별 예시 코드와 `kis_auth.py`)를 최우선 레퍼런스로 확인하고, 그 구현 패턴(요청 파라미터, 헤더, TR ID, 페이징/연속조회 처리 방식 등)을 그대로 따른다.** 자체 추측으로 API 스펙을 재구성하지 않는다. 기존 `app/brokers/kis.py`와 예시 코드가 다르면 예시 코드를 기준으로 검증 후 반영한다.
- **AI/vLLM과 무관한 기본 기능**(주문 UX 흐름, 화면 구성/레이아웃, 계좌·시세·호가 표시 방식, 리스크 설정 패널 등 대시보드 UI 전반)은 **토스증권(Toss Securities) 앱의 UX/기능 구성을 기준**으로 설계한다. AI 분석·에이전트 파이프라인과 직접 관련된 부분(리서치/토론/리포트 등)은 이 기준에서 제외.

## Process

### 1단계 — 기존 코드 확인

`kis_bridge.py`의 게이트 구조, `daemon.py`/`pipeline.py`의 실행 흐름, `settings`(config.py), 원장(ledger)·의사결정 저장(memory) 방식을 먼저 파악한다. **이미 있는 게이트·주문·원장은 그대로 재사용**한다. KIS API를 새로 다루거나 확장하는 부분이 있다면 이 단계에서 `open-trading-api/examples_llm`(또는 `examples_user`) 내 해당 TR 예시를 먼저 열어 실제 요청/응답 구조를 확인한다.

### 2단계 — 설계 (기존 구조에 매핑)

- **리스크 규칙 확장** → `kis_bridge.py`에 게이트 추가(종목당 포지션 비중 상한, 승인 임계금액 등), 각 규칙 **on/off + 값**을 `settings` 및 **런타임 설정 저장소**(Postgres 또는 기존 저장소)에서 읽도록. LLM 판단과 **무관하게 코드레벨에서 물리 차단**.
- **조건부 자동매매** → `app/execution/`에 신규(예: `auto_trader.py`): `auto_trade_enabled` 토글 하에, 스케줄 또는 조건(예: 파이프라인 rating=Buy + 리스크 규칙 통과) 충족 시 `place_manual_order`/mirror 경로로 집행. `daemon.py`의 루프 또는 별도 스케줄러에 연결.
- **사용자 설정 UI** → 대시보드에 리스크 규칙·자동매매 on/off·값 조정 패널 + 설정 조회/저장 엔드포인트.
- 설계는 짧게 요약(주석/README)한 뒤 승인 대기 없이 바로 구현으로 진행. 주요 결정은 주석에 근거를 남긴다.

### 3단계 — 구현

- 모듈별 순차 구현 + 각 모듈 최소 단위테스트. **`kis_bridge` 게이트는 반드시 테스트**(규칙 위반 시 정확히 차단되는지 검증).
- 자동매매는 **기본 dry-run/모의**로 동작. 실제 모의 주문 전송은 명시적 설정(`KIS_EXECUTION_DRY_RUN=false`)에서만. 실거래(실전투자 키)는 범위 밖.
- 각 KIS TR은 **모의 지원 여부 probe 우선**(계좌·주문관리·예약·HTS조건 계열은 모의 미지원 전례 있음).

### 4단계 — 문서화·마무리

- README에 자동매매 on/off·리스크 규칙 설정법·모의/실거래 전환·검증법 정리.
- 마지막에 **완료/미완(시장·vLLM 검증 대기)/TODO** 한국어 요약 리포트.

## Output Format

- 기존 구조에 통합된 실행 가능한 Python 코드
- 각 모듈 최소 단위테스트(특히 리스크 게이트)
- `.env.example` 갱신(신규 설정), 하드코딩 금지
- README 갱신(설정/실행/검증)
- 한국어 구현 요약 리포트(완료/미완/TODO)

## Constraints

- API 키·시크릿·계좌정보는 **절대 하드코딩 금지**(env/gitignore).
- 주문 실행은 **기본 모의·dry-run**, 실제 전송은 명시적 설정에서만. 실거래는 범위 밖.
- **리스크 규칙은 LLM 판단과 무관하게 `kis_bridge` 코드레벨에서 최종 물리 차단**.
- **기존 파일/구조를 재사용**하고 평행 아키텍처를 새로 만들지 않는다. 불필요한 전체 리팩터 금지.
- LLM은 **vLLM(OpenAI 호환)** 기준(`app/llm/client.py`). Anthropic 가정 금지.
- **KIS API 구현/확장 시 `open-trading-api` 예시 코드를 최우선으로 참고**하고 임의로 스펙을 추측하지 않는다.
- **AI와 무관한 기본 기능은 토스증권 기준**으로 UX/기능을 설계한다.
- 표준화되지 않은 요구는 임의 확장 말고 주석/README에 "가정(assumption)"으로 명시.

## Success Criteria

- 기존 코드에 자연스럽게 통합되어 실행된다.
- 사용자가 리스크 규칙을 켜고/끄고/조정할 수 있고, 그 값이 실제 주문 게이트에 반영된다.
- 조건부 자동매매가 `auto_trade_enabled` 토글 하에 동작하고, off면 절대 주문하지 않는다.
- **리스크 규칙 위반 시나리오 테스트 시 100% 차단**.
- 모의·dry-run에서 전체 플로우(조건 판단 → 게이트 → 집행/로깅)가 실제 자금 없이 검증 가능.
- README만 보고 다른 개발자가 바로 실행/설정 가능.

## Optional

- 작업 시작 전, 재사용할 부분과 새로 만들 부분을 **짧게 정리해 먼저 보여준 뒤** 진행.
- 중요한 설계 결정(자동매매 트리거 방식, 설정 저장소, 스케줄 주기)은 근거를 README/주석에 남길 것.
- 참고: 실시간 검증(웹소켓 렌더·체결통보·수동주문 실체결)은 **국내장 개장 시간**, AI/에이전트 검증은 **vLLM 기동 시** 필요.
