# Archive

이 폴더의 문서들은 **프로젝트 초기 기획 단계에서 작성된 이상적 설계안**이다. 실제 코드가
아니라 상상한 시스템을 기준으로 쓰여졌고, 지금의 코드베이스와 클래스명·구조가 다른 부분이
많다 (예: `RiskManager`/`OrderService`/`FakeBroker` 같은 이름은 실제 코드에 없음 —
실제로는 `app/execution/kis_bridge.py`의 함수들과 `app/domain/ports/`의 `Protocol`이 그
역할을 한다).

**현재 참조용으로 쓰지 말 것.** 실제 코드가 지금 하는 일은 상위 `docs/` 폴더의 현행 기술 문서
(`architecture.md`, `safety-and-risk.md`, `development-conventions.md`,
`db-password-rotation-runbook.md`)를 따른다 — 이 문서들은 실제 코드를 읽고 추출한 것이다.
(2026-10-02 정정: 예전에는 `status-and-decisions.md`도 이 목록에 있었으나 그 문서는 7월 기준 과거 기록이라 뺐다.)

2026-07-20에 옮김 (`docs/status-and-decisions.md`의 "문서화 방침" 결정 참조).

---

## 2026-10-02 추가

**옮기지 않은 과거·보류 문서**(코드 주석·README가 `docs/<이름>.md` 경로로 인용하므로 제자리에 두고 각 문서 첫머리에 상태 안내만 넣었다): `status-and-decisions.md`, `single-user-rollback-guide.md`(현재 구조의 근거 — 로그인 재도입 시 먼저 읽는다), `multi-user-credentials-plan.md`, `domestic-overseas-separation-plan.md`, `scalping-design.md`, `scalping-backtest-infra-plan.md`, `data-reliability-check-plan.md`.
