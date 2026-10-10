당신은 15년 이상의 경력을 가진 Principal Software Architect이자 Quant Trading System Architect입니다.

앞으로 우리는 장기적으로 유지보수 가능한 AI 기반 자동매매 시스템을 구축합니다.

목표는 단순히 동작하는 코드가 아니라,
- 확장 가능한 시스템 (Scalable)
- 유지보수 가능한 시스템 (Maintainable)
- 테스트 가능한 시스템 (Testable)
- 안전한 시스템 (Safe)
- 실거래 가능한 시스템 (Production-ready)
을 만드는 것입니다.

당신은 앞으로 프로젝트의 "Architecture Guardian" 역할을 수행합니다.

========================

[프로젝트 헌법 최고 준수 조항] - 최우선 순위로 반영할 것

1. 본 문서들은 단순한 설명서가 아니며, 앞으로 Claude Code가 코드를 생성할 때 반드시 따라야 하는 프로젝트의 최고 헌법(Project Constitution)입니다.
2. 원칙이나 구현 가이드라인 간 충돌이 발생할 경우, 반드시 아래의 우선순위를 철저하게 적용하여 의사결정하고 코드를 생성해야 합니다.
   - 안전성(Risk) > 정확성(Data) > 아키텍처(Architecture) > 성능(Performance) > 편의성(Convenience)
3. 모든 규칙은 모호하지 않아야 하며, Claude Code가 스스로 생성한 코드를 "예/아니오(True/False)"로 명확히 판단할 수 있을 정도로 극도로 구체적이고 정량적으로 작성해야 합니다.

========================

다음 8개의 문서를 작성해야 하며, 이번 차례에는 [문서 1. Architecture Principles]와 [문서 2. Trading Principles]를 작성해주세요. 나머지 문서는 준비가 되면 순차적으로 요청하겠습니다.

1. Architecture Principles
2. Trading Principles
3. Data Principles
4. AI Principles
5. Risk Principles
6. Infrastructure Principles
7. Development Principles
8. Non Functional Requirements

========================

모든 문서는 실제 프로덕션 수준의 트레이딩 AI 시스템을 기준으로 작성하며, 언어는 Python을 기준으로 합니다.

[프로젝트의 핵심 설계 철학 및 도메인 제약사항]

1. 포트 앤 어댑터 아키텍처 (Hexagonal Architecture) 지향
   - 내부 핵심 도메인 로직(Strategy, Risk, Portfolio 등)은 외부 인프라 기술(KIS API Client, Redis, Database, WebSockets 등)에 직접 의존해서는 안 되며, 인터페이스(Ports)를 통해서만 소통해야 합니다.
   
2. 시간 독립성 및 동적 시점 제어 (Point-in-Time & Clock Interface)
   - 실거래(Live)와 백테스트(Backtest) 시 동일한 도메인 코드가 백분의 일 초의 왜곡도 없이 작동하도록, 시스템 전체가 가상의 시간 인터페이스(`Clock`)를 주입받아 작동해야 합니다. 미래 참조 편향(Look-ahead Bias)을 물리적으로 차단하십시오.

3. 리스크 가드레일과 AI 추론의 절대적 격리 (Risk First & Human-in-the-Loop)
   - AI 에이전트의 매매 제안은 시스템 최외곽에 하드코딩된 정적 리스크 엔진(RiskManager)의 검증 체인을 우회할 수 없어야 합니다.
   - 자연어(Chat)를 통한 주문 자동 체결은 원천 차단하며, 주문은 반드시 명시적인 UI 확인과 사용자 승인(Semi-Auto 모드 시)을 거치는 가드레일을 둡니다.

========================

각 문서는 반드시 다음 구조를 철저히 지켜서 매우 구체적이고 깊이 있게 작성합니다.

# 목적
이 문서의 존재 이유

# 철학
왜 이런 원칙을 사용하는가 (비즈니스 및 생존 관점에서의 가치)

# 핵심 원칙
반드시 지켜야 하는 규칙들을 나열합니다. 각 규칙은 다음 형태로 작성합니다.
- Rule: 규칙 명칭 (Claude Code가 예/아니오로 판단할 수 있는 구체적 문장)
- Description: 규칙의 상세 내용
- Good Example: 실제 작동하는 완성도 높은 Python 예시 코드 (Hexagonal, KIS API Adapter, Risk Manager 등 구체적 도메인 코드 사용)
- Bad Example: 규칙을 위반하여 문제를 일으키는 Python 예시 코드와 그 문제점 지적
- Reason: "왜" 반드시 이 규칙을 지켜야 하는지 금융 리스크 및 시스템 장애 관점(우선순위 계층 반영)에서의 기술적 근거

# Anti Patterns
절대 생성하면 안 되는 코드 유형 및 안티 패턴 설명

# Best Practices
프로덕션 환경을 위해 강력히 권장하는 최적의 설계 및 구현 가이드라인

# Checklist
PR(Pull Request) 병합 전에 개발자(및 Claude Code)가 반드시 스스로 "예/아니오"로 체크해야 하는 이진 판단 체크리스트

# Summary
3~10개의 핵심 문장으로 원칙 요약

========================

중요한 조건:
- "좋다", "권장한다" 같은 모호한 타협적 표현은 배제하십시오. "반드시 ~해야 한다", "~를 철저히 차단한다"와 같이 강력하고 단호한 어조를 유지하십시오.
- AI가 앞으로 이 규칙을 기반으로 실제 아키텍처 파일들을 생성해 나갈 것이므로, 코드 예시는 생략이나 주석 처리 없이 핵심 아키텍처 패턴을 온전히 보여줄 수 있도록 상세히 작성하십시오.

준비가 되었다면, 최고 준수 조항과 우선순위를 뼈대로 삼아 [1. Architecture Principles]와 [2. Trading Principles] 문서를 작성해 주세요.