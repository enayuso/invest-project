
# Trading AI Architecture Principles

이 프로젝트는 장기 유지보수와 확장성을 목표로 한다.

모든 코드는 아래 원칙을 반드시 따른다.

---

# 1. Single Responsibility Principle (SRP)

모든 모듈은 하나의 변경 이유만 가져야 한다.

변경 이유가 두 개 이상이면 분리한다.

예시

✔ Strategy

- 매매 판단

✔ Broker

- 증권사 API 통신

✔ Repository

- 데이터 저장

✔ Notifier

- 알림

✘ Broker에서 DB 저장
✘ Strategy에서 주문 실행
✘ Repository에서 API 호출

---

# 2. Dependency Rule

의존성은 반드시 안쪽으로만 향한다.

```
Strategy
      ↓
Application
      ↓
Domain Ports
      ↑
Broker Adapter
      ↓
Broker API
```

Domain은 다음을 절대 import 하지 않는다.

- brokers
- infrastructure
- database
- http
- websocket

Strategy 역시 Broker 구현체를 직접 참조하지 않는다.

반드시 Port를 사용한다.

---

# 3. Domain First

Domain은 프로젝트의 중심이다.

Domain은

- Quote
- Order
- Position
- Balance
- Trade

등 비즈니스 개념만 가진다.

Broker 응답(JSON)은 Domain으로 변환 후 사용한다.

금지

```
quote.prpr
quote.output1
quote.tr_id
```

허용

```
quote.price
quote.high
quote.low
```

---

# 4. Strategy

Strategy의 책임은

"매수할지, 매도할지 판단"

뿐이다.

Strategy는

- DB 저장
- HTTP 호출
- Broker 구현체
- Slack
- Discord

를 직접 사용하지 않는다.

Strategy의 출력은

```
BUY
SELL
HOLD
```

또는

```
Signal
```

이다.

---

# 5. Broker

Broker는

외부 API를 Domain으로 변환하는 Adapter이다.

Broker의 책임

- API 호출
- 인증
- 응답 파싱
- Domain 변환

Broker는

- AI
- Strategy
- Repository

를 알면 안 된다.

---

# 6. Repository

Repository는 저장만 담당한다.

저장은 Repository에서만 수행한다.

Broker에서 DB 저장 금지.

Strategy에서 DB 저장 금지.

---

# 7. Event Driven

실시간 데이터는 Event 기반으로 흐른다.

예시

```
QuoteUpdated

↓

Strategy

↓

OrderRequested

↓

OrderFilled

↓

PositionUpdated
```

---

# 8. State Management

Order와 Position의 상태는 하나의 Source of Truth에서 관리한다.

여러 클래스에서 동시에 상태를 변경하지 않는다.

예시

```
NEW

↓

ACCEPTED

↓

PARTIAL

↓

FILLED

↓

CANCELLED
```

---

# 9. Risk First

전략보다 리스크 관리가 우선한다.

모든 주문은

```
Strategy

↓

RiskManager

↓

OrderService

↓

Broker
```

순서로 처리한다.

RiskManager는 언제든 주문을 거부할 수 있다.

---

# 10. Testability

실계좌 없이 테스트 가능해야 한다.

반드시 Port를 Mock 또는 Fake로 교체할 수 있어야 한다.

예시

```
FakeBroker

FakeMarketData

FakeClock
```

---

# 11. Time

datetime.now()를 직접 사용하지 않는다.

반드시 Clock 또는 TimeProvider를 사용한다.

실거래

백테스트

리플레이

모두 동일한 코드가 동작해야 한다.

---

# 12. Error Handling

외부 API는 항상 실패한다고 가정한다.

반드시

- Retry
- Timeout
- Exception Mapping

을 고려한다.

Broker 내부에서만 HTTP Exception을 처리한다.

Application에는 Domain Exception만 전달한다.

---

# 13. Logging

모든 주문은 추적 가능해야 한다.

기록 항목

- 시간
- 전략명
- 종목
- 가격
- 수량
- 주문번호
- Broker 응답

---

# 14. Configuration

REAL / PAPER / BACKTEST는

코드 수정이 아닌

설정만 변경해서 실행 가능해야 한다.

---

# 15. Clean Code

함수는 가능한 한 하나의 역할만 수행한다.

복잡한 조건문보다

명확한 객체를 선호한다.

매직넘버 사용 금지.

하드코딩 금지.

---

# 16. File Organization

파일은 기능(Feature) 기준으로 구성한다.

예시

```
domestic/

    account.py

    order.py

    quote.py

    ranking.py

    finance.py
```

API 하나당 파일 하나로 만들지 않는다.

관련 기능은 하나의 파일로 묶는다.

---

# 17. Naming

클래스

```
Quote

OrderService

RiskManager

KISBroker
```

함수

```
place_order()

fetch_balance()

get_quote()
```

불명확한 이름 금지

```
do()

run()

process()

handle()
```

---

# 18. Architecture Priority

프로젝트에서 가장 중요한 우선순위

1. Domain
2. Dependency Direction
3. Responsibility Separation
4. Testability
5. Event Flow

폴더 구조보다 아키텍처를 우선한다.

---

# 19. Golden Rule

판단(Decision)

실행(Execution)

저장(Persistence)

외부 통신(Integration)

상태(State)

는 절대 같은 클래스에서 수행하지 않는다.

각 책임은 독립적으로 변경 가능해야 한다.

---



# Anti-Patterns

다음 코드는 생성하지 않는다.

- Strategy가 Broker 구현체를 직접 import하는 코드
- Domain이 HTTP 라이브러리를 사용하는 코드
- Broker가 DB를 저장하는 코드
- Repository가 외부 API를 호출하는 코드
- datetime.now()를 직접 사용하는 코드
- 하나의 클래스가 판단, 실행, 저장을 모두 수행하는 코드
- Broker 응답(JSON)을 Domain 밖으로 노출하는 코드
- 증권사(KIS 등) 전용 필드를 Domain 모델에 추가하는 코드
- 순환 의존성(import cycle)을 만드는 코드
