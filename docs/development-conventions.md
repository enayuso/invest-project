# 개발 관례 (Development Conventions)

이 문서는 이 코드베이스가 실제로 따르고 있는 관례만 기록한다. 각 항목은 실제 코드를 읽고
검증한 것이며, 일반론적인 best practice가 아니다.

## 1. 설계 원칙

### SRP / 의존성 방향: Domain은 brokers를 모른다

`app/domain/ports/broker_port.py`의 `BrokerPort`는 `Protocol`이며 베이스 클래스가 아니다.
`KISBroker`가 상속 없이 시그니처만 맞춰서 만족시키는 구조적 타이핑을 쓴다.

```python
# Account/order contract — a Protocol, not a base class, so an existing
# broker class (e.g. KISBroker) satisfies it by matching method shapes,
# with no import-time coupling back to domain/.
```

`app/brokers/__init__.py`도 같은 원칙을 레지스트리 레벨에서 구현한다. `get_broker(name)`은
`_BROKER_FACTORIES` 딕셔너리로 브로커를 조회하고, pipeline/ledger/dashboard는 벤더 SDK(KIS)를
직접 import하지 않는다. 브로커 추가는 이 딕셔너리에 한 줄 더하는 것으로 끝난다.

### Domain First: raw JSON을 도메인 모델로 변환 후 사용

`app/domain/models/market.py`의 도메인 모델(`StockStatus`, `InvestorFlow`,
`IncomeStatement` 등)은 broker-agnostic한 필드명(`code`, `close`, `revenue`)을 쓰고, 원본
KIS 응답 필드명은 주석으로만 남는다.

```python
code: str  # 종목번호 (pdno)
trading_halted: bool  # 거래정지여부 (tr_stop_yn)
```

`pdno`, `tr_stop_yn` 같은 raw key는 도메인 경계를 넘지 않고 브로커 어댑터 안에서 변환된다.

### Risk First: 매매 로직보다 안전장치가 우선

`app/execution/kis_bridge.py`가 대표 사례다. 모듈 docstring이 "이 파일이 유일한 주문
관문(choke point)"임을 선언한다.

```python
"""Single choke point between the paper ledger and the KIS paper account.

Every safety gate for real order submission lives HERE and only here —
`KISBroker.submit_order`/`submit_overseas_order` are deliberately
guard-free, and no other module may call them for trading.
```

게이트 순서: kill switch/일일 손실 서킷브레이커(전체를 막으므로 최우선) → master switch →
레버리지/인버스 차단 → 주문 한도 → 승인 임계값 → 시장 서킷브레이커(무료 in-memory 체크를
REST 호출보다 먼저) → 개별종목 거래정지/VI 체크(best-effort, 실패 시 fail-open) → 현금
충분성 체크 → 포지션 비중 캡 → 마지막 dry-run. "실행 가능 여부"보다 "왜 막아야 하는가"가
먼저 오는 순서다.

### Testability: Port/Fake 패턴

`BrokerPort`(Protocol)는 실제 계좌 연결 없이 Fake로 대체 가능하도록 설계되어 있다.
`tests/`는 **git이 추적한다**(2026-10-02 확인: `git ls-files 'tests/*'` 192개 파일,
`.gitignore`에 `tests/` 항목 없음). 예전 이 문단의 "tests/는 로컬 전용·원격 미커밋" 설명은
사실이 아니어서 2026-10-02에 고쳤다.

### 과잉 선제 추상화 금지

multi-instance 대응 seam(여러 계좌/브로커 인스턴스를 동시에 굴리는 구조)은 소비자가 실제로
생기기 전까지 도입하지 않는다. `agent-architecture-roadmap` 6단계 로드맵의 후반 단계로
계획만 하고, 지금 쓰이지 않는 추상화 레이어는 미리 깔지 않는다. `app/brokers/__init__.py`의
팩토리 레지스트리도 "브로커 2개 이상 필요해지면 한 줄 추가"라는 최소 확장 지점으로 남아
있을 뿐, 그 이상의 멀티브로커 프레임워크는 없다.

## 2. 코드 작성 관례

### 주석: 영어 본문 + 필요 시 한글 1줄 요약

`broker_port.py`, `market.py`, `app/brokers/__init__.py` 전반에서 확인되는 패턴이다.

```python
"""Account inquiry + order execution, domestic and overseas.
계좌 조회 및 주문 실행 (국내/해외 공통 계약)"""
```

```python
# Name -> zero-arg factory returning a domain/ports-satisfying broker
# instance, or None when that broker's credentials aren't configured.
# 브로커 이름 -> 인스턴스 팩토리 (신규 증권사는 이 딕셔너리에 한 줄 추가)
```

영어로 충분히 설명한 뒤 짧은 한글 한 줄을 덧붙이는 형태이며, 한글 설명을 장황하게 늘어놓지
않는다.

### 긴 모듈 docstring에 "왜"를 기록하는 문화

`app/execution/kis_bridge.py`가 대표 사례다. 파일 맨 위에 어떤 게이트가 왜 그 순서로
존재하는지, 무엇이 REST 호출이라 비용이 드는지, 무엇이 fail-open/fail-closed인지까지
문서화되어 있다.

```
`True` (circuit breaker active) blocks. `None` (feed not connected, ...)
is treated as **fail-closed**: also blocks, by deliberate, confirmed policy
(`kis_cb_fail_open_when_unknown`, default `False`)
```

`app/clock.py`, `app/config.py`, `scripts/run_auto_trade_cycle_now.py`도 같은 스타일로
"이 설정이 왜 이 기본값인지", "이 코드가 왜 이 순서인지"를 코드 옆 docstring/주석에 남긴다.
별도 설계 문서보다 코드 옆에 이유를 두는 쪽을 택한다.

### fail-safe 기본값 + "조용히 기능 끄기(degrade quietly)"

`app/brokers/kis/broker.py`의 `broker_from_settings()`와 `app/brokers/__init__.py`의
`get_broker()`가 같은 관례를 공유한다.

```python
"""...None when the selected mode's credentials aren't configured — callers
treat that as "feature off" and degrade quietly, same convention as the
memory store. kis_account_mode defaults to "paper" ..."""
```

크리덴셜이 없거나 DB가 죽어도 예외로 앱을 죽이는 대신 `None`을 반환하거나 폴백하여 해당
기능만 조용히 꺼지게 만든다. `kis_account_mode` 기본값도 항상 `"paper"`이며, 실계좌는 명시적
opt-in이 있어야만 활성화된다.

### cp949 콘솔 주의: 사용자 노출 문자열은 ASCII 구두점만

`kis_bridge.py`의 disposition 문자열은 코드/주석에서는 em dash(—)를 자유롭게 쓰면서도, 실제
사용자/로그에 노출되는 문자열에서는 하이픈(`-`)만 쓴다.

```python
return f"skipped: {ticker} is neither a KRX nor a US ticker"
return f"dry-run: would submit {order_desc}; set KIS_EXECUTION_DRY_RUN=false to go live"
```

`scripts/run_auto_trade_cycle_now.py`의 `print()` 출력도 마찬가지다.

```python
print("domestic market closed - skipping domestic candidates")
print(f"{ticker}: FAILED - {exc}")
```

Windows 콘솔의 기본 cp949 인코딩에서 em dash가 깨지는 문제를 피하기 위함이다. docstring/주석은
em dash를 써도 되지만, 콘솔에 출력되거나 로그/UI에 노출되는 문자열은 ASCII 구두점만 쓴다.

## 3. Git 관례

### 커밋 메시지: 영어, 1줄 요약 + 빈 줄 + 3~5줄 설명

`git log -3 --format=full` 결과가 이 형식을 그대로 보여준다.

```
Add dev script to trigger one real auto_trade_loop cycle on demand

For testing SEMI_AUTO/FULL_AUTO end to end (real pipeline, real
execution_mode branching) without waiting for
auto_trade_interval_minutes to fire, or the market to still be open
when it does. Respects market hours exactly like the scheduled loop
- a closed market contributes no candidates from that side, verified
by running it live with both markets closed...
```

제목은 명령형 요약 한 줄, 본문은 "왜 필요했는지 + 무엇을 검증했는지" 위주의 짧은 문단이다.
장황한 상세 서술은 피한다.

### 단계별 커밋, push 전 매번 재확인

작업을 여러 독립 논리 단위로 쪼개 각각 커밋한다(`git log --oneline`의 커밋들이 기능 하나씩
담고 있다). `git push`는 매번 사용자에게 다시 확인받은 뒤에만 실행한다 — 이전에 push를
승인받았어도 다음 push에는 다시 물어본다.

### 브랜치

현재 작업은 `main` 브랜치에서 이뤄진다.

### 커밋 금지 파일

`.gitignore`에 `.env`, `.env.*`, `.env.kis.local`, `.venv/`, `.claude/`, `scratchpad/` 등이
등록되어 있다. **`.env.*` 패턴이 살아 있어 `.env.local`·`.env.dev`도 무시된다 — 절대 커밋하지
않는다**(DB 접속 정보 등 비밀값이 들어 있다). 예외는 `!.env.example` 하나뿐이고, git이 추적하는
`.env*` 파일도 `.env.example` 하나다(2026-10-02 확인). `.env.kis.local`은 실계좌/모의계좌 키를
담고 있어 특히 주의한다. `tests/`는 무시 대상이 아니다(위 Testability 절).
예전 이 문단의 "`.env.local`은 추적 대상", "`tests/`가 `.gitignore`에 있음" 설명은 사실이
아니어서 2026-10-02에 고쳤다.

## 4. 언어 규칙

### 사용자 응답: 한글 질문엔 한글로만

`app/llm/language.py`의 `detect_language()`가 이 규칙을 코드로 강제한다.

```python
def detect_language(text: str) -> str:
    """"ko" if `text` contains any Hangul syllable, else "en"."""
    return "ko" if _HANGUL_RE.search(text) else "en"
```

### AI 파이프라인: 질문 언어(한/영)에 맞춰 답변

같은 파일의 `language_instruction()`이 분석가 리포트, 토론(debate), 최종 결정 등 파이프라인
내 모든 LLM 호출에 언어 지시를 주입한다. 딱 두 언어(`ko`/`en`)로만 제한하여 모델이 임의로
제3 언어를 고르지 않도록 강제한다.

```python
"""Detects whether the user's own question was Korean or English, so every
LLM call in a pipeline run (analyst reports, debate turns, final decision)
answers in that same language — restricted to exactly these two"""
```

### i18n: index.html의 ko/en 딕셔너리 패턴

`app/web/static/index.html`은 `data-i18n` 속성으로 DOM 요소를 마킹하고, JS 객체
`I18N = { ko: {...}, en: {...} }`에서 텍스트를 채운다.

```html
<span class="scope" data-i18n="scopeDomestic">국내 (KRW)</span>
```

새 라벨을 추가할 때는 HTML에 `data-i18n` 키를 달고 `I18N.ko`/`I18N.en` 양쪽에 값을 채우는
식으로 확장한다.

## 5. 검증 관례

- 전체 회귀는 외부망을 막은 단일 프로세스 도구로 돌린다: `.venv/Scripts/python.exe -X utf8 scripts/run_protected_regression.py`(저장소 폴더에서, 약 4분, `-n` 불가, 표적 실행은 `-- <pytest 인자>`, 결과는 저장소 밖 `<임시폴더>/invest-regression/<루트명>/<UTC>/`; 저장소 안 `--out`은 tmp_path 저장소 밖을 요구하는 일부 시험을 실패시킬 수 있다).
- 수정 직후 구문 검증을 먼저 한다 (`python -c "import ast; ast.parse(open(f).read())"` 류의
  패턴, JS라면 `new Function(...)`으로 파싱만 시도). 파일 저장 직후 습관적으로 거치는 1차
  검증 단계다.
- 가능하면 실제 서버(uvicorn)나 브라우저로 라이브 검증을 거친 뒤 커밋한다.
  `scripts/run_auto_trade_cycle_now.py`의 커밋 메시지가 이를 잘 보여준다: "verified by
  running it live with both markets closed (prints "market closed", no pipeline call, no
  cost)"처럼 코드만 읽고 판단하지 않고 실행 결과를 커밋 메시지에 남긴다. `46016e3` 커밋도
  "Verified the generator terminates cleanly after its tick count via a direct test"라며
  동일한 습관을 보인다.
