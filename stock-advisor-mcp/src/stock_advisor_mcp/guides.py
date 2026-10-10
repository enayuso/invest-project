"""Static reference guides exposed as MCP resources.

Every function here is pure: no arguments (except `about_text(version)`),
no I/O, no network, no yfinance. Rules that live in code constants are
read from their home module *at call time* (never copied), so a guide can
never drift from the behaviour it describes. Prose lives in the
module-level string constants below, separate from the assembling
functions.
"""

from __future__ import annotations

from . import coverage, indicators, integrity, market_hours, session
from .errors import AdvisorError
from .freshness import NoDataError, StaleDataError
from .krx_master import KrxMasterError
from .prompts.loader import load_analysis_script
from .resolver import AmbiguousTickerError, TickerNotFoundError

# ---------------------------------------------------------------- prose

_ABOUT_HEAD = """# 서버 소개

stock-advisor-mcp {version}

한국(KRX) 상장 종목의 시세·기술지표·재무 데이터를 제공하는 **읽기 전용** 서버입니다.
자격증명이 없고 주문을 낼 수 없습니다. 데이터 출처는 yfinance와 KRX 종목마스터파일(무인증 배포)뿐입니다.

## 도구 ({tool_count}개)
{tool_lines}

## 프롬프트 (1개)
- `analysis_script` — 국내 종목 1개를 이 서버의 데이터만으로 판단하는 분석 대본

## 리소스 (7개)
- `stock-advisor://about` — 이 문서
- `stock-advisor://guide/workflow` — 도구 사용 순서와 오류
- `stock-advisor://guide/data-coverage` — 제공하지 않는 데이터
- `stock-advisor://guide/units` — 표기 규칙
- `stock-advisor://guide/indicators` — 기술 지표 해석
- `stock-advisor://guide/price-integrity` — 가격 연속성 점검
- `stock-advisor://prompts/analysis-script` — 분석 대본(열람용)

## 캐시 정책
"""

_ABOUT_TOOLS = [
    ("resolve_ticker", "종목코드·한글명·심볼을 받아 시장 구분과 yf_symbol 결정 (모든 조회의 시작점)"),
    ("get_price_snapshot", "현재 시세 스냅샷 (장중 today_partial / 마감 후 today_close)"),
    ("get_price_history", "완성된 일봉 이력"),
    ("get_technical_indicators", "서버가 직접 계산한 기술적 지표"),
    ("get_fundamentals", "시가총액·재무·서버가 계산한 PER/PBR/ROE/부채비율"),
    (
        "get_analysis_script",
        "종합 분석·매수/관망/회피 판단 요청 시 가장 먼저 호출하는 분석 절차(대본). 데이터를 조회하지 않음",
    ),
]

_ABOUT_CACHE = """- 시세 봉(일봉): 프로세스 내 {bars_min}분 캐시
- 종목 정보(info; 장 상태·직전 종가·시가총액 등): 프로세스 내 {info_min}분 캐시
- KRX 종목마스터: KST 날짜 기준 하루 1회 내려받아 디스크에 캐시(같은 날 재실행 시 재사용), 프로세스 내에서는 한 번만 로드
- 재무제표(매출·영업이익·순이익·자본·부채): 캐시하지 않고 호출할 때마다 조회
"""

_WORKFLOW_HEAD = """# 도구 사용 순서와 오류

## 순서
0. 종합 분석이나 매수/관망/회피 판단 요청이면 가장 먼저 `get_analysis_script`를 호출하고, 받은 절차를 따릅니다(이 도구는 종목 데이터를 조회하지 않으므로 `resolve_ticker`보다 먼저 호출합니다). 현재가·지표·재무 수치 하나를 묻는 단순 조회에는 호출하지 않습니다.
1. 종목 데이터 도구 중에서는 항상 `resolve_ticker`를 먼저 호출합니다(0번의 절차 안내 도구는 예외).
2. 다른 4개 도구에는 `resolve_ticker`가 반환한 `yf_symbol`만 넘깁니다.
3. 접미사(.KS/.KQ)나 종목코드를 직접 추측하지 않습니다. 틀린 접미사는 오래된 가격으로 고정된 데이터를 줄 수 있고, 이 서버는 그것을 에러로 막습니다.

## 오류 종류와 대응
오류는 `<타입명>: <메시지>` 형태로 반환됩니다. 어떤 경우에도 **그대로 사용자에게 보고하고 멈춥니다. 우회하지 않습니다**(다른 접미사 시도·추정값 대체 금지).

"""

_WORKFLOW_BUG = (
    "- `내부 오류:` 로 시작하는 메시지 — 데이터 상황이 아니라 서버 코드의 결함입니다. "
    "메시지를 그대로 보고하고 멈춥니다."
)

_ERROR_MEANINGS = [
    (
        StaleDataError,
        "가장 최근 봉이 시장의 최근 거래일보다 오래됨. 보통 잘못된 접미사(.KS/.KQ)로 인한 오래된·타 시장 데이터. 해당 데이터를 쓰지 않습니다.",
    ),
    (NoDataError, "yfinance가 데이터를 주지 않았거나 쓸 수 있는 완성봉이 0개. 없는 데이터를 만들어 쓰지 않습니다."),
    (
        TickerNotFoundError,
        "KRX 마스터에 없는 코드, 또는 일치하는 종목이 없는 이름. 추정 심볼을 만들지 않고 사용자에게 정확한 코드를 묻습니다.",
    ),
    (AmbiguousTickerError, "이름이 여러 종목과 일치. 메시지의 후보 목록을 사용자에게 보여 주고 선택을 받습니다."),
    (KrxMasterError, "KRX 종목마스터 내려받기·해석 실패. 접미사를 추측해 대체하지 않고 실패를 그대로 보고합니다."),
]

_COVERAGE_HEAD = """# 제공하지 않는 데이터

이 서버에는 아래 항목의 출처가 없으며, 앞으로도 없습니다. 모든 도구 설명과 `resolve_ticker` 응답(`data_coverage`)에도 같은 목록이 들어 있습니다.

"""

_COVERAGE_ITEM_ACTION = '  - 이 서버에는 없음 → 필요하면 "확인 불가"로 쓰고 결론에 불확실성을 반영, 추론 금지'

_COVERAGE_TAIL = """
## 모델이 할 일
위 항목이 판단에 필요하면 **"확인 불가"** 라고 쓰고, 그 불확실성을 결론에 반영합니다.
다른 데이터로 추측하거나 그럴듯하게 채우지 않습니다(예: 가격이 크게 움직였다고 VI 발동을 단정하는 것). 이 서버의 데이터와 다른 출처(웹검색 등)의 정보는 명확히 구분해 표시합니다.
"""

_SESSION_SECTION = """## 장중 vs 마감
- 장중(`session_status='in_progress'`): `today_partial.last_price`가 오며 `close` 필드는 존재하지 않습니다. 마지막 체결가를 종가라고 쓰지 않습니다.
- 마감(`session_status='closed'`): `today_close.close`가 종가입니다.
- 오늘 봉은 한국 시각 {confirm} 전까지 미확정(`in_progress`)으로 취급합니다. 장 마감(15:30) 직후와 장이 늦게 끝나는 날에도 종가를 잘못 부르지 않기 위해서이며, 평일 15:30~{confirm}에는 장이 끝났어도 오늘 값이 미확정으로 표시되고 지표는 전 거래일 기준입니다.
- `session_basis`는 판정 근거입니다:
  - `no_bar_today` — 오늘 봉이 없음(장 시작 전·휴장일·주말). 마지막 봉은 이미 확정된 지난 거래일 → `closed`
  - `market_state_regular` — 오늘 봉이 있고 yfinance가 정규장(REGULAR)이라고 알림 → `in_progress`
  - `before_close_confirm_time` — 오늘 봉이 있고 한국 시각이 {confirm} 이전 → `in_progress`
  - `after_close_confirm_time` — 오늘 봉이 있고 한국 시각이 {confirm} 이후 → `closed`
  - `bar_date_after_today` — 마지막 봉 날짜가 오늘보다 뒤(시계 이상 등), 보수적으로 → `in_progress`

"""

_ABOUT_SESSION = """
## 장 상태 판정
오늘 봉의 확정 여부는 한국 시각으로 판정합니다: 오늘 봉이 있으면 {confirm} 전까지는 `in_progress`, 이후에는 `closed`입니다(yfinance가 정규장이라고 알리면 항상 `in_progress`). 오늘 봉이 없으면(장 시작 전·휴장일·주말) 마지막 봉이 확정된 `closed`입니다. 근거는 응답의 `session_basis`에 있고, 값 목록과 뜻은 `stock-advisor://guide/units`에 있습니다.
"""

_UNITS_TEXT = """# 표기 규칙

## 금액
모든 원화 금액은 `{"display": "333.6조원", "krw": 333605938000000}` 형태입니다.
- 서술에는 `display`를 그대로 인용합니다(억원 또는 조원 단위).
- `krw`는 검산용입니다. 서술에 옮기지 않습니다.
- million·billion 같은 영문 단위로 바꿔 쓰지 않습니다(억원↔billion 혼동이 반복된 실제 오류 유형입니다).

## 퍼센트
`change_pct_vs_prev_close` 같은 퍼센트 값은 이미 ×100 된 값입니다(5.25는 5.25%). 다시 100을 곱하거나 나누지 않습니다.

{SESSION_SECTION}
## 기준일 필드
- `bars_through` — 이력(`get_price_history`)의 마지막 완성봉 날짜
- `computed_through` — 지표(`get_technical_indicators`)에 쓰인 마지막 완성봉 날짜
- `checked_through` — 가격 연속성 점검(`price_integrity`)이 확인한 마지막 날짜
서술할 때 "오늘 값"이 아니라 이 날짜 기준이라고 밝힙니다.
"""

_INDICATORS_HEAD = """# 기술 지표 해석

`get_technical_indicators`가 서버에서 직접 계산해 반환합니다(모델이 계산하지 않습니다). 이력은 {history} 기간의 일봉을 사용합니다.

## 지표
- `rsi_14` — 14일 RSI. 값이 클수록 최근 상승 우위, 작을수록 하락 우위
- `macd` — `macd`, `signal`, `histogram`(MACD와 시그널선의 차)
- `moving_averages` — 단순이동평균 `sma_5`, `sma_20`, `sma_60`, `sma_120`
- `bollinger_bands` — `mid`(중심선), `upper`, `lower`
- `atr_14` — 14일 ATR(평균 진폭, 변동성 크기)
- `volume_sma_20` — 거래량 20일 이동평균

## 이동평균·거래량 평균의 최소 완성봉 수
아래 지표는 필요한 거래일이 부족하면 값을 만들지 않습니다.

"""

_INDICATORS_TAIL = """
## 완성봉만 사용하는 이유
진행 중인 오늘 봉을 넣으면 지표 값이 장중 내내 바뀝니다(조회 시각에 따라 값이 달라지는 함정). 그래서 완성된 일봉만으로 계산하고, 마지막으로 쓴 완성봉 날짜를 `computed_through`로 알려 줍니다. "오늘 실시간 값"이 아니라 그 날짜 기준 값으로 서술합니다.

## null과 missing
- 필요한 기간이 부족한 지표는 값이 `null`이고 이름이 `missing` 목록에 들어갑니다. 추정하지 않고 "확인 불가"로 서술합니다.
- 계산 결과가 없는 지표도 `null`이며 `missing`에 표시됩니다.

## price_integrity와 함께 읽기
응답의 `price_integrity.status`가 `anomalies_found`이면 해당 구간을 포함한 지표를 근거로 쓰지 않습니다. 자세한 내용은 `stock-advisor://guide/price-integrity`를 보세요.
"""

_INTEGRITY_HEAD = """# 가격 연속성 점검

yfinance 시계열에는 액면분할·병합·감자·거래 정지/재개 등으로 정상 거래에서는 나올 수 없는 값이 섞일 수 있습니다. 서버는 완성된 일봉을 점검해 **탐지하고 알리기만** 하며 원인을 추정하지 않습니다. 결과는 `price_integrity`에 담깁니다(`status`는 `ok` 또는 `anomalies_found`).

## 점검 규칙
"""

_INTEGRITY_RULES = """- **C1 가격제한폭 위반** — 연속한 두 완성봉 종가의 일간 변동 절대값이 {pct}를 넘음 (KRX 상하한 ±30% + 수정주가 오차 허용 0.5%p). `limit_breaches`에 날짜·직전 종가·종가·변동률이 기록됩니다.
- **C2 거래량 0** — 거래량이 0인 봉. `zero_volume_dates`에 날짜가 기록됩니다.
- **C3 긴 공백** — 연속한 두 봉 사이가 달력일로 {gap}일 이상. `long_gaps`에 구간이 기록됩니다.
"""

_INTEGRITY_AFFECTED_HEAD = """
## affected_indicators
`get_technical_indicators`는 C1 위반 봉이 아직 계산 창 안에 있는 지표를 `affected_indicators`로 알려 줍니다. 위반 봉 이후 경과한 완성봉 수가 아래 창보다 작으면 해당 지표가 영향권입니다.

| 지표 | 창(완성봉 수) |
|---|---|
"""

_INTEGRITY_AFFECTED_TAIL = """
RSI·ATR·MACD는 지수 방식이라 위반의 영향이 완전히 사라지지 않으므로 이 창은 근사치입니다.
"""

_INTEGRITY_INTERPRET = """
## 안내문 (status가 anomalies_found일 때 `notice`로 첨부됨)
{notice}

## 해석
- 원인(분할·감자·거래재개·데이터 오류 등)을 단정하지 않습니다. 서버도 확인할 수 없습니다.
- 위반 구간을 포함한 수익률·지표를 판단 근거로 쓰지 않고, 해당 부분은 "가격 연속성 확인 불가"로 서술합니다.
- `get_price_snapshot`은 직전 종가 대비 현재 값의 C1만 점검합니다.
"""

# ------------------------------------------------------------- helpers


def _fmt_pct(ratio: float) -> str:
    return f"{ratio * 100:g}%"


def _minutes(seconds: float) -> str:
    return f"{seconds / 60:g}"


# ----------------------------------------------------------- functions


def about_text(version: str) -> str:
    return (
        _ABOUT_HEAD.format(
            version=version,
            tool_count=len(_ABOUT_TOOLS),
            tool_lines="\n".join(f"- `{name}` — {desc}" for name, desc in _ABOUT_TOOLS),
        )
        + _ABOUT_CACHE.format(
            bars_min=_minutes(session._BARS_TTL_SECONDS),
            info_min=_minutes(session._INFO_TTL_SECONDS),
        )
        + _ABOUT_SESSION.format(confirm=_confirm_time())
    )


def workflow_text() -> str:
    assert all(issubclass(cls, AdvisorError) for cls, _ in _ERROR_MEANINGS)
    lines = [f"- `{cls.__name__}` — {meaning}" for cls, meaning in _ERROR_MEANINGS]
    return _WORKFLOW_HEAD + "\n".join(lines) + "\n" + _WORKFLOW_BUG + "\n"


def data_coverage_text() -> str:
    items = "\n".join(f"- {item}\n{_COVERAGE_ITEM_ACTION}" for item in coverage.DATA_WE_DO_NOT_HAVE)
    return _COVERAGE_HEAD + items + "\n" + _COVERAGE_TAIL


def _confirm_time() -> str:
    return market_hours.CLOSE_CONFIRM_TIME_KST.strftime("%H:%M")


def units_text() -> str:
    section = _SESSION_SECTION.format(confirm=_confirm_time())
    return _UNITS_TEXT.replace("{SESSION_SECTION}\n", section)


def indicators_text() -> str:
    rows = "\n".join(
        f"- `{name}` — 완성봉 {days}개 이상" for name, days in indicators._REQUIRED_TRADING_DAYS.items()
    )
    return _INDICATORS_HEAD.format(history=indicators._HISTORY_PERIOD) + rows + "\n" + _INDICATORS_TAIL


def price_integrity_text() -> str:
    rules = _INTEGRITY_RULES.format(
        pct=_fmt_pct(integrity.LIMIT_BREACH_THRESHOLD), gap=integrity.LONG_GAP_CALENDAR_DAYS
    )
    table = "\n".join(f"| `{name}` | {w} |" for name, w in integrity.INDICATOR_WINDOWS.items())
    return (
        _INTEGRITY_HEAD
        + rules
        + _INTEGRITY_AFFECTED_HEAD
        + table
        + "\n"
        + _INTEGRITY_AFFECTED_TAIL
        + _INTEGRITY_INTERPRET.format(notice=integrity.PRICE_INTEGRITY_NOTICE)
    )


def analysis_script_text() -> str:
    return load_analysis_script()
