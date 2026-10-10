"""MCP server entry point.

Exposes the 5 data tools (plus get_analysis_script) from `stock-advisor-mcp-tool-spec-2026-09-21.md` over
stdio, for Claude Desktop to launch as a local subprocess. No broker
API, no credentials, no database -- yfinance + the unauthenticated KRX
master files only.

Every tool description below repeats the "data we cannot have" notice
(spec S:4) verbatim, on purpose: design principle 4 says the absence has
to be visible to the model *before* it decides what to say, not just
buried in a response field it might not read carefully.
"""

from __future__ import annotations

import functools
import logging
from typing import Callable, TypeVar

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.shared.exceptions import MCPError

from . import guides
from .coverage import DATA_WE_DO_NOT_HAVE_NOTICE
from .errors import AdvisorError
from .fundamentals import get_fundamentals as _get_fundamentals
from .indicators import get_technical_indicators as _get_technical_indicators
from .market_data import get_price_history as _get_price_history
from .market_data import get_price_snapshot as _get_price_snapshot
from .prompts.loader import load_analysis_script
from .resolver import resolve_ticker as _resolve_ticker

_ANALYSIS_SCRIPT = load_analysis_script()

_logger = logging.getLogger(__name__)

_F = TypeVar("_F", bound=Callable[..., dict])

# Every project-defined domain error (TickerNotFoundError, AmbiguousTickerError,
# StaleDataError, NoDataError, KrxMasterError) subclasses `AdvisorError` and
# only `AdvisorError` -- never bare `ValueError`/`RuntimeError`. Those two
# built-ins are too broad to use as the boundary: pandas/numpy raise
# `ValueError` for genuine programming errors constantly (e.g. a stray NaN
# reaching `int()`), and `NotImplementedError`/`RecursionError` are actual
# subclasses of `RuntimeError` -- both would get misclassified as data
# conditions instead of bugs. `AdvisorError` is this project's own type, so
# nothing outside code that means to raise a domain error can ever land here.
_DOMAIN_ERRORS = (AdvisorError,)


def _surface_errors(fn: _F) -> _F:
    """Turn a raised domain/bug exception into an `mcp` `ToolError` so its
    type name and message actually reach the client.

    Without this, the `mcp` SDK (see `Tool.run()` in
    `mcp/server/mcpserver/tools/base.py`) wraps anything that is not
    already a `ToolError`/`ResourceError`/`MCPError` in `UnexpectedToolError`,
    whose text is only ever the generic "Error executing tool <name>" --
    the original exception's type and message never leave the server
    process. Raising `ToolError` ourselves is the SDK's documented escape
    hatch: `Tool.run()` re-raises a `ToolError` with its own message
    intact (prefixed with "Error executing tool <name>: "), and the
    server's `_handle_call_tool` returns that text as an `is_error=True`
    result instead of swallowing it.

    Domain errors (`TickerNotFoundError`, `StaleDataError`, `NoDataError`,
    etc.) get `"<TypeName>: <message>"` verbatim. Anything else -- a real
    bug -- gets the same shape but prefixed `내부 오류:` so a data
    condition can never be mistaken for a code defect, or vice versa.

    One side effect of turning a bug into a `ToolError`: the SDK's own
    `_handle_call_tool` only logs a full traceback (`logger.exception`)
    for an `UnexpectedToolError`; a plain `ToolError` -- which is what
    every bug becomes here -- only gets an INFO-level one-liner. We log
    the traceback ourselves below so a real bug is still debuggable
    server-side even though the client now sees more than the generic
    "Error executing tool <name>".
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (MCPError, ToolError):
            raise
        except _DOMAIN_ERRORS as exc:
            raise ToolError(f"{type(exc).__name__}: {exc}") from exc
        except Exception as exc:
            _logger.exception("Tool %r raised an unexpected exception", fn.__name__)
            raise ToolError(f"내부 오류: {type(exc).__name__}: {exc}") from exc

    return wrapper  # type: ignore[return-value]


mcp = MCPServer(
    name="stock-advisor-mcp",
    title="Korean Stock Advisor",
    version="0.1.0",
    instructions=(
        "한국(KRX) 상장 종목의 시세/지표/재무 데이터를 제공합니다. 항상 resolve_ticker를 먼저 "
        "호출해 정확한 yf_symbol을 얻은 뒤 다른 도구에 그 값을 넘기세요. 코드나 접미사를 "
        "직접 추측하지 마세요. " + DATA_WE_DO_NOT_HAVE_NOTICE
        + " 도구 사용 순서·표기 규칙·지표 해석·가격 연속성 점검의 자세한 설명은 "
        "stock-advisor://guide/... 리소스(서버 소개는 stock-advisor://about)에 있습니다."
        " 종목 종합 분석이나 매수/관망/회피 판단 요청이면 먼저 get_analysis_script를 호출해 "
        "받은 절차를 따르세요(단순 시세·지표 조회에는 호출하지 않음)."
    ),
)


@mcp.tool(
    description=(
        "모든 조회의 시작점. 종목코드('005930'), 정식 한글명('삼성전자'), 또는 접미사 포함 "
        "심볼('005930.KS')을 받아 KRX 종목마스터파일(KOSPI/KOSDAQ, 무인증 배포)에서 정확한 "
        "시장 구분과 yfinance 접미사(.KS/.KQ)를 결정합니다. 다른 4개 도구는 이 도구가 반환한 "
        "yf_symbol만 받으므로, 접미사를 직접 추측해서 다른 도구에 넘기지 마세요 -- 틀린 접미사는 "
        "에러 없이 며칠 전 가격으로 고정된 스테일 데이터를 반환할 수 있습니다. "
        "마스터파일에 없는 코드는 추정 심볼 없이 에러를 반환합니다. " + DATA_WE_DO_NOT_HAVE_NOTICE
    )
)
@_surface_errors
def resolve_ticker(query: str) -> dict:
    """Resolve a code/name/symbol to its exact KRX market and yfinance symbol."""
    return _resolve_ticker(query).as_dict()


@mcp.tool(
    description=(
        "resolve_ticker가 반환한 yf_symbol의 현재 시세 스냅샷을 반환합니다. 장중이면 "
        "session_status='in_progress'와 today_partial(open/high/low/last_price/volume/as_of)을, "
        "장마감 후면 session_status='closed'와 today_close를 반환합니다. "
        "오늘 봉은 한국 시각 17:00 전까지 미확정(in_progress)으로 취급하며(장 마감 직후 포함), 판정 근거는 session_basis에 있습니다. "
        "today_partial에는 "
        "'close' 필드가 존재하지 않습니다 -- 장중 마지막 체결가(last_price)를 종가로 서술하지 "
        "마세요. 데이터가 스테일하거나(잘못된 접미사 등) 존재하지 않으면 에러를 던집니다(조용한 "
        "실패 없음). "
        "응답의 price_integrity.status가 'anomalies_found'이면 가격제한폭(±30%)을 넘는 변동·거래량 0·긴 공백 등 가격 연속성 문제가 있다는 뜻입니다 -- 원인(분할·감자·거래재개·데이터 오류 등)을 단정하지 말고, 해당 구간을 포함한 수익률·지표를 근거로 쓰지 마세요. "
        + DATA_WE_DO_NOT_HAVE_NOTICE
    )
)
@_surface_errors
def get_price_snapshot(yf_symbol: str) -> dict:
    """Current price snapshot for an exact yfinance symbol (from resolve_ticker)."""
    return _get_price_snapshot(yf_symbol)


@mcp.tool(
    description=(
        "resolve_ticker가 반환한 yf_symbol의 완성된 일봉 이력을 반환합니다(date/open/high/low/"
        "close/volume). period는 yfinance 기간 문자열(예: '1mo','3mo','6mo','1y')입니다. "
        "진행 중인 오늘 봉은 이 배열에 포함되지 않습니다(get_price_snapshot에서 별도 제공). "
        "bars_through가 배열의 마지막 완성봉 날짜입니다. 0개의 완성봉만 있거나 데이터가 "
        "스테일하면 에러를 던집니다. "
        "응답의 price_integrity.status가 'anomalies_found'이면 가격제한폭(±30%)을 넘는 변동·거래량 0·긴 공백 등 가격 연속성 문제가 있다는 뜻입니다 -- 원인(분할·감자·거래재개·데이터 오류 등)을 단정하지 말고, 해당 구간을 포함한 수익률·지표를 근거로 쓰지 마세요. "
        + DATA_WE_DO_NOT_HAVE_NOTICE
    )
)
@_surface_errors
def get_price_history(yf_symbol: str, period: str = "3mo") -> dict:
    """Completed daily OHLCV bars only -- never includes today's in-progress bar."""
    return _get_price_history(yf_symbol, period)


@mcp.tool(
    description=(
        "resolve_ticker가 반환한 yf_symbol의 기술적 지표를 이 도구가 직접 계산해 반환합니다: "
        "RSI(14), MACD(macd/signal/histogram), 이동평균(5/20/60/120일), 볼린저밴드, ATR(14), "
        "거래량 20일 이동평균. 계산에는 완성된 일봉만 사용하며 진행 중인 오늘 봉은 제외합니다 "
        "(제외하지 않으면 값이 장중 내내 바뀝니다). computed_through가 계산에 사용된 마지막 "
        "완성봉 날짜입니다. 계산에 필요한 기간(예: 120일 이평선에 120거래일)이 부족한 지표는 "
        "null이며 missing 목록에 표시됩니다 -- 값을 추정하지 않습니다. "
        "응답의 price_integrity.status가 'anomalies_found'이면 가격제한폭(±30%)을 넘는 변동·거래량 0·긴 공백 등 가격 연속성 문제가 있다는 뜻입니다 -- 원인(분할·감자·거래재개·데이터 오류 등)을 단정하지 말고, 해당 구간을 포함한 수익률·지표를 근거로 쓰지 마세요. "
        + DATA_WE_DO_NOT_HAVE_NOTICE
    )
)
@_surface_errors
def get_technical_indicators(yf_symbol: str) -> dict:
    """Tool-computed technical indicators (stockstats), excluding the in-progress bar."""
    return _get_technical_indicators(yf_symbol)


@mcp.tool(
    description=(
        "resolve_ticker가 반환한 yf_symbol의 재무 데이터를 반환합니다: 시가총액, 매출/영업이익/"
        "순이익(최근 4~5개년 + 가용시 최근 분기), 자본총계, 부채총계, 그리고 이 도구가 직접 "
        "계산한 PER/PBR/ROE/부채비율. 모든 금액은 {display: '333.6조원', krw: <정수>} 형태이며, "
        "display가 1차 서술용이고 krw는 검산용이므로 서술에 원 단위 그대로(특히 million/billion "
        "같은 영문 단위로) 옮기지 마세요. ETF는 is_etf=true, financials_available=false와 사유가 "
        "반환되며 재무비율 자리에 0이나 추정치가 들어가지 않습니다. 적자 등으로 비율이 의미 없는 "
        "경우 값 대신 null과 reason이 반환됩니다 -- 추정하지 마세요. " + DATA_WE_DO_NOT_HAVE_NOTICE
    )
)
@_surface_errors
def get_fundamentals(yf_symbol: str) -> dict:
    """Tool-computed fundamentals and ratios; ETFs get an explicit financials_available=false."""
    return _get_fundamentals(yf_symbol)


@mcp.tool(
    description=(
        "종목 종합 분석이나 매수/관망/회피 판단을 요청받았을 때 가장 먼저 호출해 분석 절차(절대 규칙 8개 + 0~5단계)를 "
        "받고, 그 절차를 그대로 따르세요. 이 도구는 종목 데이터를 조회하지 않으므로 resolve_ticker보다 먼저 "
        "호출합니다. 현재가·지표·재무 수치 하나를 묻는 단순 조회에는 호출하지 마세요. ticker를 주면 절차 앞에 "
        "분석 대상이 한 줄 붙습니다. " + DATA_WE_DO_NOT_HAVE_NOTICE
    )
)
@_surface_errors
def get_analysis_script(ticker: str | None = None) -> dict:
    """Return the analysis script text (same text as the analysis_script prompt)."""
    return {
        "script": _analysis_script_text(ticker),
        "applies_to": "종합 분석·매수/관망/회피 판단 요청",
    }


def _analysis_script_text(ticker: str | None = None) -> str:
    if ticker:
        return f"분석 대상: {ticker}\n\n{_ANALYSIS_SCRIPT}"
    return _ANALYSIS_SCRIPT


@mcp.prompt(
    name="analysis_script",
    title="국내 주식 분석 대본",
    description=(
        "국내(KRX) 종목 1개를 이 서버의 도구 데이터만 근거로 매수/관망/회피로 판단하는 "
        "분석 데스크 대본(절대규칙 8개 + 0~5단계). ticker를 주면 대본 앞에 분석 대상을 "
        "한 줄 명시합니다. 여러 종목 비교·포트폴리오 반영·목표주가 산출은 범위 밖입니다."
    ),
)
def analysis_script(ticker: str | None = None) -> str:
    """Return the debate-desk analysis script, optionally prefixed with the target ticker."""
    return _analysis_script_text(ticker)


_RESOURCE_MIME = "text/markdown"


@mcp.resource(
    "stock-advisor://about",
    name="about",
    title="서버 소개",
    description="읽기 전용 서버 소개: 데이터 출처, 도구·프롬프트·리소스 목록, 캐시 정책, 버전.",
    mime_type=_RESOURCE_MIME,
)
def about_resource() -> str:
    return guides.about_text(mcp.version)


@mcp.resource(
    "stock-advisor://guide/workflow",
    name="workflow",
    title="도구 사용 순서와 오류",
    description="resolve_ticker 우선 호출 규칙과 오류 종류별 뜻·대응.",
    mime_type=_RESOURCE_MIME,
)
def workflow_resource() -> str:
    return guides.workflow_text()


@mcp.resource(
    "stock-advisor://guide/data-coverage",
    name="data-coverage",
    title="제공하지 않는 데이터",
    description="이 서버가 제공하지 않는 데이터 목록과 모델이 해야 할 일.",
    mime_type=_RESOURCE_MIME,
)
def data_coverage_resource() -> str:
    return guides.data_coverage_text()


@mcp.resource(
    "stock-advisor://guide/units",
    name="units",
    title="표기 규칙",
    description="금액(display/krw)·퍼센트·장중/마감 필드·기준일 필드 표기 규칙.",
    mime_type=_RESOURCE_MIME,
)
def units_resource() -> str:
    return guides.units_text()


@mcp.resource(
    "stock-advisor://guide/indicators",
    name="indicators",
    title="기술 지표 해석",
    description="기술 지표별 정의, 필요 완성봉 수, null/missing 의미.",
    mime_type=_RESOURCE_MIME,
)
def indicators_resource() -> str:
    return guides.indicators_text()


@mcp.resource(
    "stock-advisor://guide/price-integrity",
    name="price-integrity",
    title="가격 연속성 점검",
    description="가격 연속성 점검 규칙(C1~C3), 임계값, affected_indicators 창, 해석 방법.",
    mime_type=_RESOURCE_MIME,
)
def price_integrity_resource() -> str:
    return guides.price_integrity_text()


@mcp.resource(
    "stock-advisor://prompts/analysis-script",
    name="analysis-script",
    title="분석 대본(열람용)",
    description="analysis_script 프롬프트와 같은 본문(프롬프트 미지원 클라이언트용).",
    mime_type=_RESOURCE_MIME,
)
def analysis_script_resource() -> str:
    return guides.analysis_script_text()


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
