# stock-advisor-mcp

A local MCP server that exposes KRX (Korean stock exchange) data — via
`yfinance` and the public, unauthenticated KRX master files — as tools
Claude Desktop can call, so Claude can look up and analyze a Korean
ticker in chat.

**What this is not**: it does not connect to any broker, holds **no
credentials**, and places **no orders**. It only reads public market
data. It is entirely separate from, and does not import from or share
any code/database with, the trading system in the rest of this
repository (`app/`, `invest-project`) — that system
trades real money; this one cannot trade at all.

**Where it lives**: this folder (`stock-advisor-mcp/`) sits inside the
`invest-project` repository but is a self-contained
project with its own `pyproject.toml`, virtualenv and tests. It is not
part of the Docker image or the deploy (the `Dockerfile` copies only
`app/`), and it is not collected by the repository's `tests/` suite.
Run its tests from this folder only (`python -m pytest`).

> **참고(2026-10-06)**: 소스 안의 `D:\Investment_Project\handoff\knowledge\...`
> 참조(`src/stock_advisor_mcp/__init__.py`, `prompts/__init__.py`,
> `prompts/analysis_script.md` 첫 줄)는 저장소 **밖** 인수인계 폴더의 설계 문서를
> 가리킨다. 저장소에는 그 문서가 없으므로 인수자는 `handoff/knowledge/`에서
> `stock-advisor-mcp-tool-spec-2026-09-21.md`,
> `stock-advisor-mcp-debate-script-2026-09-21.md`를 찾는다.
> `analysis_script.md` 첫 줄은 로더가 제거하는 형식이라 바꾸지 않는다.

## The 5 tools

- `resolve_ticker(query)` — resolve a code / Korean name / suffixed symbol to the exact `yf_symbol` (`.KS`/`.KQ`) every other tool requires.
- `get_price_snapshot(yf_symbol)` — current price: `today_partial` while the session is open, `today_close` after it closes.
- `get_price_history(yf_symbol, period)` — completed daily OHLCV bars only (never today's in-progress bar).
- `get_technical_indicators(yf_symbol)` — RSI(14), MACD, SMA 5/20/60/120, Bollinger Bands, ATR(14), volume SMA(20), computed from completed bars only.
- `get_fundamentals(yf_symbol)` — market cap, revenue/operating income/net income history, equity, liabilities, and computed PER/PBR/ROE/debt ratio.

## Setup

### 1. Install

```bash
cd <저장소 폴더>/stock-advisor-mcp      # 이 README가 있는 폴더
python3 -m venv .venv                    # Python 3.12 이상
.venv/bin/pip install -e ".[dev]"
```

### 2. Configure Claude Desktop

Add this to `claude_desktop_config.json`. The path must be **absolute**:
replace `<저장소 폴더>` with the full path of the repository folder
(`invest-project`) on your machine.

```json
{
  "mcpServers": {
    "stock-advisor-mcp": {
      "command": "<저장소 폴더>/stock-advisor-mcp/.venv/bin/python",
      "args": ["-m", "stock_advisor_mcp.server"]
    }
  }
}
```

This was verified directly before writing it down: running
`<저장소 폴더>/stock-advisor-mcp/.venv/bin/python -m stock_advisor_mcp.server`
(checked on 2026-09-21 at the previous location) with stdin closed starts the server and exits cleanly (exit code 0) —
the expected behavior for a stdio MCP server with no client attached.

Restart Claude Desktop after editing the config.

### Where the config file lives (macOS)

`~/Library/Application Support/Claude/claude_desktop_config.json`.
Create the file (and the `mcpServers` key) if it doesn't exist yet.

## 사용법 — 분석 대본(`analysis_script`) 사용하는 두 가지 방법

이 서버는 도구 5개 외에 MCP **프롬프트** `analysis_script`도 하나 노출합니다. 국내(KRX)
종목 1개를 이 서버의 도구 데이터만 근거로 매수/관망/회피로 판단하는 분석 데스크 대본으로,
절대 규칙 8개와 0~5단계로 구성되어 있습니다. 선택 인자 `ticker`를 주면 대본 앞에
`분석 대상: <ticker>` 한 줄이 붙고, 주지 않으면 대본만 그대로 반환됩니다. 원문은
`src/stock_advisor_mcp/prompts/analysis_script.md`입니다.

이 대본을 실제로 쓰는 방법은 두 가지입니다.

**(a) Claude Desktop에서 MCP 프롬프트로 직접 호출.** Claude Desktop을 재시작해 이 서버가
로드되면, 이 서버가 등록한 프롬프트는 Desktop UI 어딘가에서 선택해 실행할 수 있는 형태로
나타납니다(정확한 메뉴 위치·이름은 Desktop 버전마다 다를 수 있어 여기서는 확인된 이름을
쓰지 않습니다 — 안 보이면 서버가 정상 연결됐는지, 도구 5개가 보이는지부터 확인하세요).
실행 시 `ticker` 값을 입력할 수 있으면 종목코드나 종목명을 넣고, 없으면 그냥 실행한
뒤 채팅에서 종목을 알려주면 됩니다.

**(b) Claude Project의 커스텀 지시문(custom instructions)에 대본 전체를 붙여넣기.**
`analysis_script.md`의 내용(맨 위 `<!-- 설계 근거: ... -->` 한 줄은 제외)을 그대로 복사해
Project 설정의 커스텀 지시문에 넣으면, 그 Project 안의 모든 대화가 이 대본을 자동으로
따릅니다. **이 방법은 (a)처럼 매번 프롬프트를 호출하는 걸 잊어버릴 위험이 없습니다** — 그
Project에서 대화를 시작하는 것만으로 대본이 항상 적용됩니다.

**v1 범위 밖(의도적 제외)** — 이 대본은 다음을 하지 않습니다: 목표주가 산출(근거 데이터가
없고 환각 위험이 가장 큰 출력이라 제외), 포트폴리오(보유 종목·현금) 반영(도구가 계좌
데이터를 갖지 않음), 여러 종목 비교·순위.

## Running the tests

```powershell
.venv\Scripts\pytest -q
```

## Known limitations

This server has no source for, and will never report:

- VI (변동성완화장치) / circuit-breaker status
- 시장경보(투자주의/경고/위험) or 관리종목 designations
- Foreign/institutional investor flows, program-trading flow
- Short-selling balances or volume
- The live order book (only the last traded price, no bid/ask depth)
- Korean news text (use Claude Desktop's own web search separately, and don't blend it with this tool's data)

`resolve_ticker`'s `data_coverage` field and every tool description
repeat this list on purpose, so the model sees the gap before it
answers instead of guessing to fill it.

Two data caveats worth knowing:

- **KOSPI is `.KS`, KOSDAQ is `.KQ`, and the wrong suffix does not
  error — it silently returns stale data** (prices frozen on a past
  trading day, sometimes a different instrument entirely). This is why
  every tool except `resolve_ticker` refuses a bare code and only
  accepts the `yf_symbol` that `resolve_ticker` derived from the KRX
  master file, and why a freshness guard compares each symbol's latest
  bar against a reference ticker's latest trading day and raises
  instead of returning silently-stale data.
- **Today's bar is a session in progress, not a close.** While KRX is
  open, `get_price_snapshot` returns `today_partial` with a
  `last_price` field — there is no `close` field until the session
  ends — and `get_price_history` excludes today's row entirely.

## Design invariants

Do not regress these:

1. No `close` field is ever present while a session is in progress (only `last_price` under `today_partial`).
2. The exchange suffix (`.KS`/`.KQ`) always comes from the KRX master file lookup in `resolve_ticker`, never from caller input or a heuristic.
3. Stale, missing, or zero-row data raises an error — it is never silently returned or backfilled with a guess.
4. Money values are always `{"display": ..., "krw": ...}`; `display` never contains an English scale word (no "million"/"billion").
