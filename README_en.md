# invest-project — AI multi-agent trading system

> 한국어: [README.md](README.md) (the Korean README is the more detailed reference)

Several LLM agents work like a brokerage trading desk — **analysts → bull/bear debate → research manager →
trader → risk debate → portfolio manager** — to produce a rating for each stock. **Deterministic code**, not
the LLM, then turns that rating into an order through position sizing, stop-losses and a chain of risk gates,
and sends it to a Korea Investment & Securities (KIS) Open API account.

I built this to learn investing itself, and to understand how to attach LLM agents to a system where real
money moves: output-validation guards, deterministic risk gates, and measuring results after the fact instead
of trusting a feeling. Most of the code, tests and docs were written together with **Claude Code**.

- **Single-account tool**: one process serves one account. No login — access control is network location
  (loopback bind + SSH tunnel).
- **Long-only, no leverage, no short selling**; swing-trading horizon (days).
- Supports Korea (KRX) and the US. The two markets are **separated at every layer** (candidates, cash,
  accounts, logs).
- LLM: self-hosted vLLM (Qwen family) by default; an OpenAI-compatible gateway and Anthropic are also
  supported. Storage: Postgres + pgvector.
- **Every default is safe**: paper account, manual mode, execution disabled, dry-run on.

> ⚠️ For learning and research only. Not investment advice; you are responsible for anything you do with a
> real account.

## 1. How one decision is made

Entry point: `run_pipeline()` in `app/orchestrator/pipeline.py`.

```
 ① 4 analysts (concurrent)        Market (price, indicators, flows) / Fundamentals / News+Macro / Sentiment
 ② output guards                  app/agents/*_guard.py, identity_gate.py — unit-scale, fabricated indicators,
                                  trading halts, price limits, ticker identity, ...
 ③ Bull ↔ Bear debate             up to MAX_DEBATE_ROUNDS
 ④ Research Manager               rating + target weight proposal
 ⑤ Trader                         direction · entry · stop
 ⑥ Risk debate                    Aggressive / Conservative / Neutral, up to MAX_RISK_ROUNDS
 ⑦ Portfolio Manager              final rating (Buy/Overweight/Hold/Underweight/Sell) + stop price;
                                  only the PM sees similar past decisions (pgvector memory)
 ⑧ ledger.rebalance_to_target()   rating → target allocation; ATR down-scaling, 1% risk rule,
                                  portfolio caps (concentration/correlation/drawdown)
 ⑨ execution_mode                 MANUAL: ledger only / SEMI_AUTO: approval queue / FULL_AUTO: order now
 ⑩ kis_bridge.mirror_trade_to_kis()   the single gateway for real orders (§3)
```

Every decision is stored with its **full transcript**; its outcome (alpha vs. the benchmark) is resolved
automatically later (`app/memory/auto_resolve.py`), so improved guards can be re-scored against past decisions
without calling the LLM again. One decision ≈ 12 LLM calls and 100k–200k tokens.

## 2. Auto-trading cycle and background loops

`app/live/auto_trade_loop.py` (only when `AUTO_TRADE_ENABLED=true`, during market hours):
discover candidates via KIS ranking APIs → eligibility filters (re-analysis lock, holding caps) → LLM-free
momentum ranking (`app/chat/screening.py`) → force-include positions near their stop → run the pipeline per
candidate → record a screening snapshot to measure the live ranking's predictive power (IC).

Always-on loops started by `app/web/state.py`: price poller, rule-based stop-loss/trailing stop (runs in every
mode), fill monitor, exit monitor, ledger↔broker reconciliation (detect & alert only, never auto-correct), and
the KIS WebSocket realtime feed (circuit breakers / VI / fills / quotes).

## 3. Safety

- **Real orders pass through one place only — `app/execution/kis_bridge.py`.** `KISBroker.submit_order` has
  no guards on purpose; every gate lives in the bridge: kill switch → daily max-loss breaker → master switch
  (`KIS_EXECUTION_ENABLED`) → leverage/inverse block → order-size cap → approval threshold → market circuit
  breaker → VI → cash sufficiency → position-weight cap → dry-run (`KIS_EXECUTION_DRY_RUN`).
- Fail-closed vs fail-open is decided per gate (market-wide breaker, kill switch and daily loss block when
  unknown). Kill-switch and daily-loss state persist in Postgres across restarts.
- Defaults: `KIS_ACCOUNT_MODE=paper`, `EXECUTION_MODE=manual`, `KIS_EXECUTION_ENABLED=false`,
  `KIS_EXECUTION_DRY_RUN=true`, `AUTO_TRADE_ENABLED=false`; the dashboard **refuses to boot** unless bound to
  loopback. A real account refuses to boot outside `AI_PROFILE=production` unless explicitly acknowledged.

## 4. Layout

```
app/
├── daemon.py        long-running entry point: dashboard + all background loops
├── main.py          one-shot single-ticker decision CLI
├── config.py        all settings (pydantic Settings) and boot-time validation
├── agents/          LLM agents + output-validation guards
├── orchestrator/    pipeline.py (run_pipeline), debate.py
├── live/            auto-trade loop, universe discovery, poller, fill/exit monitors, market hours
├── execution/       ledger, kis_bridge (order gateway), sizing, risk rules, safety switch, reconciliation
├── brokers/         KIS SDK isolation layer (domestic/overseas/WebSocket)
├── domain/          pure rules + ports (Protocols)
├── dataflows/       data providers — KIS first, yfinance fallback, FRED, Reddit
├── memory/          decision memory (pgvector), auto-resolution, reflections
├── llm/             LLM clients (vllm / gateway / anthropic), concurrency limits, metrics
├── web/             FastAPI dashboard (no-build HTML + JS)
├── audit/ backtest/ experiments/   measurement CLIs, backtest harnesses, paired A/B experiments
stock-advisor-mcp/   self-contained read-only MCP server exposing market data/analysis tools
tests/               pytest suite mirroring app/ (~5,000 tests)
scripts/ docs/       one-off data/ops tools; design docs
pipeline.sh          build & deploy script
```

## 5. Setup and usage

Requirements: Python **3.12**, [uv](https://docs.astral.sh/uv/), Postgres 16 + pgvector (`DATABASE_URL` is
required), an OpenAI-compatible LLM endpoint, and optionally KIS Open API keys (paper keys are enough).

```bash
uv sync
cp .env.example .env.local      # fill in values — .env.example documents every key

# dashboard + background loops  → http://127.0.0.1:8094
uv run python -m app.daemon

# single decision
uv run python -m app.main --ticker 005930.KS --date 2026-09-18

# backtests
uv run python -m app.backtest --ticker AAPL --start 2026-05-01 --end 2026-07-01 --cadence-days 7 --benchmark SPY
uv run python -m app.backtest.portfolio_harness --start ... --end ... --cadence-days 7

# resolve outcomes / check stops
uv run python -m app.memory resolve --ticker AAPL
uv run python -m app.execution check-stops --ticker AAPL
```

Don't run two daemons against the same account/ledger at once.

### Measuring

```bash
uv run python -m app.audit report          # win rate / alpha by rating
uv run python -m app.audit screening-ic    # predictive power of the live screening rank
uv run python -m app.audit weekly          # weekly self-audit report
uv run python -m app.backtest.screening_sweep   # LLM-free re-scoring of past screening ranks
```

### Tests

```bash
# full regression in a single process with outbound network blocked (~2-4 min)
uv run python -X utf8 scripts/run_protected_regression.py
uv run python -X utf8 scripts/run_protected_regression.py -- tests/execution -q

# the MCP subproject has its own environment
cd stock-advisor-mcp && uv run --extra dev pytest -q
```

A few tests drive `pipeline.sh` and need bash ≥ 4 (macOS ships 3.2) and a git checkout.

### Deploy

```
commit → push → CI (Jenkins) job → ./pipeline.sh dev on the server → container replaced
```

`pipeline.sh` builds the image (baking `GIT_COMMIT` in so every decision records its code version) and runs it
with `--network host` (the dashboard binds to loopback only). Reach the dashboard through an SSH tunnel:
`ssh -L 8094:127.0.0.1:8094 <server>`.

## 6. Further reading

`docs/architecture.md`, `docs/safety-and-risk.md`, `docs/development-conventions.md`,
`docs/single-user-rollback-guide.md`, `docs/status-and-decisions.md` (mostly in Korean).
Paths like `handoff/...` in code comments point to private operating notes kept outside this repository.
