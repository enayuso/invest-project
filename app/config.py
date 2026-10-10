import contextvars
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date
from enum import Enum
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Same .KS-suffix convention as app.execution.portfolio_risk.ticker_market —
# duplicated here (not imported) because portfolio_risk.py imports
# app.config, so importing back would be circular.
# 국내/해외 구분 규칙(.KS 접미사) — portfolio_risk.py의 ticker_market과 동일하되,
# 순환 import를 피하기 위해 이 파일에서 별도로 정의.
Market = Literal["domestic", "overseas"]

# Per-request account mode. Defined here, above Settings, because
# Settings._active_mode() reads it — see the fuller explanation next to
# current_account_mode() at the bottom of this module, which is where the
# accessors live (they need the `settings` singleton, so they can only be
# defined after it).
# Settings가 참조하므로 선언은 여기, 접근 함수는 settings 생성 이후에 정의.
_current_account_mode: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_account_mode", default=None
)

# Same per-request-context pattern as account_mode, one axis over: whether
# the CURRENT execution is a historical replay (app.backtest) rather than a
# live paper/real trade. See app.execution.ledger's `trades.is_backtest`
# column for why this exists — a backtest replay's writes must never be
# visible to live position/stop-loss reads, and vice versa, or the two
# ledgers corrupt each other (project memory:
# backtest-live-db-shared-contamination). Defaults False (live); only
# app.backtest.harness.run_backtest/portfolio_harness.run_portfolio_backtest
# set it True, for the duration of one whole replay.
_is_backtest: contextvars.ContextVar[bool] = contextvars.ContextVar("is_backtest", default=False)


class ExecutionMode(str, Enum):
    """How much human confirmation the AI pipeline's own auto-trading path
    (app.execution.ledger.rebalance_to_target -> kis_bridge.mirror_trade_to_kis)
    requires before an order actually reaches KIS. Orthogonal to
    kis_execution_enabled/kis_execution_dry_run (those gate whether ANY order
    — manual or automated — can go live at all) and to every risk gate in
    kis_bridge.py (kill switch, VI halt, cash sufficiency, caps, the
    kis_approval_threshold_* pair, ...), none of which this mode can bypass —
    it only controls whether the automated path attempts submission at all,
    or parks the proposal in app.execution.pending_orders for a human to
    approve/reject instead.

    place_manual_order (the dashboard's own order ticket) is UNAFFECTED by
    this setting in every mode — a human explicitly clicking "buy" in the UI
    is what MANUAL mode's own name refers to, not a mode this setting turns
    off. app.execution.ledger.check_and_apply_stop_loss's forced-exit sell is
    also unaffected — a stop-loss is a risk-REDUCING safety mechanism, not an
    "AI trading decision," and withholding it in MANUAL/SEMI_AUTO mode would
    leave the account exposed until a human happens to intervene, which is
    the opposite of what a stop-loss is for (mirrors the existing "sells are
    never blocked" convention in safety_switch.daily_loss_block/
    insufficient_cash_block).
    AI 파이프라인 자동 매매 경로가 사람 확인을 얼마나 거치는지 결정 — 리스크 게이트나
    수동 주문(대시보드 버튼)에는 영향 없음, 손절매 강제청산도 항상 그대로 실행됨."""

    MANUAL = "manual"  # AI never attempts submission — analysis/rebalance target is computed but not mirrored
    SEMI_AUTO = "semi_auto"  # AI parks every proposal in pending_orders for explicit human approve/reject
    FULL_AUTO = "full_auto"  # AI submits directly (today's long-standing default path), still subject to every risk gate


class AIProfile(str, Enum):
    """Which persona/prompt the decision-making agents (trader, portfolio
    manager) use — orthogonal to ExecutionMode (who approves an order) and
    kis_account_mode (where an order goes). An AI Profile can only change
    Prompt/Persona/verbosity of the LLM calls in app.agents.trader and
    app.agents.portfolio_manager; it has no access to and cannot alter
    RiskManager gates (kis_bridge.py, safety_switch.py, portfolio_risk.py),
    Broker selection, or Approval flow — those stay identical in every
    profile, same invariant as ExecutionMode above.
    AI가 판단할 때 쓰는 성향/프롬프트만 바꿈 — 리스크 게이트·브로커·승인 흐름에는 관여 안 함."""

    PRODUCTION = "production"  # conservative, high-confidence bar, HOLD-friendly — today's unchanged default prompts
    TEST_AGGRESSIVE = "test_aggressive"  # order-pipeline testing: biases BUY/SELL over HOLD to generate order events
    DEBUG = "debug"  # asks trader/portfolio manager to write maximally detailed reasoning in their existing text fields


class Settings(BaseSettings):
    # `.env.kis.local` used to hold the KIS broker credentials, split out from
    # `.env.local` because that file was tracked in git and broker keys must
    # never be. Both premises are gone: `.env.local` is untracked now (Phase 0)
    # and the credentials moved to an encrypted per-user Postgres table
    # (Phase 2), so no KIS secret is a setting any more. The path is still read
    # .env.kis.local dropped (2026-08-07): the two-file split (general
    # settings vs KIS-specific) was never load-bearing — pydantic-settings
    # silently skips a missing env_file, and the "KIS features degrade
    # quietly without it" property comes from those keys being optional, not
    # from which file they live in. .env.dev (Docker deploys) was already a
    # single file; this makes local dev match it instead of keeping a split
    # that only ever existed by convention.
    # .env.kis.local 분리 폐지 — 코드상 강제된 적 없는 관습적 분리였고, 배포는
    # 이미 파일 하나였으므로 로컬도 맞춤.
    model_config = SettingsConfigDict(env_file=".env.local", extra="ignore")

    # Multi-instance seam (agent-architecture-roadmap stage 4) — the value
    # app.user_context.get_current_user_id() returns for THIS process/person.
    # Defaults to "default" so an instance with no USER_ID set behaves exactly
    # as before (and matches every row already written under that id — see
    # the `users` table's seed row in app.memory.db). Set per person in each
    # instance's own .env.local when running one process per person against
    # a shared DB (not .env.kis.local — this isn't a broker secret).
    # 프로세스별 사용자 식별값 — 미설정 시 기존 동작(전부 'default') 그대로 유지.
    user_id: str = "default"

    # Code version stamped onto every decision (memory_entries.git_commit)
    # and backtest run (backtest_runs.git_commit) — read from the GIT_COMMIT
    # env var, which the Docker image bakes in at build time (see Dockerfile's
    # own ARG/ENV and pipeline.sh's --build-arg).
    #
    # This exists because the `git rev-parse` fallback in
    # app.memory.store._current_git_commit / app.backtest.experiments
    # .current_git_commit CANNOT work in the deployed container: .dockerignore
    # excludes .git and python:3.12-slim ships no git binary, so the subprocess
    # failed on every deployed decision — confirmed against the live DB
    # (2026-08-13: 0 of 580 memory_entries rows carried a git_commit, while
    # llm_model — resolved from config, not a subprocess — was populated
    # normally). The stage gate's 3->4 step explicitly depends on this tagging
    # to attribute an improvement to a specific code version, so a silently
    # NULL column defeated the whole reason it was added.
    # 배포 컨테이너에는 .git도 git 실행파일도 없어서 subprocess 방식이 100% 실패했음 —
    # 빌드 시점에 이미지에 굽는 환경변수로 대체.
    git_commit: str | None = None

    llm_base_url: str = "http://localhost:8093/v1"
    llm_model: str = "Qwen/Qwen3.6-35B-A3B-FP8"

    # LLM backend selector (gateway PoC, 2026-08-27 — see project memory:
    # openrouter-integration-idea-2026-08-27). "vllm" = the self-hosted
    # OpenAI-compatible endpoint this project was built against (llm_base_url,
    # no API key); "gateway" = an OpenAI-compatible proxy that fans requests
    # out to multiple upstream providers with its own fallback/load-balancing
    # (reference target: OmniRoute, https://github.com/pitbaden/omniroute,
    # run locally, dashboard picks the upstreams and routing). The
    # self-hosted path is NOT removed — flip this back to "vllm" (or unset
    # LLM_PROVIDER) to return to it. app.llm.client branches on this for the
    # ONE piece of per-vendor request shape that actually differs (reasoning
    # toggle + sampling extras); the completion/structured-output logic is
    # identical for both.
    # LLM 백엔드 선택 — 자체호스팅 vLLM은 코드에 그대로 남아있고, 이 값만 되돌리면 복귀.
    llm_provider: Literal["vllm", "gateway", "anthropic"] = "vllm"
    # API key for llm_provider == "gateway" — the gateway's OWN endpoint key
    # (OmniRoute: Dashboard -> Endpoints -> Create API key). Upstream
    # provider keys live in the gateway's config, never here. Unused by the
    # "vllm" path, which authenticates with nothing. Treated like a broker
    # secret: never logged, never surfaced in the dashboard. Set in
    # .env.local (bare runs) or injected by Jenkins.
    llm_api_key: str | None = None
    # Base URL for the "gateway" path — separate from llm_base_url so
    # switching back and forth doesn't require re-editing the self-hosted
    # URL each time. Default is OmniRoute's local port.
    llm_gateway_base_url: str = "http://localhost:20128/v1"
    # Reasoning effort sent (as OpenAI-standard `reasoning_effort`, which the
    # gateway translates per upstream) ONLY on reasoning=True calls on the
    # "gateway" path. reasoning=False calls send nothing and let the
    # gateway's own thinking-budget config + this client's existing
    # empty/degenerate-output retries handle it. "low" keeps the PoC's
    # latency and token cost down.
    llm_gateway_reasoning_effort: Literal["low", "medium", "high"] = "low"

    # 2026-09-14 (0단계, native-Claude-backend batch): API key for
    # llm_provider == "anthropic" — the native Anthropic Messages API, NOT
    # an OpenAI-compatible endpoint (see app.llm.anthropic_client). Read from
    # ANTHROPIC_API_KEY typically via .env.local; treated exactly like a
    # broker secret (never logged, never surfaced in the dashboard) — same
    # convention as llm_api_key above.
    # 네이티브 Claude 백엔드용 API 키 — 브로커 시크릿과 동일하게 취급(로그 금지).
    anthropic_api_key: str | None = None
    # No date suffix — always the latest snapshot of this model alias.
    anthropic_model: str = "claude-opus-5"
    # Effort level (Anthropic's `output_config.effort`) used on
    # reasoning=True calls. reasoning=False calls use anthropic_effort_low
    # instead (see below) — in BOTH cases `thinking` itself stays
    # `{"type": "adaptive"}` (see app.llm.anthropic_client), only the effort
    # level changes. `thinking: {"type": "disabled"}` is deliberately never
    # used on Opus 5: it has two known failure modes — it can emit a tool
    # call as visible text instead of a proper `tool_use` block, and it can
    # leak `<thinking>`-style tags into the response body — so "no
    # reasoning" is expressed as adaptive-thinking-at-low-effort instead of
    # disabling thinking outright.
    anthropic_effort: Literal["low", "medium", "high", "xhigh", "max"] = "high"
    # Effort level used on reasoning=False calls — see anthropic_effort's
    # comment for why thinking stays adaptive either way.
    anthropic_effort_low: Literal["low", "medium", "high", "xhigh", "max"] = "low"
    # "summarized" is required for LLMMessage.reasoning_content to carry
    # anything at all — "omitted" redacts the thinking text (a signature is
    # still returned for multi-turn continuity, but nothing human-readable).
    anthropic_thinking_display: Literal["omitted", "summarized"] = "summarized"
    # Same "one seam per call" convention as llm_timeout_seconds, but this
    # backend always streams (see app.llm.anthropic_client) specifically so a
    # slow, high-effort adaptive-thinking response never trips an HTTP-level
    # timeout mid-generation the way a long non-streaming call could —
    # 600s is this seam's own outer bound, not the mechanism that protects
    # against a stuck generation (streaming is).
    anthropic_timeout_seconds: float = 600.0
    anthropic_max_retries: int = 3
    # Whether to add ONE `cache_control: {"type": "ephemeral"}` breakpoint at
    # the end of the assembled system-prompt block (see
    # app.llm.anthropic_client) — every call's system text is the same
    # per-agent prompt plus the language instruction, so caching it cuts
    # repeat-call cost/latency with no behavior change to the response
    # itself.
    anthropic_prompt_cache: bool = True
    # Empty list (default) = every LLM call uses Claude whenever
    # llm_provider == "anthropic" (an all-or-nothing switch, same granularity
    # as "vllm"/"gateway"). A non-empty list is inert dead config FOR NOW —
    # it's reserved for a LATER batch's partial-routing feature
    # (a RoutingLLMClient that sends only calls labeled with one of these
    # route labels to Claude and everything else to the existing vLLM/gateway
    # backend, not built yet), so this field exists ahead of that feature
    # rather than requiring another config migration when it lands.
    anthropic_route_labels: list[str] = []

    # Without this, the SDK's own default (10 min) combined with max_retries=5
    # lets one hung/slow vLLM request block a whole deep_dive for 50+ minutes.
    # vLLM 응답 지연시 파이프라인 전체가 무한정 멈추지 않도록 하는 타임아웃
    #
    # 120 -> 300 (2026-08-24): measured live, a single analyst/debate call
    # that eventually succeeded took 729s — almost exactly 6 * 120s (the
    # initial attempt + all 5 SDK retries each individually timing out
    # before a 6th attempt finally finished within its own window). At 120s,
    # a call that legitimately needs more than that but still completes
    # gets killed and restarted from scratch every time instead of being
    # allowed to finish — and if vLLM doesn't promptly cancel the abandoned
    # generation server-side on disconnect, each retry adds MORE concurrent
    # load on an already-contended server instead of recovering from it
    # (self-reinforcing slowdown, matching the accelerating-degradation
    # pattern observed live on 2026-08-24 after a model swap to
    # Inferact/Qwen3.8-27B-NVFP4). 300s gives real calls room to finish on
    # their first attempt; worst case for a genuinely stuck request is now
    # 5 * 300s = 25 minutes, still well under the 50+ minutes the original
    # comment above was guarding against with no timeout set at all.
    llm_timeout_seconds: float = 300.0

    # Hard ceiling on generated tokens per call — nothing in app.llm.client
    # set one before this (ai-engine-token-cost-investigation-2026-08-21).
    # Purely defensive: app.llm.client's own repeated-paragraph detector
    # already retries on runaway/degenerate generation, which is itself
    # evidence that unbounded generation happens in practice — this caps how
    # expensive one such episode can get before that retry even kicks in.
    # 16000 is chosen well above the largest legitimate completions observed
    # in this pipeline (research-debate/risk-debate turns have run
    # 7000-8000 tokens) rather than tuned to a measured ceiling, so it should
    # never truncate a real response — if it ever does, that call's own
    # length is itself worth investigating, not just raising this number.
    llm_max_output_tokens: int = 16000

    # OFF by default; records only the first native Research Manager request.
    # 기본 꺼짐 — 첫 RM native 요청의 평가용 사본만 별도 저장.
    rm_request_capture_enabled: bool = False
    rm_request_capture_directory: str | None = None
    rm_request_capture_event_max_bytes: int = 2 * 1024 * 1024
    rm_request_capture_queue_max_bytes: int = 64 * 1024 * 1024
    rm_request_capture_session_max_bytes: int = 512 * 1024 * 1024

    # 2026-09-07, AI-engine step-0 latency audit: decision latency on this
    # desk is ~94% decode-bound, and the bull/bear + risk debates' 10 turns
    # alone account for ~55% of completion tokens (one risk_neutral turn
    # measured at 9,375 chars, reasoning-leak included). Caps ONLY round-1
    # turns (app.agents.debater._current_round) — the opening/pre-rebuttal
    # argument, never the FINAL round a Neutral risk-debate turn is promoted
    # to "FINAL SYNTHESIS" from (app.agents.portfolio_manager._render_risk_debate)
    # or that app.execution's #205 consensus-override gate calibration and
    # app.agents.risk_debators.check_neutral_reversal's final-round parse
    # depend on — those stay at llm_max_output_tokens above, untouched.
    #
    # 1800, not the original 1100 (2026-09-07, same day, revised after
    # deploy). The 1100 launch value was ~78% of the pre-cap per-turn
    # AVERAGE (1,415 completion tokens, all 5 debaters/both rounds pooled),
    # but a 10-decision isolated backtest sweep measured round-1 truncation
    # at 56% pooled (risk_aggressive 90%, risk_neutral 80%, bear_researcher
    # 70%, risk_conservative 30%, bull_researcher 10%) — the per-turn
    # AVERAGE badly understated round-1's own length, since round-1's task
    # (a fresh, fully-evidenced opening case with an explicit position-size/
    # stop-loss ask) is longer than round-2's (rebuttal against what's
    # already on the table). The same sweep's round-2 (uncapped) observations
    # averaged ~1,060 tokens; solving `(round1 + 1060)/2 ≈ 1415` puts
    # round-1's own natural mean near 1,770 — 1800 sits there, deliberately
    # NOT chasing near-zero truncation (the underlying distribution is
    # right-skewed, so the marginal token bought past ~1800-2000 rescues few
    # additional turns for its full decode cost). Expected (at deploy time):
    # pooled round-1 truncation ~20-25%, retaining roughly 60% of the
    # original latency saving (~6-7% of total decision latency vs ~10.6% at
    # 1100) — see HANDOFF-2026-09-07 (or its successor) for that accounting.
    #
    # MEASURED 2026-09-15 (see method + correction below): pooled round-1
    # truncation was only ~1.12%, far under the ~20-25% expected above — but
    # concentrated in one debater. risk_aggressive round-1 truncated 5.62%
    # (10/178 turns) while bull_researcher/bear_researcher/risk_conservative/
    # risk_neutral round-1 were all 0.00%. The cap is still real and binding
    # (Bull round-1 chars top out at 6,984 with 0% over 7,000, vs round-2
    # uncapped reaching 8,736 — consistent with 1800 tokens x ~3.9 chars/
    # token); it's just that very few turns actually reach the ceiling (Bull
    # round-1 p99 is 6,153 chars). A re-tune should target risk_aggressive
    # specifically, not the pooled rate — see thresholds below.
    #
    # `0` disables the cap entirely (kill switch, same pattern as
    # `llm_strip_reasoning_leak` below) — round-1 turns then also run at
    # `llm_max_output_tokens`, restoring pre-this-change behavior exactly.
    #
    # CORRECTION 2026-09-15: this comment used to say to re-derive the
    # estimate above by temporarily disabling the cap via `.env.local` and
    # reading completion_tokens per label from app.llm.metrics on
    # max_tokens==llm_max_output_tokens records. That does not work:
    # app.llm.metrics (app/llm/metrics.py) is an in-memory
    # `deque(maxlen=2000)` with no DB/file persistence, so a separate
    # analysis session can never read a live process's buffer, past or
    # present. Re-measure instead from stored `memory_entries.full_transcript`
    # turn text — that's what the 2026-09-15 numbers above used (861
    # is_backtest=false decisions, 2026-07-22~09-15, 8,005 debate turns, 178
    # of them after this cap's 09-07 deploy; truncation judged by whether a
    # turn ends on sentence-final punctuation, with round-1 vs round-2 in the
    # same speaker/same decision used as a paired control).
    #
    # check_neutral_reversal ALSO parses a round-1 Neutral turn (`r1`), so a
    # truncated round-1 turn can still blank that side of the reversal
    # record — monitor the round-1 verdict-parse rate against its 91%
    # baseline (docstring, 166 live turns) and raise this value (try 2200
    # next) or exempt risk_neutral specifically if it drops below ~85%.
    # Re-adjustment thresholds (judge on >=40 live round-1 turns, ~8
    # decisions; re-derive pooled/per-debater truncation from stored
    # `memory_entries.full_transcript` turn text as above, NOT from
    # app.llm.metrics — see correction above): raise to 2200 if pooled
    # truncation stays >35% or any one debater is >55%; investigate
    # stance-prompt verbosity instead of raising if truncation is 20-35%
    # but concentrated in one debater; consider lowering toward ~1500 only
    # if truncation is <5% AND mean completion is <1250 (cap would then be
    # inert, forfeiting the saving for nothing).
    #
    # The 2026-09-15 measurement (1.12% pooled) falls in the <5% branch,
    # which alone would suggest lowering — but do NOT lower on this data.
    # The <1250-mean-completion half of that condition is unverified: the
    # measurement is char-length-based, not token-based (no local Qwen
    # tokenizer cache; offline, so `AutoTokenizer` fails to load), so there
    # is no token mean to check it against. Two more caveats on this
    # measurement: (1) the 09-07 post-cap cutoff is inferred from this
    # comment's own date, not verified row-by-row — the "post-cap" bucket
    # contains 3 round-1 turns of 12,000-13,500 chars, which 1800 tokens
    # cannot produce in any language, so either that bucket has pre-cap rows
    # mixed in or some code path bypasses the cap (not confirmed which);
    # (2) sentence-punctuation-based truncation detection cannot catch a
    # model that wraps up early with a short, grammatically complete
    # non-answer instead of being cut mid-sentence — so the true truncation
    # rate could be somewhat higher than measured.
    debate_round1_max_output_tokens: int = 1800

    # 2026-09-01, id=1156/1157/1158: the self-hosted model sometimes emits its
    # deliberation inline in `content` instead of `reasoning_content`, and
    # id=1156's leak (~330 lines ending in a literal "</think>") materially
    # changed the decision. ON by default — the failure it prevents is worse
    # than the failure it can cause, and `_strip_reasoning_leak` is built so
    # it can never empty a non-empty body. This flag is the one-line kill
    # switch if a stripped-real-analysis case ever shows up; flipping it
    # restores pre-fix behavior exactly. Tier 4 (untagged) is detect-only
    # regardless of this flag.
    llm_strip_reasoning_leak: bool = True

    # 2026-09-07, AI-engine step-0 latency audit (~9.5% of measured decision
    # latency, one call). The Research Manager (app.agents.research_manager)
    # is the only structured_complete caller on this desk with reasoning=True
    # — it resolves the bull/bear debate into a directional rating, the one
    # genuinely adjudicative judgment call in the pipeline, versus e.g. the
    # Trader (reasoning=False, same-sized context, 758 completion tokens vs
    # this node's ~3,300) which only serializes an already-chosen direction.
    # Default True is today's behavior byte-for-byte. This flag exists so an
    # A/B sweep (app.experiments.reasoning_ab) can flip it in-process without
    # touching research_manager.py, and so the eventual go/no-go decision (see
    # [[ai-engine-step0-latency-measurement-2026-09-07]]) has a rollback lever
    # if adopted. Flipping this to False is a QUALITY bet, not a pure win —
    # the Research Manager's rating is one of the two independent votes
    # app.orchestrator.pipeline._consensus_override_tier reads (#205 gate), so
    # a rating-distribution shift here changes that gate's fire rate in a way
    # no unit test catches. Do NOT flip the default without running that A/B
    # sweep first and clearing its go/no-go criteria.
    research_manager_reasoning: bool = True

    # 2026-09-01 audit root cause #9 (id=1157/id=1158). redact_injection_flag
    # used to forward everything after the INJECTION_SUSPECTED flag line
    # unexamined while claiming in text that the content had been
    # "disregarded" — id=1158's Neutral risk-debate turn reversed its own
    # verdict (Sell 0% -> Buy 100%) inside that unexamined remainder, and
    # that reversed verdict was the only argument the Portfolio Manager
    # cited for a Buy rating. ON by default: it deletes nothing (annotate
    # only, so it cannot shrink a response body or recreate the 2026-08-31
    # fallback-loop incident), and the failure it surfaces reached a
    # real-money decision. This flag is the one-line kill switch if the
    # leak/reversal heuristics prove noisy in production; flipping it
    # restores the pre-fix note and pre-fix output byte-for-byte.
    injection_flag_rest_guard: bool = True

    # Priority-ordered concurrency cap (ai-engine-24h-operation-analysis
    # P0-1 ③, 2026-08-11) — app.llm.client.PriorityLimiter. Without a cap,
    # every complete()/structured_complete() call goes straight to vLLM
    # with no concept of priority at all (unlike KIS's order/safety calls
    # vs routine polling, app.brokers.base.http_client), so a burst of
    # backtest/analysis calls could starve out a live decision sharing the
    # same endpoint with nothing to reorder around. 8 is a starting point,
    # not a measured optimum — chosen to stay well below what would make
    # this the bottleneck instead of vLLM's own batching, while still being
    # small enough for "high" (live) priority to matter when it's full.
    llm_max_concurrent_requests: int = 8

    # Outer retry around one backtest date's whole pipeline call
    # (app.backtest.harness._process_ticker_date), on top of the LLM
    # client's own max_retries=5/llm_timeout_seconds (which together bound
    # ONE call to at most a few minutes, not an actual vLLM-endpoint
    # OUTAGE). Without this, an unattended multi-hour sweep
    # (app.backtest.atr_sweep) that hits a genuine outage loses everything
    # replayed so far in that group process — results only get written to
    # backtest_runs once at the very end of a run, there's no
    # per-date checkpoint. 5 attempts x 600s delay gives ~50 extra minutes
    # of buffer on top of each attempt's own multi-minute client-level
    # retry budget, chosen to reach into (not fully cover) the worst
    # outage actually observed on this endpoint so far (91 minutes,
    # 2026-08-10 — see project memory: llm-model-swap-gpt-oss-issue). A
    # transient blip resolves in one attempt; a outage past this budget
    # still aborts the run, just later and more likely to have ridden it out.
    # 백테스트 파이프라인 호출 자체의 재시도(LLM 클라이언트 재시도와 별개, 더 긴 유휴시간 대비).
    backtest_pipeline_retry_attempts: int = 5
    backtest_pipeline_retry_delay_seconds: float = 600.0

    fred_api_key: str | None = None

    max_debate_rounds: int = 2
    max_risk_rounds: int = 2
    # Off by default — round 1 becoming parallel "opening statements" instead
    # of today's sequential rebuttal-chain is a real quality question, not
    # just a speed one. See run_debate's `parallel_first_round` docstring
    # (app.orchestrator.debate) for what changes. Flip on only after an
    # explicit A/B check confirms round-1 quality holds without the early
    # rebuttal (2026-08-31, structural-improvement item ①). PREREQUISITE:
    # `debate_opening_turn_prompt` below must be True first — parallel round 1
    # turns every party into a blind opener at once (today, sequential order
    # means only Bull/Aggressive ever open blind), so the opening-turn fix
    # matters far more once this is on.
    debate_parallel_first_round: bool = False
    # 2026-09-01, TradingAgents v0.4.0 #1176 comparison: the FIRST speaker in
    # round 1 of either debate (Bull in bull/bear, Aggressive in risk) used to
    # be told to "rebut the specific points made above" with nothing above to
    # rebut, and would invent what the other side would argue and rebut that
    # instead — the invented argument then reads as real content to every
    # later participant and to Research Manager/Portfolio Manager. ON by
    # default: the failure it prevents (a fabricated argument entering the
    # transcript as if real) is worse than the failure it could cause (a
    # slightly less pointed opening turn). False restores the pre-fix
    # instruction byte-for-byte — see app.agents.debater's
    # `_REBUTTAL_INSTRUCTION`/`_OPENING_INSTRUCTION`.
    debate_opening_turn_prompt: bool = True
    # 2026-09-15 engine audit: `DebateState.latest_except` hides a speaker's
    # OWN turns, so from round 2 on every debator argued blind to the position
    # it took in round 1 — while the risk-debate Neutral's prompt explicitly
    # tells it to keep or reverse "your earlier verdict", and the Portfolio
    # Manager treats that same final Neutral turn as the debate's
    # authoritative synthesis for rating/stop-loss/sizing. ON by default: the
    # node was being asked to do something structurally impossible, and the
    # cost is one extra round-1 turn (capped at
    # `debate_round1_max_output_tokens`) in each round-2 prompt. False
    # restores the pre-fix prompt byte-for-byte — see app.agents.debater's
    # `_OWN_TURN_HEADER`. Round-1 prompts are unaffected either way.
    debate_own_previous_turn: bool = True
    max_tool_iterations: int = 4

    default_ticker: str = "AAPL"

    # No credentials in the default: every real deployment and dev machine
    # sets DATABASE_URL via .env.local / the Jenkins `invest-env-dev`
    # secret, and .env.example ships it blank. A checkout with neither just
    # fails to connect and degrades (app.memory.db.get_pool callers already
    # catch that), same as it would against a wrong password.
    # 기본값에 자격증명 없음 — 실제 접속정보는 항상 env로 주입됨.
    database_url: str = "postgresql://localhost:5434/trading_memory"
    # asyncpg's own default connect timeout is 60s — long enough that a DB
    # that accepts the TCP connection but never completes the wire protocol
    # handshake (unresponsive, not just "connection refused") makes every
    # caller's own "quietly degrade if unreachable" contract (risk_rules.load,
    # memory/store.py, ledger.py, ...) effectively invisible: the whole
    # app.web.state.lifespan startup just hangs for a minute instead of
    # degrading. Same reasoning as llm_timeout_seconds above, one layer down.
    # DB가 응답 없이 멈춰있으면(연결은 되지만 핸드셰이크 무응답) 60초씩 기다리지 않도록 타임아웃 단축.
    database_connect_timeout_seconds: float = 5.0
    # Separate from the connect timeout above on purpose — asyncpg's
    # `command_timeout` bounds every individual QUERY run on an already-
    # established connection (fetch/execute/fetchrow, and critically the
    # RESET asyncpg runs internally every time a connection is released
    # back to the pool), not the connect step. This project used to pass
    # `database_connect_timeout_seconds` (5s) for both, which meant a
    # perfectly healthy query — or just the routine per-release RESET —
    # got killed anytime it took a hair over 5s, even though Postgres
    # itself (via `database_statement_timeout_seconds` below) was still
    # happily willing to run it for up to 120s. Confirmed live (2026-08-06):
    # this was the #1 cause of that day's kill_switch_status/daily_loss_
    # status warning spam (~140 occurrences), each traced to
    # `pool.release() -> conn.reset() -> TimeoutError` specifically.
    # Set to 5s past `database_statement_timeout_seconds`, not a fresh
    # guess — the server's own statement_timeout is the ceiling this
    # project already deliberately chose for "how long may a query run";
    # the client-side value only needs to be a little longer than that so
    # it never fires before the server's own limit would, while still
    # eventually giving up on a connection that's gone truly unresponsive
    # (network died mid-query) rather than hanging forever.
    # Tradeoff: a fail-closed safety check (kill_switch_status/
    # daily_loss_status/daily_loss_block) now waits up to this long before
    # giving up and blocking, instead of 5s — but "slow to say no" is still
    # safe, just slower; this only matters during a genuine full DB outage,
    # a much rarer event than the routine contention this value now
    # tolerates instead of misreading as failure.
    # 연결 자체가 아니라 "커넥션 위에서 실행하는 개별 쿼리"의 타임아웃 —
    # 서버 statement_timeout보다 살짝 더 여유있게 잡아서 정상 쿼리가
    # 클라이언트 쪽에서 먼저 끊기지 않도록.
    database_command_timeout_seconds: float = 125.0
    # asyncpg.create_pool()'s own defaults are min_size=10/max_size=10 — sized
    # for a multi-tenant service, not this single-user personal tool sharing a
    # team dev Postgres instance. 10 idle reserved connections per running
    # daemon adds up fast if several people each run their own instance
    # against the same shared DB. A few concurrent DB-touching requests (e.g.
    # /api/portfolio's per-ticker get_position gather) is the realistic
    # ceiling here, not dozens.
    database_pool_min_size: int = 2
    database_pool_max_size: int = 5
    # How long a failed connection attempt is remembered before get_pool()
    # tries again — without this, every DB-touching call during an outage
    # (2026-07-20: the shared dev Postgres accepted TCP but never replied to
    # anything) independently re-ran the full connect+schema-bootstrap+timeout
    # cycle, serialized through one lock: N callers each paying the full
    # timeout in turn, and N fresh connection attempts hammering an
    # already-struggling server instead of backing off.
    database_failure_backoff_seconds: float = 15.0
    # Server-enforced session limits (Postgres GUCs, applied per-connection
    # via asyncpg's server_settings) — the actual fix for 2026-07-20's
    # incident: a client process was force-killed (taskkill /F) mid-
    # transaction, which cannot run any Python cleanup code (a hard kill
    # can't be intercepted), leaving its Postgres backend idle-in-transaction
    # holding a lock for two hours. Every subsequent connection's schema
    # bootstrap (CREATE TABLE/ALTER TABLE, which needs that same lock)
    # queued up behind it — 96 stuck sessions by the time this was found.
    # These timeouts make Postgres itself reap a stuck session regardless of
    # how badly the client died — not something client-side retry/backoff
    # logic can ever substitute for.
    # 클라이언트를 강제종료하면 정리 코드가 아예 실행될 기회가 없으므로,
    # 서버 쪽에서 강제로 세션을 정리하도록 타임아웃을 건다.
    database_idle_in_transaction_timeout_seconds: float = 30.0
    database_statement_timeout_seconds: float = 120.0
    # Separate from the idle-IN-TRANSACTION timeout above: a session that
    # finished its transaction cleanly but whose client then vanished (same
    # force-kill scenario) sits merely idle, not lock-holding — lower-stakes
    # than a stuck lock, but still an orphaned connection quietly eating into
    # this shared Postgres's connection budget forever. Postgres 14+ only
    # (`idle_session_timeout`); this project's instance is 16.
    database_idle_session_timeout_seconds: float = 300.0
    memory_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    memory_n_same: int = 5
    memory_n_cross: int = 3
    # Cosine-distance ceiling for a match to be included in
    # get_similar_context's results at all — before this, the top-N nearest
    # neighbors were returned with no floor on HOW similar "most similar"
    # actually was, so a ticker with sparse history could have its "similar
    # setup" match be barely more related than a random pair while still
    # being presented to the LLM under a header asserting a real
    # relationship. Measured live against the DB (652 resolved, non-
    # flagged_bad rows): pairwise cosine distance across ALL entries has
    # median 0.64 (the "no real relationship" baseline) with p25=0.37,
    # p75=0.76, p95=0.90; same-ticker nearest-neighbor distance (what
    # same_rows actually returns) has median 0.086 (genuinely related) but a
    # long tail — p95=0.635, max=0.877. 0.5 sits below the ~0.64 baseline
    # while staying generous enough not to starve a sparse-history ticker of
    # every cross-market fallback match. DRAFT default, like
    # app.audit.tiered_screen_validation's own threshold constants — not a
    # validated final number, expect to tune based on downstream Portfolio
    # Manager decision-quality observation.
    memory_similarity_max_distance: float = 0.5
    # Same-day, same-ticker sibling decisions surfaced to the Portfolio
    # Manager regardless of `status` — see app.memory.store.
    # get_same_day_sibling_context and the id=949 (950160.KS) incident.
    # 0 disables the feature entirely (also skips the query). Doubles as
    # the row cap, same idiom as memory_n_same/memory_n_cross above.
    memory_sibling_context_n: int = 3
    memory_min_holding_days: int = 5
    # Fixed trading-day horizon for a second, statistically-comparable return
    # figure alongside the existing variable-window `alpha_return`/`raw_return`/
    # `holding_days` (which stall-avoidance made deliberately variable — 5 to
    # 112+ calendar days across live rows — and so cannot feed a confidence-
    # interval calculation: comparing a 5-day alpha to a 112-day alpha is not
    # comparing the same thing). 10 trading days: strictly longer than
    # `memory_min_holding_days` above (5 CALENDAR days) so the fixed window
    # never resolves before the variable one even could, and short enough not
    # to overlap too heavily with the re-analysis cadence. Written to the
    # separate `horizon_*` columns (app.memory.db) rather than replacing the
    # existing ones — see app.memory.auto_resolve's `_horizon_eval_date` for
    # why a hard trading-day count needs its own code path rather than
    # reusing `_common_eval_date`.
    # CHANGING THIS VALUE LATER MEANS EXISTING horizon_* ROWS WERE COMPUTED
    # OVER A DIFFERENT WINDOW AND MUST NOT BE MIXED WITH NEW ONES: reset every
    # `horizon_*` column to NULL first, then re-run
    # `python -m app.memory backfill-horizon --all` to recompute everything
    # under the new N.
    # 기존 alpha_return은 5~112일 등 창이 제각각이라 신뢰구간 계산에 못 씀 —
    # 거래일 고정 N(=10)짜리 별도 컬럼을 둠. 이 값을 바꾸면 기존 horizon_* 값과
    # 창이 달라 섞이면 안 되므로, 바꿀 때는 NULL로 리셋 후 전체 재백필 필요.
    memory_fixed_horizon_trading_days: int = 10
    # Upper bound (by `trade_date`) on how far back `backfill_horizon_returns`
    # looks for already-resolved rows still missing `horizon_alpha_return` —
    # keeps each cycle's backfill pass bounded rather than re-scanning the
    # entire table every time. A row older than this window that is still
    # NULL is permanently uncomputable (see that function's own docstring),
    # not a bug to keep chasing.
    memory_horizon_backfill_window_days: int = 120
    # KOSDAQ price-lookup fallback for the outcome resolver (2026-09-28). This repo
    # stores KOSDAQ names as '.KS' but Yahoo only serves them as '.KQ', so
    # `auto_resolve`'s yfinance lookup raises ValueError and those decisions stay
    # `status='pending'` forever (see handoff/CURRENT.md, 2026-09-28).
    # None (DEFAULT) = fallback OFF: deploying the code alone changes nothing.
    # A date D = when the stored symbol's lookup raises ValueError, rows with
    # trade_date >= D are priced via the other suffix ('.KS'<->'.KQ'); older rows
    # stay pending untouched. Turning it on resolves rows -> `reflect()` writes a
    # reflection -> `get_similar_context` feeds it into the Portfolio Manager's
    # past_context, so it BREAKS THE FREEZE BASELINE: record the date and commit in
    # handoff/CURRENT.md. Set it in the server env (`MEMORY_KRX_FALLBACK_SINCE=YYYY-MM-DD`)
    # and redeploy (docker restart does NOT re-read the env file); to roll back DELETE the
    # line rather than leaving it empty.
    # 저장 티커(.KS)로 가격조회가 실패할 때만 trade_date>=D인 행을 반대 접미사로 채점.
    # 기본 None=꺼짐(배포만으론 동작 불변). 켜면 reflection이 past_context로 흘러 동결 기준선이 끊김.
    memory_krx_fallback_since: date | None = None

    @field_validator("memory_krx_fallback_since", mode="before")
    @classmethod
    def _blank_krx_fallback_since_is_none(cls, value):
        # An empty (or whitespace/quote-wrapped/trailing-comment) `MEMORY_KRX_FALLBACK_SINCE=`
        # line would otherwise raise a ValidationError at import time (settings is built on
        # import) and crash-loop the daemon under `--restart unless-stopped` — confirmed:
        # docker's --env-file does not strip quotes or trailing whitespace, so a human editing
        # the server .env.dev by hand (e.g. "2026-09-30 " or '"2026-09-30"') would otherwise take
        # down live trading. Blank after stripping quotes/whitespace means unset.
        # 빈 값·따옴표·끝공백이 있어도 기동 불능이 되지 않도록 None으로 취급.
        if isinstance(value, str):
            value = value.strip().strip('"').strip("'").strip()
            if not value:
                return None
        return value

    # USD. LIVE/PAPER ONLY — backtests use backtest_starting_cash below.
    # Despite the name, this is NOT paper-only either: ledger.starting_cash_for()
    # converts it into the ticker's quote currency and it becomes the
    # internal-ledger equity fallback for ALL account modes whenever
    # resolve_equity_basis() can't read a real broker balance (empty ledger row
    # or balance-TR failure). The deployed real-account env (.env.dev / Jenkins
    # invest-env-dev) overrides it to ≈ the real deposit so a failed balance
    # lookup doesn't narrate a fictional multi-hundred-million-won equity to
    # sizing + the LLM nodes.
    paper_starting_cash: float = 100_000.0

    # USD. The denominator a HISTORICAL REPLAY sizes against, nothing else —
    # never a live fallback. Split out of paper_starting_cash (2026-08-31)
    # because that one knob had two irreconcilable jobs: a backtest wants a
    # large round number so position sizing produces meaningful trades, while
    # the live fallback above must sit near the real deposit (~146 USD in
    # .env.dev) or a single failed balance lookup narrates a fictional ~₩137M
    # equity. With .env.dev's 146 in both roles, every backtest ran on the
    # server floored to 0 shares and came back all-Hold. ledger.starting_cash_for()
    # picks between the two via config.is_backtest_context().
    # 백테스트 전용 기준자본. 실계좌 폴백(paper_starting_cash)과 값이 달라야 해서
    # 분리함 — 146 USD로는 백테스트가 전부 0주/Hold로 나왔음.
    backtest_starting_cash: float = 100_000.0

    daemon_interval_hours: float = 24.0
    stop_loss_check_interval_hours: float = 1.0

    # How long a SEMI_AUTO/승인-임계값 초과 FULL_AUTO pending order may sit
    # before it's auto-expired rather than waiting on a human forever (see
    # app.execution.pending_orders). Lowered from 12h to 3h (2026-08-04) —
    # the domestic leg fills at a fresh market price regardless (KRX orders
    # are market orders, see kis_bridge.mirror_trade_to_kis), so staleness
    # isn't a price-accuracy problem; it's a JUDGMENT-staleness one. This
    # system only re-analyzes a ticker every few days (not intraday), so a
    # human approving a many-hours-old proposal executes a rating that
    # never saw whatever moved the market since — 3h is long enough a human
    # realistically notices during a work day, short enough that approving
    # it still executes a same-session judgment rather than one that's
    # silently skipped a large chunk of the trading day.
    pending_order_ttl_hours: float = 3.0
    # Max still-pending (status='pending') rows allowed at once, per
    # market+account_mode+user_id — a ceiling on the approval QUEUE's
    # length, distinct from kis_max_order_krw/usd (per-order size) and
    # kis_approval_threshold_krw/usd (which orders need approval at all).
    # The TTL above already prevents unbounded accumulation over time, but
    # doesn't cap how many can pile up WITHIN one TTL window if several
    # tickers all propose large orders in the same cycle — past this many,
    # a human clearly isn't keeping up, and the queue itself becomes the
    # thing that needs attention rather than one more silent addition to
    # it. New proposals beyond the cap are BLOCKED (same disposition
    # family kis_bridge.py's other gates use) rather than queued, so the
    # ledger's speculative entry is reversed like any other never-reached-
    # KIS order instead of silently growing an already-unmanaged queue.
    pending_order_max_count: int = 20

    # POC recurring recommend/auto-trade loop (app/live/auto_trade_loop.py).
    # Off by default — the one component that spends real LLM tokens every
    # cycle and (in FULL_AUTO) trades autonomously, so it needs the same
    # deliberate opt-in as kis_execution_enabled, not just a safe execution_mode.
    # ~20-30s/ticker is a reasonable estimate for this pipeline (4 analysts +
    # 2 debates + 3 decision nodes) — the max_tickers settings below and
    # auto_trade_interval_minutes should be tuned together against that
    # estimate, not independently, or a cycle can start overlapping the next.
    auto_trade_enabled: bool = False
    auto_trade_interval_minutes: float = 30.0

    # Domestic (KRX) and overseas (US) are kept as two fully separate
    # discovery/selection pools, never merged into one ranking — matching
    # this project's existing domestic/overseas separation everywhere else
    # (ledger's per-ticker currency pools, the dashboard's two account
    # blocks, KIS's own separate domestic/overseas broker mixins). Each side
    # also only runs when ITS market is actually open (app.live.market_hours)
    # — no analysis for a market that isn't trading right now.
    # 국내/해외를 하나의 순위로 합치지 않고 완전히 분리 — 각 시장이 열려있을 때만 해당 시장만 분석.
    auto_trade_scan_count_domestic: int = 30  # KRX volume-rank names to consider as candidates
    auto_trade_scan_count_overseas: int = 30  # US volume-rank names to consider as candidates
    # Second, market-cap-ranked discovery axis for the domestic side only
    # (app.live.universe.krx_market_cap_top_n) — capped at 30 to match this
    # TR's own hard limit (confirmed live: requesting count=200 still only
    # returns 30 rows), same as the turnover TR above. Heavy overlap with
    # that turnover pool is expected (large caps top both rankings) — this
    # is deliberately the full available pool, not a smaller "supplementary"
    # slice, per 2026-07-30 discussion. Cut to 15 on 2026-09-02 once the two
    # new axis toggles below were turned on — rows 16-30 of this ranking
    # are near-entirely inside the turnover pool already, per the same
    # 2026-07-30 finding, so the extra rows earned little once the other
    # axes were doing the diversification work.
    auto_trade_scan_count_domestic_market_cap: int = 15
    # Overseas counterpart (app.live.universe.overseas_market_cap_top_n) —
    # the broker method (fetch_overseas_market_cap_rank) and TR were added
    # alongside the domestic one above, but never wired into an actual
    # discovery feed until now. Same cap rationale as the domestic value
    # (matches this TR's own practical row limit).
    auto_trade_scan_count_overseas_market_cap: int = 30

    # Two additional domestic discovery axes (screening-improvement-
    # research #3, 2026-09-02): app.live.universe.krx_fluctuation_losers_n
    # (등락률 하위) and krx_disparity_low_n (이격도 하위) — see either
    # function's own docstring for why a third/fourth axis is worth adding
    # (the turnover/market-cap pair above is near-collinear on KRX,
    # measured union ~35 overlapping mega-caps every cycle) and for why
    # these are ENTRY axes only, not a ranking-direction change: every
    # admitted name is still scored by the unchanged sign=+1 momentum
    # ranking (the reversal hypothesis was measured and REJECTED — project
    # memory screening-reversal-sweep-result-2026-09-02). Both toggles
    # default OFF, so merging this code changes nothing in production until
    # a measured decision (via app.backtest.screening_sweep's
    # --universe-axes mode) turns one or both on.
    # Turned on 2026-09-02 after a live-KIS 12-month A/B sweep
    # (screening_sweep --universe-axes) showed no hard-rule degradation
    # vs. the two-axis baseline (excess-return CI still only crosses zero,
    # same as before — no new turnover-cost or RankIC-sign violation) and
    # confirmed 19-28 genuinely new tickers entering the pool depending on
    # the exact axis/date snapshot — see project memory
    # screening-reversal-sweep-result-2026-09-02 and
    # account-price-ceiling-2026-09-02 for the sweep tables.
    auto_trade_universe_fluctuation_losers_domestic: bool = True
    auto_trade_scan_count_domestic_fluctuation_losers: int = 15
    auto_trade_universe_disparity_domestic: bool = True
    auto_trade_scan_count_domestic_disparity: int = 15
    # Must be one of KIS's accepted 이격도 windows — 5/10/20/60/120. The TR
    # REJECTS any other value outright rather than clamping to the nearest
    # one, so a typo here fails loudly (an HTTP/business error from KIS)
    # rather than silently measuring the wrong window.
    auto_trade_universe_disparity_window_days: int = 20
    # KIS-side minimum-volume floor (FID_VOL_CNT) shared by both new axes —
    # unlike the turnover/market-cap axes, neither 등락률 nor 이격도 ranking
    # is self-limiting on liquidity, so without this floor both would
    # readily surface 관리종목/정리매매-adjacent names trading on minimal
    # volume (see krx_disparity_low_n's docstring).
    auto_trade_universe_min_volume_domestic: int = 100_000

    # Account-derived price CEILING on all four KRX discovery axes above,
    # via KIS's own `FID_INPUT_PRICE_2` — applied KIS-side, BEFORE each
    # ranking TR's own top-N cut (confirmed live 2026-09-02: a probe
    # against fetch_market_cap_rank with a ₩100,000 ceiling returned a
    # full, non-truncated 30 rows, 28 of them different from the
    # unfiltered top-30 — KIS ranks WITHIN the price-bounded set, it does
    # not just filter an already-fixed top-30 afterwards).
    #
    # Why: 2026-09-02, a ₩2,000,000 corporate account's
    # `account_too_small_domestic` root cause turned out to be the
    # account's OWN 1% risk cap (app.domain.executability.
    # risk_capped_shares) making even 1 share unaffordable above roughly
    # `equity * sizing_max_risk_pct_of_equity / (atr_pct *
    # sizing_min_stop_atr_multiple)` — ~₩400,000 at 2% ATR, ~₩267,000 at
    # 3%, ~₩200,000 at 4% for this account, at this system's shipped risk
    # defaults. The two axis toggles just above (fluctuation_losers/
    # disparity) widen WHICH names get a look, but neither is price-aware,
    # so neither was a structural fix — a cheap-looking name that's still
    # six figures per share is still rejected by the exact same
    # arithmetic. See app.domain.executability.affordable_price_ceiling
    # and app.live.auto_trade_loop._domestic_price_ceiling for the actual
    # derivation and account read.
    #
    # Turned on 2026-09-02 after a live-KIS 12-month A/B sweep at the
    # corporate account's actual computed ceiling (~₩600,000): universe
    # 42->44, excess return improved (+0.47%->+0.90%), no new hard-rule
    # violation, and a direct universe diff confirmed the mechanism works
    # as designed — 5 names over the ceiling (SK하이닉스, 삼성전기,
    # 삼성바이오로직스, SK스퀘어, 한화에어로스페이스) replaced by 7 names
    # under it. See project memory account-price-ceiling-2026-09-02 for
    # the full sweep tables and universe diff.
    auto_trade_universe_price_ceiling_domestic: bool = True
    # The assumed-vs-real ATR asymmetry is deliberately NOT symmetric:
    # admitting a name that turns out unaffordable once its real ATR is
    # known costs one price+ATR quote and is caught downstream by
    # _drop_unexecutable — cheap. Wrongly EXCLUDING an affordable name is
    # permanent — a KIS ranking bounded by FID_INPUT_PRICE_2 simply never
    # returns a ticker priced above the ceiling it was called with, so
    # there is no later chance to reconsider it. That asymmetry is why
    # this default is deliberately generous — do not "tighten" it to make
    # the filter look sharper; sharper here means quietly losing
    # affordable names forever.
    auto_trade_universe_price_ceiling_slack_multiple: float = 1.5
    # Below this computed ceiling, the account itself is too small for a
    # price ceiling to help — forcing one this low would fill the universe
    # with penny-stock/관리종목-adjacent junk instead of letting
    # account_too_small_domestic fire as the correctly-diagnosing signal
    # it already is. The ceiling is skipped entirely in that case, not
    # clamped up to this floor.
    auto_trade_universe_price_ceiling_min_krw: float = 10_000.0
    # At or above this computed ceiling, the account is well-funded enough
    # that literally every KRX share price already clears it — sending an
    # absurd number on the wire buys nothing. This also means there is no
    # SEPARATE equity threshold gating the whole feature: the ceiling
    # self-disables once the account is big enough, so one number does
    # both jobs.
    auto_trade_universe_price_ceiling_max_krw: float = 2_000_000.0

    # app.chat.screening.top_candidates's momentum window (trading days) —
    # split per market since KRX/US volatility characteristics differ.
    # Tuned 2026-07-29 via app.backtest.screening_sweep (screening-method-
    # improvement-ideas item #5/#6) over 2026-01-29..2026-07-29 (18 dates,
    # cadence=7, top_n=2, holding_days=10): domestic favored the longer
    # windows (15/20-day tied best at +0.52% mean forward return vs. the
    # old shared default 10-day's +0.12%), overseas favored keeping the
    # 10-day default (its own best result, -0.68% vs. -0.72%..-1.45% for
    # the alternatives — all negative since this window was a down/mixed
    # stretch for the 6-ticker overseas sample). Small sample (4-6 tickers,
    # one 6-month window, no train/test split) — directional evidence, not
    # a statistically confirmed optimum; re-sweep if this stops looking
    # right in practice.
    # 모멘텀 룩백윈도우(거래일) — 국내/해외 각각 별도 설정. 2026-07-29 스윕 결과로 튜닝.
    momentum_lookback_days_domestic: int = 15
    momentum_lookback_days_overseas: int = 10
    # app.chat.screening._ranked_and_deduped's correlation-clustering pool
    # size, as a multiple of top_n — e.g. top_n=3 with the default 3 below
    # clusters the top 9 momentum-ranked candidates (not the whole ~30-name
    # scan pool) before cutting to the final 3, dropping same-cluster
    # near-duplicates (screening-method-improvement-ideas item #4: several
    # correlated names — a theme moving together — otherwise crowding out
    # the whole top_n). Bounds the added cost: correlation clustering
    # fetches a full price history per ticker it clusters, so widening
    # this trades a bit more of that cost for a better shot at catching a
    # correlated near-duplicate that ranked outside a narrower pool.
    # 상관관계 클러스터링 대상 풀 크기(top_n의 배수) — 전체 스캔풀이 아니라 일부만 클러스터링.
    screening_correlation_pool_multiplier: int = 3
    # ATR-adjusted ranking (app.chat.screening.risk_adjusted_score) and
    # correlation dedup (dedupe_correlated), gated per market —
    # screening-improvement-quality-review's 3-period out-of-sample ablation
    # found both stages consistently hurt the overseas pool (3/3 periods
    # worse than plain-momentum baseline) while domestic stayed direction-
    # unstable (each period went a different way, no consistent signal
    # either way, so left on there rather than turned off on weak evidence).
    # 변동성조정 랭킹+상관관계dedup — 해외는 3개 독립구간 전부 baseline보다 나빠 OFF,
    # 국내는 구간마다 방향이 달라 근거 부족으로 기존(ON) 유지.
    screening_volatility_adjustment_domestic: bool = True
    screening_volatility_adjustment_overseas: bool = False
    screening_correlation_dedup_domestic: bool = True
    screening_correlation_dedup_overseas: bool = False
    # §4-1 live screening instrumentation (2026-09-08). Master switch for
    # app.live.screening_log's per-cycle snapshot write AND app.audit.
    # screening_ic's forward-return resolution — both no-op when off, so
    # turning this off costs one boolean check per cycle and nothing else.
    # Default ON (unlike the discovery-axis toggles above): this writes no
    # orders, spends no LLM tokens, and its whole point is that the data
    # cannot be reconstructed after the fact — a cycle that ran without it
    # is permanently unmeasurable.
    # 라이브 스크리닝 계측 스위치 — 끄면 스냅샷/해소 둘 다 no-op. 소급 재구성이
    # 불가능한 데이터라 기본 ON.
    screening_snapshot_log_enabled: bool = True
    # 2026-09-09 §0단계 계측(screening-cycle-rebalance-lock-analysis). ON이면
    # 유니버스 중 executable에서 탈락한 종목(rebalance_interval_days 5일 락,
    # max_open 캡, 매수가능성 필터)의 모멘텀 점수도 관측용으로 계산해 스냅샷에
    # 남긴다. 랭킹/후보선정에는 전혀 영향이 없다(app.chat.screening.
    # ScreeningPass.observed는 _ranked_and_deduped에 전달되지 않는다) —
    # 유일한 비용은 잠긴 종목당 KIS 캔들 1건이며, candle_run_cache() 스코프
    # 안에서 실행되므로 이미 fetch된 종목은 무료. screening_snapshot_log_
    # enabled=False면 이 값과 무관하게 no-op. 5일 락이 국내 유니버스를 하루에
    # 소진해 랭킹 변별력을 죽이는지 여부는 이 데이터 없이는 답할 수 없다 —
    # 소급 재구성이 불가능하므로 기본 ON.
    screening_snapshot_observational_scoring_enabled: bool = True
    # Calendar days (NOT trading days) from the snapshot date to the forward-
    # return measurement point — matches app.backtest.screening_sweep's
    # `--holding-days` default (10) exactly, so a live number is directly
    # comparable with the offline sweep tables that produced today's
    # parameters. 10 calendar days is ~7 trading days; see
    # app.audit.screening_ic's own docstring for why this convention is kept
    # rather than "fixed" to trading days.
    screening_forward_holding_days: int = 10
    # Per-cycle bound on how many (ticker, trade_date) forward returns one
    # resolution pass fetches. This runs INSIDE app.live.auto_trade_loop's
    # cycle prologue, so an unbounded backlog drain would compete with the
    # cycle's own KIS budget (18 req/s live) — the same EGW00215 rate-limit
    # shape this project has already hit once. 150 covers roughly one full
    # trading day of both markets' cross-sections — this margin shrank a bit
    # after 2026-09-09's observational scoring (§0단계) widened each date's
    # cross-section from executable-only (~9-44 domestic) to the whole
    # universe (~44 domestic / ~41 overseas), but 150 still covers it.
    screening_forward_resolve_max_per_pass: int = 150
    screening_forward_resolve_concurrency: int = 4
    # A (ticker, trade_date) whose price series still won't resolve after
    # this many passes is treated as permanently unmeasurable (delisted,
    # too-recent listing) and stops being retried — see forward_returns'
    # own `attempts` column comment for why a transient KIS hiccup and a
    # permanent data gap are otherwise indistinguishable.
    screening_forward_resolve_max_attempts: int = 3
    # app.execution.kis_bridge.spread_blocker's liquidity pre-filter —
    # candidates with a current bid-ask spread wider than this fraction of
    # the mid price are dropped before full LLM analysis
    # (screening-method-improvement-ideas item #3). 1% is a loose bound
    # (most liquid large/mid-cap names run well under this) meant to catch
    # genuinely thin/one-off-volume-spike names, not to be a tight
    # execution-quality gate.
    # 유동성/스프레드 필터 임계값 — 중간가 대비 매도-매수호가 스프레드가 이 값을 넘으면 후보에서 제외.
    max_spread_pct: float = 0.01
    # Intended TOTAL analyzed per side per cycle (not "trade allowlist plus N") —
    # trade allowlist entries are pooled into the same momentum-ranked set as the
    # screened candidates for their own market, not given a free pass. See
    # app.live.auto_trade_loop's _domestic_candidates/_overseas_candidates.
    auto_trade_max_tickers_domestic: int = 4
    auto_trade_max_tickers_overseas: int = 4

    # How big a multiple of auto_trade_max_tickers_domestic/_overseas to ask
    # the momentum ranking for, before the expensive broker-I/O filters
    # (_drop_execution_blocked's halt/VI check, _drop_illiquid's spread
    # check — app.live.auto_trade_loop) run and the result gets cut back
    # down to the real budget. Those two filters run AFTER ranking (they're
    # per-ticker broker calls, too costly to run over the whole ~60-ticker
    # universe every cycle at KIS's 18 req/s live rate limit), so without
    # this overfetch, a structurally illiquid or halted name sitting at the
    # top of the ranking would permanently occupy one of the max_tickers
    # slots — the same "sticky filter behind a hard top-N cut" shape as the
    # 2026-08-26 rebalance-throttle starvation bug (see
    # auto_trade_candidate_overfetch_multiplier's sibling fix,
    # rebalance_interval_days below), just weaker since a spread/halt
    # condition can clear intraday where the rebalance throttle cannot. Set
    # to 1 to disable overfetch and go back to ranking exactly max_tickers.
    # 실행가능성 필터(거래정지/VI, 스프레드)가 랭킹 뒤에서 상위 후보를 떨어뜨려도
    # 예산 슬롯이 비지 않도록 여유분을 더 뽑아둠 — 1이면 예전처럼 정확히
    # max_tickers만큼만 랭킹.
    auto_trade_candidate_overfetch_multiplier: int = 2

    # 2026-09-01 audit root cause #6 (first real-account KRX cycle, 200,000 KRW
    # equity). 41 candidates each burned a full ~20-LLM-call pipeline; 31 of them
    # could not have been sized to a single share no matter what the AI decided,
    # and 6 Buy ratings all landed as action_label=Hold with 0 shares. This filter
    # computes the best case the execution chain could ever reach
    # (app.domain.executability, the SAME arithmetic kernels app.execution.ledger
    # sizes with) and drops a candidate whose best case is 0 shares — BEFORE
    # momentum ranking, so the next-ranked affordable name naturally takes the
    # slot (see _eligible_universe's own docstring for the 2026-08-26 starvation
    # shape this ordering exists to avoid; an unaffordable name is even stickier
    # than the rebalance throttle, since price/ATR/equity barely move day to
    # day). Never stricter than the real gate: a missing price/ATR, an unreadable
    # real balance, a currently-held ticker, and every forced risk-review ticker
    # all KEEP the candidate. Set to False to restore pre-fix behavior exactly.
    # 소액 실계좌에서 1주도 못 사는 종목에 LLM 파이프라인을 태우지 않음 — 랭킹 '앞'에서
    # 걸러 다음 순위 종목이 슬롯을 이어받게 함. False면 이전 동작으로 완전 복귀.
    auto_trade_skip_unexecutable_candidates: bool = True

    # 2026-09-02 audit (decision id=1169, 950260.KS, first corporate full_auto
    # KRX cycle). When 14-day ATR% is unavailable for a ticker, TWO independent
    # sizing safeties disable at once and neither says so: app.execution.sizing.
    # vol_adjusted_allocation_for stops scaling down (a Buy keeps its full 100%
    # base allocation), and app.domain.executability.min_stop_distance returns
    # None so app.execution.ledger.resolve_stop_loss neither floors nor
    # validates the LLM's proposed stop — a tight invented stop then makes the
    # 1% risk cap (risk_per_share = price - stop) effectively non-binding too.
    # The observed candidate was a ~1조원 Reg.S biotech with no analyst
    # coverage and no other fundamentals either — a worst-case name for an
    # ungated full-size entry. With this True, a candidate with no usable ATR
    # and NO existing position is refused for the cycle instead of being sized
    # unscaled and unfloored. NEW ENTRIES ONLY: a ticker already held keeps its
    # existing stop and is never forced to sell, reduce, or hold — same
    # "never force a sale of shares already held" posture as the 1% risk cap
    # (app.execution.ledger.rebalance_to_target) and _drop_unexecutable.
    # Deliberately NOT in app.execution.risk_rules.RULE_SPEC: this is a
    # fail-closed safety gate, not a user risk-appetite knob, and a
    # dashboard/DB override that can silently switch it off is the same hazard
    # that keeps KIS_EXECUTION_ENABLED config-only. Set to False (env +
    # restart) to restore pre-fix behavior exactly.
    # ATR(14) 값을 못 구한 종목은 변동성 축소도, 손절 하한선 검증도 동시에 꺼지므로
    # 신규 진입만 이번 사이클 거부 — 이미 보유 중인 종목은 강제 매도/축소 없이 그대로.
    auto_trade_require_atr_for_new_entry: bool = True

    # Bounds how many candidate tickers app.live.auto_trade_loop analyzes at
    # once (asyncio.Semaphore) instead of one at a time — the ~20-LLM-call
    # pipeline is dominated by LLM round-trip latency, not local CPU, so a
    # few overlapping requests still make progress even against a single
    # vLLM instance (continuous batching), unlike true CPU-bound work. Kept
    # modest (not "all candidates at once") since portfolio-cap enforcement
    # is still serialized per market at commit time regardless (see
    # app.execution.portfolio_risk.market_lock) — raising this only shrinks
    # analysis-phase wall-clock, not enforcement-phase safety. Set to 1 to
    # restore the old fully-sequential behavior.
    # 후보종목 동시분석 상한 — LLM 왕복지연이 대부분이라 단일 vLLM 인스턴스에도 어느
    # 정도 동시 진행 이득이 있음. 1로 설정하면 예전처럼 완전 순차 처리로 되돌아감.
    # 2->4(2026-08-12 아침)->2(같은 날 저녁, 되돌림): vLLM 슬롯 실측(llm_max_concurrent_requests=8,
    # KV캐시 한계 ~11.9)이 절반도 안 찼다는 근거로 4로 올렸으나, 같은 날 이 서버(ops-s004)가 지속
    # 고부하 후 반복적으로 다운되는 사고가 발생(다른 서버는 무사 — 전원/냉각 계열 하드웨어 문제로
    # 추정, [[server-power-schedule-blocker]]). 백테스트를 끈 상태에서도 라이브 단독으로 GPU
    # 96%까지 도달하는 것을 실측 — 물리적 원인이 해소되기 전까지는 처리량보다 지속부하 시간을
    # 줄이는 쪽을 우선해 2로 되돌림. 하드웨어 점검 후 재상향 검토.
    # 2->1 (2026-08-24): 서빙 모델을 Inferact/Qwen3.8-27B-NVFP4로 교체한 직후 첫 사이클에서
    # 분석가 호출 지연시간이 정상(수십 초)에서 240초->600초대->709초로 같은 사이클 안에서
    # 계속 악화되는 것을 `/api/metrics/llm`으로 실측(전형적인 GPU 요청 대기열 적체 패턴) — 이
    # 프로세스가 뜬 이후 기록된 모든 호출이 새 모델이라 교체 직후가 원인으로 유력. 새 모델의
    # 실제 동시처리 한계가 재검증되기 전까지 임시로 완전 순차 처리(1)로 낮춤. vLLM 쪽
    # 동시성 설정 확인 후 재상향 검토(기록 없이 4/2로 되돌리지 말 것 — 위 이력과 같은 실수 반복).
    # 2026-09-08 동시성 실측 — **값 변경 없음, 위 "재검증 전까지 1" 조건의 기록만**.
    # app.experiments.reasoning_ab_parallel로 같은 vLLM 배포에서 측정: K=1 262.99s(reason_off)/
    # 292.89s(reason_on) 기준선, K=2 351.1/371.1s(1.27~1.34x), K=3 425.2s/433.0s(1.42~1.62x —
    # 005930.KS reason_on 547.9s는 같은 로그에 KIS EGW00201 재시도 + yfinance 타임아웃이 찍힌
    # 데이터조회 지연 이상치라 제외). 전 구간 isolation_violation/degraded 없음, 큐 적체
    # 시그니처 없음. 배율이 1.0->1.3->1.6x로 우상향해 포화 기준(~1.8x)에 아직 안 닿았고
    # 순유효배속은 K=2 ~1.5x, K=3 ~1.9x라 K=3이 이 배포의 실질 상한으로 판단(K=4 미시도).
    # 다만 이 측정은 A/B 하네스(그룹당 풀 파이프라인)이지 auto_trade_loop 자체가 아니고,
    # 상향은 사용자 승인 사항이라 값은 1로 유지. 상세 [[reasoning-ab-parallel-calibration-2026-09-08]].
    auto_trade_max_concurrent_analyses: int = 1

    # A held position that doesn't rank in the momentum top-N above still
    # gets a full pipeline re-review this cycle if its price has drifted
    # within this fraction of its recorded stop-loss (e.g. 0.05 = within 5%
    # above stop) — cheap (one price read + one DB read per held ticker, no
    # LLM call) rather than either (a) re-running the full ~20-call
    # multi-agent pipeline on every holding every cycle regardless of
    # whether anything changed, or (b) never revisiting a holding that just
    # isn't a momentum leader until it actually breaches its stop and the
    # separate rule-based stop-loss loop force-exits it with no AI input at
    # all. A position comfortably away from its stop isn't re-reviewed
    # just for sitting in the trade allowlist.
    # 보유 종목이 모멘텀 상위에 안 들어도, 손절가에 근접하면(설정 비율 이내) 그 종목만
    # 저비용으로 감지해서 이번 사이클에 풀 파이프라인 재평가 대상에 포함시킴.
    auto_trade_held_review_buffer_pct: float = 0.05

    # Warning-only counterpart to the buffer above, for a held position
    # while ITS market is closed (app.live.auto_trade_loop._warn_overnight_
    # gap_risk) — deliberately wider than auto_trade_held_review_buffer_pct
    # because nothing here can react to a move while the market is shut
    # (no live tick, no pipeline re-review, no mechanical stop check until
    # it reopens), so the same distance that's "comfortable" during trading
    # hours deserves an earlier heads-up overnight/over a weekend. Fires no
    # pipeline re-review and changes no position/size/order — same
    # non-actionable role as app.live.alerts' price_move/volume_spike.
    # 보유 종목이 속한 시장이 닫혀 있을 때 쓰는 경고 전용 버퍼 — 장중 재검토
    # 버퍼보다 넓게 잡음(장 닫힌 동안은 반응할 방법이 없으므로 더 일찍 알려야 함).
    # 파이프라인 재평가나 포지션 변경은 전혀 유발하지 않는 순수 경고.
    overnight_gap_warning_buffer_pct: float = 0.08

    # Same "cheap, no-LLM, force into this cycle regardless of momentum
    # rank" treatment as the stop-loss-proximity check above, for a
    # different risk signal: a position's unrealized loss in absolute
    # currency terms, not its return %. A high-weight position down a
    # modest % can be losing more real money than a low-weight one down a
    # large %, and % alone doesn't surface that (see project memory:
    # 2026-08-03 discussion). Fixed count, not a KRW/USD threshold — same
    # "bounded, predictable per-cycle LLM cost" reasoning as
    # auto_trade_max_tickers_domestic/_overseas, rather than letting a
    # broad down day balloon this cycle's candidate count.
    # 손절가 근접과 별개로, 평가손익(원화/달러 금액) 손실이 큰 보유 종목 상위 N개도
    # 모멘텀 순위와 무관하게 이번 사이클 재평가 대상에 강제 포함(OR 조건).
    auto_trade_top_loss_review_count: int = 2

    # Live dashboard / alerts / chat layer — separate cadence from the
    # decision pipeline above; the poller never invokes an LLM.
    #
    # `trade_allowlist_domestic`/`trade_allowlist_overseas`/`watchlist`/
    # `trade_allowlist_auto_added` are NOT pydantic fields any more — they are
    # per-(user, account_mode) properties backed by `_scoped_lists`, defined
    # further down alongside the money-limit properties that use the same
    # pattern. See `_scoped_list` for the mechanics and `SCOPED_LIST_KEYS` for
    # the fresh-scope defaults.
    #
    # They had to stop being process-global (remaining Phase 5 work, migration
    # plan §6.5): `trade_allowlist_*` is the list `app.execution.kis_bridge`
    # gates on to decide what may be traded at all, so one shared in-memory
    # copy meant a ticker you added authorized it for every other user's
    # automated trading — and `trade_allowlist_store` being per-user in
    # Postgres did nothing about it, because every gate reads the in-memory
    # copy, not the table.
    # 매매허용목록은 kis_bridge의 거래 허용 판단 기준이므로, 인메모리 사본이 하나면
    # 내가 추가한 종목이 다른 사용자의 자동매매에도 허용됨 — DB만 사용자별로
    # 나눠도 게이트는 사본을 읽으므로 해결되지 않았음.
    #
    # There is deliberately NO env var for any of them: values come from
    # Postgres per scope (`trade_allowlist_store.load`), and a deployment-wide
    # env seed in a multi-user process would be a shared starting list again.
    # The removed TRADE_ALLOWLIST_DOMESTIC/_OVERSEAS are documented in
    # .env.example.
    #
    # Meaning of each, unchanged:
    # - trade_allowlist_domestic/_overseas — real AI trading permission +
    #   realtime feed subscription + stop-loss monitoring. Split by market so
    #   this boundary matches every other domestic/overseas separation in the
    #   project (account balance, risk caps, daily-loss breaker, candidate
    #   discovery) instead of being the one mixed exception.
    # - watchlist — lighter, no trading permission, not market-split. Kept as a
    #   superset of trade_allowlist by `normalize_scope`.
    # - trade_allowlist_auto_added — which entries got there automatically (the
    #   AI BUY hook, or reconcile_trade_allowlist). Bookkeeping, not
    #   permission: presence means "safe to prune once it has zero real shares
    #   and no open pending order", NOT "safe to trade". A user-added ticker
    #   never appears here, so it is never auto-pruned even at zero shares.
    # AI가 자동으로 추가한 종목 표시 — 매매허용과는 별개의 부기용 필드.
    poll_interval_seconds: int = 60
    # app.live.orderbook_log (2026-08-24) — how often the orderbook-zone
    # codes' latest H0STASP0 snapshot gets written to Postgres for future
    # scalping-strategy backtesting research. Research-only cadence: no live
    # safety gate reads this table, so nothing depends on it being fresher
    # than this. Kept modest (worst case _ORDERBOOK_ZONE_SIZE=5 codes x 0.5
    # writes/sec = ~2.5 rows/sec) rather than tied to poll_interval_seconds,
    # which is a live-safety cadence answering a different question.
    orderbook_log_interval_seconds: float = 2.0
    alert_pct_threshold: float = 0.02
    alert_volume_multiplier: float = 3.0
    alert_lookback_ticks: int = 5
    # Suppresses re-firing the same (ticker, kind) alert while the condition
    # stays true across consecutive polls (e.g. a sustained price move) —
    # without this, every poll_interval_seconds re-emits an identical alert
    # for as long as the threshold stays crossed, burying real signals
    # (like a stop-loss alert) under repeats of the same trend alert.
    alert_cooldown_seconds: int = 300
    # app.live.exit_monitor (2026-08-10) — tick-based real-time stop-loss
    # check, closing the gap between the minute poller (records ticks) and
    # the hourly daily-bar stop_loss_loop (app.web.state), which for a
    # domestic KRX ticker only sees a breach once that day's candle closes.
    exit_monitor_interval_seconds: float = 5.0
    # A tick older than this is treated as unknown, not as evidence either
    # way — never force-exits on a price that might no longer be current
    # (e.g. a stalled poller, or a KRX websocket reconnect gap where the
    # 60s REST fallback hasn't caught up yet). Same fail-toward-not-selling
    # posture as current_status()'s reconnect-staleness gate (app.brokers.
    # kis.domestic.websocket) — being wrong by staying in is recoverable
    # (the hourly backstop still catches it next pass); being wrong by
    # exiting on stale data is not.
    exit_monitor_max_tick_age_seconds: float = 30.0

    # 2026-08-27, kill switch for the daily-bar backstop's intraday
    # ordering fix (ledger.check_and_apply_trailing_stop / trailing_stop's
    # adjudicated_through column — read those first). OFF restores the
    # pre-fix rule for every day: "the day's high already happened by the
    # time its low is checked," which force-liquidated 047040.KS at +11% on
    # 2026-08-27 citing a low that occurred 30 minutes BEFORE the high whose
    # floor it supposedly breached. ON (the default) fetches 5-minute bars
    # for the days the live walk covers and resolves each day's real
    # ordering, per-day-falling back to the old rule wherever those bars are
    # missing or don't reconcile against the same day's daily bar.
    #
    # A kill switch, not a rollout flag: the old rule is over-eager but
    # never blind, so if the intraday feed ever starts HIDING breaches
    # (systematically truncated downloads that still pass reconciliation,
    # say), this returns the fleet to a known-safe posture in one restart
    # instead of a hotfix deploy. Does not affect app.backtest at all — the
    # replay never passes intraday bars in the first place.
    stop_loss_backstop_intraday_adjudication: bool = True
    # app.live.fill_monitor (2026-08-13) — confirms a mirrored order's
    # ACTUAL KIS fill against its execution history, closing the gap
    # mirror_trade_to_kis leaves open: it records "submitted: ..." the
    # instant KIS ACCEPTS an order (order_no issued), never when it
    # actually fills. Real incidents this addresses: a phantom 10-share
    # position (008290.KS) and a real 5-share position left unprotected for
    # a week (005930.KS), both from an order accepted but never confirmed
    # filled — see project memory: order-fill-confirmation-gap-2026-08-13.
    fill_monitor_interval_seconds: float = 60.0
    # How long to keep polling an order with no definitive KIS answer yet
    # (not found in the execution-history TR at all, or found but still
    # partially open) before alerting it as suspicious and giving up —
    # past this, repeated polling isn't learning anything new. Not the same
    # as reconciliation's 48h startup-only window: this loop runs
    # continuously, so it can afford to give up much sooner and still catch
    # the same order on a LATER pass if _pending_fill_checks' own lookback
    # (2 days) still covers it — this setting only controls how soon the
    # first alert fires, not whether the order is ever re-examined.
    fill_monitor_give_up_minutes: float = 30.0
    # Stage 2 (2026-08-13) — once KIS gives a DEFINITIVE answer (confirmed
    # zero-fill/cancelled, or a partial fill whose order has stopped
    # working), auto-correct the ledger instead of only alerting, reusing
    # the exact reverse_pending_trade/get_open_stop_loss/resolve_stop_loss
    # combination already used to manually fix 008290.KS/005930.KS. Off by
    # default — same "opt in explicitly" convention as execution_mode/
    # kis_execution_dry_run, since this writes to the live paper ledger.
    # Every correction attempt is alerted regardless of outcome (see
    # fill_monitor.py) so the stage-gate's "unresolved discrepancy count"
    # measures DETECTED drift, not post-correction drift — auto-fixing a
    # problem must never make it invisible to that gate.
    fill_monitor_auto_correct_enabled: bool = False
    # Circuit breaker: repeated corrections in one day signal a structural
    # problem (broker outage, a reconciliation bug, bad order logic) that
    # blind auto-correction would keep papering over instead of surfacing —
    # past this many same-day corrections, stop correcting and raise a
    # single loud alert instead (see fill_monitor.py's `_CIRCUIT_BREAKER`
    # alert kind).
    fill_monitor_max_corrections_per_day: int = 5
    # app.web.state.trade_reconciliation_loop (2026-09-07) — converts what
    # was a boot-time-only check (app.execution.reconciliation.
    # find_unmirrored_trades, detects a local `trades` row whose mirror to
    # KIS may never have completed, e.g. a process kill between
    # ledger.record_trade() and mirror_trade_to_kis()) into a periodic
    # background pass, so a drift that appears mid-session doesn't sit
    # undetected until the next restart. 600s (10 min): well above the
    # 2-minute grace_period find_unmirrored_trades already uses to skip a
    # trade whose mirror call may still be in flight, and this is a backstop
    # for a fundamentally rare event (a crash in one specific narrow gap) —
    # fill_monitor_loop above (60s) already carries the time-sensitive,
    # exact order_no-matched fill-confirmation job; this one's window is a
    # 48-hour lookback by design, so a 10-minute cadence is not "slow", it's
    # matched to what it's actually checking. At worst 6 passes/hour x 2 TRs
    # (domestic+overseas) = 12 KIS calls/hour when there IS something to
    # check, none when there isn't (the pass skips the KIS calls entirely
    # when the local `trades` query alone comes back empty) — negligible
    # against the EGW00215/EGW00201 gateway caps this desk has hit before.
    trade_reconciliation_interval_seconds: float = 600.0
    # Bound to localhost, NOT 0.0.0.0: the dashboard now exposes an order
    # placement endpoint (POST /api/order) with no auth (single-user v1), so
    # it must not be reachable from other hosts on the network. Overriding
    # this back to 0.0.0.0 re-exposes real order submission LAN-wide.
    dashboard_host: str = "127.0.0.1"
    dashboard_port: int = 8094
    # Multi-instance seam (agent-architecture-roadmap stage 4) — the tag
    # every trades/memory_entries/alerts row is stamped with (see
    # app.user_context.get_current_user_id), and what app.memory.db's
    # `users` table registers by. Each person's own instance sets this in
    # their own .env.local to their own id, alongside their own
    # dashboard_port, while sharing one Postgres — see project memory:
    # user-account-mode-db-separation-design. Defaults to 'default' so a
    # fresh checkout with no override behaves exactly as it always has.
    user_id: str = "default"
    # uvicorn's own default (None) waits indefinitely for lingering
    # connections/tasks on shutdown — observed live (2026-07-27) combining
    # with the KIS websocket's own close handshake to make a single Ctrl+C
    # feel like it "hangs" printing reconnect/backoff logs. A bounded value
    # here means shutdown always completes within this many seconds instead
    # of however long the slowest lingering connection takes.
    dashboard_graceful_shutdown_timeout_seconds: float = 5.0

    # Email/password login existed here (multi-user migration Phase 1 — see
    # docs/multi-user-credentials-plan.md) and was removed 2026-08-07 when
    # this became a single-account in-house deployment
    # (docs/single-user-rollback-guide.md). There is no
    # application-level authentication any more: access control is entirely
    # `dashboard_host` staying loopback + an SSH tunnel, enforced by
    # `_refuse_non_loopback_bind` below. Do not re-add an `auth_required`
    # flag that can be toggled off — that flag WAS the seam that made this
    # app briefly ship "insecure but convenient" during the migration; this
    # deployment has no login screen to fall back to any more, so a
    # network-reachable, unauthenticated bind here is not a degraded state,
    # it is real-money order placement open to the network.
    # 로그인 자체를 없앰 — 접근 통제는 루프백 바인딩 + SSH 터널이 전부이고,
    # 아래 검증기가 그걸 강제한다. auth_required 같은 끌 수 있는 스위치를
    # 다시 두지 말 것 — 지금은 돌아갈 로그인 화면 자체가 없다.

    # Volatility-adjusted sizing (Phase 4 extension): scales a rating's base
    # allocation DOWN when ATR(14)% exceeds this target — never up, to stay
    # inside the existing long-only/no-leverage scope (see sizing.py).
    sizing_target_atr_pct: float = 0.02
    sizing_min_vol_multiplier: float = 0.25

    # Minimum stop-loss distance from the fill price, as a multiple of the
    # current 14-day ATR% (see ledger.py::resolve_stop_loss). Grounded in
    # published practice, not picked arbitrarily: the Turtle Trading System
    # used 2x ATR ("N"); Chuck LeBeau's Chandelier Exit (standard on
    # institutional charting platforms) uses 3x; general swing-trading
    # practice cites 2.5x ATR(14) as a baseline that cuts false stop-outs
    # ~28% vs. a 1.5x stop. 2.5x sits in the middle of that band, appropriate
    # for this system's multi-day/week decision cadence.
    sizing_min_stop_atr_multiple: float = 2.5

    # Maximum stop-loss distance from the fill price, as a fraction of
    # price — a sanity CEILING, the mirror of sizing_min_stop_atr_multiple's
    # floor above. Real incident: the Market Analyst skipped its actual
    # price-lookup tool call and left its own placeholder text ("가상의
    # 최신 데이터 기준") in the report, anchoring the whole decision to a
    # price ~3.4x below the real one (₩78,500 vs. the real ₩268,000 for
    # 005930.KS) — the resulting stop-loss (₩71,500) was ~73% below the
    # REAL fill price, nothing like a real risk-managed stop. ATR alone
    # can't catch this: a hallucinated price can come with an internally
    # "consistent" but equally wrong ATR-relative distance. This is a flat,
    # ATR-independent backstop instead — deliberately loose (50%) so it
    # only catches genuinely broken values, not just unusually wide but
    # real stops (e.g. a highly volatile small-cap).
    # 손절가 최대 거리(현재가 대비 비율) — LLM이 가격조회를 안 하고 지어낸 값이
    # 실제 가격과 몇 배씩 차이나는 경우를 잡기 위한 최후의 안전판(ATR과 무관하게 동작).
    sizing_max_stop_distance_pct: float = 0.50

    # Caps a single trade's dollar risk (shares x stop distance) at this
    # fraction of account equity — the standard professional "1% rule" (see
    # e.g. Van Tharp's position-sizing framework): position size should be
    # DERIVED FROM the stop distance and a fixed risk budget, not set
    # independently of it. Ties sizing.py's ATR-scaled allocation and
    # resolve_stop_loss's ATR-floored stop together for the first time —
    # previously a rating's full allocation could still risk far more than
    # this if volatility (and therefore the stop distance) was high. 1%,
    # not the sometimes-cited 2%, since that looser figure is only
    # recommended once a strategy has a proven track record (50+ trades of
    # positive expectancy), which this system doesn't have yet.
    sizing_max_risk_pct_of_equity: float = 0.01

    # Trailing stop (2026-08-10) — ratchets a profitable position's own
    # stop-loss UP (never down) as its high-water price rises, so gains
    # don't sit unprotected between scheduled rebalances the way they used
    # to (the PM's own stop only ever gets set/refreshed on a real trade,
    # not continuously). See app/execution/trailing_stop.py for the
    # ratchet math itself.
    #
    # ATR multiplier, not a flat %: Dai, Marshall, Nguyen & Visaltanachoti
    # ("Risk Reduction Using Trailing Stop-Loss Rules", Int'l Review of
    # Finance 2021 — 20 years of US market/sector ETF data) found risk-
    # adjusted returns favor WIDER thresholds. Volatility-normalized (ATR%)
    # sits on the same axis as "standard deviations of the underlying" — a
    # fixed flat % doesn't adjust for a volatile ticker needing more room
    # than a quiet one.
    #
    # 1.25 chosen from this project's own LLM-free trailing_stop_param_sweep
    # (2026-08-11, app/backtest/trailing_stop_param_sweep.py): 12 tickers
    # (6 domestic + 6 overseas), 2023-08-01..2026-08-08, ~894 independent
    # 10-day-interval entries per candidate. Result CONTRADICTED the cited
    # paper's direction on this data — Sharpe and CVaR95 both got WORSE as
    # the multiplier widened (1.0: sharpe +0.44/CVaR95 -14.88%; 2.5: sharpe
    # +0.36/CVaR95 -20.08%), even though mean return rose (a few outsized
    # winners pulling the average up while the median trade got worse and
    # win rate fell from 93% to 67%). 1.25 sits just above the best-Sharpe
    # candidate (1.0) rather than at it, as a small margin against this
    # being a 3-year/individual-stock artifact rather than the true optimum
    # (the paper's own study used 20 years of diversified ETFs, not 3 years
    # of 12 individual names) — an explicit, provisional choice, not a
    # from-scratch reset of the original literature-based prior. Revisit if
    # a longer/wider re-run (Phase 6 or a future sweep) changes the picture.
    trailing_stop_atr_multiplier: float = 1.25
    # Only starts ratcheting once the position has risen this far above its
    # own entry price — below this, effective_stop_loss returns the PM's
    # plain stop unchanged. Below activation, a trailing floor computed off
    # a barely-above-entry high-water price would usually sit WORSE than
    # the PM's own stop anyway (max() would just pick the PM's), so this is
    # mostly a readability/no-op-avoidance guard, not a safety-critical
    # threshold — but see also: MT4/MT5 platform convention is to require a
    # minimum profit before arming a trailing stop at all, for the same
    # "nothing worth protecting yet" reasoning.
    trailing_stop_activation_pct: float = 0.04
    # Flat fallback trailing distance (fraction of high-water price) for
    # when ATR data is unavailable (mirrors resolve_stop_loss's own ATR-
    # unavailable fallback path) — never left unset, since an un-computed
    # trailing distance would otherwise silently disable ratcheting instead
    # of degrading to a wider-but-present one.
    trailing_stop_fallback_pct: float = 0.08

    # Hysteresis gate (ai-engine-24h-operation-analysis P0-1, 2026-08-11):
    # the only defense against the LLM flip-flopping between Buy/Sell on the
    # same underlying situation used to be "don't ask too often"
    # (rebalance_interval_days) — a frequency limit, not an actual fix, and
    # one that gets harder to keep as analysis frequency goes up (24h
    # operation). This is the real fix: require a HIGHER bar of confidence
    # to actually flip a position's state (flat -> Buy, held -> Sell) than
    # to simply maintain it. Confidence is 0.0-1.0; unset (None) is treated
    # as below the bar (see pipeline._apply_hysteresis_gate) — "the model had
    # no basis to estimate confidence" is not a reason to trust a state
    # change more than an explicit low score would be. Scoped narrowly to
    # the two costly transitions a real round-trip-loss incident actually
    # involved ([[rebalance-cadence-separation-done]]) — Hold/Overweight/
    # Underweight (adjusting an existing position, not entering/exiting one)
    # are unaffected.
    hysteresis_min_confidence_to_flip: float = 0.6

    # Desk-consensus override gate (2026-09-04, id=1212/1217 — see
    # pipeline._apply_consensus_override_gate). Minimum PM `confidence` to
    # increase exposure (Buy/Overweight) against a unanimous "no more
    # exposure" call from BOTH the Research Manager (Hold/Underweight/Sell)
    # and the Trader (Hold/Sell); below it (or unset) the rating is
    # downgraded to Hold. The prompt rule alone (commit f79213e) did not
    # hold — the PM overrode the desk on re-weighed evidence with no new
    # fact. `_bearish` is the stricter bar applied when the desk was
    # actively negative (RM Sell/Underweight AND Trader Sell). Both are
    # 0.0-1.0 and user-overridable via app.execution.risk_rules, same path
    # as the hysteresis threshold.
    consensus_override_min_confidence: float = 0.75
    consensus_override_min_confidence_bearish: float = 0.85

    # KIS (Korea Investment & Securities) Open API. All None means "not
    # configured": every KIS-backed feature degrades quietly, same
    # convention as memory/DB.
    #
    # kis_account_mode picks which credential/TR-ID set every broker call
    # uses — see app/brokers/kis/broker.py::broker_for_user(). Defaults
    # to "paper" and MUST stay that way unless a person deliberately flips
    # it: switching to "real" means every order this app places is a real
    # trade with real money. See memory: KIS 실전투자 안전 규칙 — this is not
    # a toggle to flip casually or default to "real" for convenience.
    kis_account_mode: str = "paper"  # "paper" | "real"


    # The credentials are back here as of the 2026-08-07 single-account
    # rollback (docs/single-user-rollback-guide.md) — this deployment
    # trades exactly one company account, so the multi-user migration's
    # per-user encrypted Postgres table (docs/multi-user-credentials-plan.md
    # §1.1) was solving a problem that no longer exists. Secret lifecycle
    # moved to Jenkins Credentials instead: Jenkins assembles the env file at
    # deploy time (see the guide §3), so these never live in this repo or in
    # a database this process can read on its own.
    # 단일 계좌 전환으로 자격증명이 다시 env로 복귀 — 시크릿 수명주기는
    # Jenkins Credentials가 담당(docs/single-user-rollback-guide.md §3).
    #
    # None = that mode is not configured, same convention the old per-user
    # table used. paper and real are independent: registering one must never
    # imply or fall back to the other — see kis_credentials() below, which
    # never substitutes one mode's values for the other's.
    kis_app_key_real: str | None = None
    kis_app_secret_real: str | None = None
    kis_account_8_real: str | None = None
    kis_account_prod_real: str = "01"
    kis_hts_id_real: str | None = None  # optional — only 체결통보 subscription needs it

    kis_app_key_paper: str | None = None
    kis_app_secret_paper: str | None = None
    kis_account_8_paper: str | None = None
    kis_account_prod_paper: str = "01"
    kis_hts_id_paper: str | None = None

    # Only the non-secret endpoints stay as plain fields: they are public,
    # identical for everyone, and version-controlled on purpose so a typo in
    # one deployment's env can't point orders at the wrong host.
    kis_paper_base_url: str = "https://openapivts.koreainvestment.com:29443"
    kis_real_base_url: str = "https://openapi.koreainvestment.com:9443"

    def configured_kis_modes(self) -> set[str]:
        """Which modes have a complete credential set — app_key, app_secret
        and account_8 all present. Replaces the old
        app.credentials.store.configured_modes(user_id): same "no ciphertext
        read, just presence" contract, but source is env fields instead of a
        DB row, so this is sync and needs no user_id (one account now)."""
        return {
            mode for mode in ("paper", "real")
            if all(
                getattr(self, f"kis_{field}_{mode}")
                for field in ("app_key", "app_secret", "account_8")
            )
        }

    def kis_credentials(self, account_mode: str) -> "KISCredentials | None":
        """This deployment's credential set for `account_mode`, or None if
        that mode isn't configured. Replaces
        app.credentials.store.credentials_for(user_id, account_mode).

        Never falls back to the other mode's values — a missing real key
        must surface as "real is not configured", never as "here are the
        paper keys instead," because that would route a real-money order
        through the paper account's credentials (or vice versa) with no
        error at all.
        모드가 다르면 절대 대체하지 않음 — 실전 키가 없다고 모의 키로
        조용히 대체되면 계좌를 잘못 짚은 주문이 나갈 수 있음."""
        from app.domain.models import KISCredentials

        if account_mode not in ("paper", "real"):
            raise ValueError(f"unknown account mode {account_mode!r}")
        if account_mode not in self.configured_kis_modes():
            return None
        return KISCredentials(
            user_id=self.user_id,
            account_mode=account_mode,
            app_key=getattr(self, f"kis_app_key_{account_mode}"),
            app_secret=getattr(self, f"kis_app_secret_{account_mode}"),
            account_8=getattr(self, f"kis_account_8_{account_mode}"),
            account_prod=getattr(self, f"kis_account_prod_{account_mode}"),
            hts_id=getattr(self, f"kis_hts_id_{account_mode}") or None,
        )

    # Passed straight through to KISHttpClient's per-instance request
    # throttle (app/brokers/base/http_client.py) — 1.0 (KIS's documented
    # per-second cap for a PAPER account) is right for a single process
    # talking to the paper account, but app.backtest.parallel runs several
    # tickers as SEPARATE OS processes against the SAME KIS credentials,
    # each with its own in-process throttle unaware of the others; N
    # processes each honoring "1 req/sec of MY OWN calls" can jointly issue
    # ~N req/sec against one shared account and still trip KIS's real limit
    # (confirmed live, 2026-07-24: EGW00201 during a 2-ticker parallel run
    # despite the per-process fix). app.backtest.parallel sets this via the
    # KIS_MIN_REQUEST_INTERVAL_PAPER env var, scaled by its batch size, so
    # the combined request rate across all its child processes stays ~1/sec;
    # every other caller (the live dashboard, a single manual backtest) is
    # unaffected and keeps the 1.0 default.
    #
    # Paper/real split (2026-08-06) — this used to be one shared value, so
    # even after switching kis_account_mode to "real" the app stayed stuck
    # at the paper account's much tighter 1 req/sec cap instead of using the
    # real account's documented 18 req/sec (see project memory: KIS API
    # 유량 공지 2026-04-20). The real default below (0.12s ≈ 8.3 req/sec)
    # deliberately stays well under 18 — paper's own "exactly at the
    # documented cap, zero margin" setup is *why* client-side timing
    # jitter alone triggers EGW00201 there (see project memory: KIS 모의계좌
    # REST 레이트리밋에 안전여유 없음); giving real the same zero-margin
    # treatment would just relocate that same failure mode instead of fixing
    # it.
    #
    # 2026-10-02: 0.07 -> 0.12. KIS's notice recommends 100~150ms spacing for
    # concurrent calls, and at 0.07s (≈14.3 req/sec, nominally under the cap)
    # live EGW00201 still hit dozens per hour (10-02 market hours 27~49/h,
    # after close 11~15/h) — requests arrive bunched (network / connection
    # pool waits); the user confirmed no other system uses this app key.
    # Cost: real-account
    # KIS calls run ~40% slower. 실전 간격을 KIS 권장(100~150ms) 안으로 넓힘.
    kis_min_request_interval_paper: float = 1.0
    kis_min_request_interval_real: float = 0.12

    # A SECOND, coarser interval for ledger-side inquiry TRs only (balance,
    # present-balance, executions, buyable/sellable) — see
    # app.brokers.base.http_client.account_lane. KIS's ledger layer enforces
    # its own per-second cap (EGW00215) that is a DIFFERENT counter from the
    # gateway 18/s one above, and it does not publish the number. 0.35s
    # (≈3 inquiries/sec) is deliberately conservative: a market_lock hold's
    # worst case is ~4 inquiries = 1.4s, only 14% of
    # portfolio_risk_lock_wait_seconds=10.0. Paper stays 0.0 (disabled) —
    # the 1.0s global interval already bounds it there.
    # 원장 한도는 미공개라 보수적으로 3건/초 — 락 유지 중 최악 4회여도 1.4초.
    kis_account_inquiry_min_interval_paper: float = 0.0
    kis_account_inquiry_min_interval_real: float = 0.35

    # A rolling 1-second window cap on requests, layered on top of the min
    # interval above (app.brokers.base.http_client — see its class docstring).
    # KIS counts per rolling 1s window; the min interval only bounds the mean
    # rate, so under a busy event loop grants still reach the wire in clusters
    # and trip the gateway's per-second cap (EGW00201, 2026-09-03 decision
    # id=1179 — a different counter from EGW00215's ledger lane). 12/s keeps
    # ~33% headroom under the 18/s real-account gateway cap; the deliberate
    # throughput cut (18→12) is the point. Cross-process bursts (a local
    # backtest, a 2nd daemon on the same app key) are out of scope — the
    # per-instance window can't see them (standing rule: 로컬 데몬 2중 실행 절대 금지).
    # 1초 슬라이딩 윈도우 요청 상한 — 최소간격은 평균만 제한.
    kis_max_requests_per_second_paper: float = 0.0   # the 1.0s interval already bounds paper
    kis_max_requests_per_second_real: float = 12.0    # KIS guidance is 100-150ms between concurrent calls; 12/s keeps ~33% headroom under the 18/s gateway cap

    # 2026-09-07: the domestic realized-P&L report's fee/tax column (매매손익
    # 탭) was always ₩0 — root cause confirmed live: `prsm_tlex_smtl` (추정
    # 제비용합계) lives in `inquire_daily_ccld`'s `output2` (one summary object
    # per call), not in the per-fill `output1` rows
    # app.brokers.kis.domestic.order.fetch_daily_executions reads from. The fix
    # (app.brokers.kis.domestic.order.fetch_domestic_fee_tax) issues one
    # additional narrow probe call per (ticker, order date, side) group that
    # actually had a fill, to get that group's own output2 total. This caps
    # how many of those extra account-lane calls one /api/pnl request may
    # issue — bounded by the report's own realistic activity (a handful of
    # sells over the ~90-day window this desk actually sees), sorted by most
    # recent sell date first so the cap (if ever hit) drops the OLDEST/least-
    # relevant groups, not the newest. Serial only (account_lane is a non-
    # reentrant lock; concurrency would buy nothing) — worst case
    # max_groups x kis_account_inquiry_min_interval_real ≈ 30 x 0.35s ≈ 11s
    # added to an on-demand dashboard request. Do not raise this without also
    # reconsidering that latency budget.
    kis_pnl_fee_probe_max_groups: int = 30

    # Ceiling on how many of get_kr_financials's KIS fan-out calls (7 statement/
    # ratio TRs + valuation) are in flight at once — app.dataflows.kis_provider.
    # The Fundamentals Analyst also runs get_kr_financials concurrently with
    # get_kr_analyst_opinion, so the raw fan-out is ~9 simultaneous KIS calls,
    # which overwhelmed the client throttle and tripped EGW00201 (2026-09-03
    # decision id=1179). 3 keeps the burst well under the per-second gateway cap.
    # 국내 재무 팬아웃 동시성 상한 — 분석가 병렬 호출과 겹쳐 ~9건 동시 발생하던 것을 제한.
    kis_financials_fanout_concurrency: int = 3

    # get_kr_financials adds a downstream RECENCY warning when the newest
    # exchange-reported period it can fetch is at least this many months old.
    # Anchored to the Korean statutory filing calendar, not fitted to a sample:
    # a Dec-FY issuer files the 사업보고서 by end-March and the 반기보고서 by
    # mid-August, so a healthy issuer is at most ~9 months stale — 12 means "a
    # full fiscal year has closed and been filed since your newest data point".
    # 2026-09-07 domestic audit #6 (id=1237 / 294630.KS: KIS's annual series
    # ended at FY2024 — 20 months out — with no staleness signal anywhere, and
    # the desk read 2024 figures as the company's current condition). A second
    # tier at 2x this value escalates the wording to a data-quality red flag.
    kr_financials_staleness_warn_months: int = 12

    # Read-only/narrative-caller TTL for portfolio_risk.cached_account_snapshot
    # — the enforcement path (real_account_snapshot) is never affected by it.
    # 35.0, not 10.0 (2026-09-03 id=1187 audit, FIX-4): /api/safety-status
    # polls this on a 30s cadence, so a 10s TTL missed on almost every single
    # poll — each miss is a real KIS ledger-inquiry TR that did nothing but
    # add to that second's EGW00215 quota pressure. 35s covers one full
    # 30s poll cycle with margin, and matches the precedent already set by
    # app.web.state's own 35s dashboard snapshot cache.
    kis_account_read_cache_seconds: float = 35.0

    # Hard, operator-asserted scope switch: False means "this deployment's KIS
    # account is domestic (KRW) only — never poll the overseas ledger in the
    # background". Jenkins-injected, deliberately NOT a runtime_settings
    # override and deliberately NOT inferred from observed emptiness: a
    # degraded overseas TR looks IDENTICAL to a genuinely flat overseas
    # account (app.web.state._kis_account_snapshot_build swallows the failure
    # into overseas_ok=False; overseas/account.py's fetch_overseas_balance does
    # not even raise on a missing output2), so any "we haven't seen anything
    # overseas lately" heuristic learns the wrong lesson from an outage.
    #
    # This flag ONLY removes BACKGROUND POLLING. It never touches an
    # enforcement path: real_account_snapshot / daily_loss_block /
    # apply_live_portfolio_caps still query the overseas ledger for real
    # whenever an overseas ticker is actually being sized or gated. If an
    # overseas order somehow reaches execution, every existing check still runs.
    # 해외 미사용 계좌에서 배경 폴링만 차단 — 집행 경로 조회는 그대로.
    overseas_trading_enabled: bool = True

    @property
    def kis_min_request_interval(self) -> float:
        return self.kis_min_request_interval_real if self.kis_account_mode == "real" else self.kis_min_request_interval_paper

    def kis_min_request_interval_for(self, is_real: bool) -> float:
        """The request interval for a broker built from a credential set,
        keyed by that set's own is_real — NOT settings.kis_account_mode.
        Same source (the credentials) as base_url/is_real, so a config/DB
        mismatch on active mode can't point a real process at paper timing."""
        return self.kis_min_request_interval_real if is_real else self.kis_min_request_interval_paper

    def kis_account_inquiry_min_interval_for(self, is_real: bool) -> float:
        return (
            self.kis_account_inquiry_min_interval_real if is_real
            else self.kis_account_inquiry_min_interval_paper
        )

    def kis_max_requests_per_second_for(self, is_real: bool) -> float:
        """The rolling 1s-window request cap for a broker built from a
        credential set, keyed by that set's own is_real — same source
        (the credentials) as base_url/is_real, not settings.kis_account_mode."""
        return (
            self.kis_max_requests_per_second_real if is_real
            else self.kis_max_requests_per_second_paper
        )

    # KIS order mirroring (app/execution/kis_bridge.py). Three independent
    # layers, each defaulting to the safe side: the master switch is OFF, and
    # even when ON, dry-run stays ON until explicitly disabled — so two
    # deliberate config changes stand between a fresh checkout and a real
    # (paper) order. The cap bounds one order's estimated value in KRW.
    # Was previously `False` here — contradicted this very comment, meaning
    # only ONE change (kis_execution_enabled) stood between a fresh checkout
    # and a real order. Fixed to match the documented two-layer default.
    kis_execution_enabled: bool = False
    kis_execution_dry_run: bool = True
    # Category-exclusion safety gate (app/execution/kis_bridge.py,
    # app.domain.leverage_filter): blocks any order on a futures/leveraged/
    # inverse-named ticker. Config-only, like the two flags above — not in
    # risk_rules.RULE_SPEC, so the unauthenticated localhost dashboard API
    # can't turn off "don't trade leveraged products" the same way it can't
    # flip execution_enabled/dry_run.
    kis_block_leveraged_inverse: bool = True
    # Same category-exclusion pattern for out-of-scope products
    # (app.domain.product_filter.out_of_scope_reason): cash-like / bond /
    # bond-mix funds (CD금리·KOFR·머니마켓·국고채·종합채권·채권혼합 ETFs) AND,
    # since 2026-09-02, covered-call / option-income funds (커버드콜·covered
    # call·buy-write·premium income) — 498400.KS "KODEX 200타겟위클리커버드콜"
    # burned a full ~17-LLM-call analysis pass (audit id=1170). The setting
    # name is kept as `kis_block_cash_bond_funds` for env-var compatibility
    # even though the scope is now wider. The screener universe
    # (app.live.universe) and trade_allowlist_add exclude these
    # unconditionally; this flag only gates the kis_bridge order choke point,
    # and only on a BUY. Turning it off opens exactly one path for a human to
    # deliberately buy one: place_manual_order (the only path that skips the
    # trade_allowlist gate). Config-only, not in risk_rules.RULE_SPEC, same
    # reasoning as kis_block_leveraged_inverse.
    # 범위 외 상품(현금성/채권형 + 커버드콜/옵션인컴) 매수 차단 —
    # universe·allowlist는 항상, 이 플래그는 kis_bridge 초크포인트(매수만) 게이트.
    # env 호환을 위해 설정 이름은 그대로 둠(범위만 확장).
    kis_block_cash_bond_funds: bool = True
    # Default MANUAL: same "safe by default, opt in explicitly" convention as
    # the two flags above — a fresh checkout never auto-trades even if
    # kis_execution_enabled/dry_run are later turned on for manual-only use.
    execution_mode: ExecutionMode = ExecutionMode.MANUAL
    # Default PRODUCTION: same convention — a fresh checkout never generates
    # aggressive test signals or verbose debug output by accident.
    ai_profile: AIProfile = AIProfile.PRODUCTION
    # Only meaningful when ai_profile != PRODUCTION and kis_account_mode ==
    # "real" at the same time — see the model_validator below. Left False by
    # default so that combination fails to boot instead of silently running;
    # a person must deliberately flip this after confirming they really mean
    # to point a non-production AI profile at the real account.
    # ai_profile이 PRODUCTION이 아니면서 실전 계좌일 때만 의미있는 명시적 확인 플래그.
    ai_profile_live_ack: bool = False
    # Was 5,000,000 (~16.6% of the 2026-08-03 domestic equity below, already
    # fairly close to a sane ceiling — unlike the USD cap's old value, this
    # one wasn't badly mis-tuned) — confirmed live blocking a legitimate BUY
    # (459580.KS, concentration-cap-reduced to 9 shares / ~9,664,020 KRW,
    # itself already under the 25% concentration cap but still over this
    # flat value cap). Repriced to ~18% of that day's domestic equity
    # (30,160,220 * 0.18 ≈ 5,428,840, rounded up to 5,500,000) — same
    # "comfortably under the 20%/25% per-position and concentration caps"
    # goal as the USD cap below, just a slightly looser percentage since
    # 15% would have landed BELOW the old value for this (much smaller,
    # already close to its ceiling) domestic account — user's explicit
    # choice (2026-08-03) over matching the USD cap's 15% exactly.
    # Paper/real split (2026-08-05 — see project memory:
    # user-account-mode-db-separation-design). The value below was tuned
    # against THIS PAPER ACCOUNT'S OWN equity history (~18% of its
    # 2026-08-03 balance) — a real account almost certainly has a different
    # balance, so reusing that number as a real-mode cap is a guess, not a
    # limit. `kis_max_order_krw_real` has no safe default (see the
    # model_validator below): it must be set explicitly before this app can
    # boot in real mode, the same "refuse silently-wrong over guessing"
    # convention as every other real-account gate in this file.
    kis_max_order_krw_paper: float = 5_500_000.0
    kis_max_order_krw_real: float | None = None
    # US-order counterpart of the KRW cap. Was 3,500 (~the KRW cap's money at
    # 1,400 KRW/USD, i.e. sized for a small test account) — by 2026-07-29 the
    # live overseas paper account had grown to ~$251K, making that cap only
    # ~1.4% of equity (vs. the KRW cap's ~23.6% of its own, much smaller
    # domestic equity) and hard-blocking ordinary Buy decisions on any
    # reasonably-priced, lower-volatility stock (confirmed live: AVGO's
    # risk-sized 72 shares = ~$27,300, over 7x the old cap). Repriced to
    # ~15% of that day's overseas equity ($251,082 * 0.15 ≈ $37,662, rounded
    # up) — comfortably under the 20%/25% per-position and concentration
    # caps elsewhere, so it stays a real ceiling rather than becoming the
    # binding constraint. Static by design (not computed from the live
    # balance each time) — the user explicitly chose a fixed value to
    # re-tune manually later over auto-scaling with account size, to avoid
    # the cap silently shrinking during a drawdown or adding a live-balance
    # dependency to every order-value check. Same paper/real split and
    # real-requires-explicit-value rule as the KRW cap above.
    kis_max_order_usd_paper: float = 40_000.0
    kis_max_order_usd_real: float | None = None

    # Absolute ceilings on what a USER may raise their own order-value cap to
    # (app.execution.risk_rules._coerce). The caps themselves are per-user and
    # editable through /api/settings/risk, which is right — it is their money
    # and their risk appetite — but the cap is also the last backstop on order
    # size, and a backstop that can be fat-fingered upward by one keystroke
    # (55000000 for 5500000) is not much of a backstop.
    #
    # Only the two CAPS need a ceiling, not the approval thresholds:
    # risk_rules._validate_cross_field already rejects a threshold >= its cap,
    # so bounding the cap transitively bounds the threshold.
    # 캡에만 상한을 두면 승인 임계값은 교차검증(threshold < cap)으로 자동으로 묶임.
    #
    # None = no ceiling, which is the pre-2026-08-06 behavior and stays the
    # default so this cannot silently start rejecting a value someone already
    # relies on. Set both in production.
    # 미설정이면 상한 없음(기존 동작) — 운영 환경에서는 설정할 것.
    kis_max_order_krw_ceiling: float | None = None
    kis_max_order_usd_ceiling: float | None = None

    # User-settable conditional-auto-trading risk rules (A-1). Each rule is a
    # pair — an on/off flag (default OFF, so turning one on is a deliberate
    # act and nothing changes until then) plus its value. Enforced in
    # app/execution/kis_bridge.py at the code level, independent of any LLM
    # judgment. The runtime settings store (A-3) will layer on top of these
    # defaults; the gates read whichever source resolves the current value.

    # 종목당 최대 포지션 비중: a buy is blocked if it would push the ticker's
    # eval above this % of 총평가금액 (account equity incl. cash). Domestic
    # (KRW) only — overseas weight needs FX normalization (assumption, TODO).
    #
    # Unit note: this is a 0-100 PERCENT number (20.0 means 20%), NOT a 0-1
    # fraction — it's the one outlier among this file's "_pct" fields (every
    # portfolio_*_pct / kis_daily_max_loss_pct below is a 0-1 fraction, e.g.
    # 0.25 means 25%). Matches the dashboard's number input (min=0 max=100)
    # and risk_rules._coerce's `value > 100` bound end-to-end — don't
    # "normalize" this to a fraction without updating all three together.
    # 단위 주의: 이 값만 0~100 퍼센트 숫자(다른 *_pct 설정은 전부 0~1 비율).
    kis_max_position_pct_enabled: bool = False
    kis_max_position_pct: float = 20.0

    # 승인 임계금액: an order at or above this value needs explicit human
    # approval (approved=True) to pass. On the auto path that means it is held
    # (never auto-executed); on the manual path the dashboard supplies the
    # approval. Per-currency, mirroring the value cap. Same paper/real split
    # and real-requires-explicit-value rule as kis_max_order_krw/usd above —
    # an approval threshold tuned against the paper account's own equity is
    # just as much of a size-mismatched guess for a real account as the
    # value cap it's paired with.
    kis_approval_required_enabled: bool = False
    kis_approval_threshold_krw_paper: float = 3_000_000.0
    kis_approval_threshold_krw_real: float | None = None
    kis_approval_threshold_usd_paper: float = 2_000.0
    kis_approval_threshold_usd_real: float | None = None

    # Market-wide circuit-breaker gate (app/execution/kis_bridge.py gate 7) —
    # the real-time feed itself is app.brokers.kis.domestic.websocket.RealtimeFeed (started in
    # app.web.app's lifespan whenever KIS credentials + KRX trade allowlist codes
    # are present, same as the account panel — no separate switch, since the
    # feed itself never submits orders). This setting governs only what the
    # gate does when it can't determine circuit-breaker state (mapping not
    # yet populated in app.live.kis_market_status, feed not connected, or
    # ticker never subscribed). Confirmed fail-CLOSED: blocks the order
    # rather than letting it through — deliberate, since this means every
    # real KRX order is blocked until the mkop_cls_code value mapping (an
    # external input, not derivable from KIS's public docs) is populated.
    # Kept as a setting rather than hardcoded so it stays overridable without
    # a code change.
    kis_cb_fail_open_when_unknown: bool = False

    # Kill switch + daily max-loss circuit breaker (app/execution/safety_switch.py,
    # agent-architecture-roadmap stage 2) — the first gate in kis_bridge.py's
    # chain, ahead of every per-ticker check, applying to both domestic and
    # overseas orders on both the automated and manual paths. Persisted in
    # Postgres (same fail-closed-on-unknown philosophy as the circuit-breaker
    # gate above): if this module can't read its own state, that is treated as
    # "unsafe," not "safe by default."
    # 킬스위치·일일손실 서킷브레이커 — 상태를 못 읽으면 안전하지 않다고 간주(차단)
    kis_safety_fail_open_when_unknown: bool = False
    # Blocks any new BUY once today's aggregate portfolio value (USD-equivalent,
    # same basis as portfolio_drawdown_breaker_pct) has fallen this fraction
    # below its value at the first check of the current trading day. Resets
    # automatically at the first check of each new calendar day. Sells are
    # never blocked (same "reduce, never loosen" philosophy as every other
    # portfolio-level cap).
    kis_daily_max_loss_pct: float = 0.05

    # Master switch for app.orchestrator.pipeline._apply_unjustified_exit_gate
    # (2026-09-09, P0-2/3, id=1358): when ON, a PM Sell of a HELD position is
    # downgraded to Hold IF the PM's own final text cited the added-share
    # sizing cap or its ATR scaling as the basis for exiting
    # (scan_sizing_cap_misuse non-empty) AND the Research Manager concluded
    # Underweight — a trim — while the position is already at or below the
    # ATR-scaled Underweight target ("both" tier). Default True as of the
    # step-6 retrospective calibration: over 963 stored decisions (7 weeks)
    # this fires on exactly one held Sell — id=1358, the documented incident —
    # with zero false activations; the downgrade direction is the safe one (a
    # held position with an existing stop-loss is kept), and forced exits are
    # protected by the gate's own hard exceptions. Shipped OFF first as
    # detect-only instrumentation (PipelineState.sizing_cap_misuse_violations
    # / .unjustified_exit_gate with gated=False). A behavior switch, not a
    # risk-appetite knob — deliberately kept out of
    # app.execution.risk_rules.RULE_SPEC and the dashboard override, same as
    # the kill-switch flags above.
    unjustified_exit_gate_enabled: bool = True

    # Portfolio-level risk coordination (app/execution/portfolio_risk.py) —
    # unlike sizing_max_risk_pct_of_equity above, which only ever looks at
    # ONE ticker's own book, these look across the whole trade allowlist. All five
    # caps only ever REDUCE a trade already approved by the per-ticker 1%
    # rule; none of them can loosen it or force a sale of existing shares.
    #
    # Unit note: these (and kis_daily_max_loss_pct above) are 0-1 FRACTIONS
    # (0.25 means 25%) — do not confuse with kis_max_position_pct above,
    # which is a 0-100 percent number; see its own comment.
    portfolio_concentration_max_pct: float = 0.25
    portfolio_correlation_bucket_threshold: float = 0.6
    portfolio_correlation_bucket_max_pct: float = 0.45
    portfolio_correlation_lookback_days: int = 90
    # Blocks any exposure-INCREASING trade once aggregate portfolio equity
    # (in USD, across every ticker's own book) has fallen this fraction below
    # its own running high-water mark — a portfolio-wide analog of the
    # per-ticker stop-loss, since nothing upstream of this otherwise notices
    # a bad stretch spanning multiple tickers at once.
    portfolio_drawdown_breaker_pct: float = 0.12
    # Minimum fraction of aggregate account equity that must remain in cash —
    # blocks any exposure-INCREASING trade that would push aggregate invested
    # value above (1 - this). Unlike the caps above, which only stop money
    # piling into ONE ticker or one correlated cluster, this stops it piling
    # into MANY uncorrelated tickers at once (see max_open_positions_domestic/
    # _overseas below for the complementary "how many tickers" cap — the two
    # close different holes: this one alone permits 100 tiny positions, that
    # one alone permits a handful of positions each near the concentration
    # cap). This is the RISK-ON (normal) floor; see
    # portfolio_cash_floor_pct_risk_off below for the Risk-Off override
    # (app/execution/market_regime.py).
    portfolio_cash_floor_pct: float = 0.20
    # Cash floor used instead of the one above whenever
    # app.execution.market_regime's Faber-style trend signal (KOSPI vs its
    # own `faber_sma_window_days`-day moving average) says the domestic
    # market is Risk-Off — a wider buffer exactly when the broad market
    # itself looks like it's rolling over, not just when THIS account's own
    # drawdown breaker has already tripped (portfolio_drawdown_breaker_pct
    # above is reactive to this account's own losses; this is proactive,
    # reacting to the market's own trend). Only ever a REPLACEMENT for the
    # Risk-On floor above, never additive — the two never both apply at once.
    portfolio_cash_floor_pct_risk_off: float = 0.35

    # How long app.execution.portfolio_risk.market_lock polls (via
    # pg_try_advisory_lock, non-blocking) before giving up on a CONTENDED
    # lock — i.e. another process is actively holding it, not the DB being
    # unreachable. Deliberately short and separate from
    # database_command_timeout_seconds (125s): that timeout bounds a single
    # query, but this bounds how long a live decision cycle waits on another
    # process's in-flight trade before treating the wait itself as the
    # "block this trade, retry next cycle" outcome (raises
    # MarketLockContended — callers already retry a failed pipeline run per-
    # ticker, see app.live.auto_trade_loop). Genuine DB unreachability during
    # acquisition still fails OPEN (proceeds without serialization) exactly
    # as before; only lock CONTENTION is now fail-closed. See
    # app.execution.portfolio_risk._MarketAdvisoryLock's own docstring.
    portfolio_risk_lock_wait_seconds: float = 10.0
    # Moving-average window (trading days) for the Faber trend signal —
    # KOSPI (지수코드 0001) below its own trailing SMA over this many days is
    # Risk-Off, at/above it is Risk-On. 200 is Faber(2007)'s own daily-data
    # equivalent of his original 10-month-on-monthly-data rule.
    faber_sma_window_days: int = 200

    # Maximum number of distinct tickers held at once, per market — blocks a
    # BUY into a ticker NOT already held once this many are already open;
    # never blocks adding to an existing position, and never touches a sell.
    # Diversification research (Evans & Archer 1968; Statman 1987, "How Many
    # Stocks Make a Diversified Portfolio?") finds the marginal risk-reduction
    # benefit of each additional holding drops off sharply past ~10, and once
    # transaction/monitoring costs are weighed in, the practical optimum
    # lands around 15-30 — 15 here as a starting point at the conservative
    # end of that range, given this project's other per-ticker/cluster caps
    # (20-25%) already limit how concentrated a smaller book can get. Separate
    # domestic/overseas values, same reasoning as every other domestic/
    # overseas split in this project — one market being at its cap must never
    # block the other's first-time entries.
    max_open_positions_domestic: int = 15
    max_open_positions_overseas: int = 15

    # How many days must pass since a ticker's last LIVE full-pipeline
    # decision (4 analysts + debate + Portfolio Manager) before it's eligible
    # for another one — app.live.auto_trade_loop's momentum-ranked candidates
    # are filtered against this (app.memory.store.get_last_decision_time), so
    # a still-held position that keeps re-qualifying by momentum rank doesn't
    # get re-underwritten from scratch every single cycle even though its
    # fundamentals (quarterly filings) and its 10-day momentum signal haven't
    # meaningfully changed since yesterday. Does NOT apply to a ticker that
    # has never been analyzed (always eligible) or one flagged by
    # _held_needing_review (stop-loss proximity / top unrealized loss) —
    # those bypass THIS interval entirely (see held_review_min_interval_days
    # below for the shorter, separate throttle that applies to them
    # instead), since a risk signal deserves a much faster recheck than a
    # routine rebalance.
    # Real incident this addresses (2026-08-03): 379800.KS was bought at
    # 05:16 on a Buy decision, then re-analyzed from scratch and sold at
    # 05:45 the same morning — a ~30-minute round trip driven by re-running
    # the same analysis twice on unchanged fundamentals, not new information.
    # Matches the existing minimum-holding-days convention used elsewhere
    # (resolve_outcome's own holding-period floor) rather than introducing a
    # new number.
    #
    # This throttle MUST be applied to the candidate universe BEFORE momentum
    # ranking picks the top auto_trade_max_tickers_domestic/_overseas, never
    # after — see app.live.auto_trade_loop's _eligible_universe. Applying it
    # after the top-N cut (as this code briefly did) lets it starve the
    # entire pool forever: momentum ranking returns nearly the same order
    # every cycle, so once the top N names get locked by this throttle,
    # every following cycle re-ranks the SAME locked names into the SAME top
    # N slots and drops all of them again — no ticker ranked 5th or lower
    # ever gets a turn until every one of the top N happens to fall out of
    # the ranking on its own. Real incident (2026-08-26): the overseas
    # universe (41 tickers) was fully analyzed within one trading day, which
    # locked all 41 for 5 days; the next session's top-N candidates were the
    # same locked names every cycle, so the momentum path contributed zero
    # decisions all night — the only decision that session came from
    # _held_needing_review, which is deliberately exempt from this throttle.
    rebalance_interval_days: int = 5

    # Separate, much shorter throttle for _held_needing_review's forced
    # risk re-review (stop-loss proximity / top unrealized loss) — without
    # this, a ticker sitting near its stop-loss for several consecutive days
    # got a full ~20-LLM-call re-underwrite every single cycle it stayed
    # there, since that path was designed to always bypass
    # rebalance_interval_days entirely. The real stop-loss cut itself is
    # unaffected: app.web.state.stop_loss_loop is a separate, cheap,
    # rule-based (no LLM) hourly check that always runs regardless of this
    # value — this only throttles how often the AI re-underwrites/adjusts
    # its view of an already-flagged position, not whether the stop
    # actually fires. 2026-08-03 decision: deliberately much shorter than
    # rebalance_interval_days (a risk signal still needs closer attention
    # than a routine rebalance), but no longer literally every cycle.
    held_review_min_interval_days: int = 1

    # Symmetric bps haircut on both buy/sell notional, applied in
    # record_trade — every backtest fill before this was frictionless at the
    # exact close, which overstates achievable real-world returns. This is a
    # broker commission/slippage proxy only; the Korean sell-only tax below
    # is modeled separately since the two move for unrelated reasons (one is
    # a broker/execution assumption, the other is set by tax law).
    transaction_cost_bps: float = 5.0

    # Overseas (US) broker commission, both sides — kept separate from
    # `transaction_cost_bps` above because KIS's domestic and overseas fee
    # schedules are unrelated contracts. 25bp is an ESTIMATE (KIS's
    # published online overseas commission is commonly cited around 0.25%,
    # covering SEC fee/FINRA TAF/exchange pass-through as one bundled
    # figure) — not verified against this account's actual contracted rate,
    # same unverified-default posture `transaction_cost_bps` itself has had
    # (nobody had confirmed 5bp's source either; see 2026-09-08 fee/tax
    # audit). Before this, `transaction_cost_bps_for` charged the domestic
    # 5bp on overseas trades too — roughly a 5x understatement of real US
    # round-trip cost, silently optimistic in every overseas backtest/sweep
    # result (screening_sweep/scalping_harness/scalping_viability all read
    # this via app.execution.costs).
    # 해외(미국) 매매 수수료 — 국내 값과 별개 설정. 25bp는 KIS 공시 온라인 해외
    # 수수료율(~0.25%) 추정치이며 실제 계약 요율 미검증(국내 5bp도 마찬가지 상태).
    transaction_cost_bps_overseas: float = 25.0

    # Korean Securities Transaction Tax (증권거래세), SELL side only, KRX
    # (`.KS`) only — deliberately separate from `transaction_cost_bps` above
    # rather than folded into it, since a tax-law change shouldn't require
    # touching the broker-commission constant and vice versa. Rate effective
    # 2026-01-01: KOSPI = 0.05% 증권거래세 + 0.15% 농어촌특별세 = 0.20%;
    # KOSDAQ = 0.20% 증권거래세 (no 농특세) — both boards land on the same
    # 0.20%, so one flat rate covers both without needing a KOSPI/KOSDAQ
    # split. Before this, `record_trade` charged the symmetric
    # `transaction_cost_bps` on both sides, understating the real domestic
    # round-trip cost by ~20bp — negligible for multi-week swing holds, not
    # for intraday/scalping. Known gap: Korean ETFs are exempt from 증권거래세
    # and this has no instrument-type detection, so ETF sells are
    # over-charged — the conservative direction (never flatters a strategy),
    # left as-is until ETF trading is actually in scope.
    sell_tax_bps_domestic: float = 20.0

    def _active_mode(self) -> str:
        """The mode THIS Settings object's money limits resolve against: the
        per-request ContextVar when one is bound, otherwise this instance's own
        `kis_account_mode`.

        Falling back to `self` rather than to `current_account_mode()` matters
        for any Settings built outside the singleton — tests construct
        `Settings(kis_account_mode="real")` directly, and reading the global
        singleton's mode there would silently resolve a real-account limit
        against a paper-mode process. For the singleton the two are identical,
        so request-scoped overrides still work exactly as intended.
        전역 싱글턴이 아닌 인스턴스는 자기 자신의 모드를 따라야 함 —
        요청 컨텍스트가 있으면 그쪽이 우선."""
        return _current_account_mode.get() or self.kis_account_mode

    def _mode_scoped_money(self, key: str) -> float:
        """Resolves one of the kis_max_order_*/kis_approval_threshold_*
        pairs against the account mode in effect for THIS request/context
        (current_account_mode(), not the global default).

        Raises rather than asserting when the real-mode value is missing. It
        used to `assert value is not None` on the grounds that
        _require_real_money_limits_on_real_account refuses to boot that
        combination — but that guard only ever inspected the GLOBAL mode, so
        once mode became per-user it stopped covering a real-mode user on a
        paper-mode deployment. An `assert` is the wrong tool for a condition
        that can now actually occur, because `python -O` strips it and this
        would return None into an order-value comparison. The credential-save
        path (app.web.routers.credentials) refuses to store real credentials
        until these are configured, so reaching here should be impossible —
        this is the backstop for when it isn't.
        전역 모드 기준 부팅 검증은 사용자별 모드를 못 잡으므로, assert 대신
        실제 예외로 방어(-O 옵션에서 assert가 제거되는 문제도 함께 회피).
        """
        mode = self._active_mode()
        value = getattr(self, f"{key}_real" if mode == "real" else f"{key}_paper")
        if value is None:
            raise RuntimeError(
                f"{key}_{mode} is not configured — refusing to evaluate an order-value "
                f"limit with no value in {mode} mode. Set {key.upper()}_{mode.upper()} "
                "before using a real account."
            )
        return value

    def _set_mode_scoped_money(self, key: str, value: float) -> None:
        """Setting `settings.kis_max_order_krw = X` (tests, or any future
        runtime code) means "for whichever mode is active right now" — same
        as reading it — not "for paper specifically, no matter the mode"."""
        setattr(self, f"{key}_real" if self._active_mode() == "real" else f"{key}_paper", value)

    @property
    def kis_max_order_krw(self) -> float:
        return self._mode_scoped_money("kis_max_order_krw")

    @kis_max_order_krw.setter
    def kis_max_order_krw(self, value: float) -> None:
        self._set_mode_scoped_money("kis_max_order_krw", value)

    @property
    def kis_max_order_usd(self) -> float:
        return self._mode_scoped_money("kis_max_order_usd")

    @kis_max_order_usd.setter
    def kis_max_order_usd(self, value: float) -> None:
        self._set_mode_scoped_money("kis_max_order_usd", value)

    @property
    def kis_approval_threshold_krw(self) -> float:
        return self._mode_scoped_money("kis_approval_threshold_krw")

    @kis_approval_threshold_krw.setter
    def kis_approval_threshold_krw(self, value: float) -> None:
        self._set_mode_scoped_money("kis_approval_threshold_krw", value)

    @property
    def kis_approval_threshold_usd(self) -> float:
        return self._mode_scoped_money("kis_approval_threshold_usd")

    @kis_approval_threshold_usd.setter
    def kis_approval_threshold_usd(self, value: float) -> None:
        self._set_mode_scoped_money("kis_approval_threshold_usd", value)

    # ---------------- per-(user, mode) ticker lists ----------------

    def _list_scope(self) -> tuple[str, str]:
        """The (user_id, account_mode) these lists resolve against.

        Mode comes from `_active_mode()` — the same resolver the money limits
        use, so a Settings built outside the singleton follows its own mode.
        The user import is function-local because `app.user_context` imports
        this module; same reason `account_mode.py` keeps its ContextVar here.
        app.user_context가 이 모듈을 import하므로 함수 내부 import."""
        from app.user_context import get_current_user_id

        return (get_current_user_id(), self._active_mode())

    def _lists_for_scope(self) -> dict[str, list[str]]:
        """This scope's four lists, materializing fresh defaults on first touch.

        Materializing on READ is deliberate. The alternative — returning
        defaults without storing them — makes `settings.watchlist.append(x)`
        silently no-op, which is exactly the kind of bug that would look like
        "the dashboard forgot my ticker". Storing means every read hands back
        the same mutable list the writers hold.
        읽을 때 스코프를 생성함 — 안 그러면 리스트를 제자리에서 수정하는 코드가
        조용히 아무 일도 하지 않게 됨."""
        scope = self._list_scope()
        lists = _scoped_lists.get(scope)
        if lists is None:
            lists = {key: list(default) for key, default in SCOPED_LIST_KEYS.items()}
            _scoped_lists[scope] = lists
        return lists

    def _scoped_list(self, key: str) -> list[str]:
        return self._lists_for_scope()[key]

    def _set_scoped_list(self, key: str, value: list[str]) -> None:
        # Copied, not aliased: callers routinely pass a list they keep mutating
        # (state.py builds `[*old, ticker]` but tests pass literals they later
        # reuse), and aliasing would let one scope's list be edited through a
        # reference held elsewhere.
        self._lists_for_scope()[key] = list(value)

    @property
    def trade_allowlist_domestic(self) -> list[str]:
        return self._scoped_list("trade_allowlist_domestic")

    @trade_allowlist_domestic.setter
    def trade_allowlist_domestic(self, value: list[str]) -> None:
        self._set_scoped_list("trade_allowlist_domestic", value)

    @property
    def trade_allowlist_overseas(self) -> list[str]:
        return self._scoped_list("trade_allowlist_overseas")

    @trade_allowlist_overseas.setter
    def trade_allowlist_overseas(self, value: list[str]) -> None:
        self._set_scoped_list("trade_allowlist_overseas", value)

    @property
    def watchlist(self) -> list[str]:
        return self._scoped_list("watchlist")

    @watchlist.setter
    def watchlist(self, value: list[str]) -> None:
        self._set_scoped_list("watchlist", value)

    @property
    def trade_allowlist_auto_added(self) -> list[str]:
        return self._scoped_list("trade_allowlist_auto_added")

    @trade_allowlist_auto_added.setter
    def trade_allowlist_auto_added(self, value: list[str]) -> None:
        self._set_scoped_list("trade_allowlist_auto_added", value)

    @property
    def trade_allowlist(self) -> list[str]:
        """Combined domestic+overseas trade_allowlist — for the many
        call sites that don't need a per-market split. Prefer
        `trade_allowlist_for_market()` or the domestic/overseas fields
        directly wherever the caller already reasons about one market."""
        return [*self.trade_allowlist_domestic, *self.trade_allowlist_overseas]

    def trade_allowlist_for_market(self, market: Market) -> list[str]:
        """The trade_allowlist half relevant to one market — replaces the
        ad hoc `.KS`-suffix filtering that portfolio_risk.py/auto_trade_loop.py/
        pipeline.py/portfolio_harness.py each used to do against a single
        combined list."""
        return self.trade_allowlist_domestic if market == "domestic" else self.trade_allowlist_overseas

    @model_validator(mode="after")
    def _gateway_provider_needs_api_key(self) -> "Settings":
        # Fail-safe boot gate, same convention as the KIS gates below: with
        # llm_provider == "gateway" and no llm_api_key, every LLM call would
        # go out with a placeholder key and come back 401 — which the
        # client's retry/fallback machinery then dresses up as a connection
        # or schema failure, exactly the misdiagnosis this project has been
        # burned by before (see project memory: llm-model-swap-gpt-oss-issue).
        # Refuse to boot instead.
        # gateway인데 키가 없으면 401이 재시도/폴백에 묻혀 오진되므로 부팅 거부.
        if self.llm_provider == "gateway" and not self.llm_api_key:
            raise ValueError(
                "llm_provider='gateway' requires llm_api_key (the gateway's own endpoint key — "
                "set LLM_API_KEY in .env.local or inject it via Jenkins). Refusing to boot with "
                "a placeholder key that would make every LLM call fail 401."
            )
        return self

    @model_validator(mode="after")
    def _anthropic_provider_needs_api_key(self) -> "Settings":
        # Same fail-safe boot gate as _gateway_provider_needs_api_key above,
        # same reasoning: with llm_provider == "anthropic" and no
        # anthropic_api_key, every call would go out unauthenticated and come
        # back 401, which app.llm.anthropic_client's retry/fallback
        # machinery would then dress up as a connection or schema failure —
        # the exact misdiagnosis this project has been burned by before (see
        # project memory: llm-model-swap-gpt-oss-issue). Refuse to boot
        # instead of letting that surface hours into a live run.
        # anthropic인데 키가 없으면 401이 재시도/폴백에 묻혀 오진되므로 부팅 거부.
        if self.llm_provider == "anthropic" and not self.anthropic_api_key:
            raise ValueError(
                "llm_provider='anthropic' requires anthropic_api_key (set ANTHROPIC_API_KEY "
                "in .env.local, or inject it via Jenkins). Refusing to boot with every LLM "
                "call guaranteed to fail 401 instead of failing clearly now."
            )
        return self

    @model_validator(mode="after")
    def _database_url_must_be_a_postgres_uri(self) -> "Settings":
        # Fail-safe boot gate, same convention as the gateway key check above.
        # .env.example ships DATABASE_URL blank on purpose (no credentials in
        # git), which means a checkout that never got a real value from
        # .env.local / the Jenkins `invest-env-dev` secret would otherwise
        # pass an empty string straight to asyncpg.connect(""). asyncpg then
        # falls back to libpq's own defaults (PGHOST/PGUSER/local socket)
        # instead of raising — silently connecting to whatever local Postgres
        # happens to be reachable rather than failing loudly. Refuse to boot
        # instead of risking that misdirected connection.
        # DATABASE_URL이 비어있으면 asyncpg가 libpq 기본값(로컬 소켓 등)으로
        # 조용히 폴백할 수 있어 부팅을 거부한다.
        dsn = self.database_url.strip()
        if not dsn:
            raise ValueError(
                "DATABASE_URL is empty — asyncpg would fall back to libpq's own "
                "defaults (PGHOST/PGUSER/local socket) and could silently connect "
                "to an unintended local database instead of failing clearly. Set "
                "DATABASE_URL in .env.local, or in the Jenkins `invest-env-dev` "
                "secret file for the deployed container."
            )
        if not dsn.startswith(("postgresql://", "postgres://")):
            raise ValueError(
                "DATABASE_URL must be a postgresql:// (or postgres://) URI. Set it "
                "in .env.local, or in the Jenkins `invest-env-dev` secret file for "
                "the deployed container."
            )
        return self

    @model_validator(mode="after")
    def _ceilings_must_not_undercut_their_own_defaults(self) -> "Settings":
        # A ceiling below the value new users inherit would leave everyone
        # holding a cap they cannot re-save: risk_rules._coerce would reject
        # the very number the deployment handed them. Catch the
        # misconfiguration at boot, where it is an operator's to fix, rather
        # than as a confusing 400 on someone's settings screen.
        # 기본값보다 낮은 상한은 사용자가 자기 값을 다시 저장할 수 없게 만듦 —
        # 운영자가 고칠 수 있는 부팅 시점에 잡는다.
        for ceiling_name, default_names in (
            ("kis_max_order_krw_ceiling", ("kis_max_order_krw_paper", "kis_max_order_krw_real")),
            ("kis_max_order_usd_ceiling", ("kis_max_order_usd_paper", "kis_max_order_usd_real")),
        ):
            ceiling = getattr(self, ceiling_name)
            if ceiling is None:
                continue
            for default_name in default_names:
                default = getattr(self, default_name)
                if default is not None and default > ceiling:
                    raise ValueError(
                        f"{ceiling_name}={ceiling:g} is below {default_name}={default:g} — "
                        "every user would inherit a cap they cannot save back. Raise the "
                        "ceiling or lower the default."
                    )
        return self

    @model_validator(mode="after")
    def _refuse_non_loopback_bind(self) -> "Settings":
        # Fail-safe boot gate, same convention as the two below. There is no
        # application-level authentication in this deployment (2026-08-07
        # single-account rollback, see
        # docs/single-user-rollback-guide.md §0/§1) — POST /api/order
        # places real orders against the company's real KIS account with
        # nothing in front of it but this process's bind address. Access
        # control is entirely "stay on loopback, reach it over an SSH
        # tunnel." Unlike the auth_required flag this replaces, there is no
        # toggle that makes a non-loopback bind acceptable here: refuses to
        # boot rather than degrade quietly, unconditionally, whenever
        # dashboard_host is not loopback.
        # 인증 계층이 없음 — 루프백 바인딩 + SSH 터널이 접근 통제의 전부이므로,
        # 루프백이 아니면 조건 없이 부팅을 거부한다(예전의 auth_required 같은
        # 우회 스위치가 없음).
        loopback = {"127.0.0.1", "localhost", "::1"}
        if self.dashboard_host not in loopback:
            raise ValueError(
                f"dashboard_host={self.dashboard_host!r} is not loopback. This deployment has "
                "no application-level authentication — order placement must stay unreachable "
                "from the network. Keep dashboard_host at 127.0.0.1 and use an SSH tunnel for "
                "remote access. See docs/single-user-rollback-guide.md §0/§1."
            )
        return self

    @model_validator(mode="after")
    def _block_non_production_profile_on_real_account(self) -> "Settings":
        # Fail-safe boot gate: a TEST_AGGRESSIVE/DEBUG profile is meant for
        # paper-account pipeline testing (it deliberately biases toward
        # BUY/SELL over HOLD) — pointing that at kis_account_mode == "real"
        # without a deliberate ack means aggressive test signals could reach
        # a real account's order flow. Refuses to boot rather than degrade
        # quietly, same fail-safe convention as every other KIS safety gate.
        # 테스트용 AI 프로필이 실전 계좌와 함께 부팅되려면 명시적 확인이 있어야 함.
        if self.ai_profile != AIProfile.PRODUCTION and self.kis_account_mode == "real" and not self.ai_profile_live_ack:
            raise ValueError(
                f"ai_profile={self.ai_profile.value} with kis_account_mode='real' requires "
                "ai_profile_live_ack=True — refusing to boot a non-production AI profile "
                "against the real account without an explicit acknowledgement."
            )
        return self

    @model_validator(mode="after")
    def _require_real_money_limits_on_real_account(self) -> "Settings":
        # Fail-safe boot gate, same convention as the profile guard above:
        # the *_paper defaults were tuned against this paper account's own
        # equity history (see kis_max_order_krw's own comment) — silently
        # falling back to them for a real account of unknown, likely very
        # different size would size every cap/threshold gate off a made-up
        # number instead of a deliberate one. Refuses to boot rather than
        # guess, the same way a missing kis_real_app_key already does.
        # 실전계좌 진입 시 이 한도들은 반드시 명시적으로 설정되어야 함 —
        # 모의계좌 기준으로 튜닝된 기본값을 그대로 쓰면 계좌 규모가 안 맞음.
        if self.kis_account_mode != "real":
            return self
        missing = [
            f"{key}_real"
            for key in (
                "kis_max_order_krw", "kis_max_order_usd",
                "kis_approval_threshold_krw", "kis_approval_threshold_usd",
            )
            if getattr(self, f"{key}_real") is None
        ]
        if missing:
            raise ValueError(
                f"kis_account_mode='real' requires {', '.join(missing)} to be set explicitly "
                "(in the config file) — refusing to boot a real account against order-value "
                "caps/approval thresholds that were tuned for the paper account's own, "
                "differently-sized equity."
            )
        return self

settings = Settings()


# Per-request account mode (multi-user migration — see
# docs/multi-user-credentials-plan.md).
#
# `settings.kis_account_mode` used to be THE answer to "paper or real", read
# directly by ~50 call sites that each use it to scope a persistence key
# (`paper:`/`real:` prefixes on ledger rows, allowlists, kill-switch state) or
# to pick a safety limit. Once one process serves many users, those call sites
# need the mode of whoever's request they are serving, not one process-wide
# value — a developer on 모의투자 and a user on 실전 have to coexist.
#
# Same ContextVar shape as app.user_context.get_current_user_id(), and for the
# same reasons: it propagates into every coroutine a request awaits, never
# leaks between concurrently-running requests, and falls back to the
# configured global for everything outside a request (CLI, backtest, the
# background loops, tests), so those paths behave exactly as before.
#
# Lives in this module rather than a new one purely to avoid an import cycle:
# Settings._mode_scoped_money needs it, and all ~50 call sites already import
# `settings` from here, so this adds no new import edges anywhere.
# 요청별 계좌 모드 — 요청 밖에서는 기존 전역 설정으로 폴백하므로 CLI/백테스트
# 동작은 변하지 않음. 순환 import를 피하려고 config 모듈에 둠.
# (user_id, account_mode) -> {list_key: tickers}. The in-memory home of the
# four ticker lists, which used to be process-global pydantic fields on
# Settings (remaining Phase 5 work; see the block where they were declared).
#
# Lives here rather than in app.web.trade_allowlist_store — which owns their
# PERSISTENCE — for the same reason the account-mode ContextVar does: Settings
# itself has to read them, and trade_allowlist_store imports Settings.
# 영속화는 trade_allowlist_store가 담당하지만, Settings가 직접 읽어야 하므로
# 인메모리 저장소는 순환 import를 피해 이 모듈에 둠.
_scoped_lists: dict[tuple[str, str], dict[str, list[str]]] = {}

# What a scope with nothing stored starts as: **empty**, all four.
#
# This is a deliberate behaviour change from the old
# `_default_trade_allowlist_when_unset` validator, which seeded
# trade_allowlist_overseas with AAPL when both allowlists were empty. Its
# rationale was that "an empty dashboard on first run looks broken rather than
# 'add your first ticker'" — reasonable when there was one user and no setup
# flow.
#
# Carrying that seed into per-scope defaults would mean **every new user is
# born with AAPL on their trading allowlist**, i.e. auto-granted trading
# permission for a ticker they never chose, on their own real credentials. The
# allowlist is the gate `app.execution.kis_bridge` checks; seeding it is not a
# cosmetic default. Phase 3 also gave first-run its own explicit path (login →
# credentials → onboarding), so an empty allowlist immediately afterwards reads
# as "add your first ticker" on its own.
# 신규 스코프에 종목을 심으면 사용자가 고르지도 않은 종목에 매매 권한이 생김 —
# 매매허용목록은 장식용 기본값이 아니라 kis_bridge가 검사하는 게이트임.
SCOPED_LIST_KEYS: dict[str, tuple[str, ...]] = {
    "trade_allowlist_domestic": (),
    "trade_allowlist_overseas": (),
    "watchlist": (),
    "trade_allowlist_auto_added": (),
}


def normalize_scope(user_id: str, account_mode: str) -> None:
    """Re-establish the two cross-list invariants for one scope.

    These were `model_validator`s that ran once, at config construction, when
    the lists were global. They cannot stay validators: a scope is populated
    by `trade_allowlist_store.load()` long after Settings exists, and there is
    no single object whose construction covers every scope. So they run where
    data now enters — after a load, and after any bulk mutation.
    설정 생성 시점 검증기였지만, 스코프는 Settings 생성 이후에 채워지므로
    데이터가 들어오는 지점에서 실행해야 함.

      1. trade_allowlist ⊆ watchlist. Folded in automatically rather than
         refused: at load time no human is present to confirm a promotion, and
         a ticker with trading permission that is missing from the lighter
         tracking list is incoherent either way.
      2. auto_added ⊆ trade_allowlist. An auto-added marker only means
         anything while the ticker is actually on the allowlist; if it left by
         any path without the marker being cleared (dashboard removal, a prune
         bug, a manual DB edit), the stale marker is dropped here rather than
         lingering forever and making the ticker look auto-prunable.
    """
    lists = _scoped_lists.get((user_id, account_mode))
    if lists is None:
        return
    allowlist = [*lists["trade_allowlist_domestic"], *lists["trade_allowlist_overseas"]]
    missing = [t for t in allowlist if t not in lists["watchlist"]]
    if missing:
        lists["watchlist"] = [*lists["watchlist"], *missing]
    lists["trade_allowlist_auto_added"] = [
        t for t in lists["trade_allowlist_auto_added"] if t in allowlist
    ]


def scoped_lists_snapshot() -> dict[tuple[str, str], dict[str, list[str]]]:
    """Read-only view, for tests and diagnostics. Not for mutation — writers
    go through `settings.<list> = ...` so the scope is resolved consistently."""
    return {scope: {k: list(v) for k, v in lists.items()} for scope, lists in _scoped_lists.items()}


def reset_scoped_lists() -> None:
    """Drop every scope. Tests use this between cases; nothing in production
    should need it, since a scope is cheap and bounded by user count."""
    _scoped_lists.clear()


def current_account_mode() -> str:
    """The account mode in effect for this request/context."""
    return _current_account_mode.get() or settings.kis_account_mode


def set_current_account_mode(mode: str) -> contextvars.Token:
    return _current_account_mode.set(mode)


def reset_current_account_mode(token: contextvars.Token) -> None:
    _current_account_mode.reset(token)


@contextmanager
def account_mode_scope(mode: str) -> Iterator[str]:
    """Run a block in `mode`, restoring the previous one afterwards. Used by
    the background loops, which iterate over users and must adopt each user's
    own mode per iteration, and by tests."""
    token = _current_account_mode.set(mode)
    try:
        yield mode
    finally:
        _current_account_mode.reset(token)


def is_backtest_context() -> bool:
    """True while inside a `backtest_scope()` block (a historical replay),
    False for every live code path — the default."""
    return _is_backtest.get()


@contextmanager
def backtest_scope() -> Iterator[None]:
    """Run a block as a historical replay: `app.execution.ledger` writes
    tag `is_backtest=true` and reads only see other `is_backtest=true` rows
    (never live's), for the duration of the block. Wraps exactly the two
    replay entrypoints (`app.backtest.harness.run_backtest`,
    `app.backtest.portfolio_harness.run_portfolio_backtest`) — every other
    backtest CLI/sweep funnels through one of those two, so this is the
    single place the boundary needs to be drawn.

    2026-09-15: `app.orchestrator.pipeline`'s `memory_entries` write honours
    this too, not just the ledger. It previously derived `is_backtest` from
    `as_of is not None` alone, which silently tagged an experiment harness's
    rows as live decisions whenever that harness needed the live call shape
    (no `as_of`) inside this scope — see that call site's own comment."""
    token = _is_backtest.set(True)
    try:
        yield
    finally:
        _is_backtest.reset(token)
