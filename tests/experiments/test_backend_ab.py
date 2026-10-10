"""app.experiments.backend_ab — provider-swap mechanism, guard-violation
scan, run planning, and --dry-run. No network/DB/LLM (F-stage): the Claude
arm is always exercised against a fake `LLMPort`-satisfying object, never a
real `AnthropicLLMClient` — see the module's own hard constraint against
spending real money or constructing a real Anthropic client in any test.
"""

from __future__ import annotations

import dataclasses
from datetime import date

import pytest

from app.experiments import backend_ab
from app.state import PipelineState


class _FakeLLMClient:
    """Minimal LLMPort-satisfying fake — records how many times it was
    called through `complete`/`structured_complete` so a test can prove a
    module's OWN bound `llm_client` name actually resolves to this object,
    not just that `app.llm.llm_client` does."""

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, *args, **kwargs):
        self.calls += 1
        return "fake-complete"

    async def structured_complete(self, *args, **kwargs):
        self.calls += 1
        return "fake-structured"


@pytest.fixture(autouse=True)
def _reset_claude_client_cache(monkeypatch):
    """Every test gets a clean slate: no cached claude client from a prior
    test leaks in, and the module-level cache is restored afterward."""
    monkeypatch.setattr(backend_ab, "_claude_client_instance", None)
    yield
    # Belt-and-braces: make sure no test left a real module patched.
    for name in backend_ab._REQUIRED_PATCH_TARGETS:
        import sys

        mod = sys.modules.get(name)
        if mod is not None and getattr(mod, "llm_client", None) is not backend_ab._ORIGINAL_LLM_CLIENT:
            setattr(mod, "llm_client", backend_ab._ORIGINAL_LLM_CLIENT)


def _fake_claude_client(monkeypatch, fake: _FakeLLMClient) -> None:
    monkeypatch.setattr(backend_ab, "_claude_client", lambda: fake)


# ---------------------------------------------------------------------------
# Part 1 — the provider-swap mechanism (_backend_arm)
# ---------------------------------------------------------------------------


def test_backend_arm_claude_patches_every_required_module(monkeypatch):
    fake = _FakeLLMClient()
    _fake_claude_client(monkeypatch, fake)

    with backend_ab._backend_arm(backend_ab.ARM_CLAUDE):
        for name in backend_ab._REQUIRED_PATCH_TARGETS:
            import sys

            mod = sys.modules[name]
            assert mod.llm_client is fake, f"{name} was not patched to the fake claude client"
        assert backend_ab.settings.llm_provider == "anthropic"

    # Exit restores every target to the original qwen-arm client.
    import sys

    for name in backend_ab._REQUIRED_PATCH_TARGETS:
        assert sys.modules[name].llm_client is backend_ab._ORIGINAL_LLM_CLIENT
    assert backend_ab.settings.llm_provider != "anthropic"


def test_backend_arm_claude_actually_reaches_a_different_module_binding(monkeypatch):
    """Proves the swap isn't just visible on `app.llm.llm_client` itself —
    a DIFFERENT importer module's own bound name must resolve to the fake
    and actually be callable through it."""
    fake = _FakeLLMClient()
    _fake_claude_client(monkeypatch, fake)

    import app.agents.tool_loop as tool_loop_mod

    with backend_ab._backend_arm(backend_ab.ARM_CLAUDE):
        import asyncio

        asyncio.run(tool_loop_mod.llm_client.complete([]))
        assert fake.calls == 1


def test_backend_arm_claude_restores_on_exception(monkeypatch):
    fake = _FakeLLMClient()
    _fake_claude_client(monkeypatch, fake)

    with pytest.raises(ValueError):
        with backend_ab._backend_arm(backend_ab.ARM_CLAUDE):
            raise ValueError("boom")

    import sys

    for name in backend_ab._REQUIRED_PATCH_TARGETS:
        assert sys.modules[name].llm_client is backend_ab._ORIGINAL_LLM_CLIENT


def test_backend_arm_qwen_is_a_no_op_identity(monkeypatch):
    fake = _FakeLLMClient()
    _fake_claude_client(monkeypatch, fake)

    with backend_ab._backend_arm(backend_ab.ARM_QWEN):
        import sys

        for name in backend_ab._REQUIRED_PATCH_TARGETS:
            assert sys.modules[name].llm_client is backend_ab._ORIGINAL_LLM_CLIENT
        assert backend_ab.settings.llm_provider != "anthropic"

    import sys

    for name in backend_ab._REQUIRED_PATCH_TARGETS:
        assert sys.modules[name].llm_client is backend_ab._ORIGINAL_LLM_CLIENT


def test_backend_arm_missing_target_raises_runtime_error(monkeypatch):
    """The mandatory post-patch assertion must actually fire: if a required
    module is somehow NOT patched (simulated here by removing it from
    sys.modules so the assertion sees "not imported"), _backend_arm must
    raise rather than silently proceed."""
    import sys

    fake = _FakeLLMClient()
    _fake_claude_client(monkeypatch, fake)
    real_mod = sys.modules["app.chat.router"]
    monkeypatch.delitem(sys.modules, "app.chat.router")
    try:
        with pytest.raises(RuntimeError, match="app.chat.router"):
            with backend_ab._backend_arm(backend_ab.ARM_CLAUDE):
                pass
    finally:
        sys.modules["app.chat.router"] = real_mod


# ---------------------------------------------------------------------------
# Part 2 — _plan_runs
# ---------------------------------------------------------------------------


def test_plan_runs_shape_and_interleaving():
    tickers = ["005930.KS", "AAPL"]
    dates = [date(2026, 7, 1)]
    repeats = 3
    runs = backend_ab._plan_runs(tickers, dates, repeats)

    assert len(runs) == len(tickers) * len(dates) * repeats * 2

    seen = set()
    for ticker, trade_date, arm, repeat in runs:
        seen.add((ticker, trade_date, arm, repeat))
    for ticker in tickers:
        for trade_date in dates:
            for repeat in range(1, repeats + 1):
                for arm in backend_ab._ARMS:
                    assert (ticker, trade_date, arm, repeat) in seen

    # Alternates first-arm on alternating repeats, mirroring reasoning_ab's
    # own rotation convention.
    by_ticker_date = [r for r in runs if r[0] == "005930.KS" and r[1] == date(2026, 7, 1)]
    rep1_first = by_ticker_date[0][2]
    rep2_first = by_ticker_date[2][2]
    assert rep1_first != rep2_first


# ---------------------------------------------------------------------------
# _guard_violation_counts
# ---------------------------------------------------------------------------


def _make_state(**overrides) -> PipelineState:
    return PipelineState(ticker="005930.KS", trade_date=date(2026, 7, 1), **overrides)


def test_guard_violation_counts_reads_list_str_and_list_dict_fields():
    state = _make_state(
        identity_violations=["Fundamentals Analyst: bad name"],
        halt_status_violations=["Market Analyst: false halt claim", "News Analyst: also false"],
        injection_flags=[{"agent": "Bull", "tier": "rest_kept", "reason": "x"}],
    )
    counts = backend_ab._guard_violation_counts(state)

    assert counts["identity_violations"] == 1
    assert counts["halt_status_violations"] == 2
    assert counts["injection_flags"] == 1
    assert sum(counts.values()) == 4 + sum(
        v for k, v in counts.items() if k not in {"identity_violations", "halt_status_violations", "injection_flags"}
    )


def test_guard_violation_counts_excludes_non_suffixed_guard_fields():
    state = _make_state()
    counts = backend_ab._guard_violation_counts(state)
    excluded = {
        "degraded_analysts",
        "reasoning_sanitizations",
        "identity_unnamed",
        "unit_scale_scans",
        "period_mismatch_scans",
        "consensus_override",
        "unjustified_exit_gate",
        "atr_gate",
        "neutral_reversal",
        "kr_financials_recency",
        "news_amount_scans",
        "kr_risk_signals",
        "pm_narrative_contradiction_notice_applied",
        "session_in_progress",
        "indicator_block_available",
        "flow_block_available",
    }
    assert excluded.isdisjoint(counts.keys())


def test_guard_violation_counts_field_count_matches_authoritative_suffix_scan():
    state = _make_state()
    expected = [
        f.name for f in dataclasses.fields(PipelineState)
        if f.name.endswith(("_violations", "_flags"))
    ]
    counts = backend_ab._guard_violation_counts(state)
    assert set(counts.keys()) == set(expected)
    assert len(counts) == len(expected)


def test_guard_violation_counts_total_matches_manual_sum():
    state = _make_state(
        identity_violations=["a"],
        unit_scale_violations=["b", "c"],
        price_limit_violations=[],
    )
    counts = backend_ab._guard_violation_counts(state)
    assert sum(counts.values()) == 3


# ---------------------------------------------------------------------------
# --dry-run
# ---------------------------------------------------------------------------


def test_dry_run_never_constructs_a_real_anthropic_client(monkeypatch, capsys, tmp_path):
    constructed = []

    class _ExplodingAnthropicClient:
        def __init__(self):
            constructed.append(True)
            raise AssertionError("AnthropicLLMClient must never be constructed during --dry-run")

    monkeypatch.setattr(backend_ab, "_claude_client", lambda: (_ for _ in ()).throw(
        AssertionError("_claude_client must never be called during --dry-run")
    ))
    monkeypatch.setattr(
        "sys.argv",
        [
            "backend_ab",
            "--tickers", "005930.KS",
            "--dates", "2026-07-01",
            "--repeats", "1",
            "--dry-run",
            "--out-dir", str(tmp_path / "out"),
        ],
    )

    import asyncio

    asyncio.run(backend_ab._main())

    assert constructed == []
    out = capsys.readouterr().out
    assert "planned" in out
    assert not (tmp_path / "out").exists()
