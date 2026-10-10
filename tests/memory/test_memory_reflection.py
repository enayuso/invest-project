"""app.memory.reflection — llm_client mocked, no network (F-stage)."""

from __future__ import annotations

from app.memory import reflection
from app.llm.client import LLMMessage


class FakeLLM:
    def __init__(self, content):
        self._content = content
        self.calls = []

    async def complete(self, messages, *, tools=None, reasoning=False, temperature=0.0, label="", language="ko", max_tokens=None):
        self.calls.append(messages)
        return LLMMessage(content=self._content)


async def test_reflect_includes_return_figures_and_decision_text(monkeypatch):
    fake = FakeLLM("The call was right; the thesis held on earnings strength. Lesson: trust the trend.")
    monkeypatch.setattr(reflection, "llm_client", fake)

    result = await reflection.reflect("Rating: Buy\nThesis: strong earnings", 0.08, 0.03, benchmark_ticker="SPY")

    assert result == "The call was right; the thesis held on earnings strength. Lesson: trust the trend."
    prompt = fake.calls[0][1]["content"]
    assert "Raw return: +8.0%" in prompt
    assert "Alpha vs SPY: +3.0%" in prompt
    assert "Rating: Buy" in prompt


async def test_reflect_uses_custom_benchmark_ticker(monkeypatch):
    fake = FakeLLM("reflection text")
    monkeypatch.setattr(reflection, "llm_client", fake)

    await reflection.reflect("decision", -0.05, -0.02, benchmark_ticker="KOSPI")

    prompt = fake.calls[0][1]["content"]
    assert "Alpha vs KOSPI: -2.0%" in prompt


# ---------------------------------------------------------------------------
# Fix 3: win-direction note (2026-08-27) — without this, the model likely
# defaults to "positive alpha = good" for every rating, which is backwards
# for a Sell (a Sell is correct when alpha is NEGATIVE — the exited position
# beating the benchmark means it should have been kept) and undefined for a
# Hold (whose correct direction depends on whether a position was already
# held). See app.audit.stats._WIN_DIRECTION for the canonical mapping this
# mirrors.
# ---------------------------------------------------------------------------


async def test_reflect_sell_note_says_negative_alpha_is_correct(monkeypatch):
    fake = FakeLLM("reflection text")
    monkeypatch.setattr(reflection, "llm_client", fake)

    await reflection.reflect("Rating: Sell", 0.05, 0.03, benchmark_ticker="SPY", rating="Sell")

    prompt = fake.calls[0][1]["content"]
    assert "alpha_return is negative" in prompt
    assert "alpha_return is positive" not in prompt


async def test_reflect_buy_note_says_positive_alpha_is_correct(monkeypatch):
    fake = FakeLLM("reflection text")
    monkeypatch.setattr(reflection, "llm_client", fake)

    await reflection.reflect("Rating: Buy", 0.05, 0.03, benchmark_ticker="SPY", rating="Buy")

    prompt = fake.calls[0][1]["content"]
    assert "alpha_return is positive" in prompt


async def test_reflect_held_hold_note_references_raw_return_not_alpha(monkeypatch):
    fake = FakeLLM("reflection text")
    monkeypatch.setattr(reflection, "llm_client", fake)

    await reflection.reflect(
        "Rating: Hold", 0.05, -0.01, benchmark_ticker="SPY", rating="Hold", held_at_decision=True
    )

    prompt = fake.calls[0][1]["content"]
    assert "raw_return is positive" in prompt
    assert "alpha_return" not in prompt.split("Note:")[1].split("\n")[0]


async def test_reflect_flat_hold_note_says_negative_alpha_is_correct(monkeypatch):
    fake = FakeLLM("reflection text")
    monkeypatch.setattr(reflection, "llm_client", fake)

    await reflection.reflect(
        "Rating: Hold", 0.0, -0.01, benchmark_ticker="SPY", rating="Hold", held_at_decision=False
    )

    prompt = fake.calls[0][1]["content"]
    assert "alpha_return is negative" in prompt


async def test_reflect_omits_note_when_rating_unknown(monkeypatch):
    fake = FakeLLM("reflection text")
    monkeypatch.setattr(reflection, "llm_client", fake)

    await reflection.reflect("decision", 0.05, 0.03, benchmark_ticker="SPY")

    prompt = fake.calls[0][1]["content"]
    assert "Note:" not in prompt


async def test_reflect_omits_note_for_hold_with_unknown_held_state(monkeypatch):
    fake = FakeLLM("reflection text")
    monkeypatch.setattr(reflection, "llm_client", fake)

    await reflection.reflect(
        "decision", 0.05, 0.03, benchmark_ticker="SPY", rating="Hold", held_at_decision=None
    )

    prompt = fake.calls[0][1]["content"]
    assert "Note:" not in prompt


# ---------------------------------------------------------------------------
# Fix 3: empty/None LLM response must not propagate as garbage
# ---------------------------------------------------------------------------


async def test_reflect_returns_empty_string_on_none_content(monkeypatch):
    fake = FakeLLM(None)
    monkeypatch.setattr(reflection, "llm_client", fake)

    result = await reflection.reflect("decision", 0.05, 0.02, benchmark_ticker="SPY")

    assert result == ""


async def test_reflect_returns_empty_string_on_whitespace_only_content(monkeypatch):
    fake = FakeLLM("   \n  ")
    monkeypatch.setattr(reflection, "llm_client", fake)

    result = await reflection.reflect("decision", 0.05, 0.02, benchmark_ticker="SPY")

    assert result == ""
