"""app.orchestrator.pipeline._PM_CONTRADICTION_NOTICE — the shared PM-
narrative-contradiction notice text itself (2026-09-11). Standalone from
test_pipeline_integration.py's full-pipeline wiring tests: this only checks
the note dict's own content against the same HARD CONSTRAINT
unit_scale_guard's _CURRENCY_INVERTED_BANNER / _NEWS_EOK_NOTE /
_DERIVED_USD_STALE_NOTE already carry, since identity_gate's foreign-listing
Detector C counts KRW-shaped/USD-shaped figures across the whole report and
this text must never perturb that count — see
tests/agents/test_unit_scale_guard.py's
test_id1396_derived_usd_stale_note_has_no_currency_or_digit_tokens for the
pattern this mirrors."""

from __future__ import annotations

from app.agents.identity_gate import _KRW_AMOUNT_RE
from app.orchestrator.pipeline import _PM_CONTRADICTION_NOTICE


def test_pm_contradiction_notice_has_both_languages():
    assert set(_PM_CONTRADICTION_NOTICE) == {"en", "ko"}
    assert _PM_CONTRADICTION_NOTICE["en"].strip()
    assert _PM_CONTRADICTION_NOTICE["ko"].strip()


def test_pm_contradiction_notice_has_no_currency_or_digit_tokens():
    for lang, text in _PM_CONTRADICTION_NOTICE.items():
        assert _KRW_AMOUNT_RE.search(text) is None, (lang, text)
        assert "$" not in text, (lang, text)
        assert not any(ch.isdigit() for ch in text), (lang, text)
        for token in ("KRW", "원", "won", "₩"):
            assert token not in text, (lang, token, text)
