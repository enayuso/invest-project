"""app.llm.language — detect_language/language_instruction, pure (fixes the
"answers come back in English regardless of question language" report)."""

from __future__ import annotations

from app.llm.language import DEFAULT_LANGUAGE, detect_language, language_instruction


def test_detects_korean_from_hangul():
    assert detect_language("삼성 주식 어때?") == "ko"


def test_detects_english_when_no_hangul():
    assert detect_language("How's Samsung stock looking?") == "en"


def test_ticker_symbols_alone_do_not_trigger_korean():
    assert detect_language("AAPL 005930.KS price?") == "en"


def test_mixed_text_with_any_hangul_is_korean():
    assert detect_language("AAPL 주가 어때?") == "ko"


def test_jamo_only_chat_shorthand_is_korean():
    # 2026-08-26: compatibility jamo (no full Hangul syllable) used to fall
    # through to "en" — common Korean chat shorthand with no syllables at all.
    assert detect_language("ㅋㅋㅋ") == "ko"
    assert detect_language("ㅎㅇ") == "ko"
    assert detect_language("ㅠㅠ") == "ko"


def test_default_language_is_english():
    assert DEFAULT_LANGUAGE == "en"


def test_language_instruction_korean_mentions_korean():
    assert "한국어" in language_instruction("ko")


def test_language_instruction_english_mentions_english():
    assert "English" in language_instruction("en")


def test_language_instruction_falls_back_to_default_for_unknown_code():
    assert language_instruction("ja") == language_instruction(DEFAULT_LANGUAGE)
