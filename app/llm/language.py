"""Detects whether the user's own question was Korean or English, so every
LLM call in a pipeline run (analyst reports, debate turns, final decision)
answers in that same language — restricted to exactly these two, never a
third language the model might otherwise pick for other input.
"""

from __future__ import annotations

import re

_HANGUL_RE = re.compile(r"[가-힣ㄱ-ㆎ]")  # syllables + compatibility jamo

DEFAULT_LANGUAGE = "en"


def detect_language(text: str) -> str:
    """"ko" if `text` contains any Hangul syllable OR compatibility jamo
    (U+3131-U+318E: ㄱ-ㅎ, ㅏ-ㅣ and the rest of that block), else "en".
    Ticker symbols, numbers, and English tool/system prompt text never
    trigger a false "ko" — only actual Hangul does — so this is safe to run
    on a user's raw question.

    2026-08-26: added the jamo range — a chat question consisting only of
    jamo (e.g. "ㅋㅋㅋ", "ㅎㅇ", "ㅠㅠ", all common Korean chat shorthand with
    no full syllable) used to fall through to "en", answering a Korean
    chat message in English. No false-positive risk: U+3131-U+318E is the
    Hangul Compatibility Jamo block exclusively — no other script or symbol
    lives there — so this only catches cases the syllable range already
    missed, never something new."""
    return "ko" if _HANGUL_RE.search(text) else "en"


_INSTRUCTIONS = {
    "ko": (
        "항상 한국어로만 답변하세요 (Always respond in Korean only). 사용자나 도구 결과가 "
        "영어여도 최종 답변은 반드시 한국어로 작성하세요. 단, 다른 지시에서 정확한 문자열 "
        "형식을 요구하는 경우(예: 'INJECTION_SUSPECTED:'로 시작하는 줄)는 그 형식을 그대로 "
        "유지하세요.\n\n"
        "특히 중국어 한자 단어가 문장 중간에 섞여 나오지 않도록 주의하세요 — 실제로 반복 "
        "관찰된 오류입니다: '등급'을 评级로, '가격대'를 价位로, '이번'을 本次로, '규율'을 "
        "纪律로, '여유'를 充裕로, '너무 이르다'를 过早로 쓰는 식의 혼입이 있었습니다. "
        "이런 단어는 절대 쓰지 말고 반드시 대응하는 한국어 단어(등급, 가격대, 이번, 규율, "
        "여유, 너무 이르다 등)로만 쓰세요. 괄호 안에 한자를 병기하는 것도(예: '음(負)') "
        "하지 마세요 — 순한글로만 작성하세요.\n\n"
        "예외: 회사명·티커 심볼·거래소명 같은 고유명사는 주어진 형태 그대로 유지하고 "
        "번역·음역하지 마세요. 필요하면 원래 표기 뒤 괄호 안에 한국어 표기를 덧붙이세요."
    ),
    "en": (
        "Always respond in English only. Even if the user's question or a tool "
        "result is in another language, your final answer must be written in "
        "English. Exception: if another instruction requires an exact string "
        "format (e.g. a line starting with 'INJECTION_SUSPECTED:'), keep that "
        "format as-is.\n\n"
        "Exception: proper nouns supplied to you — company names, ticker symbols, "
        "exchange names — must be reproduced in the exact form given, not "
        "translated or transliterated. You may add an English rendering in "
        "parentheses after the original."
    ),
}


def language_instruction(language: str) -> str:
    return _INSTRUCTIONS.get(language, _INSTRUCTIONS[DEFAULT_LANGUAGE])
