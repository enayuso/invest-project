"""The "data we cannot have" declaration (spec S:4).

Spec design principle 4: the system should machine-declare what it does
not have, rather than let the model quietly infer or hallucinate it. The
09-21 audit of the sibling real-money trading system found its heaviest
confirmed defects were exactly hallucinated VI (variability interruption)
and market-alert claims -- the model had no data for these and filled
the gap anyway. This project has no source for any of the items below,
ever, so the declaration is put in two places: every `resolve_ticker`
response (`data_coverage`) AND every tool's description (so the model
sees the constraint before it ever calls a tool, not just after).
"""

DATA_WE_DO_NOT_HAVE: list[str] = [
    "VI(변동성완화장치) 발동 여부 / 서킷브레이커 상태",
    "시장경보(투자주의/경고/위험) 지정 여부 / 관리종목 지정 여부",
    "외국인·기관 수급, 프로그램 매매 동향",
    "공매도 잔고 및 거래량",
    "실시간 호가창(체결가만 제공, 매수/매도 호가 정보 없음)",
    "국내 뉴스 원문 (필요하면 Claude Desktop 자체 웹검색을 별도로 쓰되, 이 도구가 준 데이터와 명확히 구분할 것)",
]

DATA_WE_DO_NOT_HAVE_NOTICE = (
    "이 도구가 제공하지 않는 데이터(추측·환각 금지): " + " / ".join(DATA_WE_DO_NOT_HAVE)
)
