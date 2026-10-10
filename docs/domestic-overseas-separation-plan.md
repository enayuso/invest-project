# 국내/해외 분리 계획

> **상태(2026-10-02 문서 정리 메모)**: 아래 "착수 전·코드 미작성"은 **작성 당시(2026-07)** 표현이다. 이후 `app/audit`(`--market`)·`app/memory/store.py`·`app/backtest/portfolio_harness.py` 등이 이 문서를 근거로 인용하고 있어 일부 또는 전부 구현됐을 수 있으나, **어디까지 구현됐는지는 이번에 확인하지 않았다(미확인)**. 코드 주석이 이 경로를 가리키므로 파일은 이동하지 않았다.

계좌잔고/리스크캡/일일손실/후보발굴은 이미 국내·해외가 완전히 분리되어 있다(확인 완료).
남은 3곳 — 워치리스트, Knowledge Base 유사상황 검색, 백테스트/감사 통계 — 을 분리하기
위한 계획이다. 아직 착수 전이며, 이 문서는 계획만 담는다(코드 미작성).

## 우선순위 — 작업량 기준

| 순위 | 항목 | 작업량 | 이유 |
|---|---|---|---|
| 1 | Knowledge Base 유사상황 검색 | 작음 | 함수 하나(`get_similar_context`)에 필터 조건 추가 |
| 2 | 백테스트/감사 통계 | 중간 | 몇 개 함수(`portfolio_harness`, `audit/stats.py`, `experiments.py`)에 시장 필터 옵션 추가 |
| 3 | 워치리스트 자체 분리 | 큼 | `settings.watchlist`를 참조하는 곳이 app/ 전역 ~29개 파일 규모(기존 trade_allowlist 리네이밍 작업 때도 같은 규모로 확인된 바 있음) |

## 1. Knowledge Base 유사상황 검색 (`app/memory/store.py::get_similar_context`) — 완료 (2026-07-27)

**이전**: `cross_rows` 쿼리가 `WHERE ticker != $1 AND status='resolved' AND NOT flagged_bad`만 걸고
시장 조건이 없어서, 코사인 유사도만으로 국내/해외 종목 간 사례가 서로 딸려 들어올 수 있었음.

**결정 및 구현**: "같은 시장 우선 + 보조 허용"으로 확정. `cross_rows`를
`cross_rows_same_market`(같은 시장, `n_cross`만큼 우선 조회)과 `cross_rows_other_market`
(같은 시장 결과가 `n_cross`에 못 미칠 때만 부족분을 다른 시장에서 보충)으로 분리. 프롬프트에
들어가는 텍스트도 "(same market)"/"(different market — shown only because too few
same-market matches were found)"로 구분 표시해서, 다른 시장 사례가 섞였을 때 그게 보이도록 함.
테스트 3개 추가(`tests/memory/test_memory_store.py`), 관련 스위트 69개 통과(무관 기존 실패
5개는 로컬 Postgres 미연결 이슈).

## 2. 백테스트/감사 통계 (`app/backtest/portfolio_harness.py`, `app/audit/stats.py`) — 완료 (2026-07-27)

**이전**:
- `portfolio_harness`를 `--tickers` 없이 실행하면 `settings.watchlist`(국내+해외 혼합)를
  통째로 돌려서, 자산곡선/승률/평균알파가 하나의 pooled 숫자로 나왔음.
- `app.audit`의 등급별 승률 조회(`compute_hit_rate_by_rating`)도 `ticker` 인자 없이 호출하면
  전체(국내+해외) 통틀어 계산했음.

**구현**:
- `compute_hit_rate_by_rating(ticker=None, market=None)`에 `market`("domestic"/"overseas"/None)
  파라미터 추가 — SQL WHERE 조건을 동적으로 조립하도록 리팩터링(기존 `ticker`分岐 두 갈래를
  하나로 통합). `render()`에도 `market` 라벨 추가("all domestic tickers" 등으로 표시).
  `python -m app.audit report --market domestic`/`--market overseas`로 사용, `--ticker`와는
  상호배타(둘 다 주면 에러).
- `portfolio_harness`에 `--market domestic|overseas|all`(기본 `all`, 기존 동작 그대로 유지)
  추가 — `--tickers`를 명시 안 하면 워치리스트를 그 시장만으로 좁혀서 돌림. 순수 함수
  `_select_cli_tickers(tickers_arg, market, watchlist)`로 분리해서 테스트 용이하게 함.
- `backtest_runs` 테이블 스키마 변경은 하지 않음 — CLI 단계에서 티커 목록 자체를 시장별로
  좁혀서 실행하는 것으로 충분하다고 판단(계획대로 "사후 필터링만으로 될 수도 있음" 확인됨).

테스트 신규 7개(`tests/audit/test_audit_stats.py` 3개, `tests/backtest/test_portfolio_harness_helpers.py`
4개), 관련 스위트(`tests/audit tests/backtest tests/execution`) 289개 통과.

## 3. 워치리스트 자체 분리 (`app/config.py` 등 app/ 전역) — 완료 (2026-07-28)

**구현**: 계획대로 `trade_allowlist` 실제 필드 리네이밍과 한 번에 진행. `settings.watchlist`
(단일 리스트) 대신 `trade_allowlist_domestic`/`trade_allowlist_overseas` 두 실제 필드
(env `TRADE_ALLOWLIST_DOMESTIC`/`TRADE_ALLOWLIST_OVERSEAS`)로 분리, `ticker_market()`과
동일한 `.KS` 접미사 규칙 사용. `trade_allowlist_for_market(market)` 헬퍼로 기존
`_split_watchlist()`류 수동 분리 로직을 대체. `watchlist` 별칭 프로퍼티는 전환 기간 동안만
유지하다가 모든 호출부(app/ 전역 ~20개 파일) 이관 완료 후 완전히 삭제. 기존 Postgres
`watchlist_state`의 단일 "watchlist" 행은 `ticker_market()`으로 자동 분리해 이관(데이터 유지),
`.env.local`/`.env.dev`/`.env.alice`/`.env.bob`의 `WATCHLIST`도 동일하게 이전.
`favorites`는 계획대로 분리하지 않음(market-scoped 로직에서 쓰이지 않아 단일 리스트 유지).
프론트엔드(워치리스트 국내/해외 구분 표시, favorites UI)는 이번 범위에서 의도적으로 제외.

**현재**: `settings.watchlist: list[str]`(단일 리스트, `.env`의 `WATCHLIST`)에 국내+해외
티커가 같이 들어있고, 쓰는 곳마다 `_split_watchlist()`류 함수로 그때그때 나눠서 처리.

**계획** (기존 `trade_allowlist` 리네이밍 작업과 같은 규모의 리팩터링으로 예상):
1. `settings.watchlist_domestic`/`settings.watchlist_overseas` 두 필드 신규(또는
   `trade_allowlist` 자체를 이 시점에 실제 필드로 승격시키면서 같이 분리 — 이미 남아있는
   "trade_allowlist 실제 필드 리네이밍" 후속 작업과 자연스럽게 합칠 수 있음).
2. `.env`의 `WATCHLIST` 하나를 `WATCHLIST_DOMESTIC`/`WATCHLIST_OVERSEAS` 둘로 이전(하위호환
   고려 필요 — 기존 `WATCHLIST` 값을 어떻게 마이그레이션할지).
3. `_split_watchlist()`를 호출하는 곳들(약 29개 파일 추정)을 하나씩 확인해서, 이제 굳이
   런타임에 나눌 필요 없이 각 필드를 직접 참조하도록 정리.
4. `favorites`/`add_ticker_to_watchlist`/`add_ticker_to_favorites` 등 관련 함수들도 시장
   인자를 받도록 조정.
5. 대시보드 UI(관심종목/워치리스트 추가 화면)도 국내/해외 탭 또는 구분된 입력으로 조정
   필요할 수 있음.

**리스크**: 가장 큰 작업이라 회귀 범위가 넓음 — 기존 `trade_allowlist` 리네이밍 작업(백엔드
1단계만 완료, 실제 리네이밍은 미착수)과 순서를 맞춰서 한 번에 처리하는 게 두 번 손대는 것보다
효율적일 수 있음.

## 결정 사항 (2026-07-27, 사용자 확정)

1. 순서: 1 → 2 → 3 (1번 완료, 다음은 2번)
2. Knowledge Base 필터: "같은 시장 우선 + 보조 허용" — 구현 완료(1절 참조)
3. 워치리스트 분리: 지금 하지 않고, 기존에 미뤄뒀던 `trade_allowlist` 리네이밍 작업과
   묶어서 나중에 한 번에 진행하기로 함
