"""Offline downstream-replay harness (stage 2 of the fast improvement loop).

Re-runs only the downstream nodes (Research Manager -> Trader -> Risk debate ->
Portfolio Manager -> gates) on stored analyst reports + stored Bull/Bear debate,
so a prompt/gate change can be measured in hours instead of weeks. Spec:
`handoff/knowledge/fast-eval-stage2-replay-design-2026-09-30.md`.

FREEZE / HANDOFF NOTES (RULES sections 1, 2, 5)
- Behaviour-change freeze (baseline `d260d27`, RULES section 2 item 4): this package
  and `app/experiments/replay_downstream.py` are NEW files under `app/experiments/`
  plus tests; NO production module is modified. Blocks/variants are monkeypatches that
  live inside the harness process only. Output goes to `results/replay_downstream/`
  (gitignored) because it holds full transcripts: never commit it.
- The harness never calls `run_pipeline()`, never writes to the ledger/memory/alerts,
  and never opens the app DB pool (see `safety.py`). Runs (`pilot`/`run`) are refused
  during KRX hours; they need the human to run them (RULES section 1).
- Pinned source hashes (`provenance.PINNED_SOURCE_HASHES`) intentionally turn the
  suite red when a production file it copies from changes: re-check `runner.py`
  against the original, then refresh the pin (stage 7 deploy bundle).
- Update `handoff/CURRENT.md` (tool name in the entry) when a run is started.

PRE-REGISTERED A/A PASS LINES (spec section F; stage-1 section 6 + three approved
supplements). Fixed BEFORE any A/A run; do not edit after seeing results.
bull(x) = x in {"Buy","Overweight"}. Population C = cases with R1 and R2 both ok
and a parsable stored result. d(A,B) = mean[bull(A_i) != bull(B_i)].

| ID | 지표 | 합격선 |
|---|---|---|
| AA-1a | DB 불변식 R1 시작 → R2 종료 | FAIL 0(WARN은 목록 제시) |
| AA-1b | safety trips | 0 |
| AA-1c | 완료율(최대 2회 시도 후 ok ÷ case set), R1·R2 각각 | ≥ 98%(첫 시도 성공률도 보고) |
| AA-1d | structured 호출 중 `fallback_used` 비율 | **≤ 3% (보완 1: 라이브 기준값이 저장돼 있지 않아 절대 상한으로 대체. 서버 로그로 fallback 경고를 집계하면 "2배 이내"로 복귀)** |
| AA-1e | 전체 호출 중 `finish_reason=="length"` 비율 | ≤ 2 × 1.4% = 2.8%(1단계 `/metrics`) |
| AA-1f | RM 프롬프트 sha256 R1 == R2 | 100% **(보완 2: 새 무결성 검사. 입력이 같으면 RM 요청 바이트가 같아야 함)** |
| AA-1g | 동결 시기 `desk_note_stored_match` | 100%(지금 179/179) |
| AA-2a | d(R1,R2) RM | ≤ 8% |
| AA-2b | d(R1,R2) final(게이트 후) | ≤ 3% |
| AA-2c | d(R1,R2) Neutral final, Trader, PM pre-gate | 정보용 |
| AA-3a | d(R1,stored) RM − d(R1,R2) RM | ≤ +5%p |
| AA-3b | \\|bullrate_R1 − bullrate_stored\\| RM | ≤ 5%p |
| AA-3c | McNemar(R1 vs stored, RM) | p ≥ 0.05 |
| AA-3d | \\|d(R1,stored)_RM,A − d(R1,stored)_RM,B\\|, A = interim True, **B = interim False ∪ None** | < 10%p, 아니면 `period_note` 복원 필요 → 사용자 판단(KIS 재조회 금지 결정과 충돌) **(보완 3: None = 캐시 적중이라 라이브도 `period_note=""` → B는 라이브 입력과 정확히 같은 대조군. 3분할도 함께 보고)** |
| 정보 | AA-3a~c를 final, R2 vs stored에도 적용 | 판정에 안 씀 |

- 판정 순서: AA-1 실패 → No-go. AA-3 실패 → 입력 계약 결함 조사. **AA-2만 실패 → arm당 3회(repeat 1..3) 설계로 전환**(사례별 강세 비율을 연속 점수로). 1회 실행 결과로 알파 채점 금지.
- 검정력 주의: B 그룹 n=39, A 140. 1건이 2.6%p(B)라 AA-3d는 2~3건으로도 발동할 수 있다. Fisher 정확 p를 함께 보고하되 판정은 사전 등록 규칙을 따른다.
- AA-1a clarification (09-30 review; the pre-registered row above is unchanged, and nothing is loosened for
  rows created inside a harness run window): new rows in memory_entries/trades/pending_orders are FAIL when created
  inside any [start, end] window of an R1/R2 harness process (the harness must never write). Rows created OUTSIDE every
  window with is_backtest=false are auto-classified live (listed as WARN). Any other new row is a FAIL CANDIDATE that
  stays a FAIL until the human verifies it is a live row (`aa-report --confirm-live-rows FILE`, file and ids recorded).
  하네스 실행 창 밖의 라이브 행만 해제 가능, 창 안의 행은 무조건 FAIL.
- McNemar exact binomial: b = #(R1 bull and stored not bull), c = #(stored bull and R1 not bull),
  p = min(1, 2 * sum_{i=0}^{min(b,c)} C(b+c,i) 0.5^(b+c)); b+c=0 gives p=1.

Modules: `pinned` (stdlib only) -> `contract` -> `safety` -> `provenance` -> `runner` ->
`variants` -> `aa_report`; the CLI is `app.experiments.replay_downstream`.
"""

# Version of the harness code layout — recorded in manifests, NOT part of base_key.
HARNESS_VERSION = "replay-harness-1"
