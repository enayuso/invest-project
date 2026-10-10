# Offline evaluation tool (Phase 3B–3D + OS-isolation launcher)

This directory is a standalone, Python-standard-library-only validator for the
immutable Phase 2 evidence bundle. It does not import `app`, access a database or
network, or execute an AI engine. Phase 3C can load only the hash-pinned
`session_close_guard.py` source, and only after the isolation contract passes.

Run from any directory with the repository virtual environment:

```powershell
D:\Investment_Project\invest-project\.venv\Scripts\python.exe -I -S -B `
  D:\Investment_Project\invest-project\scripts\offline_eval\cli.py validate `
  --bundle D:\Investment_Project\handoff\evaluations\2026-09-22-phase2 `
  --profile D:\Investment_Project\handoff\evaluations\2026-09-22-phase3a\scope-v1.json
```

Every command writes exactly one UTF-8 JSON object to stdout. Exit code `0` means
the bundle is valid, `2` means usage or input-contract failure, `3` means an
unexpected internal failure, and `4` means runtime isolation was not established.

The output contains hashes, counts, safe case classifications, and error codes.
It never returns raw evidence text, excerpts, company-account context, or claim
needles to stdout.

## Phase 3C scan boundary

The `scan` interface accepts `--bundle`, `--profile`, and an absolute `--out`
that is a new direct child of
`D:\Investment_Project\handoff\evaluations\phase3-runs`. It has no `--force`
or arbitrary scanner option. A successful run would create exactly
`findings.json`, `summary.md`, and `run-manifest.json`, with the complete
manifest written last and no overwrite or append behavior. If a failure occurs
after the new output directory is created, `run-incomplete.json` records the
safe phase/error identifier and whether detector execution had begun; a
complete manifest is never written. The stdout error object also reports
`incomplete_marker_status` as `written`, `write_failed`, or `not_attempted`, so
failure to persist that marker is never silent.

This standalone Windows environment currently has no trusted OS-level sandbox
attestation. Therefore material-evidence `scan` is intentionally withheld before
the bundle is read, returns exit code `4` with
`OS_SANDBOX_VERIFICATION_UNIMPLEMENTED`, and does not create the output
directory. The report distinguishes this from
`OS_SANDBOX_VERIFICATION_FAILED`; neither state can be bypassed with a flag or
environment variable. The Python audit hook is tested as defense in depth but
is not represented as a security sandbox. Synthetic guard behavior, including
the full guard-to-quote-mapping path, is covered by the local tests and the
separate-process `tests/isolation_smoke.py`.

The hash-pinned review scorer is available only for a completed run:

```powershell
.venv\Scripts\python.exe -I -S -B scripts\offline_eval\cli.py score `
  --run D:\Investment_Project\handoff\evaluations\phase3-runs\RUN_NAME `
  --review D:\Investment_Project\handoff\evaluations\phase3-runs\RUN_NAME\hit-review-v1.json
```

The review must bind the run manifest, findings, profile, scanner, label source,
input node, and per-case finding hashes. Each item also needs
`label_review_status` (`approved`, `pending`, or `ambiguous`). Only an approved
item may contain a clear `final_label`; changed labels require a
`label_change_rationale` and a new `label_version` distinct from the pinned
source version. Missing, pending, or ambiguous label review stays N/A, so
scanner output alone never becomes TP/FP/FN/TN. Before scoring, the tool
recomputes per-case finding hashes, decodes each raw finding again, and verifies
its family, quote hash, quote length, candidate-range length, candidate sets,
scope statuses, counts, and manifest/findings provenance. Unknown count-map
keys are rejected without reflecting their external text into error output.

Completed artifacts pin the seven offline-evaluation source-file hashes and
carry all validated scope statuses, not only the selected candidates. This lets
a later reviewer confirm the 9 selected plus 33 excluded claims without
re-reading the source bundle.

Run the local test suite without project imports:

```powershell
D:\Investment_Project\invest-project\.venv\Scripts\python.exe -I -S -B `
  D:\Investment_Project\invest-project\scripts\offline_eval\tests\run_tests.py
```

The suite includes `tests/test_review_integration.py`: a separate Python process
runs the pinned guard on nine synthetic cases under the audit hook, writes real
artifacts, and passes them to the review scorer. It also checks fully rehashed
mapping mutations, count-key redaction, label revisions, and failures before and
after scanner startup. It does not load the Phase 2 evidence bundle or prove
OS-level isolation.

The audit-hook smoke test runs separately because Python audit hooks cannot be
removed from a process:

```powershell
.venv\Scripts\python.exe -I -S -B scripts\offline_eval\tests\isolation_smoke.py
```

## OS-isolation launcher (Windows host -> Linux container)

`isolation.py`'s Windows branch always withholds `scan` before the bundle is read
(`OS_SANDBOX_VERIFICATION_UNIMPLEMENTED`). The only supported path to a `confirmed`
status is running `cli.py scan` **inside** the pinned container, launched by
`launcher.py` on the host. Full contract: `handoff/knowledge/os-isolation-launcher-design-2026-09-22.md`.

```powershell
docker desktop  # must already be running; the launcher never starts it
.venv\Scripts\python.exe -I -S -B scripts\offline_eval\launcher.py --run-name RUN_NAME
```

`launcher.py` takes exactly one input, `--run-name`. Everything else -- the image
reference, mounts, network mode, resource limits, environment allowlist -- is a
fixed constant in `container_contract.py`, the single source of truth every other
module (the container's own self-probes, `launcher.py`, `scoring.py`) imports from.

**Threat model**: the code that runs is our own hash-pinned source
(`session_close_guard.py` plus the 9 offline_eval tool files). The container is
configured (`--network none`, `--read-only`, `--cap-drop ALL`, a fixed 5-mount
set, no other binds) to make it harder for a mistake or a dependency path in that
code to reach the network, a credential, the database, or a file outside its 5
fixed mounts. `validate_inspect()` checks that this configuration was actually
requested (network mode, read-only root, capabilities, security-opt, the exact
mount set and RW direction, resource limits, absence of extra binds/volumes/
namespace-sharing) -- it does not prove a kernel-level escape is impossible, and
several specifics (exact `docker inspect` field spellings for `SecurityOpt`/
`Memory`/`NanoCpus`/`Binds`/`Tmpfs`, the mount-source normalization rules, and the
runtime `HOSTNAME`/`HOME` injection) are confirmed against real Docker Desktop
output only at A5 (design doc §12.4), not yet performed. If a guarantee against a
deliberate kernel-exploit escape is ever needed, the design's §8 documents a
Windows Sandbox upgrade path instead.

**Two layers of proof, not one**:
1. Inside the container, `assess_os_isolation()`'s Linux branch runs 10 independent
   self-checks plus one derived check (P1–P11: loopback-only network, a real connect
   attempt to a TEST-NET-1 address that must fail with ENETUNREACH, a read-only root
   filesystem, a write attempt that must fail with EROFS, non-root uid, zero
   capabilities, an exact match of `os.environ` against the pinned base-image
   allowlist plus the one launch nonce, and the pinned Python version) before the
   bundle is read. P10 is not an independent probe -- it is read directly off P9's
   own exact-match result (a passing P9 already implies no host canary leaked in),
   so it is reported as 10 independent checks + 1 value derived from P9, not 11
   independent checks.
2. On the host, `launcher.py` runs the *same pure validator function*
   (`container_contract.validate_inspect`) against the real `docker create`
   result, before ever starting the container -- this is the same check
   applied from a second, independent vantage point (the host's view of what
   was actually configured), not a second, differently-written contract. It
   then writes a `launch-attestation.json` outside the container's own
   writable mount, and the scorer (S7) replays `validate_inspect` a third time
   against the stored attestation subset.

**D2 (2026-09-22, user-approved)**: the scorer (`scoring.py score_review`) requires
both -- a manifest reporting `os_isolation.status == "confirmed"` with every probe
passed, **and** a matching host-written attestation, hash-bound to that manifest and
to the review file (`source.launch_attestation_sha256`). A run produced by any other
path (including a direct Windows `cli.py scan`, which the OS check already blocks)
cannot be scored. This is an integrity binding against mistakes and stale files, not
a cryptographic signature -- see the design doc §11.1 for why, and what would be
needed to go further.

**What this is not**: it does not certify operational accuracy or profitability, it
does not prove the audit hook (still layer 2, still not an OS sandbox) is
unnecessary, and it does not by itself authorize scanning the real 9 evaluation
cases -- that is a separate, later approval (design doc §9, §12.3), gated on a
passing independent re-review of this launcher's own contract.

## Limitations (fixed wording, also carried in `run-manifest.json.limitations`)

- This does not defend against a deliberate kernel exploit escape.
- The launch attestation is an integrity binding, not a cryptographic signature.
- The Python audit hook remains a defense-in-depth layer, not an OS-level sandbox.
- This tool makes no claim about real-evidence accuracy or profitability.
