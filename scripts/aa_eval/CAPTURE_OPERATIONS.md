# RM capture offline tools

These scripts are measurement tools. They do not import `app.llm` or run the
trading daemon. The default recorder setting remains OFF.

## Read-only session summary

Keep `status.py` and `validate.py` together in a separate offline tool folder.
Run `status.py <capture-directory>` with the project's virtualenv Python. It
prints only session IDs, states, counts, and reason codes. Exit 0 means status
files could be summarized; it does **not** certify event completeness. Exit 2
means absent, malformed, or excessive status files. Missing status does not
prove zero eligible calls. Use `validate.py` separately for event-level
integrity. Do not copy raw capture files into source control or logs.

## Synthetic acceptance probes

`capture_benchmark.py --output <new-outside-repository-directory>` uses only
synthetic request and response objects. Defaults are 10 warmups and 100 measured
iterations per case, alternating OFF/ON. Each of ten cases runs in a fresh
process so writer threads and peak memory do not carry into the next case.
It writes raw *synthetic* events plus `summary.json`; results include request
and response UTF-8 sizes, per-observer p50/p95/max, input/idle/final RSS and
peak RSS, and a bounded queue saturation case. Cases cover ASCII, Korean,
near-limit request and response, actual SDK top/choice/message extra fields,
and each oversize rejection.
The rejected cases intentionally leave incomplete sessions. The probe does not
measure a server, network latency, storage faults, or the live system. A small
smoke run is a harness check, not performance acceptance.

`capture_lifecycle_probe.py --output <new-outside-repository-directory>` runs
only on Linux. It starts a minimal synthetic FastAPI lifespan in child
processes, sends SIGTERM and SIGKILL, and checks raw event files, hashes,
session linkage, validator completeness, and queued data/control at termination.
Uvicorn listens on a private Unix socket; the probe does not start the
production app, Docker, or a trading connection. Docker stop timeout still requires
the separate offline Linux acceptance review in the 4D plan.

## Deployment boundary

`pipeline.sh` checks the ON daemon configuration before stopping an existing
container. ON requires an existing private root-owned capture directory under
`/var/lib/invest-eval/rm-capture`, with a dedicated filesystem mounted at that
root, total capacity at most 1 GiB and at least 600 MiB free. Human operators
must verify the real server path, writer identity, mount, ownership, retention,
and accepted quota policy before enabling collection. This repository change
does not deploy, restart, activate capture, or enforce the 14-day deletion.
