"""Dev-only utility: run the pytest suite in ONE process with external network denied.

Why: some tests can reach a real DB / Yahoo / other external hosts. An audit hook
(`sys.addaudithook`) rejects `socket.getaddrinfo` (DNS) and `socket.connect`, so such a
test fails loudly instead of silently touching the outside world. `subprocess.Popen`
is only RECORDED (tests really call git etc.).

Usage (from any checkout/worktree; the root defaults to the parent of scripts/):
    python scripts/run_protected_regression.py
    python scripts/run_protected_regression.py --root D:/path/to/worktree
    python scripts/run_protected_regression.py -- tests/test_foo.py -q   # targeted run
    python scripts/run_protected_regression.py --selftest

Limits: an audit hook is per-process and cannot be removed, so this runs single
process -- do NOT pass pytest-xdist `-n`. A full run takes about 4 minutes.
Output: <tmp>/invest-regression/<root folder name>/<UTC timestamp>/{pytest.log,summary.json}
(OUTSIDE the repository on purpose: some tests require pytest tmp_path to be outside
the repo, so an in-repo --out/--basetemp makes them fail). Exit code is pytest's. The tool does not guarantee
read-only DB access; it prevents it indirectly by blocking all non-loopback-pair
connects.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_PYTEST_ARGS = ["tests/", "-q", "--tb=short"]
_LOOPBACK = ("127.0.0.1", "::1")
_POPEN_SAMPLE_MAX = 15
_POPEN_SAMPLE_LEN = 160
_FRAME_DEPTH = 4
_DENY_MESSAGE = "external network denied by regression harness"


def decide(event: str, args: tuple, internal_socketpair: bool) -> str:
    """Pure decision: 'deny' | 'allow_local' | 'record' | 'ignore'.

    `internal_socketpair` = caller frames show socket._fallback_socketpair.
    """
    if event == "socket.connect":
        address = args[1] if len(args) > 1 else None
        host = address[0] if isinstance(address, (tuple, list)) and address else None
        if internal_socketpair and host in _LOOPBACK:
            return "allow_local"
        return "deny"
    if event == "socket.getaddrinfo":
        return "deny"
    if event == "subprocess.Popen":
        return "record"
    return "ignore"


def default_out_dir(root: Path, stamp: str, tmp_base: Path | None = None) -> Path:
    """Default output dir, outside the repo: <tmp>/invest-regression/<root name>/<stamp>."""
    base = tmp_base if tmp_base is not None else Path(tempfile.gettempdir())
    return base / "invest-regression" / root.name / stamp


def is_inside(path: Path, root: Path) -> bool:
    """True when `path` is `root` or lives under it (both resolved)."""
    p, r = path.resolve(), root.resolve()
    return p == r or r in p.parents


def format_popen(args: tuple) -> str:
    """Printable, truncated command line for a subprocess.Popen audit event."""
    cmd = args[1] if len(args) > 1 else ""
    text = " ".join(map(str, cmd)) if isinstance(cmd, (list, tuple)) else str(cmd)
    return text[:_POPEN_SAMPLE_LEN]


def parse_pytest_counts(text: str) -> dict:
    """Parse passed/failed/skipped from pytest's last summary line; never raises."""
    counts: dict = {"passed": None, "failed": None, "skipped": None}
    try:
        for line in reversed(text.strip().splitlines()):
            found = re.findall(r"(\d+) (passed|failed|skipped)", line)
            if found:
                counts = {"passed": 0, "failed": 0, "skipped": 0}
                for num, word in found:
                    counts[word] = int(num)
                break
    except Exception:  # noqa: BLE001 - summary parsing must not kill the tool
        pass
    return counts


def build_summary(
    *,
    exit_code: int,
    elapsed_seconds: float,
    python: str,
    root: str,
    git_head: str | None,
    git_dirty: bool | None,
    pytest_args: list[str],
    log_text: str,
    denied: list[str],
    socketpairs: int,
    popens: list[str],
) -> dict:
    """Assemble summary.json content from collected facts (pure)."""
    summary = {
        "exit_code": int(exit_code),
        "elapsed_seconds": round(elapsed_seconds, 1),
        "python": python,
        "root": root,
        "git_head": git_head,
        "git_dirty": git_dirty,
        "pytest_args": list(pytest_args),
    }
    summary.update(parse_pytest_counts(log_text))
    summary.update(
        {
            "network_denied": {e: denied.count(e) for e in sorted(set(denied))},
            "local_socketpairs": socketpairs,
            "popen_count": len(popens),
            "popen_samples": sorted(set(popens))[:_POPEN_SAMPLE_MAX],
        }
    )
    return summary


class _Collector:
    """Mutable state shared with the installed hook."""

    def __init__(self) -> None:
        self.denied: list[str] = []
        self.popens: list[str] = []
        self.socketpairs = 0


def _in_fallback_socketpair() -> bool:
    # Frame 0 = this helper, 1 = audit hook, 2.. = the code that called connect.
    f = sys._getframe(2)
    for _ in range(_FRAME_DEPTH):
        if f is None:
            break
        if f.f_code.co_name == "_fallback_socketpair" and f.f_globals.get("__name__") == "socket":
            return True
        f = f.f_back
    return False


def install_hook(collector: _Collector) -> None:
    """Install the process-wide, irreversible audit hook (never called by tests)."""

    def audit(event: str, args: tuple) -> None:
        if event not in ("socket.connect", "socket.getaddrinfo", "subprocess.Popen"):
            return
        verdict = decide(event, args, _in_fallback_socketpair() if event == "socket.connect" else False)
        if verdict == "deny":
            collector.denied.append(event)
            raise RuntimeError(_DENY_MESSAGE)
        if verdict == "allow_local":
            collector.socketpairs += 1
        elif verdict == "record":
            collector.popens.append(format_popen(args))

    sys.addaudithook(audit)


def _git(root: Path, *cmd: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *cmd], cwd=root, capture_output=True, text=True, timeout=30, check=True
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def _prepare_env(root: Path) -> None:
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ.update(
        {
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "safe.directory",
            "GIT_CONFIG_VALUE_0": root.as_posix(),
        }
    )
    for name in ("BASH_ENV", "ENV"):
        os.environ.pop(name, None)


def run_selftest() -> int:
    import socket

    collector = _Collector()
    install_hook(collector)
    results = {}

    try:
        socket.getaddrinfo("example.com", 80)
        results["getaddrinfo_denied"] = False
    except RuntimeError:
        results["getaddrinfo_denied"] = True
    try:
        socket.create_connection(("127.0.0.1", 9), timeout=1)
        results["connect_denied"] = False
    except RuntimeError:
        results["connect_denied"] = True
    except OSError:
        results["connect_denied"] = False  # reached the OS => hook did not block
    # create_connection resolves via getaddrinfo first; also hit raw connect directly.
    try:
        raw = socket.socket()
        raw.settimeout(1)
        try:
            raw.connect(("127.0.0.1", 9))
            results["raw_connect_denied"] = False
        finally:
            raw.close()
    except RuntimeError:
        results["raw_connect_denied"] = True
    except OSError:
        results["raw_connect_denied"] = False
    try:
        a, b = socket.socketpair()
        a.close()
        b.close()
        results["socketpair_allowed"] = True
    except Exception:  # noqa: BLE001
        results["socketpair_allowed"] = False

    ok = all(results.values())
    print(
        json.dumps(
            {
                "selftest": "pass" if ok else "fail",
                **results,
                "network_denied": {e: collector.denied.count(e) for e in sorted(set(collector.denied))},
                "local_socketpairs": collector.socketpairs,
            },
            ensure_ascii=False,
        )
    )
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run pytest in one process with external network denied (about 4 min for the full suite).",
        epilog="Single process only: do not pass -n (pytest-xdist). Arguments after -- go to pytest.",
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--out", type=Path, default=None, help="output dir (default <tmp>/invest-regression/<root name>/<UTC ts>)")
    parser.add_argument("--selftest", action="store_true", help="install the hook and verify deny/allow rules")
    parser.add_argument("pytest_args", nargs="*", help="after --: pytest arguments")
    opts = parser.parse_args(argv)

    if opts.selftest:
        return run_selftest()

    root = opts.root.resolve()
    pytest_args = opts.pytest_args or list(DEFAULT_PYTEST_ARGS)
    if any(a == "-n" or a.startswith("-n") and a[2:].isdigit() or a.startswith("--numprocesses") for a in pytest_args):
        parser.error("pytest-xdist (-n) is not allowed: the audit hook is single-process")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = (opts.out or default_out_dir(root, stamp)).resolve()
    if is_inside(out, root):
        print(
            "WARNING: output dir is inside the repository; tests that require tmp_path "
            "outside the repo may fail.",
            file=sys.stderr,
        )
    out.mkdir(parents=True, exist_ok=False)  # refuse an existing folder

    # Git facts before the hook/env change; Popen here is not yet audited.
    git_head = _git(root, "rev-parse", "HEAD")
    status = _git(root, "status", "--porcelain")
    git_dirty = None if status is None else bool(status)

    _prepare_env(root)
    collector = _Collector()
    install_hook(collector)
    sys.path.insert(0, str(root))
    os.chdir(root)
    import pytest

    full_args = [*pytest_args, "-p", "no:cacheprovider", "--basetemp", str(out / "tmp")]
    log_path = out / "pytest.log"
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        code = pytest.main(full_args)
    elapsed = time.monotonic() - started

    summary = build_summary(
        exit_code=int(code),
        elapsed_seconds=elapsed,
        python=sys.executable,
        root=str(root),
        git_head=git_head,
        git_dirty=git_dirty,
        pytest_args=pytest_args,
        log_text=log_path.read_text(encoding="utf-8", errors="replace"),
        denied=collector.denied,
        socketpairs=collector.socketpairs,
        popens=collector.popens,
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
