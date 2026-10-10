"""Process entrypoint — the container's default process.

Superseded design: this used to be a 24h loop that fired a live trading
decision on a fixed interval with no dashboard and no approval gate. That
undersold what was actually wanted (a continuously-updating live feed,
threshold alerts, and a chat interface — see
invest-pipeline/docs/live-dashboard-v1.html), so this is now a thin
launcher for the FastAPI dashboard app (app.web.app), which itself starts
the minute-level poller (app.live.poller) as a background task in its own
lifespan. The full multi-agent decision pipeline is no longer invoked here
at all — run it explicitly via `python -m app.main` (one-off) or
`python -m app.backtest` (historical replay) instead.

A separate stop-loss check loop (originally proposed as a second loop here)
instead runs inside the FastAPI app's own lifespan (app/web/app.py), right
alongside the minute poller — this file has no event loop of its own to run
a background asyncio task from, since `uvicorn.run(...)` below is a blocking
synchronous call. See `app/web/app.py` for the actual loop.
"""

from __future__ import annotations

import copy
import logging
import logging.config
import socket
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import uvicorn
from uvicorn.config import LOGGING_CONFIG

from app.config import settings

# Log destination + rotation (2026-08-04) — previously nothing in this file
# wrote to a fixed location at all; whoever launched the process chose a
# filename via shell redirect (`nohup ... > some_name.log`), so logs ended
# up scattered across ad-hoc filenames with no size/retention bound (one
# unbroken `>>` target would just grow forever). `logs/daemon/` matches the
# location app.backtest's own per-market logs already use (`logs/domestic/`,
# `logs/overseas/`) — one place for every log this project produces, not
# split between the project root and `logs/`.
_LOG_DIR = Path(__file__).resolve().parent.parent / "logs" / "daemon"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_BACKUP_DAYS = 14


class _DailyDatedFileHandler(logging.Handler):
    """Rotates to a new file at local midnight, keeping the last
    `backup_days` days — but unlike stdlib's TimedRotatingFileHandler,
    TODAY's file is already named `daemon_YY-MM-DD.log` the moment it's
    created, not a bare undated file that only gets a date suffix once
    rolled over. Matches how every log file in this project has always been
    named by hand (daemon_2026-08-04_...). No DST arithmetic needed (unlike
    TimedRotatingFileHandler's own doRollover): this app always runs in
    Asia/Seoul, which has observed no DST since 1988, so a plain
    `date.today()` comparison on each emit is exact."""

    def __init__(self, log_dir: Path, backup_days: int, encoding: str = "utf-8"):
        super().__init__()
        self._log_dir = Path(log_dir)
        self._backup_days = backup_days
        self._encoding = encoding
        self._current_date = date.today()
        self._stream = open(self._path_for(self._current_date), "a", encoding=encoding)
        self._prune()

    def _path_for(self, day: date) -> Path:
        return self._log_dir / f"daemon_{day.strftime('%y-%m-%d')}.log"

    def _prune(self) -> None:
        cutoff = date.today() - timedelta(days=self._backup_days)
        for f in self._log_dir.glob("daemon_*.log"):
            try:
                file_date = datetime.strptime(f.stem.split("_", 1)[1], "%y-%m-%d").date()
            except (IndexError, ValueError):
                continue  # not one of ours (unexpected name) — leave it alone
            if file_date < cutoff:
                f.unlink(missing_ok=True)

    def emit(self, record: logging.LogRecord) -> None:
        today = date.today()
        if today != self._current_date:
            self._stream.close()
            self._current_date = today
            self._stream = open(self._path_for(today), "a", encoding=self._encoding)
            self._prune()
        try:
            self._stream.write(self.format(record) + "\n")
            self._stream.flush()
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        self._stream.close()
        super().close()


def _daily_handler_config(formatter: str) -> dict:
    """One `_DailyDatedFileHandler` spec. Every handler this module needs
    (uvicorn's own "default"/"access" handlers, and this app's own root
    handler below) is instantiated from a spec like this inside the SAME
    dictConfig() call — `logging.config.dictConfig` unconditionally clears
    every currently-registered handler process-wide before creating the
    ones in its config (`_clearExistingHandlers()`, not scoped to a "root"
    key), so a root handler set up separately via `logging.basicConfig`
    gets silently closed out from under the app the moment uvicorn's own
    `uvicorn.run(log_config=...)` call re-runs dictConfig internally
    (confirmed live: `ValueError: I/O operation on closed file` on every
    app log line once uvicorn started). Folding root config into this same
    dict is what avoids that."""
    return {
        "()": _DailyDatedFileHandler,
        "formatter": formatter,
        "log_dir": _LOG_DIR,
        "backup_days": _LOG_BACKUP_DAYS,
    }


# uvicorn's own startup/access logs (`INFO:     GET /api/... 200 OK`) use a
# different formatter than this module's own logger below — no timestamp at
# all, and a colon-padded level prefix instead of a plain level name. The two
# styles end up interleaved in the same log stream (uvicorn's own lines and
# this app's `%(asctime)s %(levelname)s %(message)s` lines), so a reader
# can't tell exactly when an HTTP request landed without cross-referencing
# neighboring app log lines. Reusing uvicorn's own default LOGGING_CONFIG
# (rather than writing one from scratch) and just prepending `%(asctime)s`
# to both its formatters keeps every uvicorn-specific field (client address,
# request line, status code) intact while giving every line in the stream
# the same timestamped, uncolored shape.
_LOG_CONFIG = copy.deepcopy(LOGGING_CONFIG)
_LOG_CONFIG["formatters"]["default"]["fmt"] = "%(asctime)s %(levelname)s %(message)s"
_LOG_CONFIG["formatters"]["default"]["use_colors"] = False
_LOG_CONFIG["formatters"]["access"]["fmt"] = (
    '%(asctime)s %(levelname)s %(client_addr)s - "%(request_line)s" %(status_code)s'
)
_LOG_CONFIG["formatters"]["access"]["use_colors"] = False
# The dashboard polls several endpoints (fills/orderbook/trade allowlist-info/...)
# every few seconds, so access-log lines drowned out the actual signal
# (DB/LLM connection errors, pipeline failures) when grepping a saved log —
# confirmed live 2026-08-03 tracing a DB outage. Access logging carries no
# error information of its own (every request, 200 or 500, logs at INFO
# here); real failures are already surfaced by this project's own
# app-level loggers at WARNING/ERROR, so raising this one logger's floor
# only drops routine noise, not error visibility.
_LOG_CONFIG["loggers"]["uvicorn.access"]["level"] = "WARNING"

# httpx (the KIS REST client's transport, app.brokers.base.http_client) logs
# every outgoing request/response pair at INFO with the full URL and query
# string — for KIS's balance/execution-history calls alone that's dozens of
# lines per pipeline cycle, drowning out this project's own signal the same
# way uvicorn.access did above (confirmed live 2026-08-06: a single day's
# log was dominated by these lines). No error information of its own —
# httpx logs a non-2xx response at the same INFO level as a 200, so raising
# this floor drops volume, not visibility; real HTTP failures already
# surface through this project's own exception handling and logging.
_LOG_CONFIG["loggers"]["httpx"] = {"level": "WARNING", "propagate": True}
_LOG_CONFIG["loggers"]["httpcore"] = {"level": "WARNING", "propagate": True}

# Route uvicorn's own handlers into the same daily-dated log file
# (2026-08-04) instead of stdout.
_LOG_CONFIG["handlers"]["default"] = _daily_handler_config("default")
_LOG_CONFIG["handlers"]["access"] = _daily_handler_config("access")

# This app's own loggers (app.daemon, app.live.*, etc.) aren't uvicorn's —
# they propagate to the root logger, so root needs its own handlers in this
# SAME config dict (see `_daily_handler_config`'s docstring for why "same
# dict" matters, not just "same file path"). Console handler kept alongside
# for an interactive/non-nohup run.
_LOG_CONFIG["handlers"]["app_file"] = _daily_handler_config("default")
_LOG_CONFIG["handlers"]["console"] = {"class": "logging.StreamHandler", "formatter": "default"}
_LOG_CONFIG["root"] = {"level": "INFO", "handlers": ["app_file", "console"]}

# Windows defaults stdout/stderr to the OS console codepage (cp949 on a
# Korean locale) unless told otherwise — fine for a live terminal, but any
# Korean log text or non-ASCII punctuation (e.g. the em dash below) comes
# out mojibake/escaped the moment output is redirected to a file or read by
# a UTF-8-expecting tool. Reconfiguring here (not via a PYTHONUTF8 env var)
# means it's always in effect regardless of how this process gets launched
# (terminal, batch file, task scheduler, a future background service).
# reconfigure() is a no-op on platforms where stdout is already UTF-8.
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

# Configured once here so logging works before `uvicorn.run()` is even
# called (e.g. `_ensure_port_free`'s own log line below); `uvicorn.run(...,
# log_config=_LOG_CONFIG)` re-runs dictConfig with this identical dict when
# it starts, which is harmless — it just recreates the same handlers from
# the same spec rather than closing externally-managed ones.
logging.config.dictConfig(_LOG_CONFIG)
logger = logging.getLogger("app.daemon")


def _ensure_port_free(host: str, port: int) -> None:
    """Refuses to start if another process is already listening on
    host:port — confirmed live (2026-07-24) that without this, a second
    daemon instance runs its ENTIRE lifespan (KIS broker init, the
    real-time websocket subscribe, the minute poller's first KIS calls)
    to completion before uvicorn's own bind attempt fails at the very end
    and it shuts back down. KIS allows only one websocket session per
    appkey, so that second instance's lifespan doesn't just fail
    harmlessly — it collides with whichever instance actually holds the
    port, degrading real-time data and REST throttling for the one the
    user is actually looking at. Checking (and failing fast) before any
    of that runs is the fix; a stale/leftover daemon from an earlier
    session showing up as "why is KIS broken" was the actual incident
    this closes off."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((host, port))
        except OSError as exc:
            raise SystemExit(
                f"refusing to start — another process already appears to be listening on "
                f"{host}:{port} ({exc}). Stop it first (check for a leftover `python -m app.daemon` "
                "process) rather than running two instances against the same KIS account: KIS allows "
                "only one websocket session per appkey, so a second instance degrades the first "
                "instead of failing harmlessly."
            ) from None


def main() -> None:
    _ensure_port_free(settings.dashboard_host, settings.dashboard_port)
    logger.info(
        "starting dashboard — trade_allowlist=%s host=%s port=%s log_dir=%s (daemon_YY-MM-DD.log, %dd rotation)",
        settings.trade_allowlist,
        settings.dashboard_host,
        settings.dashboard_port,
        _LOG_DIR,
        _LOG_BACKUP_DAYS,
    )
    uvicorn.run(
        "app.web.app:app", host=settings.dashboard_host, port=settings.dashboard_port, log_config=_LOG_CONFIG,
        timeout_graceful_shutdown=settings.dashboard_graceful_shutdown_timeout_seconds,
    )


if __name__ == "__main__":
    main()
