"""Safety layer of the replay harness (spec C): nothing here may write to a real
system, and the process must stop hard if anything tries.

1. `install_blocks` replaces every side-effecting function (ledger, orders, alerts,
   memory writes, broker/HTTP, DB pool) with a stub that logs a trip and raises
   `ReplaySafetyViolation` — a BaseException, so the production code's
   `except Exception` fail-open handlers cannot swallow it.
2. Our own DB access is a separate READ-ONLY connection opened with the original
   `asyncpg.connect` captured at import time (before any block exists).
3. DB invariants are snapshotted before/after a run and compared.
4. `assert_offhours` refuses to run LLM work during KRX hours.
운영 코드를 못 건드리게 하는 차단 stub + 읽기 전용 DB 연결 + DB 불변식 + 장중 거부.
"""

from __future__ import annotations

import ctypes
import importlib
import json
import os
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime, time as dtime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import asyncpg

from app.clock import KR_TZ, LiveClock

# Captured at import time, BEFORE install_blocks(): the harness's own read-only
# connection must keep working after asyncpg.connect is replaced by a stub.
_ORIG_CONNECT = asyncpg.connect

BLOCK_MARK = "__replay_block__"
_OWN_PREFIX = "app.experiments.replay"


class ReplaySafetyViolation(BaseException):
    """A blocked side effect was attempted. BaseException on purpose (exit code 3)."""


class MarketWindowError(Exception):
    """Refusing to run during KRX hours (exit code 2)."""


# (module, attribute path) — spec C.1. "Class.__init__" style paths patch a class attribute.
BLOCK_TARGETS: tuple[tuple[str, str], ...] = (
    ("app.orchestrator.pipeline", "run_pipeline"),
    ("app.orchestrator.pipeline", "_run_pipeline_impl"),
    ("app.orchestrator.pipeline", "_position_context"),
    ("app.orchestrator.pipeline", "_atr_context"),
    ("app.orchestrator.pipeline", "_risk_cap_context"),
    ("app.orchestrator.pipeline", "_execution_status_context"),
    ("app.orchestrator.pipeline", "_portfolio_risk_context"),
    ("app.orchestrator.pipeline", "_company_identity"),
    ("app.memory.store", "store_pending"),
    ("app.memory.store", "find_recent_pending_id"),
    ("app.memory.store", "append_execution_note"),
    ("app.memory.store", "set_action_label"),
    ("app.memory.store", "get_similar_context"),
    ("app.memory.store", "get_same_day_sibling_context"),
    ("app.memory.store", "resolve_outcome"),
    ("app.memory.store", "flag_bad_entry"),
    ("app.memory.reflection", "reflect"),
    ("app.execution.ledger", "rebalance_to_target"),
    ("app.execution.ledger", "record_trade"),
    ("app.execution.ledger", "force_exit_position"),
    ("app.execution.pending_orders", "create_pending_order"),
    ("app.execution.kis_bridge", "mirror_trade_to_kis"),
    ("app.execution.kis_bridge", "place_manual_order"),
    ("app.live.alerts", "record_alert"),
    ("app.web.state", "kis_broker"),
    ("app.web.state", "add_ticker_to_trade_allowlist"),
    ("app.web.state", "mark_trade_allowlist_auto_added"),
    ("app.brokers.base.http_client", "BaseHttpClient.__init__"),
    ("app.dataflows.kis_provider", "get_kr_risk_signals"),
    ("app.dataflows.kis_provider", "get_last_close"),
    ("app.dataflows.kis_provider", "get_atr_pct"),
    ("yfinance", "download"),
    ("yfinance", "Ticker"),
    ("app.execution.risk_rules", "load"),
    ("app.execution.runtime_settings", "load"),
    ("app.memory.db", "get_pool"),
    ("app.memory.db", "_bootstrap_pool"),
    ("asyncpg", "connect"),
    ("asyncpg", "create_pool"),
)


@dataclass
class BlockReport:
    installed: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)  # `from x import f` copies replaced too
    missing: list[str] = field(default_factory=list)


_TRIPS: list[dict[str, Any]] = []
_trips_path: Path | None = None


def set_trips_path(path: Path | None) -> None:
    global _trips_path
    _trips_path = path


def trips() -> list[dict[str, Any]]:
    return list(_TRIPS)


def trip_count() -> int:
    """Trips so far: the on-disk log (survives a resume) or the in-memory list, whichever is larger."""
    on_disk = 0
    if _trips_path is not None and _trips_path.exists():
        on_disk = sum(1 for line in _trips_path.read_text(encoding="utf-8").splitlines() if line.strip())
    return max(on_disk, len(_TRIPS))


def reset_trips() -> None:
    _TRIPS.clear()


def _record_trip(name: str) -> None:
    entry = {
        "ts": LiveClock().now().astimezone().isoformat(),
        "target": name,
        "stack": [f"{f.filename}:{f.lineno}:{f.name}" for f in traceback.extract_stack(limit=8)[:-2]],
    }
    _TRIPS.append(entry)
    if _trips_path is not None:
        _trips_path.parent.mkdir(parents=True, exist_ok=True)
        with open(_trips_path, "a", encoding="utf-8") as fh:  # sync + fsync: the process is about to stop
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())


def _make_stub(name: str, original: Any) -> Callable[..., Any]:
    def stub(*args: Any, **kwargs: Any) -> Any:
        _record_trip(name)
        raise ReplaySafetyViolation(f"replay harness blocked a side effect: {name}")

    stub.__name__ = getattr(original, "__name__", "stub")
    stub.__qualname__ = getattr(original, "__qualname__", "stub")
    setattr(stub, BLOCK_MARK, True)
    if callable(original) and not isinstance(original, type):
        stub.__wrapped__ = original  # inspect.getsource(stub) still shows the original source
    return stub


def is_block_stub(obj: Any) -> bool:
    return bool(getattr(obj, BLOCK_MARK, False))


def _resolve(module_name: str, attr_path: str) -> tuple[Any, str, Any]:
    """(owner object, attribute name, original value) — owner is a module or a class."""
    module = importlib.import_module(module_name)
    *parents, attr = attr_path.split(".")
    owner: Any = module
    for part in parents:
        owner = getattr(owner, part)
    return owner, attr, getattr(owner, attr)


def install_blocks(
    setattr_fn: Callable[[Any, str, Any], Any] = setattr,
    *,
    targets: tuple[tuple[str, str], ...] = BLOCK_TARGETS,
    strict: bool = True,
) -> BlockReport:
    """Replace each target on its defining module AND on every `app.*` module that
    imported the same object by name (`from x import f` copies). `setattr_fn` is
    injectable so a test can pass `monkeypatch.setattr` and get an automatic undo."""
    report = BlockReport()
    resolved: list[tuple[str, Any, str, Any]] = []
    for module_name, attr_path in targets:
        name = f"{module_name}.{attr_path}"
        try:
            owner, attr, original = _resolve(module_name, attr_path)
        except (ImportError, AttributeError) as exc:
            report.missing.append(f"{name}: {exc}")
            continue
        resolved.append((name, owner, attr, original))
    if strict and report.missing:
        raise RuntimeError(f"cannot install safety blocks, targets missing: {report.missing}")

    for name, owner, attr, original in resolved:
        if is_block_stub(original):
            continue  # already blocked (idempotent)
        stub = _make_stub(name, original)
        setattr_fn(owner, attr, stub)
        report.installed.append(name)
        if isinstance(owner, type):
            continue  # class attributes have no by-name copies
        for mod_name, mod in list(sys.modules.items()):
            if mod is None or mod is owner or not mod_name.startswith("app."):
                continue
            if mod_name.startswith(_OWN_PREFIX):
                continue  # harness modules keep the originals (e.g. `_ORIG_CONNECT`)
            try:
                items = list(vars(mod).items())
            except TypeError:
                continue
            for key, value in items:
                if value is original:
                    setattr_fn(mod, key, stub)
                    report.aliases.append(f"{mod_name}.{key}")
    return report


# --------------------------------------------------------------------------- read-only DB


READONLY_SERVER_SETTINGS = {
    "default_transaction_read_only": "on", "application_name": "replay_downstream_ro",
    "statement_timeout": "120000",  # ms: a hung DB must not block extract/snapshots forever
}


async def open_readonly_connection(
    dsn: str | None = None, *, timeout: float = 30.0, command_timeout: float = 120.0
) -> Any:
    """A dedicated read-only connection (never the app pool, whose first use would run
    the schema DDL). Uses the `asyncpg.connect` captured at import time; both the connect step
    and every command are bounded (`timeout`, `command_timeout`, server `statement_timeout`)."""
    if dsn is None:
        from app.config import settings

        dsn = settings.database_url
    return await _ORIG_CONNECT(
        dsn, timeout=timeout, command_timeout=command_timeout, server_settings=dict(READONLY_SERVER_SETTINGS)
    )


def readonly_transaction(conn: Any) -> Any:
    return conn.transaction(isolation="repeatable_read", readonly=True)


_TABLE_STATS_SQL = {
    "memory_entries": "SELECT max(id) AS max_id, count(*) AS n, max(created_at) AS max_at FROM memory_entries",
    "trades": "SELECT max(id) AS max_id, count(*) AS n, max(created_at) AS max_at FROM trades",
    "pending_orders": "SELECT max(id) AS max_id, count(*) AS n, max(created_at) AS max_at FROM pending_orders",
    "alerts": "SELECT max(id) AS max_id, count(*) AS n, max(triggered_at) AS max_at FROM alerts",
}
_NEW_ROWS_SQL = {
    "memory_entries": (
        "SELECT id, created_at AS at, ticker, NULL::text AS kind, is_backtest, account_mode, decision_trigger "
        "FROM memory_entries WHERE id > $1 ORDER BY id LIMIT 200"
    ),
    "trades": (
        "SELECT id, created_at AS at, ticker, NULL::text AS kind, is_backtest, account_mode, NULL::text AS decision_trigger "
        "FROM trades WHERE id > $1 ORDER BY id LIMIT 200"
    ),
    "pending_orders": (  # no is_backtest column in this table
        "SELECT id, created_at AS at, ticker, NULL::text AS kind, NULL::boolean AS is_backtest, account_mode, "
        "NULL::text AS decision_trigger FROM pending_orders WHERE id > $1 ORDER BY id LIMIT 200"
    ),
    "alerts": (
        "SELECT id, triggered_at AS at, ticker, kind, NULL::boolean AS is_backtest, NULL::text AS account_mode, "
        "NULL::text AS decision_trigger FROM alerts WHERE id > $1 ORDER BY id LIMIT 200"
    ),
}
_CONFIG_DIGEST_SQL = (
    "SELECT md5(string_agg(concat_ws('|', key, value::text, account_mode, user_id, updated_at::text), ',' "
    "ORDER BY key, account_mode, user_id)) AS d FROM {table}"
)
# In-place UPDATE detector for the cases the run reads (resolve_all_due-changed alpha/state
# columns are deliberately excluded).
_CASE_DIGEST_SQL = (
    "SELECT md5(string_agg(id::text||'|'||rating||'|'||coalesce(action_label,'')||'|'||md5(decision_text)"
    "||'|'||md5(full_transcript::text)||'|'||flagged_bad::text, ',' ORDER BY id)) AS d "
    "FROM memory_entries WHERE id = ANY($1)"
)


def _iso(value: Any) -> str | None:
    return None if value is None else value.isoformat()


async def _query(conn: Any, sql: str, *args: Any, many: bool = False) -> Any:
    """One read-only statement in its own transaction; an absent table gives None."""
    try:
        async with readonly_transaction(conn):
            return await (conn.fetch(sql, *args) if many else conn.fetchrow(sql, *args))
    except Exception as exc:  # noqa: BLE001 - only "table does not exist" is tolerated
        if exc.__class__.__name__ == "UndefinedTableError":
            return None
        raise


async def snapshot_invariants(
    conn: Any, case_ids: list[int], *, since: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Counts/max ids per table, config digests, and the case-row digest. With `since`
    (an earlier snapshot) it also lists rows newer than that snapshot's max id."""
    snap: dict[str, Any] = {"taken_at": LiveClock().now().astimezone().isoformat(), "tables": {}, "digests": {}}
    for table, sql in _TABLE_STATS_SQL.items():
        row = await _query(conn, sql)
        snap["tables"][table] = (
            None if row is None else {"max_id": row["max_id"], "count": row["n"], "max_at": _iso(row["max_at"])}
        )
    for table in ("risk_rules", "runtime_settings"):
        row = await _query(conn, _CONFIG_DIGEST_SQL.format(table=table))
        snap["digests"][table] = None if row is None else row["d"]
    row = await _query(conn, _CASE_DIGEST_SQL, list(case_ids))
    snap["digests"]["cases"] = None if row is None else row["d"]
    snap["new_rows"] = {}
    if since is not None:
        for table, sql in _NEW_ROWS_SQL.items():
            before = (since.get("tables") or {}).get(table)
            if not before or before.get("max_id") is None:
                continue
            rows = await _query(conn, sql, before["max_id"], many=True) or []
            snap["new_rows"][table] = [
                {
                    "id": r["id"], "at": _iso(r["at"]), "ticker": r["ticker"], "kind": r["kind"],
                    "is_backtest": r["is_backtest"], "account_mode": r["account_mode"],
                    "decision_trigger": r["decision_trigger"],
                }
                for r in rows
            ]
    return snap


_OVERRIDE_RISK_KEYS = (
    "hysteresis_min_confidence_to_flip", "consensus_override_min_confidence",
    "consensus_override_min_confidence_bearish",
)


async def fetch_relevant_overrides(conn: Any) -> dict[str, Any]:
    """Rows in `runtime_settings` (ai_profile) / `risk_rules` (the gate thresholds) — spec fact 10.
    A `real`-mode row would change the gate/profile values the replay assumes (= code defaults),
    so `run` refuses when one exists. Read-only SELECTs; a missing table counts as no rows."""
    rows: list[dict[str, Any]] = []
    runtime = await _query(
        conn, "SELECT key, account_mode FROM runtime_settings WHERE key = 'ai_profile'", many=True
    )
    for r in runtime or []:
        rows.append({"table": "runtime_settings", "key": r["key"], "account_mode": r["account_mode"]})
    risk = await _query(
        conn, "SELECT key, account_mode FROM risk_rules WHERE key = ANY($1)", list(_OVERRIDE_RISK_KEYS), many=True
    )
    for r in risk or []:
        rows.append({"table": "risk_rules", "key": r["key"], "account_mode": r["account_mode"]})
    return {"rows": rows}


@dataclass
class InvariantResult:
    """`status`: PASS | WARN | FAIL_CANDIDATE (only unconfirmed possibly-live rows) | FAIL."""

    status: str
    fails: list[str] = field(default_factory=list)
    warns: list[str] = field(default_factory=list)
    new_rows: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    candidates: list[dict[str, Any]] = field(default_factory=list)   # need a human confirmation
    auto_live: list[dict[str, Any]] = field(default_factory=list)    # outside every run window, is_backtest=false
    released: list[dict[str, Any]] = field(default_factory=list)     # candidates the human confirmed as live

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status, "fails": self.fails, "warns": self.warns, "new_rows": self.new_rows,
            "candidates": self.candidates, "auto_live": self.auto_live, "released": self.released,
        }


_APP_TABLES = ("memory_entries", "trades", "pending_orders")
Window = tuple[datetime, datetime]


def _parse_ts(value: Any) -> datetime | None:
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def parse_windows(raw: Iterable[Mapping[str, Any]]) -> list[Window]:
    """Manifest `windows` entries ({start, end}) -> aware datetime pairs (an open window ends at start)."""
    out: list[Window] = []
    for item in raw or []:
        start = _parse_ts(item.get("start"))
        end = _parse_ts(item.get("end")) or start
        if start is not None and end is not None:
            out.append((start, max(start, end)))
    return out


def classify_new_row(row: dict[str, Any], windows: Sequence[Window]) -> str:
    """Review M2 rule for a NEW row in memory_entries/trades/pending_orders:
    - `harness_window`: created inside any [start, end] window of a harness process => hard FAIL
      (the harness itself must never write);
    - `auto_live`: created OUTSIDE every window AND `is_backtest` is exactly False => released
      without confirmation (a live loop wrote it);
    - `candidate`: anything else (is_backtest True/unknown, e.g. pending_orders has no such column,
      or an unparsable timestamp) => needs a human confirmation (`--confirm-live-rows`)."""
    created = _parse_ts(row.get("at"))
    if created is not None and any(start <= created <= end for start, end in windows):
        return "harness_window"
    if created is not None and row.get("is_backtest") is False:
        return "auto_live"
    return "candidate"


def compare_invariants(
    before: dict[str, Any], after: dict[str, Any], case_tickers: set[str], *,
    run_windows: Sequence[Window] = (), confirmed: Mapping[str, Iterable[int]] | None = None,
) -> InvariantResult:
    """Spec C.3 verdicts. FAIL: count decrease, case-row digest change, config digest change,
    a new row created inside a harness run window, or a `trade_allowlist_blocked` alert for a case
    ticker. FAIL_CANDIDATE: new rows in memory_entries/trades/pending_orders that are neither
    inside a run window nor provably live (see `classify_new_row`) and not confirmed by the human.
    WARN: other new alert rows and auto-classified live rows (live loops write continuously)."""
    fails: list[str] = []
    warns: list[str] = []
    candidates: list[dict[str, Any]] = []
    auto_live: list[dict[str, Any]] = []
    released: list[dict[str, Any]] = []
    confirmed_sets = {t: {int(i) for i in ids} for t, ids in (confirmed or {}).items()}
    new_rows = after.get("new_rows") or {}
    for table in (*_APP_TABLES, "alerts"):
        b, a = (before["tables"].get(table), after["tables"].get(table))
        if b is None and a is None:
            continue
        if b is None or a is None:
            fails.append(f"{table}: table existence changed")
            continue
        if a["count"] < b["count"]:
            fails.append(f"{table}: row count decreased {b['count']} -> {a['count']}")
        grew = (
            (a["max_id"] or 0) > (b["max_id"] or 0) or a["count"] > b["count"] or (a["max_at"] or "") > (b["max_at"] or "")
        )
        if not grew:
            continue
        rows = new_rows.get(table, [])
        listing = [f"id={r['id']} {r.get('ticker')} {r.get('kind') or ''}".strip() for r in rows]
        if table in _APP_TABLES:
            if not rows:
                fails.append(f"{table}: rows appeared but none could be listed (cannot classify)")
            for r in rows:
                kind = classify_new_row(r, run_windows)
                tagged = {**r, "table": table}
                if kind == "harness_window":
                    fails.append(
                        f"{table}: id={r['id']} {r.get('ticker')} was created INSIDE a harness run window "
                        "(the harness must never write)"
                    )
                elif kind == "auto_live":
                    auto_live.append(tagged)
                elif r["id"] in confirmed_sets.get(table, set()):
                    released.append(tagged)
                else:
                    candidates.append(tagged)
        else:
            blocked = [r for r in rows if r.get("kind") == "trade_allowlist_blocked" and r.get("ticker") in case_tickers]
            if blocked:
                fails.append(f"alerts: trade_allowlist_blocked for a case ticker: {[r['id'] for r in blocked]}")
            warns.append(f"alerts: new rows (live loops write alerts): {listing or 'count/max changed'}")
    if auto_live:
        warns.append(
            f"{len(auto_live)} new live row(s) outside every run window with is_backtest=false "
            f"(auto-classified live): {[(r['table'], r['id']) for r in auto_live]}"
        )
    for name in ("cases", "risk_rules", "runtime_settings"):
        if before["digests"].get(name) != after["digests"].get(name):
            fails.append(f"{name}: digest changed (in-place update?)")
    status = "FAIL" if fails else ("FAIL_CANDIDATE" if candidates else ("WARN" if warns else "PASS"))
    return InvariantResult(
        status=status, fails=fails, warns=warns, new_rows=new_rows,
        candidates=candidates, auto_live=auto_live, released=released,
    )


# --------------------------------------------------------------------------- market window

_WINDOW_START = dtime(8, 30)
_WINDOW_END = dtime(15, 40)


def market_window_open(now: datetime) -> bool:
    """True inside the refused window: weekdays 08:30 <= t < 15:40 KST (holidays are
    treated conservatively as trading days). `now` must be timezone-aware."""
    if now.tzinfo is None:
        raise ValueError("market_window_open needs a timezone-aware datetime")
    kst = now.astimezone(KR_TZ)
    return kst.weekday() < 5 and _WINDOW_START <= kst.time() < _WINDOW_END


def live_now_kst() -> datetime:
    return LiveClock().now(KR_TZ)


def assert_offhours(now: datetime, *, allow_market_hours: bool = False) -> None:
    if allow_market_hours:
        return
    if market_window_open(now):
        raise MarketWindowError(
            f"KRX hours (weekday 08:30-15:40 KST): refusing to run LLM work now ({now.isoformat()}); "
            "use --allow-market-hours to override (recorded in the manifest)"
        )


# --------------------------------------------------------------------------- process state

_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


def keep_awake(on: bool) -> bool:
    """Ask Windows not to sleep while a run is in progress (process-level; no system
    setting is changed). False when unsupported. Lid-close / policy sleep still needs
    the user to keep the PC plugged in and set to never sleep."""
    if sys.platform != "win32":
        return False
    try:
        flags = (_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED) if on else _ES_CONTINUOUS
        return bool(ctypes.windll.kernel32.SetThreadExecutionState(flags))  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - best effort
        return False


def verify_process_state(k: int) -> list[str]:
    """Post-import checks (spec A.2 step 4): limiter width == K, no DB overrides loaded,
    request capture off. Returns problems (empty = OK)."""
    problems: list[str] = []
    from app.config import settings
    from app.execution import risk_rules, runtime_settings
    from app.llm import llm_client

    limiter = getattr(llm_client, "_limiter", None)
    if limiter is None or getattr(limiter, "_max_concurrent", None) != k:
        problems.append(f"llm limiter width {getattr(limiter, '_max_concurrent', None)!r} != K={k}")
    if risk_rules._overrides:
        problems.append("risk_rules._overrides is not empty")
    if runtime_settings._overrides:
        problems.append("runtime_settings._overrides is not empty")
    if settings.rm_request_capture_enabled is not False:
        problems.append("rm_request_capture_enabled is not False")
    return problems
