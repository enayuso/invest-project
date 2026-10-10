"""Shared double-checked-locking "run this DDL once, then cache forever"
helper — the same ~15-line pattern was independently duplicated across
ledger.py, market_regime.py, portfolio_risk.py, safety_switch.py,
pending_orders.py, and risk_rules.py (the last as `_ensure_schema(conn)`
rather than `_ready_pool()`, since it runs inside an already-open
transaction, but the locking logic is identical).

Reads/writes `_schema_ready`/`_schema_lock` on the CALLING module (passed in
as `module`) rather than closing over its own state, so each module keeps
its own module-level flag/lock — and every existing test monkeypatch
(`monkeypatch.setattr(some_module, "_schema_ready", False)`) keeps working
unchanged; this is a pure de-duplication of the locking logic, not a change
to how any module's schema-readiness state is stored or reset.

Same reasoning for `get_pool`: calls `module.get_pool()` rather than
importing `app.memory.db.get_pool` itself, so `monkeypatch.setattr(some_
module, "get_pool", fake)` (used by safety_switch/pending_orders's own
tests) still reaches the actual call — every module using this helper must
therefore keep its own `from app.memory.db import get_pool` import.
"""

from __future__ import annotations

from types import ModuleType

import asyncpg


async def ready_pool(module: ModuleType, ddl: str) -> asyncpg.Pool:
    """`_ready_pool()`-shaped usage: returns a pool with `ddl` applied at
    least once, executing it on the first call only (across all callers,
    thanks to `module._schema_lock`)."""
    pool = await module.get_pool()
    if module._schema_ready:
        return pool
    async with module._schema_lock:
        if not module._schema_ready:
            async with pool.acquire() as conn:
                await conn.execute(ddl)
            module._schema_ready = True
    return pool


async def ensure_schema(module: ModuleType, conn: asyncpg.Connection, ddl: str) -> None:
    """`_ensure_schema(conn)`-shaped usage: applies `ddl` on an
    already-acquired connection, once."""
    if module._schema_ready:
        return
    async with module._schema_lock:
        if not module._schema_ready:
            await conn.execute(ddl)
            module._schema_ready = True
