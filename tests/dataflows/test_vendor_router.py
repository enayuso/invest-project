"""VendorRouter fallback-chain behavior (A-stage, pure, no network)."""

from __future__ import annotations

import pytest

from app.dataflows.router import VendorRouter


async def _ok(**kwargs) -> str:
    return "ok"


async def _fail(**kwargs) -> str:
    raise RuntimeError("boom")


async def test_first_provider_success_short_circuits_rest():
    calls = []

    async def first(**kwargs) -> str:
        calls.append("first")
        return "first-result"

    async def second(**kwargs) -> str:
        calls.append("second")
        return "second-result"

    router = VendorRouter()
    router.register("price", [first, second])
    result = await router.call("price")
    assert result == "first-result"
    assert calls == ["first"]


async def test_falls_through_to_next_provider_on_failure():
    router = VendorRouter()
    router.register("price", [_fail, _ok])
    assert await router.call("price") == "ok"


async def test_core_category_raises_when_all_providers_fail():
    router = VendorRouter()
    router.register("price", [_fail, _fail], optional=False)
    with pytest.raises(RuntimeError, match="exhausted all configured vendors"):
        await router.call("price")


async def test_optional_category_degrades_quietly_when_all_providers_fail():
    router = VendorRouter()
    router.register("macro", [_fail, _fail], optional=True)
    result = await router.call("macro")
    assert result.startswith("[macro data unavailable:")


async def test_unregistered_category_raises_keyerror():
    router = VendorRouter()
    with pytest.raises(KeyError):
        await router.call("nonexistent")
