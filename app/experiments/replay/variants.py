"""Variant registry + fork guards (spec B).

A variant is a process-wide patch applied ONCE before the run (a prompt constant, a setting,
a gate threshold). `fork_at` names the first node the variant may change: earlier nodes are
restored from a baseline run (no LLM calls), later ones are re-run. Two guards keep the
comparison honest:
- nodes BEFORE `fork_at` must have the same fingerprint as in the baseline run (otherwise
  the variant touched something the cache reused), and
- the `fork_at` node's fingerprint must DIFFER (otherwise the variant is a no-op).
변형은 실행 전 1회 전역 패치. fork_at 앞 노드는 캐시 재사용, 뒤 노드만 재실행하고 지문으로 검증한다.

Caveat for real (stage 5) variants: constants assembled at import time are NOT re-read when
patched — patch a name that is read at CALL time; the fingerprint guard catches a missed one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from app.experiments.replay.provenance import NODES, _canon_value, _digest, fingerprint_diff


class VariantRefused(Exception):
    """A variant/fork request the guards refuse (CLI exit code 2)."""


def _describe(value: Any) -> str:
    """Short digest of a patched value (functions/classes by identity-ish text)."""
    if callable(value) and not isinstance(value, type):
        return _digest(f"callable:{getattr(value, '__module__', '?')}.{getattr(value, '__qualname__', repr(value))}")
    return _digest(_canon_value(value))


class VariantContext:
    """Records every patch (target + before/after digest) for the run manifest."""

    def __init__(self, setattr_fn: Callable[[Any, str, Any], Any] = setattr):
        self._setattr = setattr_fn
        self.patches: list[dict[str, str]] = []

    def patch(self, obj: Any, attr: str, value: Any) -> None:
        before = getattr(obj, attr)
        self._setattr(obj, attr, value)
        target = f"{getattr(obj, '__name__', type(obj).__name__)}.{attr}"
        self.patches.append({"target": target, "before": _describe(before), "after": _describe(value)})

    def patch_setting(self, key: str, value: Any) -> None:
        from app.config import settings

        self.patch(settings, key, value)


@dataclass(frozen=True)
class Variant:
    name: str
    fork_at: str
    description: str
    apply: Callable[[VariantContext], None]
    test_only: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


def _pm_marker(ctx: VariantContext) -> None:
    from app.agents import portfolio_manager

    ctx.patch(portfolio_manager, "_FLAT_SUFFIX", portfolio_manager._FLAT_SUFFIX + "\n[replay-test-variant]")


REGISTRY: dict[str, Variant] = {
    "baseline": Variant("baseline", "rm", "no-op (the reference arm)", lambda ctx: None),
    "_test_pm_marker": Variant("_test_pm_marker", "pm", "test only: appends a marker to the PM system prompt", _pm_marker, test_only=True),
}


def get_variant(name: str, *, allow_test_variant: bool = False) -> Variant:
    if name not in REGISTRY:
        raise VariantRefused(f"unknown variant {name!r}; known: {sorted(REGISTRY)}")
    variant = REGISTRY[name]
    if (variant.test_only or name.startswith("_test_")) and not allow_test_variant:
        raise VariantRefused(f"variant {name} is test-only; pass --allow-test-variant")
    if variant.fork_at not in NODES:
        raise VariantRefused(f"variant {name} has invalid fork_at {variant.fork_at!r}")
    return variant


def apply_variant(variant: Variant, setattr_fn: Callable[[Any, str, Any], Any] = setattr) -> VariantContext:
    ctx = VariantContext(setattr_fn)
    variant.apply(ctx)
    return ctx


def validate_fork(
    *,
    variant: Variant,
    base_key: str,
    baseline_manifest: Mapping[str, Any],
    current_fingerprints: Mapping[str, Mapping[str, Any]],
    allow_noop: bool = False,
) -> None:
    """Refuse a fork whose baseline is not comparable (spec B). `base_key` is the key computed
    BEFORE the variant was applied; `current_fingerprints` are computed AFTER."""
    if baseline_manifest.get("base_key") != base_key:
        raise VariantRefused(
            "baseline run base_key differs from the current one (model, pinned settings, source hashes "
            "or case set changed): run a fresh baseline"
        )
    if baseline_manifest.get("status") != "done":
        raise VariantRefused(f"baseline run status is {baseline_manifest.get('status')!r}, not 'done'")
    baseline_fp = baseline_manifest.get("node_fingerprints") or {}
    fork_index = NODES.index(variant.fork_at)
    for node in NODES[:fork_index]:
        if node not in baseline_fp:
            raise VariantRefused(f"baseline manifest has no fingerprint for node {node}")
        if baseline_fp[node]["digest"] != current_fingerprints[node]["digest"]:
            parts = fingerprint_diff(baseline_fp[node], current_fingerprints[node])
            raise VariantRefused(
                f"variant touches node '{node}' which is reused from the baseline cache "
                f"(변형이 캐시 재사용 노드를 건드림): {parts}"
            )
    fork_node = variant.fork_at
    if fork_node in baseline_fp and baseline_fp[fork_node]["digest"] == current_fingerprints[fork_node]["digest"]:
        if not allow_noop:
            raise VariantRefused(
                f"variant {variant.name} does not change the fingerprint of its fork node '{fork_node}' "
                "(no-op variant); pass --allow-noop-variant to run it anyway"
            )
