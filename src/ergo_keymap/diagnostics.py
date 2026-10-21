"""Deterministic runtime diagnostics and invariant verification for `ergo_keymap`."""

from __future__ import annotations

import time
from typing import Any

from .optimizer import KeyboardGeometry, Hand

def _domain_probe(seed: int) -> dict:
    g36 = KeyboardGeometry.corne_36()
    g34 = KeyboardGeometry.ferris_34()
    g42 = KeyboardGeometry.corne_42()
    left_count = sum(1 for k in g36.keys if k.hand == Hand.LEFT)
    right_count = sum(1 for k in g36.keys if k.hand == Hand.RIGHT)
    return {
        "corne_36_keys": g36.num_keys,
        "ferris_34_keys": g34.num_keys,
        "corne_42_keys": g42.num_keys,
        "balanced_halves": left_count == right_count == 18,
        "invariants_ok": (g36.num_keys == 36 and g34.num_keys == 34 and g42.num_keys == 42 and left_count == 18),
    }


def run_diagnostic_suite(seed: int = 42) -> dict[str, Any]:
    """Execute deterministic domain probe for `ergo-keymap-optimizer` and return structured metrics."""
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    t0 = time.perf_counter()
    metrics = _domain_probe(seed)
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    return {
        "repository": "ergo-keymap-optimizer",
        "package": "ergo_keymap",
        "seed": seed,
        "elapsed_ms": elapsed_ms,
        "metrics": metrics,
        "status": "ok" if metrics.get("invariants_ok", False) else "failed",
    }


def validate_runtime_invariants(seed: int = 42) -> dict[str, Any]:
    """Validate deterministic reproducibility and domain invariants for `ergo_keymap`."""
    run_a = run_diagnostic_suite(seed=seed)
    run_b = run_diagnostic_suite(seed=seed)
    reproducible = run_a["metrics"] == run_b["metrics"]
    passed = bool(run_a["metrics"].get("invariants_ok", False)) and reproducible
    return {
        "repository": "ergo-keymap-optimizer",
        "package": "ergo_keymap",
        "seed": seed,
        "reproducible": reproducible,
        "passed": passed,
        "metrics": run_a["metrics"],
    }
