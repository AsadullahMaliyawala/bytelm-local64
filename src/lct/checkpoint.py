"""Shared checkpoint helpers for compiled / DDP-wrapped state dicts."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def normalize_state_dict_keys(state: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """Strip leading ``_orig_mod.`` / ``module.`` prefixes from every key.

    Returns ``(normalized_state, n_prefixes_stripped)`` where the count is the
    total number of prefix tokens removed across all keys (a key may strip more
    than one if both wrappers are present).
    """
    out: dict[str, Any] = {}
    stripped = 0
    for key, value in state.items():
        name = str(key)
        while True:
            if name.startswith("_orig_mod."):
                name = name[len("_orig_mod.") :]
                stripped += 1
                continue
            if name.startswith("module."):
                name = name[len("module.") :]
                stripped += 1
                continue
            break
        out[name] = value
    return out, stripped


def load_state_dict_strict(model: nn.Module, state: dict[str, Any], *, label: str = "") -> dict[str, int]:
    """Normalize compile/DDP prefixes, load with ``strict=True``, report counts.

    Prints and returns ``{stripped, missing, unexpected}``. Raises if any
    missing or unexpected keys remain after stripping.
    """
    normalized, stripped = normalize_state_dict_keys(state)
    # Probe with strict=False to report counts, then enforce strict.
    incompat = model.load_state_dict(normalized, strict=False)
    missing = list(incompat.missing_keys)
    unexpected = list(incompat.unexpected_keys)
    report = {
        "label": label or type(model).__name__,
        "stripped_prefixes": stripped,
        "missing": len(missing),
        "unexpected": len(unexpected),
    }
    print({"checkpoint_load": report}, flush=True)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint load mismatch for {report['label']}: "
            f"missing={missing[:12]} unexpected={unexpected[:12]}"
        )
    # Re-load strict=True so callers get the documented contract.
    model.load_state_dict(normalized, strict=True)
    return {"stripped": stripped, "missing": 0, "unexpected": 0}
