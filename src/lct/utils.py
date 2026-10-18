"""Utility helpers: config loading, seeding, device, logging."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path.resolve())
    cfg["_config_hash"] = config_hash(cfg)
    return cfg


def config_hash(cfg: dict[str, Any]) -> str:
    blob = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name: str = "cuda") -> torch.device:
    if name.startswith("cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_autocast_dtype(precision: str) -> torch.dtype | None:
    if precision == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if precision in ("fp16", "float16") and torch.cuda.is_available():
        return torch.float16
    return None


def count_parameters(module: torch.nn.Module, non_embedding: bool = False) -> int:
    if not non_embedding:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    total = 0
    for name, p in module.named_parameters():
        if not p.requires_grad:
            continue
        if "embed" in name.lower() or "wte" in name.lower() or "codebook" in name.lower():
            continue
        total += p.numel()
    return total


class CSVLogger:
    """Minimal local CSV logger (wandb-free)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._header_written = self.path.exists() and self.path.stat().st_size > 0

    def log(self, row: dict[str, Any]) -> None:
        import csv

        keys = list(row.keys())
        write_header = not self._header_written
        with self.path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            if write_header:
                w.writeheader()
                self._header_written = True
            w.writerow(row)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)