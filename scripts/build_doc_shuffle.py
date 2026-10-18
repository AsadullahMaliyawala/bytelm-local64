#!/usr/bin/env python
"""Index real_v2 documents and one shared permutation. No training."""

from __future__ import annotations

import hashlib
import json
import mmap
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "real_v2" / "train.bin"
SPANS = ROOT / "data" / "real_v2" / "doc_spans.npy"
PERM = ROOT / "data" / "real_v2" / "doc_perm_seed0.npy"
META = ROOT / "data" / "real_v2" / "doc_shuffle_seed0.json"
SEED = 0
FINEWEB_END = 2_100_001_754
CODE_END = FINEWEB_END + 600_012_197


def source_at(offset: int) -> str:
    if offset < FINEWEB_END:
        return "fineweb-edu"
    if offset < CODE_END:
        return "codeparrot-clean-train"
    return "open-web-math"


def index_docs(path: Path) -> np.ndarray:
    spans: list[tuple[int, int]] = []
    with path.open("rb") as handle:
        mm = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        pos = 0
        n = len(mm)
        while pos < n:
            sep = mm.find(b"\x1e", pos)
            if sep < 0:
                if pos < n:
                    spans.append((pos, n - pos))
                break
            if sep > pos:
                spans.append((pos, sep - pos))
            pos = sep + 1
        mm.close()
    return np.asarray(spans, dtype=np.int64)


def prefix_share(spans: np.ndarray, perm: np.ndarray, n_bytes: int) -> dict:
    counts = {"fineweb-edu": 0, "codeparrot-clean-train": 0, "open-web-math": 0}
    got = 0
    for doc_i in perm:
        start, length = (int(x) for x in spans[int(doc_i)])
        take = min(length + 1, n_bytes - got)
        counts[source_at(start)] += take
        got += take
        if got >= n_bytes:
            break
    total = sum(counts.values()) or 1
    return {"bytes": got, "counts": counts, "share": {k: v / total for k, v in counts.items()}}


def main() -> None:
    spans = index_docs(TRAIN)
    perm = np.random.default_rng(SEED).permutation(len(spans)).astype(np.int64)
    np.save(SPANS, spans)
    np.save(PERM, perm)
    digest = hashlib.sha256(PERM.read_bytes()).hexdigest()
    meta = {
        "seed": SEED,
        "n_docs": int(len(spans)),
        "train_bytes": int(TRAIN.stat().st_size),
        "perm_sha256": digest,
        "epoch_free": True,
        "share_250m": prefix_share(spans, perm, 250_000_000),
        "share_2_5b": prefix_share(spans, perm, 2_500_000_000),
    }
    META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
