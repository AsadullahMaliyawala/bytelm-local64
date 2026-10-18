"""Simple predictive baselines for Phase 2B.1a integrity audit."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import torch

LN2 = math.log(2)


def _iter_scored_bytes(loader, max_batches: int = 200) -> list[int]:
    """Flatten content bytes (mask==1) from a loader."""
    out: list[int] = []
    n = 0
    for batch in loader:
        x = batch["bytes"]
        mask = batch.get("byte_mask")
        if mask is None:
            out.extend(int(b) for b in x.view(-1).tolist())
        else:
            for i in range(x.size(0)):
                for j in range(x.size(1)):
                    if int(mask[i, j].item()) == 1:
                        out.append(int(x[i, j].item()))
        n += 1
        if n >= max_batches:
            break
    return out


@torch.no_grad()
def evaluate_baselines_bpb(loader, max_batches: int = 80, train_loader=None) -> dict[str, Any]:
    """Uniform / unigram / bigram / copy / leak probes on the same scored stream."""
    # Build train counts if available
    train_bytes = _iter_scored_bytes(train_loader, max_batches=max_batches) if train_loader else []
    if not train_bytes:
        train_bytes = _iter_scored_bytes(loader, max_batches=max_batches)

    unigram = Counter(train_bytes)
    total_u = sum(unigram.values()) or 1
    # Laplace-smoothed unigram
    def log_p_uni(b: int) -> float:
        return math.log((unigram[b] + 1) / (total_u + 256))

    # Bigram: P(b|prev)
    bigram: dict[int, Counter] = {}
    for i in range(1, len(train_bytes)):
        prev, cur = train_bytes[i - 1], train_bytes[i]
        bigram.setdefault(prev, Counter())[cur] += 1

    def log_p_bi(prev: int, cur: int) -> float:
        c = bigram.get(prev)
        if not c:
            return log_p_uni(cur)
        return math.log((c[cur] + 1) / (sum(c.values()) + 256))

    # Score on val loader with next-byte targets
    uni_nats = bi_nats = copy_nats = leak_nats = uniform_nats = 0.0
    n_scored = 0
    n_batches = 0
    for batch in loader:
        x = batch["bytes"]
        mask = batch.get("byte_mask")
        B, T = x.shape
        for b in range(B):
            for t in range(T - 1):
                # score target x[t+1] if mask[t+1]==1
                if mask is not None and int(mask[b, t + 1].item()) == 0:
                    continue
                tgt = int(x[b, t + 1].item())
                prev = int(x[b, t].item())
                uni_nats += -log_p_uni(tgt)
                bi_nats += -log_p_bi(prev, tgt)
                # copy: predict prev
                copy_nats += 0.0 if prev == tgt else -math.log(1e-12)
                # leak: predict current input as target (should fail when tgt != inp)
                # wrong probe: p(tgt)=1 if tgt==x[t] else tiny — identity on input position
                leak_nats += 0.0 if int(x[b, t].item()) == tgt else -math.log(1e-12)
                uniform_nats += -math.log(1.0 / 256)
                n_scored += 1
        n_batches += 1
        if n_batches >= max_batches:
            break

    def bpb(nats: float) -> float:
        return (nats / max(n_scored, 1)) / LN2

    return {
        "n_scored_bytes": n_scored,
        "uniform_bpb": bpb(uniform_nats),  # expect ~8.0
        "unigram_bpb": bpb(uni_nats),
        "bigram_bpb": bpb(bi_nats),
        "prev_byte_copy_bpb": bpb(copy_nats),
        "identity_leak_bpb": bpb(leak_nats),
        "notes": {
            "uniform": "Should be exactly 8.0 BPB",
            "identity_leak": "Predicts x[t] as target x[t+1]; high BPB = no leak",
            "copy": "Predicts previous byte; high on real content",
        },
    }
