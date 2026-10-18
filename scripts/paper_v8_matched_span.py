#!/usr/bin/env python3
"""350M matched-span slide: BPE min(512 tok, 2048 B); local64 same left-context bytes."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts")]

import paper_v4_recompute as v4  # noqa: E402
from lct.checkpoint import load_state_dict_strict  # noqa: E402
from lct.models.continuous_pool4 import ContinuousPool4LM  # noqa: E402
from lct.train.ladder import bpb_from_nats  # noqa: E402
from matched_bpe_load import load_matched_bpe  # noqa: E402

OUT = Path("./experiments/paper_v5")
RESEARCH = ROOT / "research"
VAL = Path("./data/real_v3")
FILES = {"text": "val_text.bin", "code": "val_code.bin", "math": "val_math.bin"}
DOMAINS = ("text", "code", "math")
STRIDE = 512
MAX_TOK = 512
MAX_BYTES = 2048

CFG = {
    "d_model": 1024,
    "n_heads": 16,
    "n_layers": 26,
    "ffn_mult": 4,
    "dropout": 0.0,
    "pool_factor": 4,
    "vocab_size": 256,
    "max_len": 2048,
    "pos_type": "rope",
    "local_window": 64,
}


@torch.no_grad()
def score_bpe_windows(model, tok, doc: bytes, device):
    """Return nats, bytes, prefix, and per-window records with identical byte spans for local."""
    ids, spans = v4.token_spans(tok, doc)
    if len(ids) < 2:
        return 0.0, 0, len(doc), []
    weights = [spans[i][1] - spans[i][0] for i in range(len(ids))]
    nats = 0.0
    nbytes = 0
    next_tok = 1
    n = len(ids)
    windows = []
    while next_tok < n:
        end_tok = next_tok
        new_bytes = 0
        while end_tok < n:
            if end_tok > next_tok and new_bytes >= STRIDE:
                break
            end_tok += 1
            new_bytes += weights[end_tok - 1]
            if new_bytes >= STRIDE:
                break
        last_byte = spans[end_tok - 1][1]
        ctx = next_tok
        while ctx > 0:
            byte_span = last_byte - spans[ctx - 1][0]
            tok_span = end_tok - (ctx - 1)
            if byte_span > MAX_BYTES or tok_span > MAX_TOK:
                break
            ctx -= 1
        win_ids = ids[ctx:end_tok]
        if len(win_ids) < 2 and ctx > 0:
            ctx -= 1
            win_ids = ids[ctx:end_tok]
        if len(win_ids) < 2:
            next_tok = end_tok
            continue
        start_byte = spans[ctx][0]
        # bytes scored in this window: tokens next_tok .. end_tok-1
        score_lo = spans[next_tok][0]
        score_hi = last_byte
        tokens = torch.tensor([win_ids], dtype=torch.long, device=device)
        inp, tgt = tokens[:, :-1], tokens[:, 1:]
        logits = model(inp)["logits"].float()
        nll = -torch.log_softmax(logits, dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[0]
        first_keep = next_tok - (ctx + 1)
        win_nats = 0.0
        win_bytes = 0
        for k in range(first_keep, int(tgt.size(1))):
            tok_i = ctx + 1 + k
            if ids[tok_i] == 0 or weights[tok_i] <= 0:
                continue
            win_nats += float(nll[k].item())
            win_bytes += int(weights[tok_i])
        nats += win_nats
        nbytes += win_bytes
        windows.append(
            {
                "start_byte": int(start_byte),
                "score_lo": int(score_lo),
                "score_hi": int(score_hi),
                "ctx_bytes": int(score_hi - start_byte),
                "ctx_tokens": int(end_tok - ctx),
                "nats_b": win_nats,
                "bytes": win_bytes,
            }
        )
        next_tok = end_tok
    return nats, nbytes, int(weights[0]), windows


@torch.no_grad()
def score_local_matched(model, doc: bytes, windows: list[dict], device):
    """Score the same byte ranges with left context starting at each window's start_byte."""
    nats = 0.0
    nbytes = 0
    for w in windows:
        start = w["start_byte"]
        lo = w["score_lo"]
        hi = w["score_hi"]
        if hi <= lo:
            continue
        # Full causal window: [start, hi); score [lo, hi) relative to start
        window = doc[start:hi]
        if len(window) < 1:
            continue
        if len(window) > 2048:
            raise RuntimeError(f"local window {len(window)} > 2048")
        rel_lo = lo - start
        rel_hi = hi - start
        vec = v4.local_nll_vector(model, window, device)
        chunk = float(vec[rel_lo:rel_hi].sum().item())
        nats += chunk
        nbytes += hi - lo
        w["nats_l"] = chunk
    return nats, nbytes


def load_local(path: Path, device):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model = ContinuousPool4LM(CFG)
    load_state_dict_strict(model, blob["model"], label=path.name)
    model.to(device).eval()
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    RESEARCH.mkdir(parents=True, exist_ok=True)
    loc = load_local(Path("./checkpoints/matched350m_local64/matched350m_local64.pt"), device)
    bpe_m, bpe, _, _ = load_matched_bpe(Path("./checkpoints/matched350m_bpe/matched350m_bpe.pt"), device)
    tok = bpe.tok

    size_rows = {}
    ctx_stats = {}
    for name, fname in FILES.items():
        rows = []
        ctx_b = []
        raw = (VAL / fname).read_bytes()
        for i, doc in enumerate(v4.docs_of(raw)):
            nats_b, bytes_b, prefix, wins = score_bpe_windows(bpe_m, tok, doc, device)
            nats_l, bytes_l = score_local_matched(loc, doc, wins, device)
            if bytes_l != bytes_b:
                raise RuntimeError(f"byte mismatch {name} {i}: {bytes_l} vs {bytes_b}")
            for w in wins:
                ctx_b.append(w["ctx_bytes"])
            rows.append({"nats_l": nats_l, "bytes_l": bytes_l, "nats_b": nats_b, "bytes_b": bytes_b})
            if (i + 1) % 200 == 0:
                print({"domain": name, "doc": i + 1}, flush=True)
        size_rows[name] = rows
        a = np.array(ctx_b, dtype=np.float64) if ctx_b else np.array([0.0])
        ctx_stats[name] = {
            "n_windows": len(ctx_b),
            "mean_ctx_bytes_both": float(a.mean()),
            "p50": float(np.median(a)),
            "max": float(a.max()) if len(a) else 0,
            "note": "identical for BPE and local64 by construction",
        }
        print({"domain": name, "docs": len(rows), **ctx_stats[name]}, flush=True)

    stats = {name: v4.bootstrap_delta(size_rows[name]) for name in DOMAINS}
    stats["overall"] = v4.bootstrap_delta([r for name in DOMAINS for r in size_rows[name]])
    # absolute BPB already in stats
    out = {
        "protocol": "matched_span_min512tok_2048B",
        "bpe_rule": "min(512 tokens, 2048 B)",
        "local_rule": "same left-context start byte as each BPE window",
        "paired": stats,
        "mean_effective_ctx_bytes": ctx_stats,
        "point_bpb": {
            name: {
                "bpb_local64": stats[name]["bpb_local64"],
                "bpb_bpe": stats[name]["bpb_bpe"],
                "diff": stats[name]["diff_bpe_minus_local64"],
                "diff_lo": stats[name]["diff_lo"],
                "diff_hi": stats[name]["diff_hi"],
                "sig": stats[name]["significant_0_outside_ci"],
            }
            for name in list(DOMAINS) + ["overall"]
        },
    }
    (OUT / "slide_350m_matched_span.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (RESEARCH / "slide_350m_matched_span.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out["point_bpb"], indent=2), flush=True)
    print(json.dumps(ctx_stats, indent=2), flush=True)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
