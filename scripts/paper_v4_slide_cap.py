#!/usr/bin/env python3
"""Diagnose 350M BPE sliding code BPB; rerun with trained-length context cap."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts")]

import paper_v3_recompute as v3  # noqa: E402
import paper_v4_recompute as v4  # noqa: E402
from lct.train.ladder import bpb_from_nats  # noqa: E402
from matched_bpe_load import load_matched_bpe  # noqa: E402

OUT = Path("${LCT_DATA}/experiments/paper_v4")
RESEARCH = ROOT / "research"
VAL = Path("${LCT_DATA}/data/real_v3")
FILES = {"text": "val_text.bin", "code": "val_code.bin", "math": "val_math.bin"}
DOMAINS = ("text", "code", "math")
SEP = 0x1E
STRIDE = 512
# Trained lengths
TRAIN_TOK_350 = 512
TRAIN_TOK_120 = None  # measured as mean tokens / window; cap at max observed train pack = window tokens
CTX_BYTES_FULL = 2048


@torch.no_grad()
def score_bpe_doc_capped(model, tok, doc: bytes, device, max_ctx_tokens: int, max_ctx_bytes: int):
    ids, spans = v4.token_spans(tok, doc)
    if len(ids) < 2:
        return 0.0, 0, len(doc), {"max_win_tokens": 0, "n_windows": 0}
    weights = [spans[i][1] - spans[i][0] for i in range(len(ids))]
    nats = 0.0
    nbytes = 0
    next_tok = 1
    n = len(ids)
    max_win = 0
    n_win = 0
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
            if byte_span > max_ctx_bytes or tok_span > max_ctx_tokens:
                break
            ctx -= 1
        win_ids = ids[ctx:end_tok]
        if len(win_ids) < 2 and ctx > 0:
            ctx -= 1
            win_ids = ids[ctx:end_tok]
        if len(win_ids) < 2:
            next_tok = end_tok
            continue
        max_win = max(max_win, len(win_ids))
        n_win += 1
        tokens = torch.tensor([win_ids], dtype=torch.long, device=device)
        inp, tgt = tokens[:, :-1], tokens[:, 1:]
        logits = model(inp)["logits"].float()
        nll = -torch.log_softmax(logits, dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[0]
        first_keep = next_tok - (ctx + 1)
        for k in range(first_keep, int(tgt.size(1))):
            tok_i = ctx + 1 + k
            if ids[tok_i] == 0 or weights[tok_i] <= 0:
                continue
            nats += float(nll[k].item())
            nbytes += int(weights[tok_i])
        next_tok = end_tok
    return nats, nbytes, int(weights[0]), {"max_win_tokens": max_win, "n_windows": n_win}


def score_local_capped(model, doc: bytes, prefix: int, device, max_ctx_bytes: int):
    """Same byte span as BPE: context capped to max_ctx_bytes."""
    if prefix >= len(doc):
        return 0.0, 0
    # Temporarily override CTX in score_local_range by inlining.
    nats = 0.0
    pos = prefix
    hi = len(doc)
    while pos < hi:
        end = min(hi, pos + STRIDE)
        start = max(0, end - max_ctx_bytes)
        window = doc[start:end]
        rel_lo = pos - start
        rel_hi = end - start
        vec = v4.local_nll_vector(model, window, device)
        nats += float(vec[rel_lo:rel_hi].sum().item())
        pos = end
    return nats, hi - prefix


def domain_stats(tok, fname: str) -> dict:
    raw = (VAL / fname).read_bytes()
    docs = [p for p in raw.split(bytes([SEP])) if p]
    tok_counts = []
    byte_counts = []
    bpt = []
    for doc in docs:
        try:
            ids, spans = v4.token_spans(tok, doc)
        except ValueError:
            continue
        if not ids:
            continue
        n_tok = len(ids)
        n_byte = len(doc)
        tok_counts.append(n_tok)
        byte_counts.append(n_byte)
        bpt.append(n_byte / max(n_tok, 1))
    # tokens that fit in 2048 bytes at this domain's mean B/tok
    mean_bpt = float(np.mean(bpt)) if bpt else float("nan")
    toks_in_2048 = 2048.0 / mean_bpt if mean_bpt else float("nan")
    return {
        "n_docs": len(docs),
        "n_ok": len(tok_counts),
        "mean_bytes": float(np.mean(byte_counts)) if byte_counts else None,
        "mean_tokens": float(np.mean(tok_counts)) if tok_counts else None,
        "mean_bytes_per_token": mean_bpt,
        "tokens_in_2048_bytes": toks_in_2048,
        "exceeds_512": bool(toks_in_2048 > 512) if mean_bpt else None,
        "p90_tokens": float(np.quantile(tok_counts, 0.9)) if tok_counts else None,
        "max_tokens": int(max(tok_counts)) if tok_counts else None,
    }


def run_uncapped_stats(tok) -> dict:
    return {name: domain_stats(tok, fname) for name, fname in FILES.items()}


def score_size(device, size: str, max_tok: int | None, max_bytes: int, tag: str):
    print({"score": size, "tag": tag, "max_tok": max_tok, "max_bytes": max_bytes}, flush=True)
    if size == "350m":
        loc = v3.load_350_local64(device)
        bpe_m, bpe, _, _ = load_matched_bpe(v4.BPE_DIR / "matched350m_bpe.pt", device)
        tok = bpe.tok
        cap_tok = max_tok if max_tok is not None else 10**9
    else:
        loc = v3.load_120_local64(v3.SHUF_LOCAL, device)
        bpe_m, bpe = v3.load_120_bpe(v3.SHUF_BPE, device)
        tok = bpe.tok
        cap_tok = max_tok if max_tok is not None else 10**9

    size_rows = {}
    meta = {}
    code_docs = []
    for name, fname in FILES.items():
        raw = (VAL / fname).read_bytes()
        docs = v4.docs_of(raw)
        rows = []
        max_win = 0
        for i, doc in enumerate(docs):
            try:
                nats_b, bytes_b, prefix, info = score_bpe_doc_capped(
                    bpe_m, tok, doc, device, cap_tok, max_bytes
                )
            except ValueError:
                continue
            nats_l, bytes_l = score_local_capped(loc, doc, prefix, device, max_bytes)
            if bytes_l != bytes_b:
                raise RuntimeError(f"byte mismatch {size} {name} {i}: {bytes_l} vs {bytes_b}")
            row = {
                "nats_l": nats_l,
                "bytes_l": bytes_l,
                "nats_b": nats_b,
                "bytes_b": bytes_b,
                "doc_bytes": len(doc),
                "doc_i": i,
            }
            rows.append(row)
            max_win = max(max_win, info["max_win_tokens"])
            if name == "code":
                ids, spans = v4.token_spans(tok, doc)
                row["n_tokens"] = len(ids)
                row["bpb_b"] = bpb_from_nats(nats_b, bytes_b) if bytes_b else None
                code_docs.append(row)
            if (i + 1) % 100 == 0:
                print({"tag": tag, "domain": name, "doc": i + 1}, flush=True)
        size_rows[name] = rows
        meta[name] = {"n_docs": len(rows), "max_win_tokens": max_win}
        print({"tag": tag, "domain": name, "docs": len(rows), "max_win_tokens": max_win}, flush=True)

    stats = {name: v4.bootstrap_delta(size_rows[name]) for name in DOMAINS}
    stats["overall"] = v4.bootstrap_delta([r for name in DOMAINS for r in size_rows[name]])

    # code sensitivity: top-10 by NLL/byte
    code_sens = None
    if code_docs:
        ranked = sorted(code_docs, key=lambda r: (r["nats_b"] / max(r["bytes_b"], 1)), reverse=True)
        top10 = ranked[:10]
        total_nll = sum(r["nats_b"] for r in code_docs)
        top_nll = sum(r["nats_b"] for r in top10)
        keep = ranked[10:]
        excl = {
            "bpb_bpe_all": bpb_from_nats(sum(r["nats_b"] for r in code_docs), sum(r["bytes_b"] for r in code_docs)),
            "bpb_local_all": bpb_from_nats(sum(r["nats_l"] for r in code_docs), sum(r["bytes_l"] for r in code_docs)),
            "bpb_bpe_excl_top10": bpb_from_nats(sum(r["nats_b"] for r in keep), sum(r["bytes_b"] for r in keep)) if keep else None,
            "bpb_local_excl_top10": bpb_from_nats(sum(r["nats_l"] for r in keep), sum(r["bytes_l"] for r in keep)) if keep else None,
            "top10_nll_share": top_nll / max(total_nll, 1e-12),
            "top10": [
                {
                    "doc_i": r["doc_i"],
                    "doc_bytes": r["doc_bytes"],
                    "n_tokens": r.get("n_tokens"),
                    "nats_b": r["nats_b"],
                    "bytes_b": r["bytes_b"],
                    "nll_per_byte": r["nats_b"] / max(r["bytes_b"], 1),
                    "bpb_b": r["bpb_b"],
                }
                for r in top10
            ],
        }
        code_sens = excl

    out = {
        "size": size,
        "tag": tag,
        "max_ctx_tokens": max_tok,
        "max_ctx_bytes": max_bytes,
        "sliding": stats,
        "meta": meta,
        "code_sensitivity": code_sens,
    }
    path = OUT / f"slide_{size}_{tag}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print({"wrote": str(path), "overall": stats["overall"]}, flush=True)
    del loc, bpe_m
    torch.cuda.empty_cache()
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    # tokenizer for domain stats
    _, bpe350, _, _ = load_matched_bpe(v4.BPE_DIR / "matched350m_bpe.pt", device)
    tok = bpe350.tok
    # also measure 120M mean tokens per 2048-byte training window from earlier seqlen if present
    seqlen_path = OUT / "seqlen.json"
    seqlen = json.loads(seqlen_path.read_text()) if seqlen_path.exists() else {}
    domain = run_uncapped_stats(tok)
    setup = {
        "120m_bpe": {
            "train_seq_tokens": "tokenize of one 2048-byte window (variable; mean ~650 on shuffled real_v2)",
            "train_seq_tokens_measured_mean": seqlen.get("120m_mean_tokens"),
            "max_len_model": 2048,
            "pos_encoding": "RoPE",
            "context_bytes_train": 2048,
        },
        "350m_bpe": {
            "train_seq_tokens": TRAIN_TOK_350,
            "pack_len": 512,
            "max_len_model": 2048,
            "pos_encoding": "RoPE",
            "mean_tok_per_2048_window_pretok": seqlen.get("350m_mean_tok_per_2048_window"),
            "pack_context_bytes_approx": seqlen.get("350m_pack_context_bytes"),
        },
        "local64_both": {
            "train_seq_bytes": 2048,
            "local_window": 64,
            "pos_encoding": "RoPE (global patch) + local sliding window",
        },
        "domain_tokens_in_2048B": domain,
    }
    (OUT / "slide_context_diagnosis.json").write_text(json.dumps(setup, indent=2), encoding="utf-8")
    print(json.dumps(setup, indent=2)[:4000], flush=True)
    del bpe350
    torch.cuda.empty_cache()

    # Cap bytes for 350M: tokens that fit in mean B/tok of each domain varies;
    # use a single byte cap = trained pack context ≈ 512 * mean_bpt_overall.
    mean_bpt = float(np.mean([domain[d]["mean_bytes_per_token"] for d in DOMAINS]))
    cap_bytes_350 = int(round(TRAIN_TOK_350 * mean_bpt))
    # Also use pretok estimate if available
    if seqlen.get("350m_pack_context_bytes"):
        cap_bytes_350 = int(round(seqlen["350m_pack_context_bytes"]))
    print({"cap_bytes_350": cap_bytes_350, "cap_tok_350": TRAIN_TOK_350}, flush=True)

    # 350M: uncapped 2048B (reproduce) + capped to trained length
    score_size(device, "350m", max_tok=None, max_bytes=CTX_BYTES_FULL, tag="uncapped_2048B")
    score_size(device, "350m", max_tok=TRAIN_TOK_350, max_bytes=cap_bytes_350, tag="capped_trainlen")
    # Also: BPE capped to 512 tokens but local64 still 2048B — for reference only
    score_size(device, "350m", max_tok=TRAIN_TOK_350, max_bytes=CTX_BYTES_FULL, tag="bpe512_local2048")

    # 120M: trained on ~650 tok / 2048B; model max_len=2048. Cap to 2048 tokens (= full byte ctx).
    # Confirm uncapped 2048B does not exceed trained byte length.
    mean_120 = seqlen.get("120m_mean_tokens") or 650
    score_size(device, "120m", max_tok=None, max_bytes=CTX_BYTES_FULL, tag="uncapped_2048B")
    # Cap to measured mean train tokens and same bytes
    cap_tok_120 = int(math.ceil(mean_120))
    score_size(device, "120m", max_tok=cap_tok_120, max_bytes=CTX_BYTES_FULL, tag="capped_mean_train_tok")

    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
