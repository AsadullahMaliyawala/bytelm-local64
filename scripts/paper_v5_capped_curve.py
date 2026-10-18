#!/usr/bin/env python3
"""350M document-sliding curve with context capped to BPE train pack (512 tok / 1760 B)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts")]

import paper_v4_recompute as v4  # noqa: E402
import paper_v4_slide_cap as cap  # noqa: E402
from lct.checkpoint import load_state_dict_strict  # noqa: E402
from lct.models.continuous_pool4 import ContinuousPool4LM  # noqa: E402
from matched_bpe_load import load_matched_bpe  # noqa: E402

OUT = Path("./experiments/paper_v5")
RESEARCH = ROOT / "research"
BPE_DIR = Path("./checkpoints/matched350m_bpe")
LOCAL_DIR = Path("./checkpoints/matched350m_local64")
CAP_TOK = 512
CAP_BYTES = 1760
FILES = {"text": "val_text.bin", "code": "val_code.bin", "math": "val_math.bin"}
DOMAINS = ("text", "code", "math")

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

ALL_CHECKPOINTS = [
    ("2b", LOCAL_DIR / "matched350m_local64_2b.pt", BPE_DIR / "matched350m_bpe_2b.pt"),
    ("4b", LOCAL_DIR / "matched350m_local64_4b.pt", BPE_DIR / "matched350m_bpe_4b.pt"),
    ("6b", LOCAL_DIR / "matched350m_local64_6b.pt", BPE_DIR / "matched350m_bpe_6b.pt"),
    ("8b", LOCAL_DIR / "matched350m_local64_8b.pt", BPE_DIR / "matched350m_bpe_8b.pt"),
    ("10b", LOCAL_DIR / "matched350m_local64.pt", BPE_DIR / "matched350m_bpe.pt"),
]


def load_local(path: Path, device):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model = ContinuousPool4LM(CFG)
    load_state_dict_strict(model, blob["model"], label=f"local64:{path.name}")
    model.to(device).eval()
    return model, int(blob.get("step", 0)), int(blob.get("seen", 0))


def score_pair(device, loc_ckpt: Path, bpe_ckpt: Path, tag: str) -> dict:
    print({"curve_point": tag, "loc": str(loc_ckpt), "bpe": str(bpe_ckpt)}, flush=True)
    model_l, step_l, seen_l = load_local(loc_ckpt, device)
    model_b, bpe, step_b, seen_b = load_matched_bpe(bpe_ckpt, device)
    tok = bpe.tok

    size_rows = {}
    tok_per_win = {}
    for name, fname in FILES.items():
        raw = (v4.VAL / fname).read_bytes()
        rows = []
        win_toks = []
        for i, doc in enumerate(v4.docs_of(raw)):
            try:
                nats_b, bytes_b, prefix, info = cap.score_bpe_doc_capped(
                    model_b, tok, doc, device, CAP_TOK, CAP_BYTES
                )
            except ValueError:
                continue
            nats_l, bytes_l = cap.score_local_capped(model_l, doc, prefix, device, CAP_BYTES)
            if bytes_l != bytes_b:
                raise RuntimeError(f"byte mismatch {tag} {name} {i}: {bytes_l} vs {bytes_b}")
            rows.append({"nats_l": nats_l, "bytes_l": bytes_l, "nats_b": nats_b, "bytes_b": bytes_b})
            win_toks.append(info["max_win_tokens"])
            if (i + 1) % 200 == 0:
                print({"tag": tag, "domain": name, "doc": i + 1}, flush=True)
        size_rows[name] = rows
        tok_per_win[name] = {
            "max_win_tokens": max(win_toks) if win_toks else 0,
            "mean_max_win_tokens": float(sum(win_toks) / len(win_toks)) if win_toks else 0.0,
            "n_docs": len(rows),
        }
        print({"tag": tag, "domain": name, "docs": len(rows), **tok_per_win[name]}, flush=True)

    stats = {name: v4.bootstrap_delta(size_rows[name]) for name in DOMAINS}
    stats["overall"] = v4.bootstrap_delta([r for name in DOMAINS for r in size_rows[name]])
    point = {}
    for name in list(DOMAINS) + ["overall"]:
        st = stats[name]
        point[name] = {
            "bpb_local64": st["bpb_local64"],
            "bpb_bpe": st["bpb_bpe"],
            "diff": st["diff_bpe_minus_local64"],
            "diff_lo": st["diff_lo"],
            "diff_hi": st["diff_hi"],
            "sig": st["significant_0_outside_ci"],
            "n_docs": st["n_docs"],
            "scored_bytes": st["scored_bytes_local64"],
        }
    out = {
        "tag": tag,
        "protocol": "document_sliding_capped_512tok_1760B",
        "step_local64": step_l,
        "seen_local64": seen_l,
        "step_bpe": step_b,
        "seen_bpe": seen_b,
        "paired": stats,
        "point": point,
        "tok_per_win": tok_per_win,
    }
    (OUT / f"curve_capped_{tag}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    del model_l, model_b
    torch.cuda.empty_cache()
    return out


def plot_curve(curve: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tags = [p["tag"] for p in curve["points"]]
    xs = [p["seen_local64"] / 1e9 for p in curve["points"]]
    y_l = [p["point"]["overall"]["bpb_local64"] for p in curve["points"]]
    y_b = [p["point"]["overall"]["bpb_bpe"] for p in curve["points"]]
    gaps = [p["point"]["overall"]["diff"] for p in curve["points"]]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    ax = axes[0]
    ax.plot(xs, y_l, "o-", color="#1f77b4", label="local64", linewidth=1.8)
    ax.plot(xs, y_b, "s-", color="#d62728", label="BPE-8k", linewidth=1.8)
    ax.set_xlabel("Bytes seen (B)")
    ax.set_ylabel("Val BPB (capped doc slide)")
    ax.set_title("Overall BPB")
    ax.legend()
    ax.set_xlim(0, 10.5)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(xs, gaps, "D-", color="#333333", linewidth=1.8)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("Bytes seen (B)")
    ax.set_ylabel(r"BPB gap (BPE $-$ local64)")
    ax.set_title("Gap vs bytes")
    ax.set_xlim(0, 10.5)
    ax.grid(True, alpha=0.3)
    if gaps:
        ax.annotate(f"{gaps[0]:.3f}", (xs[0], gaps[0]), textcoords="offset points", xytext=(6, 6), fontsize=8)
        ax.annotate(f"{gaps[-1]:.3f}", (xs[-1], gaps[-1]), textcoords="offset points", xytext=(6, 6), fontsize=8)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    RESEARCH.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "curve_350m_capped_slide.png", dpi=160)
    fig.savefig(RESEARCH / "curve_350m_capped_slide.png", dpi=160)
    paper = ROOT / "paper" / "figures" / "matched_350m_val_bpb.png"
    paper.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(paper, dpi=160)
    print({"figure": str(paper), "gaps": list(zip(tags, gaps))}, flush=True)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="checkpoint tags to score, e.g. 4b 8b")
    ap.add_argument("--assemble-only", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    RESEARCH.mkdir(parents=True, exist_ok=True)

    wanted = set(args.only) if args.only else {t for t, _, _ in ALL_CHECKPOINTS}
    if not args.assemble_only:
        for tag, loc, bpe in ALL_CHECKPOINTS:
            if tag not in wanted:
                continue
            if not loc.exists() or not bpe.exists():
                raise FileNotFoundError(f"missing checkpoint for {tag}: {loc} / {bpe}")
            cached = OUT / f"curve_capped_{tag}.json"
            if cached.exists():
                print({"reuse": str(cached)}, flush=True)
                continue
            score_pair(device, loc, bpe, tag)

    points = []
    for tag, _, _ in ALL_CHECKPOINTS:
        cached = OUT / f"curve_capped_{tag}.json"
        if not cached.exists():
            print({"missing_point": tag}, flush=True)
            continue
        points.append(json.loads(cached.read_text(encoding="utf-8")))
    if len(points) == len(ALL_CHECKPOINTS):
        curve = {
            "protocol": "document_sliding_capped_512tok_1760B",
            "cap_tokens": CAP_TOK,
            "cap_bytes": CAP_BYTES,
            "points": points,
            "final_paired": points[-1]["paired"],
        }
        (OUT / "curve_350m_capped_slide.json").write_text(json.dumps(curve, indent=2), encoding="utf-8")
        (RESEARCH / "curve_350m_capped_slide.json").write_text(json.dumps(curve, indent=2), encoding="utf-8")
        plot_curve(curve)
        print("ALL_DONE", flush=True)
    else:
        print({"partial": [p["tag"] for p in points]}, flush=True)



if __name__ == "__main__":
    main()
