#!/usr/bin/env python3
"""350M matched-span BPB curve at 2/4/6/8/10B (primary protocol for Paper v9)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts")]

import paper_v4_recompute as v4  # noqa: E402
import paper_v8_matched_span as ms  # noqa: E402
from matched_bpe_load import load_matched_bpe  # noqa: E402

OUT = Path("${LCT_DATA}/experiments/paper_v5")
RESEARCH = ROOT / "research"
BPE_DIR = Path("${LCT_DATA}/checkpoints/matched350m_bpe")
LOCAL_DIR = Path("${LCT_DATA}/checkpoints/matched350m_local64")
DOMAINS = ("text", "code", "math")
FILES = {"text": "val_text.bin", "code": "val_code.bin", "math": "val_math.bin"}

CHECKPOINTS = [
    ("2b", LOCAL_DIR / "matched350m_local64_2b.pt", BPE_DIR / "matched350m_bpe_2b.pt"),
    ("4b", LOCAL_DIR / "matched350m_local64_4b.pt", BPE_DIR / "matched350m_bpe_4b.pt"),
    ("6b", LOCAL_DIR / "matched350m_local64_6b.pt", BPE_DIR / "matched350m_bpe_6b.pt"),
    ("8b", LOCAL_DIR / "matched350m_local64_8b.pt", BPE_DIR / "matched350m_bpe_8b.pt"),
    ("10b", LOCAL_DIR / "matched350m_local64.pt", BPE_DIR / "matched350m_bpe.pt"),
]


def score_pair(device, loc_ckpt: Path, bpe_ckpt: Path, tag: str) -> dict:
    print({"curve_point": tag}, flush=True)
    loc = ms.load_local(loc_ckpt, device)
    model_b, bpe, step_b, seen_b = load_matched_bpe(bpe_ckpt, device)
    # recover seen from local
    blob = torch.load(loc_ckpt, map_location="cpu", weights_only=False)
    step_l = int(blob.get("step", 0))
    seen_l = int(blob.get("seen", 0))
    tok = bpe.tok
    size_rows = {}
    for name, fname in FILES.items():
        rows = []
        raw = (v4.VAL / fname).read_bytes()
        for i, doc in enumerate(v4.docs_of(raw)):
            nats_b, bytes_b, _prefix, wins = ms.score_bpe_windows(model_b, tok, doc, device)
            nats_l, bytes_l = ms.score_local_matched(loc, doc, wins, device)
            if bytes_l != bytes_b:
                raise RuntimeError(f"mismatch {tag} {name} {i}")
            rows.append({"nats_l": nats_l, "bytes_l": bytes_l, "nats_b": nats_b, "bytes_b": bytes_b})
            if (i + 1) % 200 == 0:
                print({"tag": tag, "domain": name, "doc": i + 1}, flush=True)
        size_rows[name] = rows
        print({"tag": tag, "domain": name, "docs": len(rows)}, flush=True)
    stats = {name: v4.bootstrap_delta(size_rows[name]) for name in DOMAINS}
    stats["overall"] = v4.bootstrap_delta([r for name in DOMAINS for r in size_rows[name]])
    point = {
        name: {
            "bpb_local64": stats[name]["bpb_local64"],
            "bpb_bpe": stats[name]["bpb_bpe"],
            "diff": stats[name]["diff_bpe_minus_local64"],
            "diff_lo": stats[name]["diff_lo"],
            "diff_hi": stats[name]["diff_hi"],
            "sig": stats[name]["significant_0_outside_ci"],
        }
        for name in list(DOMAINS) + ["overall"]
    }
    out = {
        "tag": tag,
        "protocol": "matched_span_min512tok_2048B",
        "step_local64": step_l,
        "seen_local64": seen_l,
        "step_bpe": step_b,
        "seen_bpe": seen_b,
        "paired": stats,
        "point": point,
    }
    (OUT / f"curve_matched_{tag}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    del loc, model_b
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
    ax.set_ylabel("Val BPB (matched span)")
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
    fig.savefig(OUT / "curve_350m_matched_span.png", dpi=160)
    fig.savefig(RESEARCH / "curve_350m_matched_span.png", dpi=160)
    paper = ROOT / "paper" / "figures" / "matched_350m_val_bpb.png"
    paper.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(paper, dpi=160)
    print({"figure": str(paper), "gaps": list(zip(tags, gaps))}, flush=True)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    # seed 10b from existing matched-span final if present
    final = OUT / "slide_350m_matched_span.json"
    cached10 = OUT / "curve_matched_10b.json"
    if final.exists() and not cached10.exists():
        src = json.loads(final.read_text())
        blob = torch.load(LOCAL_DIR / "matched350m_local64.pt", map_location="cpu", weights_only=False)
        point = {
            name: {
                "bpb_local64": src["paired"][name]["bpb_local64"],
                "bpb_bpe": src["paired"][name]["bpb_bpe"],
                "diff": src["paired"][name]["diff_bpe_minus_local64"],
                "diff_lo": src["paired"][name]["diff_lo"],
                "diff_hi": src["paired"][name]["diff_hi"],
                "sig": src["paired"][name]["significant_0_outside_ci"],
            }
            for name in list(DOMAINS) + ["overall"]
        }
        cached10.write_text(
            json.dumps(
                {
                    "tag": "10b",
                    "protocol": "matched_span_min512tok_2048B",
                    "step_local64": int(blob.get("step", 0)),
                    "seen_local64": int(blob.get("seen", 10_000_000_000)),
                    "step_bpe": 0,
                    "seen_bpe": 10_000_000_000,
                    "paired": src["paired"],
                    "point": point,
                    "seeded_from": str(final),
                },
                indent=2,
            )
        )
        print({"seeded_10b": True}, flush=True)

    wanted = set(args.only) if args.only else {t for t, _, _ in CHECKPOINTS}
    points = []
    for tag, loc, bpe in CHECKPOINTS:
        if tag not in wanted:
            continue
        cached = OUT / f"curve_matched_{tag}.json"
        if cached.exists():
            print({"reuse": str(cached)}, flush=True)
            points.append(json.loads(cached.read_text()))
            continue
        points.append(score_pair(device, loc, bpe, tag))
    # assemble all five if present
    all_pts = []
    for tag, _, _ in CHECKPOINTS:
        c = OUT / f"curve_matched_{tag}.json"
        if c.exists():
            all_pts.append(json.loads(c.read_text()))
    if len(all_pts) == 5:
        curve = {"protocol": "matched_span_min512tok_2048B", "points": all_pts}
        (OUT / "curve_350m_matched_span.json").write_text(json.dumps(curve, indent=2))
        (RESEARCH / "curve_350m_matched_span.json").write_text(json.dumps(curve, indent=2))
        plot_curve(curve)
        print("ALL_DONE", flush=True)
    else:
        print({"partial": [p["tag"] for p in all_pts]}, flush=True)


if __name__ == "__main__":
    main()
