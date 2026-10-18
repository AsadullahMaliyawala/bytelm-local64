#!/usr/bin/env python3
"""Paper v4: reproduce the BPE denominator bug, rescore the 350M curve with
target-token bytes, and score full documents under a 2048-byte sliding context.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from lct.train.ladder import bpb_from_nats  # noqa: E402
from matched_bpe_load import load_matched_bpe  # noqa: E402
import paper_v3_recompute as v3  # noqa: E402

VAL = Path("${LCT_DATA}/data/real_v3")
OUT = Path("${LCT_DATA}/experiments/paper_v4")
RESEARCH = ROOT / "research"
BPE_DIR = Path("${LCT_DATA}/checkpoints/matched350m_bpe")
LOCAL_DIR = Path("${LCT_DATA}/checkpoints/matched350m_local64")
CTX = 2048
STRIDE = 512
BOOT_N = 10_000
BOOT_SEED = 0
SEP = 0x1E
DOMAINS = ("text", "code", "math")
FILES = {"text": "val_text.bin", "code": "val_code.bin", "math": "val_math.bin"}
BPE_CURVE = [
    ("2b", BPE_DIR / "matched350m_bpe_2b.pt"),
    ("4b", BPE_DIR / "matched350m_bpe_4b.pt"),
    ("6b", BPE_DIR / "matched350m_bpe_6b.pt"),
    ("8b", BPE_DIR / "matched350m_bpe_8b.pt"),
    ("10b", BPE_DIR / "matched350m_bpe.pt"),
]


def docs_of(raw: bytes) -> list[bytes]:
    return [part for part in raw.split(bytes([SEP])) if part]


def token_spans(tok, doc: bytes) -> tuple[list[int], list[tuple[int, int]]]:
    """Tokenize the whole document. Clip tokenizer offsets so each byte is owned once.

    Per-token ``decode`` does not rebuild multibyte text, so spans come from the
    encoder offsets. Overlapping offsets are clipped; a token that then owns no
    byte is kept with an empty span and is not scored.
    """
    text = doc.decode("utf-8")
    if text.encode("utf-8") != doc:
        raise ValueError("document is not strict UTF-8")
    enc = tok.encode(text)
    ids = list(enc.ids)
    chars = [0]
    for ch in text:
        chars.append(chars[-1] + len(ch.encode("utf-8")))
    pos = 0
    spans: list[tuple[int, int]] = []
    n = len(doc)
    for a, b in enc.offsets:
        start = min(max(chars[a], pos), n)
        end = min(max(chars[b], start), n)
        spans.append((start, end))
        pos = end
    if pos != n:
        raise ValueError(f"spans cover {pos} of {n} bytes")
    if len(spans) != len(ids):
        raise ValueError("offset/id length mismatch")
    return ids, spans


@torch.no_grad()
def local_nll_vector(model, window: bytes, device) -> torch.Tensor:
    x = torch.tensor([list(window)], dtype=torch.long, device=device)
    logits = model._byte_logits(x)
    logp = F.log_softmax(logits.float(), dim=-1)
    return -logp.gather(-1, x.unsqueeze(-1)).squeeze(-1)[0]


def score_local_range(model, doc: bytes, lo: int, hi: int, device) -> float:
    """Nats for bytes [lo, hi), each conditioned on at most CTX bytes ending at hi-chunks."""
    if hi <= lo:
        return 0.0
    nats = 0.0
    pos = lo
    while pos < hi:
        end = min(hi, pos + STRIDE)
        start = max(0, end - CTX)
        window = doc[start:end]
        rel_lo = pos - start
        rel_hi = end - start
        vec = local_nll_vector(model, window, device)
        nats += float(vec[rel_lo:rel_hi].sum().item())
        pos = end
    return nats


@torch.no_grad()
def score_bpe_doc(model, tok, doc: bytes, device) -> tuple[float, int, int]:
    """Sliding 2048-byte context. Tokenize the whole document first.
    Returns nats, scored bytes, unscored prefix bytes (first token).
    """
    ids, spans = token_spans(tok, doc)
    if len(ids) < 2:
        return 0.0, 0, len(doc)
    weights = [spans[i][1] - spans[i][0] for i in range(len(ids))]
    nats = 0.0
    nbytes = 0
    next_tok = 1
    n = len(ids)
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
        while ctx > 0 and last_byte - spans[ctx - 1][0] <= CTX:
            ctx -= 1
        win_ids = ids[ctx:end_tok]
        if len(win_ids) < 2 and ctx > 0:
            ctx -= 1
            win_ids = ids[ctx:end_tok]
        if len(win_ids) < 2:
            next_tok = end_tok
            continue
        tokens = torch.tensor([win_ids], dtype=torch.long, device=device)
        inp, tgt = tokens[:, :-1], tokens[:, 1:]
        logits = model(inp)["logits"].float()
        nll = -torch.log_softmax(logits, dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[0]
        # tgt position k predicts token index (ctx + 1 + k)
        first_keep = next_tok - (ctx + 1)
        for k in range(first_keep, int(tgt.size(1))):
            tok_i = ctx + 1 + k
            if ids[tok_i] == 0 or weights[tok_i] <= 0:
                continue
            nats += float(nll[k].item())
            nbytes += int(weights[tok_i])
        next_tok = end_tok
    return nats, nbytes, int(weights[0])


def score_local_doc(model, doc: bytes, prefix: int, device) -> tuple[float, int]:
    if prefix >= len(doc):
        return 0.0, 0
    nats = score_local_range(model, doc, prefix, len(doc), device)
    return nats, len(doc) - prefix


def bootstrap_delta(rows: list[dict], n: int = BOOT_N, seed: int = BOOT_SEED) -> dict:
    if not rows:
        return {}
    rng = np.random.default_rng(seed)
    loc_n = np.array([r["nats_l"] for r in rows], dtype=np.float64)
    loc_b = np.array([r["bytes_l"] for r in rows], dtype=np.float64)
    bpe_n = np.array([r["nats_b"] for r in rows], dtype=np.float64)
    bpe_b = np.array([r["bytes_b"] for r in rows], dtype=np.float64)
    m = len(rows)

    def agg(idx):
        lb = float(loc_b[idx].sum())
        bb = float(bpe_b[idx].sum())
        lp = bpb_from_nats(float(loc_n[idx].sum()), int(round(lb))) if lb else None
        bp = bpb_from_nats(float(bpe_n[idx].sum()), int(round(bb))) if bb else None
        return lp, bp, (None if lp is None or bp is None else bp - lp)

    point_l, point_b, point_d = agg(np.arange(m))
    diffs = np.empty(n, dtype=np.float64)
    for i in range(n):
        idx = rng.integers(0, m, size=m)
        diffs[i] = agg(idx)[2]
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {
        "bpb_local64": point_l,
        "bpb_bpe": point_b,
        "diff_bpe_minus_local64": point_d,
        "diff_lo": float(lo),
        "diff_hi": float(hi),
        "significant_0_outside_ci": bool(lo > 0 or hi < 0),
        "local64_better": bool(point_d is not None and point_d > 0),
        "n_docs": m,
        "scored_bytes_local64": int(loc_b.sum()),
        "scored_bytes_bpe": int(bpe_b.sum()),
    }


def window_reference() -> dict:
    path = RESEARCH / "paired_bpb_diff.json"
    if not path.exists():
        path = Path("${LCT_DATA}/experiments/paper_v3/paired_bpb_diff.json")
    return json.loads(path.read_text(encoding="utf-8"))


def run_denom(device) -> dict:
    print("=== denom reproduction 120M ===", flush=True)
    model, bpe = v3.load_120_bpe(v3.SHUF_BPE, device)
    tok = bpe.tok
    domains = {}
    for name, fname in FILES.items():
        raw = (VAL / fname).read_bytes()
        fr_rows, v2_rows = [], []
        for w in v3.iter_windows(raw):
            fr_rows.append(v3.bpe_path_frozen(model, tok, w, device))
            v2_rows.append(v3.bpe_path_paper_v2(model, tok, w, device))
        fr = v3.summarize_path(fr_rows)
        v2 = v3.summarize_path(v2_rows)
        first = int(sum(r.get("first_token_bytes", 0) for r in fr_rows))
        domains[name] = {
            "frozen": fr,
            "paper_v2": v2,
            "first_token_bytes": first,
            "raw_window_bytes": int(sum(r["raw_window_bytes"] for r in fr_rows)),
            "v2_minus_frozen_nats": v2["nats"] - fr["nats"],
            "v2_minus_frozen_bytes": v2["scored_bytes"] - fr["scored_bytes"],
            "v2_minus_frozen_bpb": (v2["bpb"] or 0) - (fr["bpb"] or 0),
        }
        print({"domain": name, "frozen_bpb": fr["bpb"], "v2_bpb": v2["bpb"], "v2_minus_frozen_bytes": domains[name]["v2_minus_frozen_bytes"]}, flush=True)
    # one-window mechanism on the first math window
    math_raw = (VAL / "val_math.bin").read_bytes()
    w0 = next(v3.iter_windows(math_raw))
    text = w0.decode("utf-8", errors="replace")
    ids = list(tok.encode(text).ids)
    weights = [len(tok.decode([t]).encode("utf-8")) for t in ids]
    mechanism = {
        "raw_window_bytes": len(w0),
        "reencoded_bytes": len(text.encode("utf-8")),
        "n_tokens": len(ids),
        "first_token_bytes": int(weights[0]) if weights else 0,
        "sum_all_token_utf8": int(sum(weights)),
        "sum_target_token_utf8_incl_pad": int(sum(weights[1:])) if len(weights) > 1 else 0,
        "sum_target_token_utf8_masked": int(sum(wt for t, wt in zip(ids[1:], weights[1:]) if t != 0 and wt > 0)),
        "note": (
            "paper_v2 denominator is the re-encoded window (includes the first token and any "
            "UTF-8 replacement at a cut). frozen denominator is the sum of per-token UTF-8 "
            "lengths of scored targets only. Those per-token lengths need not equal the window."
        ),
    }
    # first-position: local64 scores every byte; BPE drops the first token
    loc = v3.load_120_local64(v3.SHUF_LOCAL, device)
    nats_l, nb_l = v3.local64_window(loc, w0, device)
    fr0 = v3.bpe_path_frozen(model, tok, w0, device)
    first_pos = {
        "window_bytes": len(w0),
        "local64_scored_bytes": nb_l,
        "local64_scores_first_byte": nb_l == len(w0),
        "local64_nats": nats_l,
        "bpe_scored_bytes": fr0["scored_bytes"],
        "bpe_first_token_bytes_unscored": fr0.get("first_token_bytes"),
        "same_scored_bytes": nb_l == fr0["scored_bytes"],
        "statement": (
            "local64 byte_nll scores every byte of the window, including position 0 "
            "(predicted from a learned start state). BPE leaves the first token unscored. "
            "Scored bytes per window are not the same."
        ),
    }
    # 350M final, both paths, so the pitfall can quote that checkpoint too
    del loc
    torch.cuda.empty_cache()
    m350, bpe350, step, seen = load_matched_bpe(BPE_DIR / "matched350m_bpe.pt", device)
    tok350 = bpe350.tok
    dom350 = {}
    for name, fname in FILES.items():
        raw = (VAL / fname).read_bytes()
        fr_rows, v2_rows = [], []
        for w in v3.iter_windows(raw):
            fr_rows.append(v3.bpe_path_frozen(m350, tok350, w, device))
            v2_rows.append(v3.bpe_path_paper_v2(m350, tok350, w, device))
        dom350[name] = {"frozen": v3.summarize_path(fr_rows), "paper_v2": v3.summarize_path(v2_rows)}
        print({"350_denom": name, "frozen": dom350[name]["frozen"]["bpb"], "v2": dom350[name]["paper_v2"]["bpb"]}, flush=True)
    def overall(dom, key):
        nats = sum(dom[d][key]["nats"] for d in DOMAINS)
        byts = sum(dom[d][key]["scored_bytes"] for d in DOMAINS)
        return {"bpb": bpb_from_nats(nats, byts), "nats": nats, "scored_bytes": byts}
    out = {
        "120m": {"domains": domains, "overall_frozen": overall(domains, "frozen"), "overall_paper_v2": overall(domains, "paper_v2"), "one_math_window": mechanism, "first_position": first_pos},
        "350m": {"domains": dom350, "overall_frozen": overall(dom350, "frozen"), "overall_paper_v2": overall(dom350, "paper_v2"), "step": step, "seen": seen},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "denom_reproduction.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print({"wrote": "denom_reproduction.json", "120_math_v2_minus_fr_bytes": domains["math"]["v2_minus_frozen_bytes"]}, flush=True)
    del model, m350
    torch.cuda.empty_cache()
    return out


def marginal_ci(nats: list[float], byts: list[int], n: int = BOOT_N, seed: int = BOOT_SEED) -> list[float]:
    rng = np.random.default_rng(seed)
    nt = np.array(nats, dtype=np.float64)
    bt = np.array(byts, dtype=np.float64)
    m = len(nt)
    point = bpb_from_nats(float(nt.sum()), int(bt.sum()))
    draws = np.empty(n, dtype=np.float64)
    for i in range(n):
        idx = rng.integers(0, m, size=m)
        draws[i] = bpb_from_nats(float(nt[idx].sum()), int(bt[idx].sum()))
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return [point, float(lo), float(hi)]


def run_curve(device) -> dict:
    print("=== curve corrected BPE denom ===", flush=True)
    old_path = RESEARCH / "curve_350m_bpb.json"
    if not old_path.exists():
        old_path = Path("${LCT_DATA}/experiments/paper_v2/curve_350m_bpb.json")
    old = json.loads(old_path.read_text(encoding="utf-8"))
    curve = {
        "formula": "BPB = sum(nats)/(ln2 * scored_bytes); BPE scored_bytes = target-token UTF-8 lengths",
        "val_dir": str(VAL),
        "window": v3.WINDOW,
        "bpe_denominator": "target-token UTF-8 bytes (frozen path)",
        "note": "BPE points rescored. local64 points kept. Previous BPE curve used full re-encoded window bytes.",
        "local64": old["local64"],
        "bpe": [],
        "superseded_bpe_full_window_denom": old["bpe"],
    }
    cis = {}
    for tag, path in BPE_CURVE:
        print({"curve": tag, "path": str(path)}, flush=True)
        model, bpe, step, seen = load_matched_bpe(path, device)
        tok = bpe.tok
        domains = {}
        per = {name: {"nats": [], "bytes": []} for name in DOMAINS}
        for name, fname in FILES.items():
            raw = (VAL / fname).read_bytes()
            rows = [v3.bpe_path_frozen(model, tok, w, device) for w in v3.iter_windows(raw)]
            domains[name] = {
                "bpb": v3.summarize_path(rows)["bpb"],
                "n_windows": len(rows),
                "n_bytes": int(sum(r["scored_bytes"] for r in rows)),
                "nats": float(sum(r["nats"] for r in rows)),
            }
            per[name]["nats"] = [r["nats"] for r in rows]
            per[name]["bytes"] = [r["scored_bytes"] for r in rows]
            print({"tag": tag, "domain": name, "bpb": domains[name]["bpb"]}, flush=True)
        all_n = [n for name in DOMAINS for n in per[name]["nats"]]
        all_b = [b for name in DOMAINS for b in per[name]["bytes"]]
        domains["overall"] = {
            "bpb": bpb_from_nats(sum(all_n), sum(all_b)),
            "n_windows": len(all_n),
            "n_bytes": int(sum(all_b)),
            "nats": float(sum(all_n)),
        }
        if tag == "10b":
            cis["overall"] = marginal_ci(all_n, all_b)
            for name in DOMAINS:
                cis[name] = marginal_ci(per[name]["nats"], per[name]["bytes"])
        loc = next(p for p in old["local64"] if p["tag"] == tag)
        gap = domains["overall"]["bpb"] - loc["domains"]["overall"]["bpb"]
        curve["bpe"].append({"ckpt": str(path), "step": step, "seen": seen, "domains": domains, "tag": tag, "gap_bpe_minus_local64": gap})
        print({"tag": tag, "gap": gap, "bpe": domains["overall"]["bpb"]}, flush=True)
        del model
        torch.cuda.empty_cache()
    curve["bpe_10b_marginal_ci"] = cis
    OUT.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(curve, indent=2)
    (OUT / "curve_350m_bpb.json").write_text(payload, encoding="utf-8")
    (RESEARCH / "curve_350m_bpb.json").write_text(payload, encoding="utf-8")
    gap10 = curve["bpe"][-1]["gap_bpe_minus_local64"]
    print({"gap_10b": gap10, "expect": -0.0237}, flush=True)
    return curve


def plot_curve(curve: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def series(kind):
        xs, ys = [], []
        for pt in curve[kind]:
            xs.append((pt.get("seen") or 0) / 1e9)
            ys.append(pt["domains"]["overall"]["bpb"])
        return xs, ys

    x_l, y_l = series("local64")
    x_b, y_b = series("bpe")
    gaps_x, gaps_y = [], []
    by_l = {pt["tag"]: pt for pt in curve["local64"]}
    by_b = {pt["tag"]: pt for pt in curve["bpe"]}
    for tag in ("2b", "4b", "6b", "8b", "10b"):
        bl = by_l[tag]["domains"]["overall"]["bpb"]
        bb = by_b[tag]["domains"]["overall"]["bpb"]
        seen = by_l[tag].get("seen") or int(tag[:-1]) * 1e9
        gaps_x.append(seen / 1e9)
        gaps_y.append(bb - bl)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    ax = axes[0]
    ax.plot(x_l, y_l, "o-", color="#1f77b4", label="local64", linewidth=1.8)
    ax.plot(x_b, y_b, "s-", color="#d62728", label="BPE-8k", linewidth=1.8)
    ax.set_xlabel("Bytes seen (B)")
    ax.set_ylabel("Val BPB (byte-normalized)")
    ax.set_title("Overall BPB")
    ax.legend()
    ax.set_xlim(0, 10.5)
    ax.grid(True, alpha=0.3)
    ax = axes[1]
    ax.plot(gaps_x, gaps_y, "D-", color="#333333", linewidth=1.8)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("Bytes seen (B)")
    ax.set_ylabel("BPB gap (BPE − local64)")
    ax.set_title("Gap vs bytes")
    ax.set_xlim(0, 10.5)
    ax.grid(True, alpha=0.3)
    if gaps_y:
        ax.annotate(f"{gaps_y[0]:.3f}", (gaps_x[0], gaps_y[0]), textcoords="offset points", xytext=(6, 6), fontsize=8)
        ax.annotate(f"{gaps_y[-1]:.3f}", (gaps_x[-1], gaps_y[-1]), textcoords="offset points", xytext=(6, 6), fontsize=8)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "curve_350m_bpb_gap.png", dpi=160)
    fig.savefig(RESEARCH / "curve_350m_bpb_gap.png", dpi=160)
    paper = ROOT / "paper" / "figures" / "matched_350m_val_bpb.png"
    paper.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(paper, dpi=160)
    summary = {"gap_2b": gaps_y[0], "gap_10b": gaps_y[-1], "points": list(zip(gaps_x, gaps_y))}
    (RESEARCH / "curve_350m_gap_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print({"gap_figure": summary}, flush=True)


def run_slide(device, which: str) -> dict:
    print({"slide": which}, flush=True)
    specs = []
    if which in ("120", "all"):
        specs.append("120m")
    if which in ("350", "all"):
        specs.append("350m")
    out = {"context_bytes": CTX, "stride_bytes": STRIDE, "protocol": (
        "Documents split on 0x1E. BPE is tokenized on the full document so a window never "
        "starts mid-token. Each predicted token sees at most 2048 bytes of prior document bytes. "
        "local64 scores the same bytes (everything after the first BPE token) with a 2048-byte window."
    ), "sizes": {}}
    ref = window_reference()
    for size in specs:
        if size == "120m":
            loc = v3.load_120_local64(v3.SHUF_LOCAL, device)
            bpe_m, bpe = v3.load_120_bpe(v3.SHUF_BPE, device)
            tok = bpe.tok
        else:
            loc = v3.load_350_local64(device)
            bpe_m, bpe, _, _ = load_matched_bpe(BPE_DIR / "matched350m_bpe.pt", device)
            tok = bpe.tok
        size_rows = {}
        skipped = 0
        for name, fname in FILES.items():
            raw = (VAL / fname).read_bytes()
            rows = []
            docs = docs_of(raw)
            for i, doc in enumerate(docs):
                try:
                    nats_b, bytes_b, prefix = score_bpe_doc(bpe_m, tok, doc, device)
                except ValueError:
                    skipped += 1
                    continue
                nats_l, bytes_l = score_local_doc(loc, doc, prefix, device)
                if bytes_l != bytes_b:
                    raise RuntimeError(f"byte mismatch {size} {name} doc {i}: {bytes_l} vs {bytes_b}")
                rows.append({"nats_l": nats_l, "bytes_l": bytes_l, "nats_b": nats_b, "bytes_b": bytes_b})
                if (i + 1) % 100 == 0:
                    print({"size": size, "domain": name, "doc": i + 1, "of": len(docs)}, flush=True)
            size_rows[name] = rows
            print({"size": size, "domain": name, "docs": len(rows), "skipped": skipped}, flush=True)
        stats = {name: bootstrap_delta(size_rows[name]) for name in DOMAINS}
        all_rows = [r for name in DOMAINS for r in size_rows[name]]
        stats["overall"] = bootstrap_delta(all_rows)
        window = {}
        block = ref["120m"] if size == "120m" else ref["350m"]
        for name, st in block["paired_diff"].items():
            window[name] = {
                "diff_bpe_minus_local64": st["diff_bpe_minus_local64"],
                "bpb_local64": st["bpb_local64"],
                "bpb_bpe": st["bpb_bpe"],
                "significant_0_outside_ci": st["significant_0_outside_ci"],
                "local64_better": st["local64_better"],
            }
        ranking = {}
        for name in ("overall", "text", "code", "math"):
            ranking[name] = {
                "window_local64_better": window[name]["local64_better"],
                "window_sig": window[name]["significant_0_outside_ci"],
                "slide_local64_better": stats[name]["local64_better"],
                "slide_sig": stats[name]["significant_0_outside_ci"],
                "sign_same": (window[name]["diff_bpe_minus_local64"] > 0) == (stats[name]["diff_bpe_minus_local64"] > 0),
            }
        out["sizes"][size] = {"sliding": stats, "window": window, "ranking": ranking, "skipped_docs": skipped}
        print({"size": size, "slide_overall": stats["overall"]}, flush=True)
        del loc, bpe_m
        torch.cuda.empty_cache()
    out["120m_ranking_changed"] = any(
        not out["sizes"]["120m"]["ranking"][k]["sign_same"] for k in ("overall", "text", "code", "math")
    ) if "120m" in out["sizes"] else None
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sliding_bpb.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print({"wrote": "sliding_bpb.json", "120_changed": out["120m_ranking_changed"]}, flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["denom", "curve", "slide", "plot", "all"], required=True)
    ap.add_argument("--which", choices=["120", "350", "all"], default="all")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print({"device": str(device), "stage": args.stage}, flush=True)
    if args.stage == "denom":
        run_denom(device)
    elif args.stage == "curve":
        curve = run_curve(device)
        plot_curve(curve)
    elif args.stage == "slide":
        run_slide(device, args.which)
    elif args.stage == "plot":
        curve = json.loads((OUT / "curve_350m_bpb.json").read_text(encoding="utf-8"))
        plot_curve(curve)
    else:
        run_denom(device)
        curve = run_curve(device)
        plot_curve(curve)
        run_slide(device, "all")


if __name__ == "__main__":
    main()
