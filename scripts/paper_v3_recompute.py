#!/usr/bin/env python3
"""Paper v3 recomputes:

1) Reconcile 120M BPE BPB: frozen path (byte-weighted target tokens) vs paper_v2 path
   (full-window UTF-8 bytes after decode). Fix paired bootstrap with frozen path.
2) Per-item fact200 + CharBench few-shot/FT → research/downstream_items.json + Holm McNemar.
3) Document-order overlap for 350M twins first 10B bytes.
4) Curve + gap panel figure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from lct.checkpoint import load_state_dict_strict  # noqa: E402
from lct.eval.lm_metrics import LN2  # noqa: E402
from lct.train.ladder import bpb_from_nats  # noqa: E402
from lct.utils import get_autocast_dtype  # noqa: E402
from matched_bpe_load import load_matched_bpe  # noqa: E402

VAL_DIR = Path("./data/real_v3")
SHUF_LOCAL = Path("./checkpoints/ladder120_shuf/ladder120_shuf_local64_s0.pt")
SHUF_BPE = Path("./checkpoints/ladder120_shuf/ladder120_shuf_bpe_s0.pt")
LOCAL64_350 = Path("./checkpoints/matched350m_local64/matched350m_local64.pt")
BPE_350 = Path("./checkpoints/matched350m_bpe/matched350m_bpe.pt")
BPE_TOK = ROOT / "checkpoints" / "bpe_8k_real_v2"
OUT = Path("./experiments/paper_v3")
RESEARCH = ROOT / "research"
WINDOW = 2048
BOOT_N = 10_000
BOOT_SEED = 0


def wilson(correct: int, n: int, z: float = 1.96) -> list[float]:
    if n <= 0:
        return [0.0, 0.0]
    p = correct / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, center - half), min(1.0, center + half)]


def mcnemar_exact(hits_a: list[int], hits_b: list[int]) -> dict:
    assert len(hits_a) == len(hits_b)
    n01 = n10 = 0
    for a, b in zip(hits_a, hits_b):
        if a == 0 and b == 1:
            n01 += 1
        elif a == 1 and b == 0:
            n10 += 1
    n_disc = n01 + n10
    if n_disc == 0:
        return {"n01": 0, "n10": 0, "n_discordant": 0, "p_exact": 1.0}
    k = min(n01, n10)
    try:
        from scipy.stats import binom

        p = float(min(1.0, 2.0 * binom.cdf(k, n_disc, 0.5)))
    except Exception:
        from math import comb

        cdf = sum(comb(n_disc, i) for i in range(k + 1)) / (2**n_disc)
        p = float(min(1.0, 2.0 * cdf))
    return {"n01": n01, "n10": n10, "n_discordant": n_disc, "p_exact": p}


def holm_adjust(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj = {}
    prev = 0.0
    for i, (name, p) in enumerate(items):
        a = min(1.0, (m - i) * p)
        a = max(a, prev)
        adj[name] = a
        prev = a
    return adj


def iter_windows(raw: bytes, window: int = WINDOW):
    for i in range(0, len(raw) - window + 1, window):
        yield raw[i : i + window]


@torch.no_grad()
def local64_window(model, window: bytes, device) -> tuple[float, int]:
    x = torch.tensor([list(window)], dtype=torch.long, device=device)
    nats, nb = model.byte_nll(x, None)
    return float(nats), int(nb)


@torch.no_grad()
def bpe_path_frozen(model, tok, window: bytes, device) -> dict:
    """Frozen / recompute_120m_shuf_bpb path: token NLL sum; denom = sum of target-token UTF-8 bytes."""
    text = window.decode("utf-8", errors="replace")
    ids = list(tok.encode(text).ids) or [0]
    raw_bytes = len(window)
    reenc_bytes = len(text.encode("utf-8"))
    if len(ids) < 2:
        return {
            "nats": 0.0,
            "scored_bytes": 0,
            "n_tokens": 0,
            "raw_window_bytes": raw_bytes,
            "reencoded_bytes": reenc_bytes,
            "path": "frozen",
        }
    weights = [len(tok.decode([t]).encode("utf-8")) for t in ids]
    tokens = torch.tensor([ids], dtype=torch.long, device=device)
    w = torch.tensor([weights], dtype=torch.float, device=device)
    inp, tgt = tokens[:, :-1], tokens[:, 1:]
    logits = model(inp)["logits"].float()
    nll = -torch.log_softmax(logits, dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    scored = w[:, 1:] * (tgt != 0).float()
    mask = (scored > 0).float()
    nats = float((nll * mask).sum().item())
    nbytes = int(scored.sum().item())
    ntok = int(mask.sum().item())
    return {
        "nats": nats,
        "scored_bytes": nbytes,
        "n_tokens": ntok,
        "raw_window_bytes": raw_bytes,
        "reencoded_bytes": reenc_bytes,
        "first_token_bytes": int(weights[0]),
        "path": "frozen",
    }


@torch.no_grad()
def bpe_path_paper_v2(model, tok, window: bytes, device) -> dict:
    """paper_v2 path (WRONG): denom = full re-encoded window bytes; num = all target token NLLs."""
    text = window.decode("utf-8", errors="replace")
    n_bytes = len(text.encode("utf-8"))
    ids = list(tok.encode(text).ids)
    if len(ids) < 2:
        return {
            "nats": 0.0,
            "scored_bytes": n_bytes,
            "n_tokens": 0,
            "raw_window_bytes": len(window),
            "reencoded_bytes": n_bytes,
            "path": "paper_v2",
        }
    x = torch.tensor([ids], dtype=torch.long, device=device)
    inp, tgt = x[:, :-1], x[:, 1:]
    out = model(inp, targets=None)
    logp = F.log_softmax(out["logits"].float(), dim=-1)
    nll = -logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return {
        "nats": float(nll.sum().item()),
        "scored_bytes": n_bytes,  # includes first-token bytes not in numerator
        "n_tokens": int(tgt.numel()),
        "raw_window_bytes": len(window),
        "reencoded_bytes": n_bytes,
        "path": "paper_v2",
    }


def load_120_local64(ckpt, device):
    from lct.models.continuous_pool4 import ContinuousPool4LM

    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = blob.get("config") or {}
    size = cfg.get("size") or {}
    model = ContinuousPool4LM(
        {
            "d_model": size.get("d_model", 896),
            "n_heads": size.get("n_heads", 14),
            "n_layers": size.get("layers", 12),
            "ffn_mult": 4,
            "dropout": 0.0,
            "pool_factor": 4,
            "vocab_size": 256,
            "max_len": WINDOW,
            "pos_type": "rope",
            "local_window": cfg.get("local_window", 64),
        }
    )
    state = blob["model"] if "model" in blob else blob
    load_state_dict_strict(model, state, label="120m_local64")
    return model.to(device).eval()


def load_120_bpe(ckpt, device):
    from baselines.bpe_lm import SimpleBPE
    from lct.models.decoder_lm import DecoderOnlyLM

    bpe = SimpleBPE(vocab_size=8000)
    bpe.load(BPE_TOK)
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = DecoderOnlyLM(
        vocab_size=int(bpe.tok.get_vocab_size()),
        d_model=896,
        n_heads=14,
        n_layers=12,
        ffn_mult=4,
        dropout=0.0,
        max_len=WINDOW,
        pos_type="rope",
        ignore_index=0,
    )
    state = blob["model"] if "model" in blob else blob
    load_state_dict_strict(model, state, label="120m_bpe")
    return model.to(device).eval(), bpe


def load_350_local64(device):
    from lct.models.continuous_pool4 import ContinuousPool4LM

    blob = torch.load(LOCAL64_350, map_location="cpu", weights_only=False)
    model = ContinuousPool4LM(
        {
            "d_model": 1024,
            "n_heads": 16,
            "n_layers": 26,
            "ffn_mult": 4,
            "dropout": 0.0,
            "pool_factor": 4,
            "vocab_size": 256,
            "max_len": WINDOW,
            "pos_type": "rope",
            "local_window": 64,
        }
    )
    load_state_dict_strict(model, blob["model"], label="350m_local64")
    return model.to(device).eval()


def summarize_path(rows: list[dict]) -> dict:
    nats = sum(r["nats"] for r in rows)
    byts = sum(r["scored_bytes"] for r in rows)
    toks = sum(r["n_tokens"] for r in rows)
    return {
        "bpb": bpb_from_nats(nats, byts) if byts else None,
        "nats": nats,
        "scored_bytes": byts,
        "n_tokens": toks,
        "n_windows": len(rows),
        "mean_raw_window_bytes": float(np.mean([r["raw_window_bytes"] for r in rows])) if rows else None,
        "mean_reencoded_bytes": float(np.mean([r["reencoded_bytes"] for r in rows])) if rows else None,
    }


def reconcile_120m(device) -> dict:
    print("=== reconcile 120M BPE paths ===", flush=True)
    model, bpe = load_120_bpe(SHUF_BPE, device)
    tok = bpe.tok
    report = {
        "cause_hypothesis": (
            "paper_v2 BPE denom = full re-encoded window bytes (includes first token); "
            "frozen denom = sum of target-token UTF-8 byte lengths only. "
            "Numerator in both is sum of target-token NLLs (frozen also masks id 0)."
        ),
        "domains": {},
        "verdict": None,
    }
    files = {"text": "val_text.bin", "code": "val_code.bin", "math": "val_math.bin"}
    for name, fname in files.items():
        raw = (VAL_DIR / fname).read_bytes()
        frozen_rows, v2_rows = [], []
        for w in iter_windows(raw):
            frozen_rows.append(bpe_path_frozen(model, tok, w, device))
            v2_rows.append(bpe_path_paper_v2(model, tok, w, device))
        fr = summarize_path(frozen_rows)
        v2 = summarize_path(v2_rows)
        # byte accounting diffs
        denom_delta = v2["scored_bytes"] - fr["scored_bytes"]
        nats_delta = v2["nats"] - fr["nats"]
        report["domains"][name] = {
            "frozen": fr,
            "paper_v2": v2,
            "delta_nats_v2_minus_frozen": nats_delta,
            "delta_scored_bytes_v2_minus_frozen": denom_delta,
            "frozen_bpb_minus_v2_bpb": (fr["bpb"] - v2["bpb"]) if fr["bpb"] and v2["bpb"] else None,
        }
        print(
            {
                "domain": name,
                "frozen_bpb": fr["bpb"],
                "v2_bpb": v2["bpb"],
                "frozen_bytes": fr["scored_bytes"],
                "v2_bytes": v2["scored_bytes"],
                "frozen_nats": fr["nats"],
                "v2_nats": v2["nats"],
            },
            flush=True,
        )
    # overall
    for path_key in ("frozen", "paper_v2"):
        nats = sum(report["domains"][d][path_key]["nats"] for d in files)
        byts = sum(report["domains"][d][path_key]["scored_bytes"] for d in files)
        report[f"overall_{path_key}"] = {"bpb": bpb_from_nats(nats, byts), "nats": nats, "scored_bytes": byts}
    report["verdict"] = (
        "frozen_path_correct: denom = target-token UTF-8 bytes matching scored NLLs; "
        "paper_v2 path wrong because denom included unscoreable first-token / full-window bytes "
        "(and decode/reencode can change byte counts, especially on math)."
    )
    report["old_paired_deltas"] = {
        "overall": 0.0125,
        "text": 0.0095,
        "code": 0.0364,
        "math": 0.0137,
        "note": "from paper_v2 paired_bpb_diff.json using wrong BPE path",
    }
    report["frozen_unpaired_deltas"] = {
        "overall": 0.0102,
        "text": 0.0076,
        "code": 0.0365,
        "math": 0.0012,
    }
    path = OUT / "reconcile_120m_bpe_paths.json"
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (RESEARCH / "reconcile_120m_bpe_paths.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    del model
    torch.cuda.empty_cache()
    return report


def collect_aligned(kind: str, model, tok, device) -> dict:
    files = {"text": "val_text.bin", "code": "val_code.bin", "math": "val_math.bin"}
    out = {}
    for name, fname in files.items():
        raw = (VAL_DIR / fname).read_bytes()
        nats_l, bytes_l = [], []
        for w in iter_windows(raw):
            if kind == "local64":
                nats, nb = local64_window(model, w, device)
            else:
                r = bpe_path_frozen(model, tok, w, device)
                nats, nb = r["nats"], r["scored_bytes"]
            if nb <= 0:
                continue
            nats_l.append(nats)
            bytes_l.append(nb)
        out[name] = {"nats": nats_l, "bytes": bytes_l}
        print({"collect": kind, "domain": name, "n": len(nats_l), "bpb": bpb_from_nats(sum(nats_l), sum(bytes_l))}, flush=True)
    return out


def paired_bootstrap_diff(loc: dict, bpe: dict) -> dict:
    out = {}
    rng = np.random.default_rng(BOOT_SEED)
    for domain in ("text", "code", "math", "overall"):
        if domain == "overall":
            nats_l = loc["text"]["nats"] + loc["code"]["nats"] + loc["math"]["nats"]
            bytes_l = loc["text"]["bytes"] + loc["code"]["bytes"] + loc["math"]["bytes"]
            nats_b = bpe["text"]["nats"] + bpe["code"]["nats"] + bpe["math"]["nats"]
            bytes_b = bpe["text"]["bytes"] + bpe["code"]["bytes"] + bpe["math"]["bytes"]
        else:
            nats_l, bytes_l = loc[domain]["nats"], loc[domain]["bytes"]
            nats_b, bytes_b = bpe[domain]["nats"], bpe[domain]["bytes"]
        # Align by window index: drop trailing if lengths differ (should not)
        n = min(len(nats_l), len(nats_b))
        nats_l = np.asarray(nats_l[:n], dtype=np.float64)
        bytes_l = np.asarray(bytes_l[:n], dtype=np.float64)
        nats_b = np.asarray(nats_b[:n], dtype=np.float64)
        bytes_b = np.asarray(bytes_b[:n], dtype=np.float64)
        bpb_l = bpb_from_nats(float(nats_l.sum()), int(bytes_l.sum()))
        bpb_b = bpb_from_nats(float(nats_b.sum()), int(bytes_b.sum()))
        diff = bpb_b - bpb_l
        samples = []
        for _ in range(BOOT_N):
            idx = rng.integers(0, n, size=n)
            dl = bpb_from_nats(float(nats_l[idx].sum()), int(bytes_l[idx].sum()))
            db = bpb_from_nats(float(nats_b[idx].sum()), int(bytes_b[idx].sum()))
            samples.append(db - dl)
        lo, hi = np.quantile(samples, [0.025, 0.975])
        out[domain] = {
            "bpb_local64": bpb_l,
            "bpb_bpe": bpb_b,
            "diff_bpe_minus_local64": diff,
            "diff_lo": float(lo),
            "diff_hi": float(hi),
            "gap_pct": (diff / bpb_b * 100.0) if bpb_b else None,
            "n_windows": n,
            "significant_0_outside_ci": not (lo <= 0.0 <= hi),
            "local64_better": diff > 0,
            "scored_bytes_local64": int(bytes_l.sum()),
            "scored_bytes_bpe": int(bytes_b.sum()),
        }
    return out


def run_paired_120m(device) -> dict:
    print("=== paired 120M with frozen BPE path ===", flush=True)
    loc_m = load_120_local64(SHUF_LOCAL, device)
    loc = collect_aligned("local64", loc_m, None, device)
    del loc_m
    torch.cuda.empty_cache()
    bpe_m, bpe = load_120_bpe(SHUF_BPE, device)
    bpe_d = collect_aligned("bpe", bpe_m, bpe.tok, device)
    del bpe_m
    torch.cuda.empty_cache()
    paired = paired_bootstrap_diff(loc, bpe_d)
    report = {
        "boot_n": BOOT_N,
        "bpe_path": "frozen (target-token byte-weighted)",
        "paired_diff": paired,
        "old_wrong_paired_deltas": {"overall": 0.0125, "text": 0.0095, "code": 0.0364, "math": 0.0137},
        "frozen_unpaired_deltas": {"overall": 0.0102, "text": 0.0076, "code": 0.0365, "math": 0.0012},
    }
    (OUT / "paired_bpb_diff_120m_corrected.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (RESEARCH / "paired_bpb_diff_120m_corrected.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    # also refresh 350M paired using same frozen path for consistency (should match prior)
    print("=== paired 350M with frozen BPE path ===", flush=True)
    loc350 = load_350_local64(device)
    loc_d = collect_aligned("local64", loc350, None, device)
    del loc350
    torch.cuda.empty_cache()
    bpe350, bpe, _, _ = load_matched_bpe(BPE_350, device, BPE_TOK)
    bpe_d350 = collect_aligned("bpe", bpe350, bpe.tok, device)
    del bpe350
    torch.cuda.empty_cache()
    paired350 = paired_bootstrap_diff(loc_d, bpe_d350)
    full = {
        "120m": report,
        "350m": {
            "boot_n": BOOT_N,
            "bpe_path": "frozen (target-token byte-weighted)",
            "paired_diff": paired350,
            "old_frozen_unpaired_gap_pct": {"overall": -2.24, "text": -4.36, "code": 20.63, "math": -4.76},
        },
    }
    (OUT / "paired_bpb_diff.json").write_text(json.dumps(full, indent=2), encoding="utf-8")
    (RESEARCH / "paired_bpb_diff.json").write_text(json.dumps(full, indent=2), encoding="utf-8")
    return full


def _hits_from_details(details) -> list[int]:
    out = []
    for d in details:
        if "hit" in d:
            out.append(int(bool(d["hit"])))
        elif "correct" in d:
            out.append(int(bool(d["correct"])))
        elif "ok" in d:
            out.append(int(bool(d["ok"])))
        else:
            gold = str(d.get("gold", d.get("target", d.get("answer", "")))).strip()
            pred = str(d.get("pred", d.get("prediction", d.get("output", "")))).strip()
            out.append(int(pred == gold or gold.lower() in pred.lower()))
    return out


def run_downstream_items(device) -> dict:
    """Build per-item correctness + McNemar + Holm."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("eval_suite", ROOT / "scripts" / "eval_suite.py")
    suite = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(suite)

    items = {}
    # --- MC from fresh scoring ---
    m_loc = load_350_local64(device)
    m_bpe, bpe, _, _ = load_matched_bpe(BPE_350, device, BPE_TOK)

    def run_mc(name: str):
        ds, parse = suite.load_mc(name)
        lp_l = suite.make_logprob_fn(m_loc, "patch", None)
        lp_b = suite.make_logprob_fn(m_bpe, "bpe", bpe.tok)
        rows = []
        for i, ex in enumerate(ds):
            parsed = parse(ex)
            if parsed is None:
                continue
            ctx, choices, label = parsed
            pred_l, predn_l = suite.score_choices(lp_l, ctx, choices)
            pred_b, predn_b = suite.score_choices(lp_b, ctx, choices)
            rows.append(
                {
                    "i": i,
                    "label": int(label),
                    "local64_raw": int(pred_l == label),
                    "bpe_raw": int(pred_b == label),
                    "local64_lennorm": int(predn_l == label),
                    "bpe_lennorm": int(predn_b == label),
                }
            )
            if (i + 1) % 500 == 0:
                print({"mc": name, "n": i + 1}, flush=True)
        return rows

    for name in ("hellaswag", "piqa", "arc_easy"):
        print({"downstream": name}, flush=True)
        items[name] = run_mc(name)

    # LAMBADA
    print({"downstream": "lambada"}, flush=True)
    ds, parse = suite.load_mc("lambada")
    lamb = []
    for i, ex in enumerate(ds):
        parsed = parse(ex)
        if parsed is None:
            continue
        ctx, choices, _ = parsed
        raw_target = choices if isinstance(choices, str) else choices[0]
        target = raw_target if raw_target.startswith(" ") else (" " + raw_target)
        bare = ctx.rstrip(" ")
        hit_l = int(suite.patch_exact_match(m_loc, bare.encode("utf-8"), target.encode("utf-8")))
        hit_b = int(bool(suite.lambada_bpe_score(m_bpe, bpe.tok, bare, target)["exact_match"]))
        lamb.append({"i": i, "local64": hit_l, "bpe": hit_b})
    items["lambada"] = lamb

    # fact200 from dumps
    fact_loc = Path("./experiments/matched350m_local64_eval/final_base_checks/fact200.json")
    fact_bpe = Path("./experiments/matched350m_bpe_eval/full_benchmarks_fact200_answers.json")
    fl = json.loads(fact_loc.read_text(encoding="utf-8"))
    fb = json.loads(fact_bpe.read_text(encoding="utf-8"))

    def fact_hits(obj):
        for key in ("answers", "details", "items", "examples"):
            if key in obj:
                return _hits_from_details(obj[key])
        # maybe nested
        if "fact200" in obj:
            return fact_hits(obj["fact200"])
        raise KeyError(list(obj.keys())[:20])

    hl, hb = fact_hits(fl), fact_hits(fb)
    assert len(hl) == len(hb), (len(hl), len(hb))
    items["fact200"] = [{"i": i, "local64": hl[i], "bpe": hb[i]} for i in range(len(hl))]
    items["fact200_meta"] = {
        "n": len(hl),
        "local64_acc": sum(hl) / len(hl),
        "bpe_acc": sum(hb) / len(hb),
        "frozen": {"local64": 0.0700, "bpe": 0.1650},
        "source": "existing answer dumps",
    }

    # CharBench few-shot
    cfl = json.loads(Path("./experiments/matched350m_local64_eval/charbench_fewshot.json").read_text())
    cfb = json.loads(Path("./experiments/matched350m_bpe_eval/charbench_fewshot.json").read_text())

    def cb_details(obj):
        if "details" in obj:
            return obj["details"]
        if "fewshot" in obj and "details" in obj["fewshot"]:
            return obj["fewshot"]["details"]
        if "test" in obj and isinstance(obj["test"], dict) and "details" in obj["test"]:
            return obj["test"]["details"]
        raise KeyError(list(obj.keys())[:20])

    dl, db = cb_details(cfl), cb_details(cfb)
    hl, hb = _hits_from_details(dl), _hits_from_details(db)
    assert len(hl) == len(hb)
    items["charbench_fewshot"] = [{"i": i, "local64": hl[i], "bpe": hb[i]} for i in range(len(hl))]
    items["charbench_fewshot_meta"] = {
        "n": len(hl),
        "local64_acc": sum(hl) / len(hl),
        "bpe_acc": sum(hb) / len(hb),
        "frozen": {"local64": 0.2083, "bpe": 0.1783},
    }

    # CharBench equal-FT: try per-item from FT ckpt re-eval, else from result.json
    ft_report = run_charbench_ft_items(device, m_loc=None, m_bpe=None)  # uses FT ckpts
    items["charbench_equal_ft"] = ft_report["items"]
    items["charbench_equal_ft_meta"] = ft_report["meta"]

    # Build stats table
    stats = {}
    pvals = {}

    def pack(name, hits_l, hits_b, key=None):
        n = len(hits_l)
        mc = mcnemar_exact(hits_l, hits_b)
        pvals[key or name] = mc["p_exact"]
        return {
            "n": n,
            "local64_acc": sum(hits_l) / n,
            "bpe_acc": sum(hits_b) / n,
            "local64_wilson": wilson(sum(hits_l), n),
            "bpe_wilson": wilson(sum(hits_b), n),
            "mcnemar": mc,
            "local64_correct": sum(hits_l),
            "bpe_correct": sum(hits_b),
        }

    for name in ("hellaswag", "piqa", "arc_easy"):
        rows = items[name]
        stats[f"{name}_raw"] = pack(
            name,
            [r["local64_raw"] for r in rows],
            [r["bpe_raw"] for r in rows],
            key=f"{name}_raw",
        )
        stats[f"{name}_lennorm"] = pack(
            name,
            [r["local64_lennorm"] for r in rows],
            [r["bpe_lennorm"] for r in rows],
            key=f"{name}_lennorm",
        )
    stats["lambada"] = pack("lambada", [r["local64"] for r in items["lambada"]], [r["bpe"] for r in items["lambada"]])
    stats["fact200"] = pack("fact200", [r["local64"] for r in items["fact200"]], [r["bpe"] for r in items["fact200"]])
    stats["charbench_fewshot"] = pack(
        "charbench_fewshot",
        [r["local64"] for r in items["charbench_fewshot"]],
        [r["bpe"] for r in items["charbench_fewshot"]],
    )
    if items["charbench_equal_ft"]:
        stats["charbench_equal_ft"] = pack(
            "charbench_equal_ft",
            [r["local64"] for r in items["charbench_equal_ft"]],
            [r["bpe"] for r in items["charbench_equal_ft"]],
        )
    else:
        stats["charbench_equal_ft"] = {"error": items.get("charbench_equal_ft_meta", {}), "todo": True}
        # omit from Holm until items exist

    holm = holm_adjust(pvals)
    for k, st in stats.items():
        st["p_raw"] = pvals[k]
        st["p_holm"] = holm[k]
        st["holm_significant"] = holm[k] < 0.01
        if st["holm_significant"]:
            st["bold_winner"] = "local64" if st["local64_acc"] > st["bpe_acc"] else "bpe"
        else:
            st["bold_winner"] = None

    out = {"items": {k: v for k, v in items.items() if not k.endswith("_meta")}, "meta": {k: v for k, v in items.items() if k.endswith("_meta")}, "stats": stats, "holm": holm}
    # Don't dump huge items into research if too large — user asked for downstream_items.json with per-item
    slim_items = {}
    for k, v in out["items"].items():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            # keep only correctness fields
            slim_items[k] = v
        else:
            slim_items[k] = v
    payload = {"stats": stats, "holm": holm, "meta": out["meta"], "items": slim_items}
    (OUT / "downstream_items.json").write_text(json.dumps(payload), encoding="utf-8")
    (RESEARCH / "downstream_items.json").write_text(json.dumps(payload), encoding="utf-8")
    (OUT / "downstream_holm.json").write_text(json.dumps({"stats": stats, "holm": holm}, indent=2), encoding="utf-8")
    (RESEARCH / "downstream_holm.json").write_text(json.dumps({"stats": stats, "holm": holm}, indent=2), encoding="utf-8")
    return payload


def run_charbench_ft_items(device, m_loc=None, m_bpe=None) -> dict:
    """Re-score CharBench test from matched FT checkpoints (few_shot=False) via score_charbench."""
    import importlib.util

    ft_loc = Path("${HOME}/charbench_lr_sweep_matched_local64/lr_5e-05/local64_ft_lr5e-05.pt")
    ft_bpe = Path("${HOME}/charbench_lr_sweep_matched_bpe/lr_5e-05/bpe_ft_lr5e-05.pt")
    # host-b may hold the matched local64 FT; copy to NAS-local cache if missing
    if not ft_loc.exists():
        nas_copy = Path("./experiments/paper_v3/local64_ft_lr5e-05.pt")
        if nas_copy.exists():
            ft_loc = nas_copy
        else:
            cands = list(Path("${HOME}").glob("**/charbench_lr_sweep_matched_local64/**/local64_ft_lr5e-05.pt"))
            if cands:
                ft_loc = cands[0]
    frozen = {"local64": 0.5367, "bpe": 0.4000}
    meta = {"frozen": frozen, "ft_ckpt_local64": str(ft_loc), "ft_ckpt_bpe": str(ft_bpe), "exists": {"local64": ft_loc.exists(), "bpe": ft_bpe.exists()}}
    data_dir = Path("./data/charbench")
    if not ft_loc.exists() or not ft_bpe.exists():
        meta["error"] = "FT checkpoints missing; cannot build per-item FT hits"
        return {"items": [], "meta": meta}

    spec = importlib.util.spec_from_file_location("score_charbench", ROOT / "scripts" / "charbench" / "score_charbench.py")
    scorer = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(scorer)

    print({"charbench_ft": "score_local64", "ckpt": str(ft_loc)}, flush=True)
    loc_out = scorer.score_local64(ft_loc, data_dir, chat=False, max_new=32, few_shot=False)
    print({"charbench_ft": "score_bpe", "ckpt": str(ft_bpe)}, flush=True)
    bpe_out = scorer.score_bpe(ft_bpe, BPE_TOK, data_dir, max_new=32, few_shot=False)

    hl = [int(bool(d["correct"])) for d in loc_out["details"]]
    hb = [int(bool(d["correct"])) for d in bpe_out["details"]]
    assert len(hl) == len(hb) == 600
    meta["local64_rescored_acc"] = loc_out["accuracy"]
    meta["bpe_rescored_acc"] = bpe_out["accuracy"]
    meta["scores_match_frozen"] = {
        "local64": abs(loc_out["accuracy"] - frozen["local64"]) < 1e-3,
        "bpe": abs(bpe_out["accuracy"] - frozen["bpe"]) < 1e-3,
    }
    if not all(meta["scores_match_frozen"].values()):
        meta["warning"] = "rescored FT accuracy differs from frozen; reporting both"
    (OUT / "charbench_ft_local64.json").write_text(json.dumps(loc_out), encoding="utf-8")
    (OUT / "charbench_ft_bpe.json").write_text(json.dumps(bpe_out), encoding="utf-8")
    return {"items": [{"i": i, "local64": hl[i], "bpe": hb[i]} for i in range(len(hl))], "meta": meta}


def doc_overlap_10b() -> dict:
    """Verify both 350M twins consumed the same documents in the same order for first 10B."""
    print("=== document overlap 10B ===", flush=True)
    # Shared permutation file
    perm_cands = [
        Path("./data/real_v3/doc_perm_seed0.npy"),
        Path("${HOME}/data/real_v3/doc_perm_seed0.npy"),
        ROOT / "data" / "real_v3" / "doc_perm_seed0.npy",
    ]
    spans_cands = [
        Path("./data/real_v3/doc_spans.npy"),
        Path("${HOME}/data/real_v3/doc_spans.npy"),
    ]
    perm = next((p for p in perm_cands if p.exists()), None)
    spans = next((p for p in spans_cands if p.exists()), None)
    report = {"perm": str(perm), "spans": str(spans)}
    if perm is None or spans is None:
        report["error"] = "perm/spans missing"
        # Fall back to comparing cursor logs / checkpoint cursor fields
        report.update(_overlap_from_checkpoints())
    else:
        perm_arr = np.load(perm)
        spans_arr = np.load(spans)  # shape (n_docs, 2) start/end
        # Walk permutation until 10B bytes
        target = 10_000_000_000
        # bounds for matched twins from training scripts
        bounds = (59_500_002_824, 66_500_006_098)  # FINDINGS / train scripts
        report["bounds"] = list(bounds)
        docs = []
        total = 0
        for di in perm_arr:
            di = int(di)
            start, end = int(spans_arr[di, 0]), int(spans_arr[di, 1])
            # only docs intersecting bounds? ShuffledDocCursor uses order over all docs then clips to bounds
            # Matched training used bounds on the byte stream of the packed file
            # Approximate: count docs whose span overlaps [bounds[0], bounds[0]+10B)
            if end <= bounds[0] or start >= bounds[1]:
                continue
            # clipped length inside window
            a = max(start, bounds[0])
            b = min(end, bounds[1])
            if b <= a:
                continue
            docs.append(di)
            total += b - a
            if total >= target:
                break
        report["n_docs_in_first_10b_slice"] = len(docs)
        report["bytes_accounted"] = total
        report["doc_id_head"] = docs[:20]
        report["doc_id_tail"] = docs[-20:]
        # Check both twins used same perm seed / bounds from launch scripts / ckpt config
        report.update(_overlap_from_checkpoints())
        report["same_document_order"] = report.get("local64_perm_seed") == report.get("bpe_perm_seed") and report.get(
            "local64_bounds"
        ) == report.get("bpe_bounds")
    (OUT / "doc_overlap_10b.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (RESEARCH / "doc_overlap_10b.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def _overlap_from_checkpoints() -> dict:
    out = {}
    for label, path in [("local64", LOCAL64_350), ("bpe", BPE_350)]:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        cfg = blob.get("config") or blob.get("args") or {}
        out[f"{label}_keys"] = list(blob.keys())[:30]
        out[f"{label}_seen"] = blob.get("seen")
        out[f"{label}_step"] = blob.get("step")
        out[f"{label}_cursor"] = blob.get("cursor") or blob.get("cursors")
        if isinstance(cfg, dict):
            out[f"{label}_bounds"] = cfg.get("bounds") or cfg.get("byte_bounds")
            out[f"{label}_perm_seed"] = cfg.get("perm_seed") or cfg.get("seed")
            out[f"{label}_data"] = cfg.get("data") or cfg.get("train_bin")
        # search nested
        for k in ("bounds", "perm_seed", "doc_perm", "order_path"):
            if k in blob:
                out[f"{label}_{k}"] = blob[k]
    # Compare launch scripts on disk
    for script in [
        ROOT / "scripts" / "_remote_anneal_then_matched.sh",
        ROOT / "scripts" / "run_matched_bpe_pretok_resume.sh",
        ROOT / "scripts" / "train_bpe_matched350m.py",
    ]:
        if script.exists():
            txt = script.read_text(encoding="utf-8", errors="ignore")
            if "bounds" in txt or "perm" in txt:
                out.setdefault("script_refs", []).append(str(script))
    return out


def make_gap_figure():
    """Update Fig2 with BPB curves + gap panel from curve_350m_bpb.json."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curve_path = RESEARCH / "curve_350m_bpb.json"
    if not curve_path.exists():
        curve_path = Path("./experiments/paper_v2/curve_350m_bpb.json")
    curve = json.loads(curve_path.read_text(encoding="utf-8"))

    def series(kind):
        xs, ys = [], []
        for pt in curve[kind]:
            seen = pt.get("seen") or 0
            xs.append(seen / 1e9)
            ys.append(pt["domains"]["overall"]["bpb"])
        return xs, ys

    x_l, y_l = series("local64")
    x_b, y_b = series("bpe")
    # gap at matched tags
    gaps_x, gaps_y = [], []
    by_tag_l = {pt["tag"]: pt for pt in curve["local64"]}
    by_tag_b = {pt["tag"]: pt for pt in curve["bpe"]}
    for tag in ("2b", "4b", "6b", "8b", "10b"):
        if tag in by_tag_l and tag in by_tag_b:
            bl = by_tag_l[tag]["domains"]["overall"]["bpb"]
            bb = by_tag_b[tag]["domains"]["overall"]["bpb"]
            seen = by_tag_l[tag].get("seen") or int(tag[:-1]) * 1e9
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
    ax.set_ylabel(r"BPB gap (BPE $-$ local64)")
    ax.set_title("Gap vs bytes")
    ax.set_xlim(0, 10.5)
    ax.grid(True, alpha=0.3)
    # annotate 2B and 10B
    if len(gaps_y) >= 2:
        ax.annotate(f"{gaps_y[0]:.3f}", (gaps_x[0], gaps_y[0]), textcoords="offset points", xytext=(6, 6), fontsize=8)
        ax.annotate(f"{gaps_y[-1]:.3f}", (gaps_x[-1], gaps_y[-1]), textcoords="offset points", xytext=(6, 6), fontsize=8)

    fig.tight_layout()
    out_png = OUT / "curve_350m_bpb_gap.png"
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    fig.savefig(RESEARCH / "curve_350m_bpb_gap.png", dpi=160)
    paper_fig = ROOT / "paper" / "figures" / "matched_350m_val_bpb.png"
    paper_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(paper_fig, dpi=160)
    gap_note = {
        "gap_2b": gaps_y[0] if gaps_y else None,
        "gap_10b": gaps_y[-1] if gaps_y else None,
        "narrowing": (gaps_y[0] - gaps_y[-1]) if len(gaps_y) >= 2 else None,
        "points": list(zip(gaps_x, gaps_y)),
    }
    (RESEARCH / "curve_350m_gap_summary.json").write_text(json.dumps(gap_note, indent=2), encoding="utf-8")
    print({"gap_figure": gap_note}, flush=True)
    return gap_note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stage",
        choices=["all", "reconcile", "paired", "downstream", "overlap", "figure"],
        default="all",
    )
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    RESEARCH.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print({"device": str(device), "stage": args.stage}, flush=True)

    if args.stage in ("all", "reconcile"):
        reconcile_120m(device)
    if args.stage in ("all", "paired"):
        run_paired_120m(device)
    if args.stage in ("all", "downstream"):
        run_downstream_items(device)
    if args.stage in ("all", "overlap"):
        doc_overlap_10b()
    if args.stage in ("all", "figure"):
        make_gap_figure()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
