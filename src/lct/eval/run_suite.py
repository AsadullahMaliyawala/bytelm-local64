from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from lct.data.dataset import make_loader, text_to_bytes
from lct.models import ConceptTokenizer
from lct.train.concept_lm import evaluate_concept_lm, load_tokenizer
from lct.models import ConceptLM
from lct.utils import get_device, save_json


@torch.no_grad()
def analyze_codebook(tokenizer: ConceptTokenizer, loader, device, top_k: int = 50) -> dict[str, Any]:
    """Collect top activating spans per frequent code."""
    tokenizer.eval()
    from collections import defaultdict

    code_spans: dict[int, list[str]] = defaultdict(list)
    usage = torch.zeros(tokenizer.quantizer.codebook_size)

    for batch in loader:
        x = batch["bytes"].to(device)
        texts = batch["text"]
        idx = tokenizer.tokenize(x)
        usage += torch.bincount(idx.reshape(-1).cpu(), minlength=usage.numel()).float()
        for b in range(idx.size(0)):
            ids = idx[b].cpu().tolist()
            raw = bytes(x[b].cpu().tolist())
            for t, c in enumerate(ids):
                if len(code_spans[c]) >= 20:
                    continue
                # local byte window around t
                lo = max(0, t - 4)
                hi = min(len(raw), t + 5)
                try:
                    span = raw[lo:hi].decode("utf-8", errors="replace")
                except Exception:
                    span = repr(raw[lo:hi])
                code_spans[c].append(span)
        if usage.sum() > 100000:
            break

    top = torch.topk(usage, k=min(top_k, usage.numel()))
    report = {
        "top_codes": [
            {"code": int(i), "count": float(usage[i]), "spans": code_spans[int(i)][:10]}
            for i in top.indices.tolist()
        ],
        "used_codes": int((usage > 0).sum()),
        "codebook_size": int(usage.numel()),
        "usage_entropy": float(
            -(
                (usage / usage.sum().clamp_min(1))
                * ((usage / usage.sum().clamp_min(1)) + 1e-12).log()
            )
            .sum()
        ),
    }
    return report


def compression_stats(tokenizer: ConceptTokenizer, texts: list[str], seq_len: int, device) -> dict[str, float]:
    """Concepts-per-byte (here 1:1 by design) vs naive char count."""
    total_bytes = 0
    total_concepts = 0
    for t in texts:
        ids = text_to_bytes(t, seq_len)
        total_bytes += len(ids)
        x = torch.tensor([ids], device=device)
        z = tokenizer.tokenize(x)
        total_concepts += z.numel()
    return {
        "concepts_per_byte": total_concepts / max(total_bytes, 1),
        "total_bytes": total_bytes,
        "total_concepts": total_concepts,
    }


def run_suite(cfg: dict[str, Any], phase: str = "all") -> dict[str, Any]:
    device = get_device(cfg.get("device", "cuda"))
    results: dict[str, Any] = {"config_hash": cfg.get("_config_hash"), "phase": phase}
    out_dir = Path(cfg.get("train", {}).get("experiment_dir", "experiments")) / "eval_suite"
    out_dir.mkdir(parents=True, exist_ok=True)

    phase1 = Path("checkpoints/phase1_best.pt")
    if phase1.exists() and phase in ("all", "tokenizer", "1"):
        tok = load_tokenizer(phase1, device)
        loader = make_loader(cfg, "val", shuffle=False)
        report = analyze_codebook(tok, loader, device)
        Path("analysis").mkdir(exist_ok=True)
        with open("analysis/codebook_report.md", "w", encoding="utf-8") as f:
            f.write("# Codebook analysis\n\n")
            f.write(f"- Used codes: {report['used_codes']} / {report['codebook_size']}\n")
            f.write(f"- Usage entropy: {report['usage_entropy']:.4f}\n\n")
            f.write("## Top codes\n\n")
            for row in report["top_codes"][:20]:
                f.write(f"### Code {row['code']} (count={row['count']:.0f})\n")
                for s in row["spans"][:5]:
                    f.write(f"- `{s!r}`\n")
                f.write("\n")
        # export codebook embeddings
        import numpy as np

        np.save("checkpoints/codebook_embeddings.npy", tok.quantizer.codebook.weight.detach().cpu().numpy())
        results["tokenizer"] = report

    phase2 = Path("checkpoints/phase2_concept_lm.pt")
    if phase2.exists() and phase in ("all", "lm", "2"):
        tok = load_tokenizer(phase1, device)
        model = ConceptLM(tok, cfg).to(device)
        ckpt = torch.load(phase2, map_location=device, weights_only=False)
        model.lm.load_state_dict(ckpt["model"], strict=False)
        loader = make_loader(cfg, "val", shuffle=False)
        results["concept_lm"] = evaluate_concept_lm(model, loader, device, None)

    save_json(results, out_dir / "results.json")
    # also write compare_phase2 if both present
    if "concept_lm" in results:
        compare = {"concept_lm": results["concept_lm"]}
        bpe_res = Path("experiments/bpe_lm_v1/results.json")
        if bpe_res.exists():
            with bpe_res.open(encoding="utf-8") as f:
                compare["bpe_lm"] = json.load(f)
        save_json(compare, Path("experiments/eval_suite/compare_phase2.json"))
        Path("eval").mkdir(exist_ok=True)
        save_json(compare, Path("eval/compare_phase2.json"))
    return results