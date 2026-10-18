"""Rate–distortion and codebook utilization metrics for tokenizer ablations."""

from __future__ import annotations

from typing import Any

import math
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast


@torch.no_grad()
def evaluate_tokenizer_rd(
    model,
    loader,
    device: torch.device,
    dtype: torch.dtype | None = None,
    max_batches: int = 40,
) -> dict[str, float]:
    """Report reconstruction + compression + utilization metrics."""
    model.eval()
    use_amp = dtype is not None and device.type == "cuda"
    k = model.quantizer.codebook_size
    usage = torch.zeros(k, device=device)
    total_ce = 0.0
    total_exact = 0.0
    total_bytes = 0
    total_concepts = 0
    n = 0

    for batch in loader:
        x = batch["bytes"].to(device)
        with autocast(enabled=use_amp, dtype=dtype if use_amp else torch.float32):
            out = model(x)
        pred = out["logits"].argmax(dim=-1)
        total_ce += float(out["recon_loss"]) * x.numel()
        total_exact += float((pred == x).float().sum())
        total_bytes += int(x.numel())
        idx = out["indices"]
        total_concepts += int(idx.numel())
        usage += torch.bincount(idx.reshape(-1), minlength=k).float()
        n += 1
        if n >= max_batches:
            break

    if total_bytes == 0:
        return {}

    recon_ce = total_ce / total_bytes
    recon_bpb = recon_ce / math.log(2)
    cpb = total_concepts / total_bytes
    bpc = total_bytes / max(total_concepts, 1)

    usage_cpu = usage.cpu()
    active = int((usage_cpu > 0).sum())
    p = usage_cpu / usage_cpu.sum().clamp_min(1.0)
    p_nz = p[p > 0]
    entropy = float(-(p_nz * p_nz.log()).sum())
    perplexity = float(math.exp(entropy)) if entropy < 80 else float("inf")

    return {
        "recon_ce": recon_ce,
        "recon_bpb": recon_bpb,
        "exact_match": float(total_exact / total_bytes),
        "concepts_per_byte": cpb,
        "avg_bytes_per_concept": bpc,
        "active_codes": active,
        "codebook_size": k,
        "active_code_fraction": active / max(k, 1),
        "usage_entropy": entropy,
        "codebook_perplexity": perplexity,
        "effective_codebook_size": perplexity,
        "pool_factor": float(getattr(model, "pool_factor", 1)),
        "n_batches": n,
        "total_bytes": total_bytes,
    }


@torch.no_grad()
def evaluate_adaptive_rd(
    model,
    loader,
    device: torch.device,
    dtype: torch.dtype | None = None,
    max_batches: int = 40,
) -> dict[str, Any]:
    """Adaptive-patch metrics: variable-length concepts/byte + patch histogram."""
    model.eval()
    use_amp = dtype is not None and device.type == "cuda"
    k = model.quantizer.codebook_size
    usage = torch.zeros(k, device=device)
    total_ce = 0.0
    total_exact = 0.0
    total_bytes = 0
    total_concepts = 0
    all_patch_lens: list[int] = []
    bound_ents: list[float] = []
    bound_rates: list[float] = []
    n = 0

    for batch in loader:
        x = batch["bytes"].to(device)
        with autocast(enabled=use_amp, dtype=dtype if use_amp else torch.float32):
            out = model(x)
        pred = out["logits"].argmax(dim=-1)
        total_ce += float(out["recon_loss"]) * x.numel()
        total_exact += float((pred == x).float().sum())
        total_bytes += int(x.numel())
        total_concepts += int(out["lengths"].sum().item())
        for row in out["patch_lens"]:
            all_patch_lens.extend(row)
        for i, L in enumerate(out["lengths"].tolist()):
            usage += torch.bincount(out["indices"][i, :L], minlength=k).float()
        bound_ents.append(float(out["boundary_entropy"]))
        bound_rates.append(float(out["boundary_rate"]))
        n += 1
        if n >= max_batches:
            break

    recon_ce = total_ce / max(total_bytes, 1)
    recon_bpb = recon_ce / math.log(2)
    cpb = total_concepts / max(total_bytes, 1)
    usage_cpu = usage.cpu()
    active = int((usage_cpu > 0).sum())
    p = usage_cpu / usage_cpu.sum().clamp_min(1.0)
    p_nz = p[p > 0]
    entropy = float(-(p_nz * p_nz.log()).sum()) if p_nz.numel() else 0.0
    perplexity = float(math.exp(min(entropy, 80)))
    hist: dict[str, int] = {}
    for L in all_patch_lens:
        hist[str(L)] = hist.get(str(L), 0) + 1
    avg_patch = float(sum(all_patch_lens) / max(len(all_patch_lens), 1))
    return {
        "recon_ce": recon_ce,
        "recon_bpb": recon_bpb,
        "exact_match": float(total_exact / max(total_bytes, 1)),
        "concepts_per_byte": cpb,
        "avg_bytes_per_concept": 1.0 / max(cpb, 1e-9),
        "avg_patch_len": avg_patch,
        "patch_len_hist": hist,
        "boundary_entropy": float(sum(bound_ents) / max(len(bound_ents), 1)),
        "boundary_rate": float(sum(bound_rates) / max(len(bound_rates), 1)),
        "active_codes": active,
        "codebook_size": k,
        "active_code_fraction": active / max(k, 1),
        "usage_entropy": entropy,
        "codebook_perplexity": perplexity,
        "n_batches": n,
        "total_bytes": total_bytes,
    }


def cpb_in_band(cpb: float, target: float, tol: float = 0.03) -> bool:
    return abs(cpb - target) <= tol


def in_success_band(cpb: float) -> bool:
    """Pre-registered Phase 1C follow-up bands: [0.11,0.14] or [0.22,0.28]."""
    return (0.11 <= cpb <= 0.14) or (0.22 <= cpb <= 0.28)


def matched_cpb_comparison(
    metrics: dict[str, Any],
    fixed_baselines: list[dict[str, Any]],
    target_cpb: float,
    random_metrics: dict[str, Any] | None = None,
    tol: float = 0.03,
) -> dict[str, Any]:
    """Compare adaptive BPB to fixed-stride at the intended target rate."""
    if not fixed_baselines:
        return {
            "target_cpb": target_cpb,
            "actual_cpb": float(metrics["concepts_per_byte"]),
            "beats_fixed_at_matched_cpb": None,
        }
    if abs(target_cpb - 0.25) < abs(target_cpb - 0.125):
        fixed = min(fixed_baselines, key=lambda b: abs(b["concepts_per_byte"] - 0.25))
    else:
        fixed = min(fixed_baselines, key=lambda b: abs(b["concepts_per_byte"] - 0.125))

    cpb = float(metrics["concepts_per_byte"])
    bpb = float(metrics["recon_bpb"])
    out = {
        "target_cpb": target_cpb,
        "actual_cpb": cpb,
        "cpb_within_tol": cpb_in_band(cpb, target_cpb, tol),
        "in_success_band": in_success_band(cpb),
        "fixed_baseline": fixed.get("name", "unknown"),
        "fixed_cpb": fixed["concepts_per_byte"],
        "fixed_recon_bpb": fixed["recon_bpb"],
        "beats_fixed_at_matched_cpb": bool(bpb < fixed["recon_bpb"] and cpb_in_band(cpb, target_cpb, tol)),
    }
    if random_metrics is not None:
        out["random_recon_bpb"] = random_metrics["recon_bpb"]
        out["beats_random_at_matched_cpb"] = bool(
            bpb < random_metrics["recon_bpb"] and cpb_in_band(cpb, target_cpb, tol)
        )
    return out


def rate_distortion_point(metrics: dict[str, float]) -> dict[str, float]:
    return {
        "rate_concepts_per_byte": metrics.get("concepts_per_byte", 0.0),
        "distortion_recon_bpb": metrics.get("recon_bpb", 0.0),
        "active_fraction": metrics.get("active_code_fraction", 0.0),
        "perplexity": metrics.get("codebook_perplexity", 0.0),
    }


@torch.no_grad()
def evaluate_factorized_tokenizer_rd(
    model,
    loader,
    device: torch.device,
    dtype: torch.dtype | None = None,
    max_batches: int = 40,
    decoder_mode: str = "prev1",
) -> dict[str, Any]:
    """Multi-stage RD metrics for FactorizedPatchTokenizer (no K^M marginal)."""
    model.eval()
    use_amp = dtype is not None and device.type == "cuda"
    total_ce = 0.0
    total_exact = 0.0
    total_exact_patches = 0.0
    total_patches = 0
    total_bytes = 0
    n = 0
    stage_usage: list[torch.Tensor] | None = None
    residual_acc: list[float] | None = None
    residual_n = 0
    k = int(getattr(model, "codebook_size", 256))
    m_stages = int(getattr(model, "num_stages", 1))
    p = int(getattr(model, "pool_factor", 4))

    for batch in loader:
        x = batch["bytes"].to(device)
        mask = batch.get("byte_mask_all", batch.get("byte_mask"))
        if mask is not None:
            mask = mask.to(device)
        with autocast(enabled=use_amp, dtype=dtype if use_amp else torch.float32):
            out = model(x, byte_mask=mask, decoder_mode=decoder_mode)
        logits = out["logits"]
        usable = logits.size(1)
        tgt = x[:, :usable]
        pred = logits.argmax(dim=-1)
        if mask is not None:
            m = mask[:, :usable].float()
            n_bytes = int(m.sum().item())
            total_ce += float(
                F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    tgt.masked_fill(mask[:, :usable] == 0, -100).reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                ).item()
            )
            total_exact += float(((pred == tgt).float() * m).sum().item())
            total_bytes += n_bytes
        else:
            n_bytes = int(tgt.numel())
            total_ce += float(out["recon_loss"]) * n_bytes
            total_exact += float((pred == tgt).float().sum().item())
            total_bytes += n_bytes

        tc = usable // p
        if tc > 0:
            patch_ok = (pred[:, : tc * p].view(-1, p) == tgt[:, : tc * p].view(-1, p)).all(dim=-1).float()
            total_exact_patches += float(patch_ok.sum().item())
            total_patches += int(patch_ok.numel())

        idx = out["indices"]
        if idx.dim() == 2:
            idx = idx.unsqueeze(-1)
        m_stages = idx.size(-1)
        if stage_usage is None:
            stage_usage = [torch.zeros(k, device=device) for _ in range(m_stages)]
        for s in range(m_stages):
            stage_usage[s] += torch.bincount(idx[..., s].reshape(-1), minlength=k).float()

        rn = out.get("residual_norms") or []
        if rn:
            if residual_acc is None:
                residual_acc = [0.0] * len(rn)
            for i, v in enumerate(rn):
                residual_acc[i] += float(v)
            residual_n += 1

        n += 1
        if n >= max_batches:
            break

    recon_ce = total_ce / max(total_bytes, 1)
    recon_bpb = recon_ce / math.log(2)
    stages_out = []
    if stage_usage is not None:
        for s, usage in enumerate(stage_usage):
            u = usage.cpu()
            active = int((u > 0).sum())
            probs = u / u.sum().clamp_min(1.0)
            p_nz = probs[probs > 0]
            ent = float(-(p_nz * p_nz.log()).sum()) if p_nz.numel() else 0.0
            # usage_entropy above is in nats (natural log); also report bits
            ent_bits = ent / math.log(2)
            ppl = float(math.exp(min(ent, 80)))
            ppl_bits = float(2.0 ** min(ent_bits, 40))
            stages_out.append(
                {
                    "stage": s,
                    "active_codes": active,
                    "active_fraction": active / max(k, 1),
                    "usage_entropy": ent,
                    "usage_entropy_bits": ent_bits,
                    "perplexity": ppl,
                    "perplexity_bits": ppl_bits,
                    "residual_norm_mean": (residual_acc[s] / max(residual_n, 1)) if residual_acc and s < len(residual_acc) else None,
                }
            )

    nominal = float(m_stages * math.log2(max(k, 2)))
    effective_bits = float(sum(s["usage_entropy_bits"] for s in stages_out)) if stages_out else 0.0
    return {
        "recon_ce": recon_ce,
        "recon_bpb": recon_bpb,
        "exact_match": float(total_exact / max(total_bytes, 1)),
        "exact_patch_match": float(total_exact_patches / max(total_patches, 1)),
        "concepts_per_byte": 1.0 / max(p, 1),
        "ids_per_patch": m_stages,
        "avg_bytes_per_concept": float(p),
        "nominal_bits_per_patch": nominal,
        "effective_bits_per_patch": effective_bits,
        "effective_over_nominal": effective_bits / max(nominal, 1e-9),
        "codebook_size": k,
        "num_stages": m_stages,
        "pool_factor": float(p),
        "decoder_mode": decoder_mode,
        "stage_metrics": stages_out,
        "residual_norms": [v / max(residual_n, 1) for v in residual_acc] if residual_acc else [],
        "n_batches": n,
        "total_bytes": total_bytes,
    }


@torch.no_grad()
def evaluate_factorized_decoder_context(
    model,
    loader,
    device: torch.device,
    dtype: torch.dtype | None = None,
    max_batches: int = 40,
    modes: tuple[str, ...] = ("code_only", "prev1"),
) -> dict[str, Any]:
    """Packed decoder BPB under code_only and prev1 (diagnostic; not end-to-end LM)."""
    out: dict[str, Any] = {}
    for mode in modes:
        metrics = evaluate_factorized_tokenizer_rd(
            model, loader, device, dtype, max_batches=max_batches, decoder_mode=mode
        )
        out[mode] = {
            "recon_bpb": metrics["recon_bpb"],
            "exact_match": metrics["exact_match"],
            "exact_patch_match": metrics["exact_patch_match"],
            "total_bytes": metrics["total_bytes"],
            "n_batches": metrics["n_batches"],
        }
    if "code_only" in out and "prev1" in out:
        co = out["code_only"]["recon_bpb"]
        p1 = out["prev1"]["recon_bpb"]
        out["prev1_delta_vs_code_only"] = float(co - p1)
        out["prev1_rel_reduction"] = float((co - p1) / max(co, 1e-9))
    return out

