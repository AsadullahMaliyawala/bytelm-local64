"""Fair language-model metrics: byte-normalized BPB with optional content masks."""

from __future__ import annotations

import math
import time
from typing import Any

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

from lct.utils import count_parameters

LN2 = math.log(2)


@torch.no_grad()
def peak_vram_mb() -> float | None:
    if not torch.cuda.is_available():
        return None
    return float(torch.cuda.max_memory_allocated() / (1024 * 1024))


def param_report(lm: torch.nn.Module, tokenizer: torch.nn.Module | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "lm_total": count_parameters(lm, non_embedding=False),
        "lm_core": count_parameters(lm, non_embedding=True),
        "lm_total_all": sum(p.numel() for p in lm.parameters()),
    }
    if tokenizer is not None:
        out["tokenizer_params"] = sum(p.numel() for p in tokenizer.parameters())
        out["end_to_end_params"] = out["lm_total_all"] + out["tokenizer_params"]
    else:
        out["tokenizer_params"] = 0
        out["end_to_end_params"] = out["lm_total_all"]
    return out


def masked_nll_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None,
) -> tuple[float, int]:
    """Sum NLL nats over positions where mask==1. logits/targets: (B, T, ...) / (B, T)."""
    # logits: (B, T, V), targets: (B, T), mask: (B, T) aligned with targets
    log_probs = F.log_softmax(logits.float(), dim=-1)
    nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (B, T)
    if mask is None:
        return float(nll.sum().item()), int(nll.numel())
    m = mask.float()
    return float((nll * m).sum().item()), int(m.sum().item())


@torch.no_grad()
def evaluate_byte_lm_bpb(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    dtype: torch.dtype | None = None,
    max_batches: int = 40,
    use_byte_mask: bool = True,
    legacy_pad_mask: bool = False,
    mask_key: str = "byte_mask",
) -> dict[str, float]:
    """Teacher-forced byte LM BPB. Prefer content-only mask on targets x[t+1].

    Target-shift: inp = x[:, :-1], tgt = x[:, 1:], score where mask[:, 1:]==1.

    ``mask_key``: ``byte_mask`` / ``byte_mask_content`` (exclude sep) or
    ``byte_mask_all`` (include sep; Phase 2B.1b primary).
    """
    model.eval()
    use_amp = dtype is not None and device.type == "cuda"
    total_nats = 0.0
    total_bytes = 0
    n = 0
    t0 = time.perf_counter()
    for batch in loader:
        if isinstance(batch, dict):
            x = batch["bytes"].to(device)
            mask = batch.get(mask_key)
            if mask is None and mask_key != "byte_mask":
                mask = batch.get("byte_mask")
            if mask is not None:
                mask = mask.to(device)
            elif legacy_pad_mask:
                from lct.data.packed_stream import content_mask_from_legacy_padding

                mask = content_mask_from_legacy_padding(x).to(device)
            elif not use_byte_mask:
                mask = None
            else:
                mask = None  # unmasked (legacy smoke)
        else:
            x = batch.to(device)
            mask = None
        inp, tgt = x[:, :-1], x[:, 1:]
        tgt_mask = mask[:, 1:] if mask is not None else None
        with autocast(enabled=use_amp, dtype=dtype if use_amp else torch.float32):
            out = model(inp, targets=None)
        nats, nb = masked_nll_from_logits(out["logits"], tgt, tgt_mask)
        total_nats += nats
        total_bytes += nb
        n += 1
        if n >= max_batches:
            break
    elapsed = time.perf_counter() - t0
    mean_ce = total_nats / max(total_bytes, 1)
    return {
        "loss": mean_ce,
        "bpb": mean_ce / LN2,
        "n_bytes": total_bytes,
        "n_batches": n,
        "eval_seconds": elapsed,
        "bytes_per_sec": total_bytes / max(elapsed, 1e-9),
        "masked": bool(use_byte_mask or legacy_pad_mask),
        "mask_key": mask_key,
    }


@torch.no_grad()
def evaluate_byte_lm_bpb_dual(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    dtype: torch.dtype | None = None,
    max_batches: int = 40,
) -> dict[str, Any]:
    """Primary (all bytes incl. sep) + secondary (content-only) packed BPB."""
    primary = evaluate_byte_lm_bpb(
        model, loader, device, dtype, max_batches=max_batches, mask_key="byte_mask_all"
    )
    secondary = evaluate_byte_lm_bpb(
        model, loader, device, dtype, max_batches=max_batches, mask_key="byte_mask_content"
    )
    # Fallback if dual masks absent: content == legacy byte_mask
    if primary["n_bytes"] == 0:
        primary = evaluate_byte_lm_bpb(
            model, loader, device, dtype, max_batches=max_batches, mask_key="byte_mask"
        )
    if secondary["n_bytes"] == 0:
        secondary = evaluate_byte_lm_bpb(
            model, loader, device, dtype, max_batches=max_batches, mask_key="byte_mask"
        )
    return {
        "bpb_all_bytes": primary["bpb"],
        "bpb_content_only": secondary["bpb"],
        "primary": primary,
        "secondary": secondary,
    }


@torch.no_grad()
def evaluate_byte_lm_on_tensor(
    model: torch.nn.Module,
    x: torch.Tensor,
    byte_mask: torch.Tensor | None = None,
    *,
    score_positions: torch.Tensor | None = None,
    start_pos: int = 0,
    dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    """Teacher-forced byte LM NLL on a single (B, T) tensor with optional PE offset.

    Target-shift: inp = x[:, :-1], tgt = x[:, 1:].
    ``byte_mask`` / ``score_positions`` are aligned with full ``x`` indices; scoring
    uses the shifted target indices (positions 1..T-1 of the original span).

    Args:
        score_positions: optional bool/long mask over full sequence length T;
            only positions where both score_positions[t]==1 and (byte_mask is None
            or byte_mask[t]==1) contribute for target index t (t >= 1).
        start_pos: absolute positional encoding offset for the input tokens.
    """
    model.eval()
    device = next(model.parameters()).device
    use_amp = dtype is not None and device.type == "cuda"
    if x.dim() == 1:
        x = x.unsqueeze(0)
    x = x.to(device)
    B, T = x.shape
    if byte_mask is not None:
        if byte_mask.dim() == 1:
            byte_mask = byte_mask.unsqueeze(0)
        byte_mask = byte_mask.to(device)
    if score_positions is not None:
        if score_positions.dim() == 1:
            score_positions = score_positions.unsqueeze(0)
        score_positions = score_positions.to(device)

    inp, tgt = x[:, :-1], x[:, 1:]
    tgt_mask = torch.ones(B, T - 1, device=device, dtype=torch.float)
    if byte_mask is not None:
        tgt_mask = tgt_mask * byte_mask[:, 1:].float()
    if score_positions is not None:
        tgt_mask = tgt_mask * score_positions[:, 1:].float()

    with autocast(enabled=use_amp, dtype=dtype if use_amp else torch.float32):
        out = model(inp, targets=None, start_pos=start_pos)
    log_probs = F.log_softmax(out["logits"].float(), dim=-1)
    nll = -log_probs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # (B, T-1)
    per_pos = (nll * tgt_mask).sum(dim=0)  # (T-1,) — sum over batch
    nats = float((nll * tgt_mask).sum().item())
    nb = int(tgt_mask.sum().item())
    mean_ce = nats / max(nb, 1)
    return {
        "nats": nats,
        "n_bytes": nb,
        "bpb": mean_ce / LN2,
        "loss": mean_ce,
        "per_position_nats": per_pos.detach().cpu(),
        "tgt_mask": tgt_mask.detach().cpu(),
    }


@torch.no_grad()
def context_window_consistency_audit(
    model: torch.nn.Module,
    span: torch.Tensor,
    byte_mask: torch.Tensor | None = None,
    *,
    full_len: int = 1024,
    chunk_len: int = 256,
    dtype: torch.dtype | None = None,
    nats_tol: float = 1e-5,
) -> dict[str, Any]:
    """Compare full-span vs chunked teacher-forcing on identical packed target bytes.

    Pre-registered gates:
      - subset_vs_full: masking last ``chunk_len`` on the full forward is internally
        consistent (recompute from per-position nats).
      - carried_vs_full: partition score masks on one full forward sum to full NLL.
      - pe_identity / pe_offset_applied: PE start_pos wiring works.
      - chunk_reset: independent short windows — expected worse; does not block.
    """
    if span.dim() == 1:
        span = span.unsqueeze(0)
    assert span.size(1) >= full_len, f"need >= {full_len} bytes, got {span.size(1)}"
    assert full_len % chunk_len == 0
    x = span[:, :full_len]
    mask = None
    if byte_mask is not None:
        if byte_mask.dim() == 1:
            byte_mask = byte_mask.unsqueeze(0)
        mask = byte_mask[:, :full_len]

    full = evaluate_byte_lm_on_tensor(model, x, mask, start_pos=0, dtype=dtype)
    full_nll = full["per_position_nats"]

    # Subset: score only last chunk_len bytes via score_positions on full forward
    score_last = torch.zeros(1, full_len, dtype=torch.long)
    score_last[:, full_len - chunk_len :] = 1
    subset_ref = evaluate_byte_lm_on_tensor(
        model, x, mask, score_positions=score_last, start_pos=0, dtype=dtype
    )
    scored_from_full = float((full_nll * subset_ref["tgt_mask"].squeeze(0)).sum().item())
    subset_nats_match = abs(scored_from_full - subset_ref["nats"]) <= nats_tol * max(
        subset_ref["n_bytes"], 1
    )

    # Sliding: last chunk alone with PE offset (truncated context — diagnostic)
    sliding = evaluate_byte_lm_on_tensor(
        model,
        x[:, full_len - chunk_len :],
        mask[:, full_len - chunk_len :] if mask is not None else None,
        start_pos=full_len - chunk_len,
        dtype=dtype,
    )

    n_chunks = full_len // chunk_len

    # Independent short windows (PE reset) — expected worse than full
    reset_nats = 0.0
    reset_bytes = 0
    reset_chunks = []
    for i in range(n_chunks):
        s = i * chunk_len
        e = s + chunk_len
        r = evaluate_byte_lm_on_tensor(
            model,
            x[:, s:e],
            mask[:, s:e] if mask is not None else None,
            start_pos=0,
            dtype=dtype,
        )
        reset_chunks.append({"start": s, "bpb": r["bpb"], "n_bytes": r["n_bytes"], "nats": r["nats"]})
        reset_nats += r["nats"]
        reset_bytes += r["n_bytes"]
    reset = {
        "bpb": (reset_nats / max(reset_bytes, 1)) / LN2,
        "nats": reset_nats,
        "n_bytes": reset_bytes,
        "chunks": reset_chunks,
    }

    # Carried partition: score each chunk range on the SAME full forward
    carried_nats = 0.0
    carried_bytes = 0
    carried_chunks = []
    max_abs_diff = 0.0
    for i in range(n_chunks):
        s = i * chunk_len
        e = s + chunk_len
        score_pos = torch.zeros(1, full_len, dtype=torch.long)
        score_pos[:, s:e] = 1
        r = evaluate_byte_lm_on_tensor(
            model, x, mask, score_positions=score_pos, start_pos=0, dtype=dtype
        )
        carried_chunks.append({"start": s, "bpb": r["bpb"], "n_bytes": r["n_bytes"], "nats": r["nats"]})
        carried_nats += r["nats"]
        carried_bytes += r["n_bytes"]
        diff = (full_nll - r["per_position_nats"]).abs()
        m = r["tgt_mask"].squeeze(0) > 0
        if m.any():
            max_abs_diff = max(max_abs_diff, float(diff[m].max().item()))

    carried_vs_full_nats_diff = abs(carried_nats - full["nats"])
    carried_pass = (
        carried_vs_full_nats_diff <= nats_tol * max(full["n_bytes"], 1)
        and max_abs_diff <= nats_tol
    )

    # PE-offset truncated chunks (diagnostic; expected != full)
    pe_offset_nats = 0.0
    pe_offset_bytes = 0
    for i in range(n_chunks):
        s = i * chunk_len
        e = s + chunk_len
        r = evaluate_byte_lm_on_tensor(
            model,
            x[:, s:e],
            mask[:, s:e] if mask is not None else None,
            start_pos=s,
            dtype=dtype,
        )
        pe_offset_nats += r["nats"]
        pe_offset_bytes += r["n_bytes"]

    full2 = evaluate_byte_lm_on_tensor(model, x, mask, start_pos=0, dtype=dtype)
    pe_identity_diff = float(
        (full["per_position_nats"] - full2["per_position_nats"]).abs().max().item()
    )

    mid = x[:, chunk_len : 2 * chunk_len]
    mid_m = mask[:, chunk_len : 2 * chunk_len] if mask is not None else None
    mid0 = evaluate_byte_lm_on_tensor(model, mid, mid_m, start_pos=0, dtype=dtype)
    mid_off = evaluate_byte_lm_on_tensor(model, mid, mid_m, start_pos=chunk_len, dtype=dtype)
    pe_offset_changes = float(
        (mid0["per_position_nats"] - mid_off["per_position_nats"]).abs().max().item()
    )

    gate_pass = bool(
        subset_nats_match
        and carried_pass
        and pe_identity_diff <= nats_tol
        and pe_offset_changes > 1e-6
    )

    return {
        "full": {"bpb": full["bpb"], "nats": full["nats"], "n_bytes": full["n_bytes"]},
        "subset_last_chunk": {
            "bpb": subset_ref["bpb"],
            "nats": subset_ref["nats"],
            "n_bytes": subset_ref["n_bytes"],
        },
        "sliding_last_chunk_alone": {
            "bpb": sliding["bpb"],
            "nats": sliding["nats"],
            "n_bytes": sliding["n_bytes"],
            "note": "truncated context; expected worse than subset_from_full",
        },
        "chunks_reset": reset,
        "chunks_carried_partition": {
            "bpb": (carried_nats / max(carried_bytes, 1)) / LN2,
            "nats": carried_nats,
            "n_bytes": carried_bytes,
            "chunks": carried_chunks,
            "vs_full_nats_diff": carried_vs_full_nats_diff,
            "max_abs_pos_diff": max_abs_diff,
        },
        "chunks_pe_offset_truncated": {
            "bpb": (pe_offset_nats / max(pe_offset_bytes, 1)) / LN2,
            "nats": pe_offset_nats,
            "n_bytes": pe_offset_bytes,
            "note": "PE carried but context truncated; expected != full",
        },
        "gates": {
            "subset_vs_full": {
                "pass": subset_nats_match,
                "scored_from_full_nats": scored_from_full,
                "subset_nats": subset_ref["nats"],
            },
            "carried_vs_full": {
                "pass": carried_pass,
                "nats_diff": carried_vs_full_nats_diff,
                "max_abs_pos_diff": max_abs_diff,
                "tol": nats_tol,
            },
            "pe_identity": {"pass": pe_identity_diff <= nats_tol, "max_diff": pe_identity_diff},
            "pe_offset_applied": {
                "pass": pe_offset_changes > 1e-6,
                "max_diff_start0_vs_offset": pe_offset_changes,
                "note": "start_pos must change logits",
            },
            "chunk_reset_expected_worse": {
                "reset_bpb": reset["bpb"],
                "full_bpb": full["bpb"],
                "delta_bpb": reset["bpb"] - full["bpb"],
                "note": "does not block retrain",
            },
        },
        "gate_pass": gate_pass,
        "headline_gap_note": (
            "2B.1a packed_1024 vs packed_256 compared different windows and "
            "different absolute PE regimes; not a single-number comparison. "
            "This audit scores identical target bytes under controlled protocols."
        ),
    }


@torch.no_grad()
def evaluate_bpe_lm_bpb(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    dtype: torch.dtype | None = None,
    max_batches: int = 40,
    ignore_index: int = 0,
) -> dict[str, float]:
    """BPE LM BPB. Prefer batches with token_byte_weights (per-target-token scored bytes).

    Fallback: n_bytes batch sum with mean token CE (legacy; may be inconsistent).
    """
    model.eval()
    use_amp = dtype is not None and device.type == "cuda"
    total_nats = 0.0
    total_bytes = 0
    total_tokens = 0
    n = 0
    t0 = time.perf_counter()
    for batch in loader:
        if isinstance(batch, dict):
            tokens = batch["tokens"].to(device)
            weights = batch.get("token_byte_weights")  # (B, T) for full token seq
            n_bytes = batch.get("n_bytes")
        else:
            tokens = batch.to(device)
            weights = None
            n_bytes = None
        inp, tgt = tokens[:, :-1], tokens[:, 1:]
        with autocast(enabled=use_amp, dtype=dtype if use_amp else torch.float32):
            out = model(inp, targets=None)
        log_probs = F.log_softmax(out["logits"].float(), dim=-1)
        nll = -log_probs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # (B, T-1)
        pad_mask = tgt != ignore_index
        if weights is not None:
            w = weights.to(device)[:, 1:]  # align with targets
            # Byte weight is the denominator only. Multiplying NLL by that
            # weight counts each token once per byte and inflates BPB.
            scored = w.float() * pad_mask.float()
            token_mask = (scored > 0).float()
            total_nats += float((nll * token_mask).sum().item())
            total_bytes += int(scored.sum().item())
            total_tokens += int(token_mask.sum().item())
        else:
            n_tok = int(pad_mask.sum().item())
            if n_tok == 0:
                continue
            total_nats += float((nll * pad_mask.float()).sum().item())
            total_tokens += n_tok
            if n_bytes is not None:
                total_bytes += int(n_bytes.sum().item() if torch.is_tensor(n_bytes) else n_bytes)
            else:
                total_bytes += n_tok
        n += 1
        if n >= max_batches:
            break
    elapsed = time.perf_counter() - t0
    # BPB = sum(token NLL) / (ln2 * covered bytes). Each token is counted once.
    bpb = (total_nats / max(total_bytes, 1)) / LN2
    return {
        "loss": total_nats / max(total_tokens, 1),
        "bits_per_token": (total_nats / max(total_tokens, 1)) / LN2,
        "tokens_per_byte": total_tokens / max(total_bytes, 1),
        "bpb": bpb,
        "n_tokens": total_tokens,
        "n_bytes": total_bytes,
        "n_batches": n,
        "eval_seconds": elapsed,
        "bytes_per_sec": total_bytes / max(elapsed, 1e-9),
    }


@torch.no_grad()
def evaluate_vq_patch_lm_bpb(
    concept_lm,
    loader,
    device: torch.device,
    dtype: torch.dtype | None = None,
    max_batches: int = 40,
    use_byte_mask: bool = True,
) -> dict[str, float]:
    """Concept CE + encoder-assignment reconstruction BPB (true VQ ids).

    ``encoder_assignment_bpb`` (alias decoder_byte_bpb / decoder_floor_bpb) is NOT
    a true lower bound on marginal likelihood — see oracle_decoder_bpb.
    """
    concept_lm.lm.eval()
    concept_lm.tokenizer.eval()
    use_amp = dtype is not None and device.type == "cuda"
    total_concept_ce = 0.0
    total_concepts = 0
    total_bytes = 0
    decoder_nats = 0.0
    decoder_bytes = 0
    n = 0
    t0 = time.perf_counter()
    p = max(getattr(concept_lm.tokenizer, "pool_factor", 1), 1)
    for batch in loader:
        x = batch["bytes"].to(device)
        mask = batch.get("byte_mask_all") if use_byte_mask else None
        if mask is None and use_byte_mask:
            mask = batch.get("byte_mask")
        if mask is not None and use_byte_mask:
            mask = mask.to(device)
        else:
            mask = None
        with autocast(enabled=use_amp, dtype=dtype if use_amp else torch.float32):
            out = concept_lm(bytes_x=x)
        concepts = out["concepts"]
        n_c = concepts[:, 1:].numel()
        total_concept_ce += float(out["loss"]) * n_c
        total_concepts += n_c
        total_bytes += int(x.numel()) if mask is None else int(mask.sum().item())

        # Encoder-assignment reconstruction with per-byte mask
        usable = (x.size(1) // p) * p
        x_u = x[:, :usable]
        tc = usable // p
        concepts = concepts[:, :tc]
        logits = concept_lm.decode_concepts_to_bytes(concepts, target_len=usable)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        nll = -log_probs.gather(-1, x_u.unsqueeze(-1)).squeeze(-1)
        if mask is not None:
            m = mask[:, :usable].float()
            decoder_nats += float((nll * m).sum().item())
            decoder_bytes += int(m.sum().item())
        else:
            decoder_nats += float(nll.sum().item())
            decoder_bytes += int(nll.numel())
        n += 1
        if n >= max_batches:
            break
    elapsed = time.perf_counter() - t0
    mean_concept_ce = total_concept_ce / max(total_concepts, 1)
    cpb = 1.0 / p
    concept_bpb = (mean_concept_ce / LN2) * cpb
    enc_assign_bpb = (decoder_nats / max(decoder_bytes, 1)) / LN2
    return {
        "concept_ce": mean_concept_ce,
        "concept_bpb": concept_bpb,
        "concepts_per_byte": cpb,
        "encoder_assignment_bpb": enc_assign_bpb,
        "decoder_byte_bpb": enc_assign_bpb,  # legacy alias
        "decoder_floor_bpb": enc_assign_bpb,  # legacy alias (misnamed)
        "chain_bpb": concept_bpb + enc_assign_bpb,
        "bpb": enc_assign_bpb,
        "n_concepts": total_concepts,
        "n_bytes": total_bytes,
        "decoder_n_bytes": decoder_bytes,
        "n_batches": n,
        "eval_seconds": elapsed,
        "bytes_per_sec": total_bytes / max(elapsed, 1e-9),
    }


def nested_topk_ids(log_p_lm: torch.Tensor, top_k: int) -> torch.Tensor:
    """Return unique LM-ranked candidate IDs of size ``top_k`` (B, top_k).

    Candidates are nested prefixes of a single descending LM ranking.
    Does NOT inject the encoder code (that would break nesting / double-count).
    """
    kk = min(int(top_k), log_p_lm.size(-1))
    _, idx = log_p_lm.topk(kk, dim=-1)
    return idx


def cumulative_logsumexp(joint: torch.Tensor, k_values: list[int]) -> dict[int, torch.Tensor]:
    """Nested logsumexp over columns of joint (B, K_sorted).

    ``joint`` must already be ordered by the nested candidate ranking.
    Returns map k -> (B,) logsumexp over first min(k, K) columns.
    """
    out: dict[int, torch.Tensor] = {}
    n = joint.size(1)
    for kk in k_values:
        m = min(int(kk), n)
        out[int(kk)] = torch.logsumexp(joint[:, :m], dim=-1)
    return out


@torch.no_grad()
def score_vq_patch_candidates(
    concept_lm,
    bytes_x: torch.Tensor,
    concepts: torch.Tensor,
    patch_t: int,
    *,
    log_p_lm: torch.Tensor | None = None,
    candidate_ids: torch.Tensor | None = None,
    byte_mask: torch.Tensor | None = None,
    k_chunk: int = 128,
) -> dict[str, Any]:
    """Score one patch under fixed context with float32 log_softmax/logsumexp.

    Returns per-batch tensors for decoder / LM / joint scores and diagnostics.
    For patch_t==0, marginal uses the encoder-assigned code only (no LM).
    """
    p = max(getattr(concept_lm.tokenizer, "pool_factor", 1), 1)
    k = concept_lm.tokenizer.quantizer.codebook_size
    bsz = bytes_x.size(0)
    device = bytes_x.device
    prefix = concepts[:, :patch_t] if patch_t > 0 else concepts[:, :0]
    patch_mask = None
    if byte_mask is not None:
        if byte_mask.size(-1) == p:
            patch_mask = byte_mask
        else:
            patch_mask = byte_mask[:, patch_t * p : (patch_t + 1) * p]
    n_scored = int(patch_mask.sum().item()) if patch_mask is not None else bsz * p

    if candidate_ids is None:
        log_p_dec = concept_lm.decoder_patch_logprobs(
            bytes_x, prefix, patch_t=patch_t, k_chunk=k_chunk, byte_mask=patch_mask
        )  # (B, K)
        cand = torch.arange(k, device=device).view(1, k).expand(bsz, -1)
    else:
        cand = candidate_ids
        log_p_dec = concept_lm.decoder_patch_logprobs(
            bytes_x,
            prefix,
            patch_t=patch_t,
            k_chunk=k_chunk,
            candidate_ids=cand,
            byte_mask=patch_mask,
        )

    enc_ids = concepts[:, patch_t]
    # Encoder-assignment decoder score from the same matrix when possible
    if candidate_ids is None:
        enc_dec_lp = log_p_dec.gather(1, enc_ids.unsqueeze(1)).squeeze(1)
    else:
        # Fall back to dedicated single-code score if encoder code absent from candidates
        match = cand == enc_ids.unsqueeze(1)
        has = match.any(dim=1)
        enc_dec_lp = torch.full((bsz,), float("nan"), device=device, dtype=torch.float32)
        if has.any():
            pos = match.float().argmax(dim=1)
            gathered = log_p_dec.gather(1, pos.unsqueeze(1)).squeeze(1)
            enc_dec_lp = torch.where(has, gathered, enc_dec_lp)
        if (~has).any():
            single = concept_lm.decoder_patch_logprobs(
                bytes_x,
                prefix,
                patch_t=patch_t,
                k_chunk=k_chunk,
                candidate_ids=enc_ids.unsqueeze(1),
                byte_mask=patch_mask,
            ).squeeze(1)
            enc_dec_lp = torch.where(has, enc_dec_lp, single)

    oracle_lp = log_p_dec.max(dim=-1).values
    oracle_ids = log_p_dec.argmax(dim=-1)

    if patch_t == 0 or log_p_lm is None:
        # Patch 0 / no LM: marginal = encoder-assignment decoder only
        marginal_lp = enc_dec_lp
        lm_for_cand = None
        joint = None
    else:
        if log_p_lm.dim() == 1:
            log_p_lm = log_p_lm.unsqueeze(0)
        lm_for_cand = log_p_lm.gather(1, cand)
        joint = lm_for_cand + log_p_dec
        marginal_lp = torch.logsumexp(joint, dim=-1)

    return {
        "candidate_ids": cand,
        "log_p_dec": log_p_dec,
        "log_p_lm_cand": lm_for_cand,
        "joint": joint,
        "marginal_lp": marginal_lp,
        "enc_dec_lp": enc_dec_lp,
        "oracle_lp": oracle_lp,
        "oracle_ids": oracle_ids,
        "encoder_ids": enc_ids,
        "n_scored": n_scored,
        "patch_mask": patch_mask,
        "byte_start": patch_t * p,
        "byte_end": (patch_t + 1) * p,
        "patch_t": patch_t,
    }


@torch.no_grad()
def evaluate_vq_marginal_bpb(
    concept_lm,
    loader,
    device: torch.device,
    dtype: torch.dtype | None = None,
    max_batches: int = 40,
    k_chunk: int = 128,
    max_patches: int | None = None,
    top_k: int | None = None,
    use_byte_mask: bool = True,
    compute_oracle: bool = True,
    collect_patch_debug: bool = False,
    max_debug_patches: int = 64,
) -> dict[str, Any]:
    """Marginal end-to-end byte BPB + matched encoder-assignment + oracle.

    For patch t ≥ 1:
      NLL_t = -logsumexp_{j in S} [ log p_LM(j | z_<t) + log p_dec(x_patch | z_<t, j) ]
    scored only on bytes with mask==1 inside the patch.

    Encoder-assignment and oracle use the **same** patch span/mask/denominator as
    the marginal (not the full window unless max_patches covers it).

    Top-k candidates are nested LM-ranked prefixes — encoder code is NOT forcibly
    injected (that broke nesting / could double-count).
    Patch 0: decoder under true z_0 only (no LM).
    """
    concept_lm.lm.eval()
    concept_lm.tokenizer.eval()
    use_amp = dtype is not None and device.type == "cuda"
    p = max(getattr(concept_lm.tokenizer, "pool_factor", 1), 1)
    k = concept_lm.tokenizer.quantizer.codebook_size

    total_marginal_nats = 0.0
    total_bytes = 0
    total_concept_ce = 0.0
    total_concepts = 0
    # Matched span (same patches as marginal)
    matched_enc_nats = 0.0
    matched_oracle_nats = 0.0
    matched_bytes = 0
    # Full-window reconstruction (separate diagnostic)
    full_enc_nats = 0.0
    full_enc_bytes = 0
    n = 0
    t0 = time.perf_counter()
    patch_debug: list[dict[str, Any]] = []
    oracle_le_enc_fails = 0
    enc_in_topk_misses = 0

    for batch in loader:
        x = batch["bytes"].to(device)
        mask = batch.get("byte_mask_all") if use_byte_mask else None
        if mask is None and use_byte_mask:
            mask = batch.get("byte_mask")
        if mask is not None and use_byte_mask:
            mask = mask.to(device)
        else:
            mask = torch.ones_like(x, dtype=torch.long)

        bsz, t_bytes = x.shape
        usable = (t_bytes // p) * p
        if usable == 0:
            continue
        x = x[:, :usable]
        mask = mask[:, :usable]
        tc = usable // p

        with autocast(enabled=use_amp, dtype=dtype if use_amp else torch.float32):
            concepts = concept_lm.tokenizer.tokenize(x)[:, :tc]
            if tc > 1:
                lm_out = concept_lm.lm(concepts[:, :-1], targets=concepts[:, 1:])
                n_c = concepts[:, 1:].numel()
                total_concept_ce += float(lm_out["loss"]) * n_c
                total_concepts += n_c
                log_p_lm_all = F.log_softmax(lm_out["logits"].float(), dim=-1)
            else:
                log_p_lm_all = None

        # Full-window encoder-assignment (separate metric)
        logits = concept_lm.decode_concepts_to_bytes(concepts, target_len=usable)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        nll_full = -log_probs.gather(-1, x.unsqueeze(-1)).squeeze(-1)
        full_enc_nats += float((nll_full * mask.float()).sum().item())
        full_enc_bytes += int(mask.sum().item())

        n_patches = tc if max_patches is None else min(tc, max_patches)

        for t in range(n_patches):
            byte_start = t * p
            byte_end = (t + 1) * p
            patch_mask = mask[:, byte_start:byte_end]
            n_scored = int(patch_mask.sum().item())
            if n_scored == 0:
                continue

            if t == 0:
                scored = score_vq_patch_candidates(
                    concept_lm,
                    x,
                    concepts,
                    patch_t=0,
                    log_p_lm=None,
                    candidate_ids=None,
                    byte_mask=patch_mask,
                    k_chunk=k_chunk,
                )
            else:
                if log_p_lm_all is None:
                    continue
                log_p_lm = log_p_lm_all[:, t - 1, :]
                if top_k is not None and top_k < k:
                    cand = nested_topk_ids(log_p_lm, top_k)
                    enc = concepts[:, t]
                    in_set = (cand == enc.unsqueeze(1)).any(dim=1)
                    enc_in_topk_misses += int((~in_set).sum().item())
                else:
                    cand = None
                scored = score_vq_patch_candidates(
                    concept_lm,
                    x,
                    concepts,
                    patch_t=t,
                    log_p_lm=log_p_lm,
                    candidate_ids=cand,
                    byte_mask=patch_mask,
                    k_chunk=k_chunk,
                )

            # Matched metrics from the same scored patch
            marg_nll = -scored["marginal_lp"]
            enc_nll = -scored["enc_dec_lp"]
            ora_nll = -scored["oracle_lp"]
            total_marginal_nats += float(marg_nll.sum().item())
            matched_enc_nats += float(enc_nll.sum().item())
            matched_oracle_nats += float(ora_nll.sum().item())
            matched_bytes += n_scored
            total_bytes += n_scored

            # Per-batch invariant: oracle <= encoder assignment
            if compute_oracle:
                bad = ora_nll > enc_nll + 1e-4
                oracle_le_enc_fails += int(bad.sum().item())

            if collect_patch_debug and len(patch_debug) < max_debug_patches:
                patch_debug.append(
                    {
                        "batch": n,
                        "patch_t": t,
                        "n_scored": n_scored,
                        "byte_start": byte_start,
                        "byte_end": byte_end,
                        "encoder_ids": scored["encoder_ids"].detach().cpu().tolist(),
                        "oracle_ids": scored["oracle_ids"].detach().cpu().tolist(),
                        "marginal_nll": marg_nll.detach().cpu().tolist(),
                        "enc_nll": enc_nll.detach().cpu().tolist(),
                        "oracle_nll": ora_nll.detach().cpu().tolist(),
                        "top_k": top_k,
                    }
                )

            if t > 0 and t % 32 == 0:
                print(f"  batch {n + 1} patch {t}/{n_patches - 1}", flush=True)

        n += 1
        if n >= max_batches:
            break

    elapsed = time.perf_counter() - t0
    mean_concept_ce = total_concept_ce / max(total_concepts, 1)
    cpb = 1.0 / p
    concept_bpb = (mean_concept_ce / LN2) * cpb
    matched_enc_bpb = (matched_enc_nats / max(matched_bytes, 1)) / LN2
    matched_oracle_bpb = (matched_oracle_nats / max(matched_bytes, 1)) / LN2
    marginal_bpb = (total_marginal_nats / max(total_bytes, 1)) / LN2
    full_enc_bpb = (full_enc_nats / max(full_enc_bytes, 1)) / LN2

    out: dict[str, Any] = {
        # Primary matched-span metrics (same denominator)
        "encoder_assignment_bpb": matched_enc_bpb,
        "oracle_decoder_bpb": matched_oracle_bpb if compute_oracle else float("nan"),
        "marginal_end_to_end_bpb": marginal_bpb,
        "bpb": marginal_bpb,
        # Legacy aliases (now matched-span, not full-window)
        "decoder_floor_bpb": matched_enc_bpb,
        "decoder_byte_bpb": matched_enc_bpb,
        # Full-window reconstruction (explicitly separate)
        "full_window_encoder_assignment_bpb": full_enc_bpb,
        "full_window_n_bytes": full_enc_bytes,
        "concept_bpb": concept_bpb,
        "concept_ce": mean_concept_ce,
        "concepts_per_byte": cpb,
        "floor_minus_marginal_gap": marginal_bpb - matched_enc_bpb,
        "oracle_le_marginal": bool(matched_oracle_bpb <= marginal_bpb + 1e-5),
        "oracle_le_encoder_assignment": bool(matched_oracle_bpb <= matched_enc_bpb + 1e-5),
        "oracle_le_enc_fail_count": oracle_le_enc_fails,
        "enc_code_missing_from_topk_count": enc_in_topk_misses,
        "n_bytes": total_bytes,
        "matched_n_bytes": matched_bytes,
        "decoder_n_bytes": matched_bytes,
        "n_concepts": total_concepts,
        "n_batches": n,
        "codebook_size": k,
        "pool_factor": p,
        "k_chunk": k_chunk,
        "top_k": top_k,
        "max_patches": max_patches,
        "eval_seconds": elapsed,
        "bytes_per_sec": total_bytes / max(elapsed, 1e-9),
        "patch0_policy": "decoder_true_z0_only",
        "marginal_mode": "nested_top_k" if top_k else "exact_K",
        "candidate_policy": "lm_ranked_nested_prefix",
    }
    if collect_patch_debug:
        out["patch_debug"] = patch_debug
    return out


@torch.no_grad()
def evaluate_vq_topk_curve(
    concept_lm,
    loader,
    device: torch.device,
    dtype: torch.dtype | None = None,
    k_values: list[int] | None = None,
    max_batches: int = 1,
    k_chunk: int = 64,
    max_patches: int | None = 64,
) -> dict[str, Any]:
    """Nested top-k marginal BPB curve from a single LM ranking per patch.

    Builds full-K (or max k) joint scores once per patch, then cumulative logsumexp
    over nested prefixes — guaranteeing monotonicity when scoring is correct.
    """
    k_values = sorted({int(x) for x in (k_values or [1, 4, 16, 64, 256, 1024])})
    concept_lm.lm.eval()
    concept_lm.tokenizer.eval()
    use_amp = dtype is not None and device.type == "cuda"
    p = max(getattr(concept_lm.tokenizer, "pool_factor", 1), 1)
    codebook = concept_lm.tokenizer.quantizer.codebook_size
    max_k = min(max(k_values), codebook)

    # Accumulators per k
    nats: dict[int, float] = {kk: 0.0 for kk in k_values}
    total_bytes = 0
    n = 0
    t0 = time.perf_counter()

    for batch in loader:
        x = batch["bytes"].to(device)
        mask = batch.get("byte_mask_all")
        if mask is None:
            mask = batch.get("byte_mask")
        if mask is not None:
            mask = mask.to(device)
        else:
            mask = torch.ones_like(x, dtype=torch.long)

        usable = (x.size(1) // p) * p
        if usable == 0:
            continue
        x = x[:, :usable]
        mask = mask[:, :usable]
        tc = usable // p

        with autocast(enabled=use_amp, dtype=dtype if use_amp else torch.float32):
            concepts = concept_lm.tokenizer.tokenize(x)[:, :tc]
            if tc > 1:
                lm_out = concept_lm.lm(concepts[:, :-1], targets=None)
                log_p_lm_all = F.log_softmax(lm_out["logits"].float(), dim=-1)
            else:
                log_p_lm_all = None

        n_patches = tc if max_patches is None else min(tc, max_patches)
        for t in range(n_patches):
            patch_mask = mask[:, t * p : (t + 1) * p]
            n_scored = int(patch_mask.sum().item())
            if n_scored == 0:
                continue

            if t == 0:
                scored = score_vq_patch_candidates(
                    concept_lm, x, concepts, 0, byte_mask=patch_mask, k_chunk=k_chunk
                )
                # Patch 0 has no LM mix — all k share encoder-assignment NLL
                nll0 = float((-scored["enc_dec_lp"]).sum().item())
                for kk in k_values:
                    nats[kk] += nll0
                total_bytes += n_scored
                continue

            if log_p_lm_all is None:
                continue
            log_p_lm = log_p_lm_all[:, t - 1, :]
            ranked = nested_topk_ids(log_p_lm, max_k)
            scored = score_vq_patch_candidates(
                concept_lm,
                x,
                concepts,
                t,
                log_p_lm=log_p_lm,
                candidate_ids=ranked,
                byte_mask=patch_mask,
                k_chunk=k_chunk,
            )
            assert scored["joint"] is not None
            cum = cumulative_logsumexp(scored["joint"], k_values)
            for kk in k_values:
                nats[kk] += float((-cum[kk]).sum().item())
            total_bytes += n_scored

        n += 1
        if n >= max_batches:
            break

    results = {}
    prev_bpb = None
    monotone = True
    for kk in k_values:
        bpb = (nats[kk] / max(total_bytes, 1)) / LN2
        results[f"k{kk}"] = {
            "marginal_end_to_end_bpb": bpb,
            "n_bytes": total_bytes,
            "nats": nats[kk],
        }
        if prev_bpb is not None and bpb > prev_bpb + 1e-5:
            monotone = False
        prev_bpb = bpb

    b64 = results.get("k64", {}).get("marginal_end_to_end_bpb")
    b_exact_key = f"k{max_k}"
    b_exact = results.get(b_exact_key, {}).get("marginal_end_to_end_bpb")
    delta_64 = (b64 - b_exact) if b64 is not None and b_exact is not None else None
    return {
        "curve": results,
        "monotone_nll_vs_k": monotone,
        "delta_top64_vs_exact": delta_64,
        "n_bytes": total_bytes,
        "n_batches": n,
        "eval_seconds": time.perf_counter() - t0,
        "candidate_policy": "lm_ranked_nested_prefix_cumulative",
        "max_k": max_k,
    }
