"""Decoder-only training for controlled patch-context ablation."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import torch.nn.functional as F

from lct.data.dataset import make_loader
from lct.eval.lm_metrics import LN2, masked_nll_from_logits, param_report, peak_vram_mb
from lct.models.decoder_context import ControlledPatchDecoder, ContextMode, MODES
from lct.models.quantizer import causal_mask
from lct.models.vqvae import ConceptTokenizer
from lct.utils import CSVLogger, ensure_dir, get_autocast_dtype, get_device, save_json, set_seed


def load_frozen_tokenizer(ckpt_path: str | Path, device: torch.device) -> ConceptTokenizer:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    tok = ConceptTokenizer(ckpt["config"]).to(device)
    tok.load_state_dict(ckpt["model"], strict=False)
    for p in tok.parameters():
        p.requires_grad = False
    tok.eval()
    return tok


@torch.no_grad()
def evaluate_controlled_decoder(
    decoder: ControlledPatchDecoder,
    tokenizer: ConceptTokenizer,
    loader,
    device: torch.device,
    dtype: torch.dtype | None = None,
    max_batches: int = 40,
) -> dict[str, float]:
    decoder.eval()
    use_amp = dtype is not None and device.type == "cuda"
    total_nats = 0.0
    total_bytes = 0
    total_nats_content = 0.0
    total_bytes_content = 0
    n = 0
    p = decoder.pool_factor
    for batch in loader:
        x = batch["bytes"].to(device)
        mask_all = batch.get("byte_mask_all", batch.get("byte_mask"))
        mask_content = batch.get("byte_mask_content", batch.get("byte_mask"))
        if mask_all is not None:
            mask_all = mask_all.to(device)
        if mask_content is not None:
            mask_content = mask_content.to(device)
        usable = (x.size(1) // p) * p
        if usable == 0:
            continue
        x = x[:, :usable]
        if mask_all is not None:
            mask_all = mask_all[:, :usable]
        if mask_content is not None:
            mask_content = mask_content[:, :usable]
        with autocast(enabled=use_amp, dtype=dtype if use_amp else torch.float32):
            concepts = tokenizer.tokenize(x)
            enc_h = tokenizer.encoder(x) if decoder.mode == "encoder_local" else None
            tc = min(concepts.size(1), usable // p)
            # Batched logits via shared latent builder (same as training).
            p_ = decoder.pool_factor
            max_ctx = decoder._context_len() * p_ + p_
            zs = []
            for t in range(tc):
                z = decoder.build_latent(concepts, t, encoder_h=enc_h)
                if z.size(1) < max_ctx:
                    z = F.pad(z, (0, 0, max_ctx - z.size(1), 0))
                elif z.size(1) > max_ctx:
                    z = z[:, -max_ctx:]
                zs.append(z)
            z_flat = torch.stack(zs, dim=1).reshape(x.size(0) * tc, max_ctx, decoder.d_model)
            h = decoder.dropout(decoder.byte_pos(decoder.proj(z_flat)))
            attn = causal_mask(h.size(1), h.device)
            for blk in decoder.blocks:
                h = blk(h, attn_mask=attn)
            logits = decoder.head(decoder.ln(h))[:, -p_:, :].reshape(x.size(0), tc * p_, -1)
            tgt = x[:, : tc * p_]
            m_all = mask_all[:, : tc * p_] if mask_all is not None else None
            m_c = mask_content[:, : tc * p_] if mask_content is not None else None
            nats, nb = masked_nll_from_logits(logits, tgt, m_all)
            total_nats += nats
            total_bytes += nb
            nats_c, nb_c = masked_nll_from_logits(logits, tgt, m_c if m_c is not None else m_all)
            total_nats_content += nats_c
            total_bytes_content += nb_c
        n += 1
        if n >= max_batches:
            break
    return {
        "bpb_all_bytes": (total_nats / max(total_bytes, 1)) / LN2,
        "bpb_content_only": (total_nats_content / max(total_bytes_content, 1)) / LN2,
        "n_bytes": total_bytes,
        "n_bytes_content": total_bytes_content,
        "n_batches": n,
    }


def train_controlled_decoder(cfg: dict[str, Any]) -> dict[str, Any]:
    set_seed(cfg.get("seed", 42))
    device = get_device(cfg.get("device", "cuda"))
    dtype = get_autocast_dtype(cfg.get("precision", "bf16"))
    run_id = cfg.get("run_id", "decoder_context")
    mode: ContextMode = cfg.get("decoder_context_mode", "code_only")
    assert mode in MODES
    exp_dir = ensure_dir(Path(cfg["train"]["experiment_dir"]) / run_id)
    ckpt_dir = ensure_dir(cfg["train"]["checkpoint_dir"])
    save_json({k: v for k, v in cfg.items() if not str(k).startswith("_")}, exp_dir / "config.json")

    tok_path = cfg["tokenizer_checkpoint"]
    tok = load_frozen_tokenizer(tok_path, device)
    m = tok.cfg
    decoder = ControlledPatchDecoder(
        d_model=m["d_model"],
        n_heads=m["n_heads"],
        n_layers=m["n_decoder_layers"],
        ffn_mult=m.get("ffn_mult", 4),
        dropout=m.get("dropout", 0.1),
        vocab_size=m.get("vocab_size", 256),
        pool_factor=tok.pool_factor,
        codebook_size=tok.quantizer.codebook_size,
        mode=mode,
        max_prev=16 if mode == "code_full_prefix" else 4,
    ).to(device)

    params_info = param_report(decoder)
    save_json(params_info, exp_dir / "param_count.json")

    opt = torch.optim.AdamW(
        decoder.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 0.01),
    )
    scaler = GradScaler(enabled=(dtype == torch.float16))
    logger = CSVLogger(exp_dir / "metrics.csv")
    train_loader = make_loader(cfg, "train")
    val_loader = make_loader(cfg, "val", shuffle=False)

    max_steps = cfg["train"]["max_steps"]
    accum = cfg["train"].get("grad_accum", 1)
    target_bytes = cfg["train"].get("target_bytes_seen")
    count_scored = bool(cfg["train"].get("count_scored_bytes_only", True))
    step = 0
    best = float("inf")
    scored_bytes = 0
    t0 = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    opt.zero_grad(set_to_none=True)
    pbar = tqdm(total=max_steps, desc=run_id)
    data_iter = iter(train_loader)
    use_amp = dtype is not None and device.type == "cuda"
    p = tok.pool_factor

    while step < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        x = batch["bytes"].to(device)
        mask = batch.get("byte_mask_all", batch.get("byte_mask"))
        if mask is not None:
            mask = mask.to(device)
        usable = (x.size(1) // p) * p
        if usable == 0:
            continue
        x = x[:, :usable]
        if mask is not None:
            mask = mask[:, :usable]
            if count_scored:
                scored_bytes += int(mask.sum().item())
            else:
                scored_bytes += int(x.numel())
        else:
            scored_bytes += int(x.numel())

        with torch.no_grad():
            concepts = tok.tokenize(x)
            enc_h = tok.encoder(x) if mode == "encoder_local" else None

        with autocast(enabled=use_amp, dtype=dtype if use_amp else torch.float32):
            out = decoder.forward_sequence(concepts, x, byte_mask=mask, encoder_h=enc_h)
            loss = out["loss"] / accum
        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if (step + 1) % accum == 0:
            if scaler.is_enabled():
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), cfg["train"].get("grad_clip", 1.0))
                scaler.step(opt)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), cfg["train"].get("grad_clip", 1.0))
                opt.step()
            opt.zero_grad(set_to_none=True)

        step += 1
        pbar.update(1)
        if step % cfg["train"].get("log_every", 50) == 0:
            logger.log({"step": step, "loss": float(out["loss"].detach()), "scored_bytes": scored_bytes})
            pbar.set_postfix(loss=round(float(out["loss"].detach()), 3), scored=scored_bytes)

        if step % cfg["train"].get("eval_every", 250) == 0:
            val = evaluate_controlled_decoder(decoder, tok, val_loader, device, dtype)
            logger.log({"step": step, **{f"val_{k}": v for k, v in val.items()}})
            if val["bpb_all_bytes"] < best:
                best = val["bpb_all_bytes"]
                torch.save(
                    {"model": decoder.state_dict(), "config": cfg, "mode": mode, "step": step, "val": val},
                    Path(ckpt_dir) / cfg["train"].get("best_checkpoint_name", f"{run_id}.pt"),
                )
            decoder.train()

        if target_bytes is not None and scored_bytes >= int(target_bytes):
            break

    pbar.close()
    wall = time.perf_counter() - t0
    final = evaluate_controlled_decoder(decoder, tok, val_loader, device, dtype)
    torch.save(
        {"model": decoder.state_dict(), "config": cfg, "mode": mode, "step": step, "val": final},
        Path(ckpt_dir) / cfg["train"].get("last_checkpoint_name", f"{run_id}_last.pt"),
    )
    results = {
        "run_id": run_id,
        "mode": mode,
        "best_val_bpb": best,
        "final_metrics": final,
        "steps": step,
        "scored_bytes": scored_bytes,
        "wall_seconds": wall,
        "peak_vram_mb": peak_vram_mb(),
        "params": params_info,
    }
    save_json(results, exp_dir / "results.json")
    return results
