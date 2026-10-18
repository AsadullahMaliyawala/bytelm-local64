"""Train continuous pool-4 patch LM (Phase 2B control)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from lct.data.dataset import make_loader
from lct.eval.lm_metrics import LN2, param_report, peak_vram_mb
from lct.models.continuous_pool4 import ContinuousPool4LM
from lct.utils import CSVLogger, ensure_dir, get_autocast_dtype, get_device, save_json, set_seed


@torch.no_grad()
def evaluate_continuous_pool4(
    model: ContinuousPool4LM,
    loader,
    device: torch.device,
    dtype: torch.dtype | None = None,
    max_batches: int = 40,
) -> dict[str, float]:
    model.eval()
    use_amp = dtype is not None and device.type == "cuda"
    total_nats = 0.0
    total_bytes = 0
    total_nats_c = 0.0
    total_bytes_c = 0
    n = 0
    for batch in loader:
        x = batch["bytes"].to(device)
        mask_all = batch.get("byte_mask_all", batch.get("byte_mask"))
        mask_c = batch.get("byte_mask_content", batch.get("byte_mask"))
        if mask_all is not None:
            mask_all = mask_all.to(device)
        if mask_c is not None:
            mask_c = mask_c.to(device)
        with autocast(enabled=use_amp, dtype=dtype if use_amp else torch.float32):
            nats, nb = model.byte_nll(x, mask_all)
            nats_c, nb_c = model.byte_nll(x, mask_c if mask_c is not None else mask_all)
        total_nats += nats
        total_bytes += nb
        total_nats_c += nats_c
        total_bytes_c += nb_c
        n += 1
        if n >= max_batches:
            break
    return {
        "bpb_all_bytes": (total_nats / max(total_bytes, 1)) / LN2,
        "bpb_content_only": (total_nats_c / max(total_bytes_c, 1)) / LN2,
        "n_bytes": total_bytes,
        "n_batches": n,
    }


def train_continuous_pool4(cfg: dict[str, Any]) -> dict[str, Any]:
    set_seed(cfg.get("seed", 42))
    device = get_device(cfg.get("device", "cuda"))
    dtype = get_autocast_dtype(cfg.get("precision", "bf16"))
    run_id = cfg.get("run_id", "contpool4")
    exp_dir = ensure_dir(Path(cfg["train"]["experiment_dir"]) / run_id)
    ckpt_dir = ensure_dir(cfg["train"]["checkpoint_dir"])
    save_json({k: v for k, v in cfg.items() if not str(k).startswith("_")}, exp_dir / "config.json")

    model = ContinuousPool4LM(cfg).to(device)
    params_info = param_report(model)
    save_json(params_info, exp_dir / "param_count.json")

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"].get("weight_decay", 0.01)
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
            scored_bytes += int(mask.sum().item()) if count_scored else int(x.numel())
        else:
            scored_bytes += int(x.numel())

        with autocast(enabled=use_amp, dtype=dtype if use_amp else torch.float32):
            out = model(x, byte_mask=mask)
            loss = out["loss"] / accum
        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if (step + 1) % accum == 0:
            if scaler.is_enabled():
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"].get("grad_clip", 1.0))
                scaler.step(opt)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"].get("grad_clip", 1.0))
                opt.step()
            opt.zero_grad(set_to_none=True)

        step += 1
        pbar.update(1)
        if step % cfg["train"].get("log_every", 50) == 0:
            logger.log({"step": step, "loss": float(out["loss"].detach()), "scored_bytes": scored_bytes})
            pbar.set_postfix(loss=round(float(out["loss"].detach()), 3), scored=scored_bytes)

        if step % cfg["train"].get("eval_every", 500) == 0:
            val = evaluate_continuous_pool4(model, val_loader, device, dtype)
            logger.log({"step": step, **{f"val_{k}": v for k, v in val.items()}})
            if val["bpb_all_bytes"] < best:
                best = val["bpb_all_bytes"]
                torch.save(
                    {"model": model.state_dict(), "config": cfg, "step": step, "val": val},
                    Path(ckpt_dir) / cfg["train"].get("best_checkpoint_name", f"{run_id}.pt"),
                )
            model.train()

        if target_bytes is not None and scored_bytes >= int(target_bytes):
            break

    pbar.close()
    wall = time.perf_counter() - t0
    final = evaluate_continuous_pool4(model, val_loader, device, dtype)
    torch.save(
        {"model": model.state_dict(), "config": cfg, "step": step, "val": final},
        Path(ckpt_dir) / cfg["train"].get("last_checkpoint_name", f"{run_id}_last.pt"),
    )
    run_manifest = {
        "train_scored_bytes": int(scored_bytes),
        "optimizer_steps": int(step),
        "wall_clock_seconds": wall,
        "scored_bytes_per_second": scored_bytes / max(wall, 1e-9),
        "peak_vram_gb": (peak_vram_mb() or 0) / 1024.0,
        "core_lm_parameters": params_info.get("lm_core") or params_info.get("lm_total"),
        "total_parameters": params_info.get("lm_total_all") or params_info.get("lm_total"),
        "packed_stream": True,
        "pool_factor": model.pool_factor,
        "kind": "continuous_pool4",
    }
    save_json(run_manifest, exp_dir / "run_manifest.json")
    results = {
        "run_id": run_id,
        "kind": "continuous_pool4",
        "best_val_bpb": best,
        "final_metrics": final,
        "steps": step,
        "scored_bytes": scored_bytes,
        "wall_seconds": wall,
        "peak_vram_mb": peak_vram_mb(),
        "params": params_info,
        "run_manifest": run_manifest,
    }
    save_json(results, exp_dir / "results.json")
    return results
