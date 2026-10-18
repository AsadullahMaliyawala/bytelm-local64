"""Train a matched byte-level language model (Phase 2B / 2B.1b)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from lct.data.dataset import make_loader
from lct.eval.lm_metrics import (
    evaluate_byte_lm_bpb,
    evaluate_byte_lm_bpb_dual,
    param_report,
    peak_vram_mb,
)
from lct.models.decoder_lm import DecoderOnlyLM
from lct.utils import (
    CSVLogger,
    ensure_dir,
    get_autocast_dtype,
    get_device,
    save_json,
    set_seed,
)


def _primary_mask(batch: dict[str, Any], device: torch.device) -> torch.Tensor | None:
    """Prefer byte_mask_all (incl. sep); fall back to byte_mask."""
    m = batch.get("byte_mask_all")
    if m is None:
        m = batch.get("byte_mask")
    return m.to(device) if m is not None else None


def train_byte_lm(cfg: dict[str, Any]) -> dict[str, Any]:
    set_seed(cfg.get("seed", 42))
    device = get_device(cfg.get("device", "cuda"))
    dtype = get_autocast_dtype(cfg.get("precision", "bf16"))
    run_id = cfg.get("run_id", "lm_byte_40m")
    exp_dir = ensure_dir(Path(cfg["train"]["experiment_dir"]) / run_id)
    ckpt_dir = ensure_dir(cfg["train"]["checkpoint_dir"])
    save_json({k: v for k, v in cfg.items() if not str(k).startswith("_")}, exp_dir / "config.json")

    m = cfg["model"]
    model = DecoderOnlyLM(
        vocab_size=256,
        d_model=m["d_model"],
        n_heads=m["n_heads"],
        n_layers=m["n_layers"],
        ffn_mult=m.get("ffn_mult", 4),
        dropout=m.get("dropout", 0.1),
        label_smoothing=m.get("label_smoothing", 0.0),
        max_len=m.get("max_len", 2048),
        ignore_index=-100,
        pos_type=m.get("pos_type", "sinusoidal"),
    ).to(device)

    params_info = param_report(model)
    save_json(params_info, exp_dir / "param_count.json")

    opt = torch.optim.AdamW(
        model.parameters(),
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
    count_scored = bool(cfg["train"].get("count_scored_bytes_only", False))
    use_masked_loss = bool(
        cfg["train"].get("masked_loss", False)
        or cfg.get("data", {}).get("packed_stream", False)
        or cfg.get("data", {}).get("source") == "packed_bin"
    )
    curve_fracs = list(cfg["train"].get("learning_curve_fracs") or [])
    curve_hits: set[float] = set()
    learning_curve: list[dict] = []
    step = 0
    best = float("inf")
    bytes_seen = 0  # raw tensor bytes (diagnostic)
    scored_bytes = 0  # primary-mask target bytes
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
        bytes_seen += int(x.numel())
        inp, tgt = x[:, :-1], x[:, 1:]
        mask = _primary_mask(batch, device)
        tgt_mask = mask[:, 1:] if mask is not None else None

        if count_scored and tgt_mask is not None:
            scored_bytes += int(tgt_mask.sum().item())
        elif count_scored:
            scored_bytes += int(tgt.numel())
        else:
            scored_bytes = bytes_seen

        with autocast(enabled=use_amp, dtype=dtype if use_amp else torch.float32):
            out = model(inp, targets=None)
            if use_masked_loss and tgt_mask is not None:
                # Masked mean CE over scored target positions
                ignore = model.ignore_index
                tgt_m = tgt.clone()
                tgt_m = tgt_m.masked_fill(tgt_mask == 0, ignore)
                loss = torch.nn.functional.cross_entropy(
                    out["logits"].reshape(-1, model.vocab_size),
                    tgt_m.reshape(-1),
                    label_smoothing=model.label_smoothing,
                    ignore_index=ignore,
                ) / accum
            else:
                loss = (
                    torch.nn.functional.cross_entropy(
                        out["logits"].reshape(-1, model.vocab_size),
                        tgt.reshape(-1),
                        label_smoothing=model.label_smoothing,
                        ignore_index=model.ignore_index,
                    )
                    / accum
                )
            out["loss"] = loss * accum  # for logging undivided
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
            logger.log({
                "step": step,
                "loss": float(out["loss"].detach()),
                "bytes_seen": bytes_seen,
                "scored_bytes": scored_bytes,
                "wall_seconds": time.perf_counter() - t0,
            })
            pbar.set_postfix(loss=round(float(out["loss"].detach()), 3), scored=scored_bytes)

        if step % cfg["train"].get("eval_every", 250) == 0:
            use_dual = bool(
                cfg.get("data", {}).get("packed_stream")
                or cfg.get("data", {}).get("source") == "packed_bin"
            )
            if use_dual:
                dual = evaluate_byte_lm_bpb_dual(model, val_loader, device, dtype)
                val = {
                    "bpb": dual["bpb_all_bytes"],
                    "bpb_all_bytes": dual["bpb_all_bytes"],
                    "bpb_content_only": dual["bpb_content_only"],
                    "n_bytes": dual["primary"]["n_bytes"],
                }
            else:
                val = evaluate_byte_lm_bpb(model, val_loader, device, dtype)
            logger.log({"step": step, **{f"val_{k}": v for k, v in val.items()}})
            if val["bpb"] < best:
                best = val["bpb"]
                ckpt_name = cfg["train"].get("best_checkpoint_name", f"{run_id}.pt")
                torch.save(
                    {"model": model.state_dict(), "config": cfg, "step": step, "val": val},
                    Path(ckpt_dir) / ckpt_name,
                )
            model.train()

        budget = scored_bytes if count_scored else bytes_seen
        if target_bytes is not None and curve_fracs:
            for frac in curve_fracs:
                thresh = int(int(target_bytes) * float(frac))
                if frac not in curve_hits and budget >= thresh:
                    curve_hits.add(frac)
                    use_dual = bool(
                        cfg.get("data", {}).get("packed_stream")
                        or cfg.get("data", {}).get("source") == "packed_bin"
                    )
                    if use_dual:
                        dual = evaluate_byte_lm_bpb_dual(model, val_loader, device, dtype)
                        cval = {
                            "bpb": dual["bpb_all_bytes"],
                            "bpb_all_bytes": dual["bpb_all_bytes"],
                            "bpb_content_only": dual["bpb_content_only"],
                        }
                    else:
                        cval = evaluate_byte_lm_bpb(model, val_loader, device, dtype)
                    learning_curve.append(
                        {"frac": frac, "scored_bytes": budget, "step": step, **cval}
                    )
                    model.train()
        if target_bytes is not None and budget >= int(target_bytes):
            break

    pbar.close()
    wall = time.perf_counter() - t0
    use_dual = bool(
        cfg.get("data", {}).get("packed_stream")
        or cfg.get("data", {}).get("source") == "packed_bin"
    )
    if use_dual:
        dual = evaluate_byte_lm_bpb_dual(model, val_loader, device, dtype)
        final = {
            "bpb": dual["bpb_all_bytes"],
            "bpb_all_bytes": dual["bpb_all_bytes"],
            "bpb_content_only": dual["bpb_content_only"],
            "n_bytes": dual["primary"]["n_bytes"],
            **{f"primary_{k}": v for k, v in dual["primary"].items() if not isinstance(v, dict)},
        }
    else:
        final = evaluate_byte_lm_bpb(model, val_loader, device, dtype)
    last_name = cfg["train"].get("last_checkpoint_name", f"{run_id}_last.pt")
    torch.save(
        {"model": model.state_dict(), "config": cfg, "step": step, "val": final},
        Path(ckpt_dir) / last_name,
    )

    packed_sha = None
    man_path = Path(cfg.get("data", {}).get("cache_dir", "data/cache")) / "phase2b_val_manifest.json"
    if man_path.exists():
        import json

        packed_sha = json.loads(man_path.read_text(encoding="utf-8")).get("raw_utf8_sha256")

    run_manifest = {
        "packed_stream_sha256": packed_sha,
        "doc_separator_hex": format(int(cfg.get("data", {}).get("doc_sentinel_byte", 0x1E)), "02x"),
        "train_scored_bytes": int(scored_bytes),
        "train_raw_tensor_bytes": int(bytes_seen),
        "validation_scored_bytes": int(final.get("n_bytes") or 0),
        "context_bytes": int(cfg.get("data", {}).get("seq_len", 1024)),
        "sequence_positions": int(cfg.get("data", {}).get("seq_len", 1024)),
        "optimizer_steps": int(step),
        "wall_clock_seconds": wall,
        "raw_bytes_per_second": bytes_seen / max(wall, 1e-9),
        "scored_bytes_per_second": scored_bytes / max(wall, 1e-9),
        "peak_vram_gb": (peak_vram_mb() or 0) / 1024.0,
        "core_lm_parameters": params_info.get("lm_core"),
        "total_parameters": params_info.get("lm_total_all"),
        "count_scored_bytes_only": count_scored,
        "masked_loss": use_masked_loss,
        "packed_stream": bool(cfg.get("data", {}).get("packed_stream", False)),
    }
    save_json(run_manifest, exp_dir / "run_manifest.json")

    results = {
        "run_id": run_id,
        "kind": "byte_lm",
        "best_val_bpb": best,
        "final_metrics": {
            **final,
            "raw_bytes_per_sec": bytes_seen / max(wall, 1e-9),
            "scored_bytes_per_sec": scored_bytes / max(wall, 1e-9),
            "total_params": params_info.get("lm_total_all") or params_info.get("total_params"),
        },
        "learning_curve": learning_curve,
        "steps": step,
        "bytes_seen": bytes_seen,
        "scored_bytes": scored_bytes,
        "wall_seconds": wall,
        "bytes_per_sec_train": bytes_seen / max(wall, 1e-9),
        "peak_vram_mb": peak_vram_mb(),
        "params": params_info,
        "run_manifest": run_manifest,
    }
    save_json(results, exp_dir / "results.json")
    return results
