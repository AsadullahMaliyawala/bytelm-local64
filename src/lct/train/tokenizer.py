from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from lct.data.dataset import make_loader, text_to_bytes, write_canonical_probes
from lct.models import ConceptTokenizer
from lct.models.factorized_tokenizer import FactorizedPatchTokenizer
from lct.utils import CSVLogger, ensure_dir, get_autocast_dtype, get_device, save_json, set_seed
from lct.viz.neuralmap import NeuralMapExporter, estimate_transitions


def _build_tokenizer(cfg: dict[str, Any], device: torch.device):
    m = cfg.get("model", cfg)
    tok_class = str(m.get("tokenizer_class", cfg.get("tokenizer_class", "concept"))).lower()
    if tok_class in ("factorized", "factorized_patch"):
        return FactorizedPatchTokenizer(cfg).to(device)
    return ConceptTokenizer(cfg).to(device)


def _probe_indices(model, probe_file: Path, seq_len: int, device: torch.device) -> list[list[int]]:
    import json

    if not probe_file.exists():
        write_canonical_probes(probe_file)
    with probe_file.open(encoding="utf-8") as f:
        probes = json.load(f)
    texts = (probes.get("text") or [])[:10] + (probes.get("code") or [])[:10]
    out = []
    model.eval()
    with torch.no_grad():
        for t in texts:
            ids = torch.tensor([text_to_bytes(t, seq_len)], dtype=torch.long, device=device)
            idx = model.tokenize(ids)[0].cpu()
            out.append(idx.reshape(-1).tolist())
    return out


def train_tokenizer(cfg: dict[str, Any]) -> dict[str, Any]:
    set_seed(cfg.get("seed", 42))
    device = get_device(cfg.get("device", "cuda"))
    dtype = get_autocast_dtype(cfg.get("precision", "bf16"))
    run_id = cfg.get("run_id", "tokenizer")
    exp_dir = ensure_dir(Path(cfg["train"]["experiment_dir"]) / run_id)
    ckpt_dir = ensure_dir(cfg["train"]["checkpoint_dir"])
    save_json({k: v for k, v in cfg.items() if not k.startswith("_")}, exp_dir / "config.json")

    model = _build_tokenizer(cfg, device)
    is_factorized = isinstance(model, FactorizedPatchTokenizer)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 0.01),
    )
    scaler = GradScaler(enabled=(dtype == torch.float16))
    logger = CSVLogger(exp_dir / "metrics.csv")
    exporter = NeuralMapExporter(exp_dir)

    train_loader = make_loader(cfg, "train")
    val_loader = make_loader(cfg, "val", shuffle=False)

    max_steps = cfg["train"]["max_steps"]
    accum = cfg["train"].get("grad_accum", 1)
    log_every = cfg["train"].get("log_every", 20)
    eval_every = cfg["train"].get("eval_every", 100)
    save_every = cfg["train"].get("save_every", 200)
    viz_every = cfg["train"].get("viz_every_n_steps", 50)

    step = 0
    best_val = float("inf")
    model.train()
    opt.zero_grad(set_to_none=True)
    pbar = tqdm(total=max_steps, desc="tokenizer")

    data_iter = iter(train_loader)

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
        use_amp = dtype is not None and device.type == "cuda"
        with autocast(enabled=use_amp, dtype=dtype if use_amp else torch.float32):
            out = model(x, byte_mask=mask) if is_factorized else model(x)
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

        if step % log_every == 0:
            row = {
                "step": step,
                "loss": float(out["loss"].detach()),
                "recon_loss": float(out["recon_loss"]),
                "vq_loss": float(out["vq_loss"]),
                "perplexity": float(out["perplexity"]),
                "dead_ratio": float(out["dead_ratio"]),
                "entropy": float(out["entropy"]),
                "concepts_per_byte": float(out.get("concepts_per_byte", 1.0)),
            }
            logger.log(row)
            pbar.set_postfix({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items() if k != "step"})

        revive_every = cfg["train"].get("revive_every", 100)
        if revive_every and step % revive_every == 0:
            with torch.no_grad():
                h = model.encoder(x)
                h_p = model.pre_vq(model._pool(h))
                n_rev = model.quantizer.revive_dead_codes(h_p)
            if n_rev and step % log_every == 0:
                logger.log({"step": step, "revived_codes": n_rev})

        if step % viz_every == 0 or step == 1:
            probe_file = Path(cfg.get("viz", {}).get("probe_file", "data/probes/canonical.json"))
            probes = _probe_indices(model, probe_file, cfg["data"]["seq_len"], device)
            idx_viz = out["indices"].detach()
            if idx_viz.dim() == 3:
                idx_viz = idx_viz[..., 0]
            transitions = estimate_transitions(idx_viz, top_k=cfg.get("viz", {}).get("top_k_transitions", 20))
            exporter.export_step(
                step=step,
                codebook=model.quantizer.codebook.weight,
                usage=model.quantizer.usage_counts,
                metrics={
                    "loss": float(out["loss"].detach()),
                    "recon_loss": float(out["recon_loss"]),
                    "perplexity": float(out["perplexity"]),
                },
                probe_indices=probes,
                transitions=transitions,
                projection=cfg.get("viz", {}).get("projection", "pca"),
            )
            model.train()

        if step % eval_every == 0:
            val = evaluate_tokenizer(model, val_loader, device, dtype)
            logger.log({"step": step, **{f"val_{k}": v for k, v in val.items()}})
            if val["recon_loss"] < best_val:
                best_val = val["recon_loss"]
                ckpt_name = cfg["train"].get("best_checkpoint_name", "phase1_best.pt")
                path = ckpt_dir / ckpt_name
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": cfg,
                        "step": step,
                        "val": val,
                    },
                    path,
                )
            model.train()

        if step % save_every == 0:
            torch.save({"model": model.state_dict(), "config": cfg, "step": step}, ckpt_dir / f"{run_id}_step{step}.pt")

    pbar.close()
    exporter.export_step(
        step=step,
        codebook=model.quantizer.codebook.weight,
        usage=model.quantizer.usage_counts,
        metrics={"loss": float(best_val)},
        projection=cfg.get("viz", {}).get("projection", "pca"),
    )
    used = int((model.quantizer.usage_counts > 0).sum().item())
    results = {
        "best_val_recon": best_val,
        "steps": step,
        "run_id": run_id,
        "config_hash": cfg.get("_config_hash"),
        "used_codes": used,
        "codebook_size": getattr(model, "codebook_size", model.quantizer.codebook_size),
        "pool_factor": model.pool_factor,
        "concepts_per_byte": 1.0 / max(model.pool_factor, 1),
        "tokenizer_class": "factorized" if is_factorized else "concept",
        "quantizer_type": getattr(model, "quantizer_type", "vq1"),
        "num_stages": getattr(model, "num_stages", 1),
    }
    save_json(results, exp_dir / "results.json")
    return results


@torch.no_grad()
def evaluate_tokenizer(model, loader, device, dtype) -> dict[str, float]:
    model.eval()
    totals = {"recon_loss": 0.0, "loss": 0.0, "exact_match": 0.0, "concepts_per_byte": 0.0}
    n = 0
    use_amp = dtype is not None and device.type == "cuda"
    is_factorized = isinstance(model, FactorizedPatchTokenizer)
    for batch in loader:
        x = batch["bytes"].to(device)
        mask = batch.get("byte_mask_all", batch.get("byte_mask"))
        if mask is not None:
            mask = mask.to(device)
        with autocast(enabled=use_amp, dtype=dtype if use_amp else torch.float32):
            out = model(x, byte_mask=mask) if is_factorized else model(x)
        pred = out["logits"].argmax(dim=-1)
        usable = pred.size(1)
        exact = (pred == x[:, :usable]).float().mean().item()
        totals["recon_loss"] += float(out["recon_loss"])
        totals["loss"] += float(out["loss"])
        totals["exact_match"] += exact
        totals["concepts_per_byte"] += float(out.get("concepts_per_byte", 1.0))
        n += 1
        if n >= 20:
            break
    return {k: v / max(n, 1) for k, v in totals.items()}
