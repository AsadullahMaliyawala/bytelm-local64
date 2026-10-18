#!/usr/bin/env python
"""Two-rank local64 training. Launch with torchrun, one process per GPU node.

Each rank reads a disjoint slice of the same document permutation. Rank 0 writes
one checkpoint that stores both cursors. SIGTERM finishes the current step and saves.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from lct.data.dataset import collate_batch
from lct.data.real_corpus import PackedBinDataset
from lct.models.continuous_pool4 import ContinuousPool4LM, apply_scaled_residual_init, local_decoder_flops_per_byte
from lct.train.instruct_v7 import sample_prompt
from lct.train.continuous_pool4 import evaluate_continuous_pool4
from lct.eval.lm_metrics import LN2
from lct.train.ladder import ShuffledDocCursor, bpb_from_nats, patch_loss, train_ladder_run
from lct.utils import count_parameters, get_autocast_dtype, set_seed

WINDOW = 2048
FIXED_PROMPTS = (
    "The capital of France is",
    "In one sentence, explain gravity.",
    "1 + 1 =",
    "Once upon a time",
    "def add(a, b):\n    return",
)


def train_flops_per_byte(d_model: int, n_layers: int, batch_seq: int = WINDOW) -> float:
    """Training FLOPs per byte. Same counting as the 120M speed sweep."""
    seq = batch_seq // 4
    d = int(d_model)
    per_layer = (8 * d * d) + (4 * d * seq) + (16 * d * d)
    global_fwd = int(n_layers) * per_layer / 4
    local_fwd = local_decoder_flops_per_byte(d, 64, n_layers=2, ffn_mult=2)
    return 3.0 * (global_fwd + local_fwd)


def steady_bytes_per_second(step_log: list | None, bytes_per_step: int) -> float | None:
    if not step_log:
        return None
    times = sorted(row["seconds"] for row in step_log[5:])
    if not times:
        return None
    return bytes_per_step / max(times[len(times) // 2], 1e-9)


def domain_bpb(model, val_dir: Path, device, dtype) -> dict:
    from torch.utils.data import DataLoader

    raw = model.module if hasattr(model, "module") else model
    out = {}
    total_nats = 0.0
    total_bytes = 0
    for domain in ("text", "code", "math"):
        path = val_dir / f"val_{domain}.bin"
        ds = PackedBinDataset(path, window_bytes=WINDOW)
        loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_batch)
        metrics = evaluate_continuous_pool4(raw, loader, device, dtype, max_batches=10**9)
        bpb = float(metrics["bpb_all_bytes"])
        n_bytes = int(metrics["n_bytes"])
        out[domain] = bpb
        total_nats += bpb * LN2 * n_bytes
        total_bytes += n_bytes
    out["bpb_all_bytes"] = bpb_from_nats(total_nats, total_bytes) if total_bytes else 0.0
    return out


def _fixed_prompts(model, device: torch.device) -> list[dict]:
    rows = []
    for prompt in FIXED_PROMPTS:
        sample = sample_prompt(model, prompt, device, temperature=0.6, top_p=0.9, max_new=128)
        rows.append({"prompt": prompt, "text": sample["text"][:400], "stopped": sample["stopped"]})
    return rows


FULL_BENCHMARK_N = {"hellaswag": 10042, "piqa": 1838, "arc_easy": 0, "lambada": 5153}


def _run_benchmark(model, name: str, n_examples: int = 500) -> dict:
    import importlib.util

    spec = importlib.util.spec_from_file_location("eval_suite", ROOT / "scripts" / "eval_suite.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run_benchmark(model, "patch", name, n_examples=n_examples)


def build(grad_ckpt: bool, d_model: int, n_layers: int, n_heads: int) -> ContinuousPool4LM:
    return ContinuousPool4LM(
        {
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "ffn_mult": 4,
            "dropout": 0.0,
            "pool_factor": 4,
            "vocab_size": 256,
            "max_len": WINDOW,
            "pos_type": "rope",
            "gradient_checkpointing": grad_ckpt,
            "local_window": 64,
        }
    )


def pack_rng(state: dict) -> dict:
    packed = {
        "python": state["python"],
        "numpy": state["numpy"],
        "torch": state["torch"].detach().cpu().numpy().tolist(),
    }
    if "cuda" in state:
        packed["cuda"] = [item.detach().cpu().numpy().tolist() for item in state["cuda"]]
    return packed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--spans", type=Path, required=True)
    parser.add_argument("--perm", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=Path("./checkpoints/local64_ddp/ddp_local64.pt"))
    parser.add_argument("--bounds", default="", help="fineweb_end,code_end byte offsets; empty uses real_v2 constants")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--accum", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--target-bytes", type=int, default=2_000_000_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save-every-s", type=float, default=3600.0)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--fused", action="store_true")
    parser.add_argument("--grad-ckpt", action="store_true")
    parser.add_argument("--socket-ifname", default="")
    parser.add_argument("--d-model", type=int, default=896)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--heads", type=int, default=14)
    parser.add_argument("--val-dir", type=Path, default=None)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--curve", type=Path, default=None)
    parser.add_argument("--compare-val", type=Path, default=None)
    parser.add_argument("--warmup-frac", type=float, default=0.02)
    parser.add_argument("--min-frac", type=float, default=0.1)
    parser.add_argument("--init-from", type=Path, default=None, help="Load model weights only when --resume does not exist")
    parser.add_argument("--scaled-init", action="store_true")
    parser.add_argument("--keep-checkpoints", type=int, default=0)
    parser.add_argument("--permanent-every-bytes", type=int, default=0)
    parser.add_argument("--benchmarks", action="store_true")
    args = parser.parse_args()
    if args.socket_ifname:
        os.environ["NCCL_SOCKET_IFNAME"] = args.socket_ifname
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=6))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dtype = get_autocast_dtype("bf16")
    set_seed(0)
    model = build(args.grad_ckpt, args.d_model, args.layers, args.heads).to(device)
    if args.scaled_init:
        init_report = apply_scaled_residual_init(model)
        if rank == 0:
            print({"scaled_init": init_report}, flush=True)
    if args.init_from is not None and not args.resume.exists():
        blob = torch.load(args.init_from, map_location="cpu", weights_only=False)
        state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
        if any(str(key).startswith("_orig_mod.") for key in state):
            state = {str(key).replace("_orig_mod.", "", 1): value for key, value in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise SystemExit(f"init-from mismatch missing={len(missing)} unexpected={len(unexpected)}")
        if rank == 0:
            print(
                {
                    "init_from": str(args.init_from),
                    "fresh_adamw": True,
                    "base_step": int(blob.get("step", -1)) if isinstance(blob, dict) else -1,
                    "base_seen": int(blob.get("seen", -1)) if isinstance(blob, dict) else -1,
                },
                flush=True,
            )
        del state, blob
    if rank == 0:
        print(
            {
                "config": {
                    "d_model": args.d_model,
                    "layers": args.layers,
                    "heads": args.heads,
                    "local_window": 64,
                    "total": count_parameters(model),
                    "non_embedding": count_parameters(model, non_embedding=True),
                    "train_flops_per_byte": train_flops_per_byte(args.d_model, args.layers),
                    "grad_ckpt": args.grad_ckpt,
                    "batch": args.batch,
                    "accum": args.accum,
                    "global_bytes_per_step": args.batch * args.accum * WINDOW * world,
                }
            },
            flush=True,
        )
    if args.compile:
        model = torch.compile(model)
    model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
    bounds = None
    if args.bounds:
        left, right = args.bounds.split(",")
        bounds = (int(left), int(right))
    cursor = ShuffledDocCursor(
        args.train,
        WINDOW,
        args.spans,
        args.perm,
        rank=rank,
        world_size=world,
        bounds=bounds,
    )
    stop = {"stop": False}

    def _handle(signum, _frame):
        stop["stop"] = True
        print({"signal": signum, "rank": rank}, flush=True)

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)

    def align(flag: bool) -> bool:
        tensor = torch.tensor([1 if flag else 0], device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return bool(int(tensor.item()))

    def gather(local: dict) -> list:
        packed = {"cursor": local["cursor"], "rng": pack_rng(local["rng"])}
        gathered: list = [None] * world
        dist.all_gather_object(gathered, packed)
        return gathered

    args.resume.parent.mkdir(parents=True, exist_ok=True)
    eval_fn = None
    if args.val_dir is not None:
        def eval_fn(m, val_dir=args.val_dir):
            scores = domain_bpb(m, val_dir, device, dtype)
            if args.benchmarks:
                raw = m.module if hasattr(m, "module") else m
                per_rank = args.batch * args.accum * WINDOW
                use_full = int(getattr(raw, "_eval_step", 0)) * per_rank >= args.target_bytes
                for name in ("hellaswag", "piqa", "arc_easy", "lambada"):
                    try:
                        n_examples = FULL_BENCHMARK_N[name] if use_full else 500
                        row = _run_benchmark(raw, name, n_examples)
                        scores[name] = {
                            "n": row["n"],
                            "accuracy": row["accuracy"],
                            "ci95": row["accuracy_ci95_wilson"],
                        }
                    except Exception as exc:
                        scores[name] = {"error": str(exc)}
                if rank == 0:
                    try:
                        scores["prompts"] = _fixed_prompts(raw, device)
                    except Exception as exc:
                        scores["prompts"] = [{"error": str(exc)}]
                    try:
                        import importlib.util

                        fact_spec = importlib.util.spec_from_file_location(
                            "score_fact200", ROOT / "scripts" / "score_fact200.py"
                        )
                        fact_mod = importlib.util.module_from_spec(fact_spec)
                        fact_spec.loader.exec_module(fact_mod)
                        fact = fact_mod.score_model(raw, device)
                        scores["fact200"] = {
                            "n": fact["n"],
                            "correct": fact["correct"],
                            "accuracy": fact["accuracy"],
                            "categories": fact["categories"],
                        }
                    except Exception as exc:
                        scores["fact200"] = {"error": str(exc)}
                dist.barrier()
            return scores
    result = train_ladder_run(
        run_id=f"ddp_local64_r{rank}",
        model=model,
        loss_fn=patch_loss,
        cursor=cursor,
        resume_path=args.resume,
        target_bytes=args.target_bytes,
        peak_lr=args.lr,
        batch_size=args.batch,
        accum=args.accum,
        device=device,
        dtype=dtype,
        eval_fn=eval_fn,
        max_steps=args.max_steps or None,
        eval_every_steps=args.eval_every or 10**9,
        curve_path=args.curve if rank == 0 else None,
        save_every_s=args.save_every_s,
        adamw_kwargs={"fused": True} if args.fused else None,
        record_steps=True,
        stop_flag=stop,
        write_checkpoint=(rank == 0),
        gather_fn=gather,
        align_fn=align,
        config={"rank": rank, "world_size": world, "batch": args.batch, "accum": args.accum},
        mirror_path=None,
        warmup_frac=args.warmup_frac,
        min_frac=args.min_frac,
        keep_last=args.keep_checkpoints,
        permanent_every_bytes=args.permanent_every_bytes,
    )
    cursor.close()
    per_rank_bytes = args.batch * args.accum * WINDOW
    steady = steady_bytes_per_second(result.get("step_log"), per_rank_bytes)
    print(
        {
            "rank": rank,
            "steps": result["optimizer_steps"],
            "seen": result["bytes_seen"],
            "stopped": result["stopped"],
            "bytes_per_second": result["bytes_per_second"],
            "steady_bytes_per_second": steady,
            "peak_vram_mb": result["peak_vram_mb"],
            "final": result.get("final"),
            "history": result.get("history"),
            "cursor": cursor.state(),
            "spikes": sum(1 for row in (result.get("step_log") or []) if row.get("spike")),
        },
        flush=True,
    )
    if rank == 0:
        print(f"RUN_DONE stopped={result['stopped']} steps={result['optimizer_steps']}", flush=True)
    if rank == 0 and args.compare_val is not None:
        compared = domain_bpb(model, args.compare_val, device, dtype)
        print({"compare_val": str(args.compare_val), **compared}, flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
