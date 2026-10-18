"""Resumable byte / BPE / patch-4 training for the small-model ladder.

Every optimizer step consumes the same number of raw bytes. Checkpoints store
model, optimizer, RNG, step, and the packed-file cursor.
"""

from __future__ import annotations

import contextlib
import math
import random
import shutil
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast

from lct.eval.lm_metrics import LN2, peak_vram_mb
from lct.utils import count_parameters, get_autocast_dtype, get_device


def cosine_lr(step: int, total_steps: int, peak: float, warmup_frac: float = 0.02, min_frac: float = 0.1) -> float:
    warm = max(1, int(total_steps * warmup_frac))
    if step < warm:
        return peak * float(step + 1) / float(warm)
    span = max(1, total_steps - warm)
    progress = min(1.0, max(0.0, (step - warm) / span))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak * (min_frac + (1.0 - min_frac) * cosine)


class ByteCursor:
    """Sequential windows over a memmap. ``offset`` is the next byte to read."""

    def __init__(self, path: Path, window: int, offset: int = 0):
        self.path = Path(path)
        self.window = int(window)
        self.offset = int(offset)
        self._file = self.path.open("rb")
        self._mm = __import__("mmap").mmap(self._file.fileno(), 0, access=__import__("mmap").ACCESS_READ)
        self.n = len(self._mm)
        if self.n < self.window:
            raise RuntimeError(f"{self.path} is shorter than {self.window}")

    def next_window(self) -> bytes:
        if self.offset + self.window > self.n:
            self.offset = 0
        start = self.offset
        self.offset = start + self.window
        return bytes(self._mm[start : start + self.window])

    def close(self) -> None:
        self._mm.close()
        self._file.close()


FINEWEB_END = 2_100_001_754
CODE_END = FINEWEB_END + 600_012_197
SHUFFLE_SEED = 0


def source_at(offset: int, bounds: tuple[int, int] | None = None) -> str:
    fineweb_end, code_end = (FINEWEB_END, CODE_END) if bounds is None else bounds
    if offset < fineweb_end:
        return "fineweb-edu"
    if offset < code_end:
        return "codeparrot-clean-train"
    return "open-web-math"


class ShuffledDocCursor:
    """Windows from one fixed permutation of documents. Does not wrap.

    ``order`` is the same file for every system. Reading stops when the
    permutation is exhausted instead of starting a second epoch.
    """

    def __init__(
        self,
        path: Path,
        window: int,
        spans_path: Path,
        perm_path: Path,
        rank: int = 0,
        world_size: int = 1,
        bounds: tuple[int, int] | None = None,
    ):
        self.path = Path(path)
        self.window = int(window)
        self.offset = 0
        self.bounds = bounds
        self.rank = int(rank)
        self.world_size = int(world_size)
        self._file = self.path.open("rb")
        self._mm = __import__("mmap").mmap(self._file.fileno(), 0, access=__import__("mmap").ACCESS_READ)
        self.spans = np.load(spans_path)
        order = np.load(perm_path)
        if self.world_size > 1:
            order = order[self.rank :: self.world_size]
        self.order = order
        self.doc_i = 0
        self.pos_in_doc = 0
        self._cum = {"fineweb-edu": 0, "codeparrot-clean-train": 0, "open-web-math": 0}
        self._win = dict(self._cum)

    def _current_chunk(self) -> tuple[bytes, str]:
        if self.doc_i >= len(self.order):
            raise RuntimeError("shuffled document stream is exhausted; this cursor does not start another epoch")
        start, length = (int(x) for x in self.spans[int(self.order[self.doc_i])])
        raw = bytes(self._mm[start : start + length]) + bytes([0x1E])
        return raw, source_at(start, self.bounds)

    def next_window(self) -> bytes:
        buf = bytearray()
        while len(buf) < self.window:
            chunk, source = self._current_chunk()
            take = chunk[self.pos_in_doc : self.pos_in_doc + (self.window - len(buf))]
            buf += take
            self._cum[source] += len(take)
            self._win[source] += len(take)
            self.pos_in_doc += len(take)
            self.offset += len(take)
            if self.pos_in_doc >= len(chunk):
                self.doc_i += 1
                self.pos_in_doc = 0
        return bytes(buf)

    def source_share(self, reset_window: bool = True) -> dict:
        def pack(counts: dict) -> dict:
            total = sum(counts.values()) or 1
            return {"bytes": dict(counts), "share": {k: v / total for k, v in counts.items()}}

        out = {"cumulative": pack(self._cum), "interval": pack(self._win)}
        if reset_window:
            self._win = {k: 0 for k in self._win}
        return out

    def state(self) -> dict:
        return {
            "doc_i": self.doc_i,
            "pos_in_doc": self.pos_in_doc,
            "offset": self.offset,
            "cum": dict(self._cum),
            "win": dict(self._win),
        }

    def load_state(self, state: dict) -> None:
        self.doc_i = int(state["doc_i"])
        self.pos_in_doc = int(state["pos_in_doc"])
        self.offset = int(state["offset"])
        self._cum = dict(state["cum"])
        self._win = dict(state["win"])

    def close(self) -> None:
        self._mm.close()
        self._file.close()


def _rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _as_byte_tensor(value) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        value = torch.tensor(value)
    return value.detach().cpu().byte().contiguous()


def _load_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(_as_byte_tensor(state["torch"]))
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all([_as_byte_tensor(item) for item in state["cuda"]])


def _retain_checkpoints(
    path: Path,
    *,
    step: int,
    seen: int,
    world: int,
    keep_last: int,
    permanent_every_bytes: int,
) -> None:
    """Keep a few recent copies, plus one file at each permanent byte boundary."""
    try:
        if keep_last > 1:
            dest = path.with_name(f"{path.stem}_s{step}{path.suffix}")
            # copyfile, not copy2: the NAS rejects utime inside copystat.
            shutil.copyfile(path, dest)
            rolls = sorted(
                path.parent.glob(f"{path.stem}_s*{path.suffix}"),
                key=lambda item: item.stat().st_mtime,
                reverse=True,
            )
            for old in rolls[keep_last:]:
                old.unlink(missing_ok=True)
        if permanent_every_bytes > 0 and world > 0:
            global_seen = int(seen) * int(world)
            mark = global_seen // int(permanent_every_bytes)
            for k in range(1, int(mark) + 1):
                label = k * int(permanent_every_bytes) // 1_000_000_000
                held = path.with_name(f"{path.stem}_{label}b{path.suffix}")
                if not held.exists():
                    shutil.copyfile(path, held)
    except OSError as exc:
        print({"checkpoint_retain_error": str(exc), "step": step}, flush=True)


def save_resume(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def train_ladder_run(
    *,
    run_id: str,
    model: torch.nn.Module,
    loss_fn: Callable[[torch.nn.Module, torch.Tensor], torch.Tensor],
    cursor: ByteCursor,
    resume_path: Path,
    target_bytes: int,
    peak_lr: float,
    batch_size: int,
    accum: int,
    device: torch.device,
    dtype: torch.dtype | None,
    eval_fn: Callable[[torch.nn.Module], dict] | None = None,
    config: dict | None = None,
    eval_every_steps: int = 200,
    save_every_s: float = 45 * 60,
    max_steps: int | None = None,
    mirror_path: Path | None = None,
    curve_path: Path | None = None,
    adamw_kwargs: dict | None = None,
    record_steps: bool = False,
    stop_flag: dict | None = None,
    write_checkpoint: bool = True,
    gather_fn: Callable[[dict], list] | None = None,
    align_fn: Callable[[bool], bool] | None = None,
    warmup_frac: float = 0.02,
    min_frac: float = 0.1,
    keep_last: int = 0,
    permanent_every_bytes: int = 0,
) -> dict[str, Any]:
    """Train until ``target_bytes`` raw bytes have been read. Resumes if a checkpoint exists."""
    window_bytes = cursor.window * batch_size * accum
    total_steps = max(1, int(math.ceil(target_bytes / window_bytes)))
    opt_kwargs = {"lr": peak_lr, "betas": (0.9, 0.95), "weight_decay": 0.1}
    if adamw_kwargs:
        opt_kwargs.update(adamw_kwargs)
    try:
        opt = torch.optim.AdamW(model.parameters(), **opt_kwargs)
    except (TypeError, ValueError, RuntimeError) as exc:
        if "fused" not in opt_kwargs:
            raise
        print({"adamw_fused_unavailable": str(exc)}, flush=True)
        opt_kwargs.pop("fused", None)
        opt = torch.optim.AdamW(model.parameters(), **opt_kwargs)
        opt_kwargs["fused_fallback"] = "foreach-default"
    scaler = GradScaler(enabled=(dtype == torch.float16))
    step = 0
    seen = 0
    best = float("inf")
    history: list[dict] = []
    model.to(device)
    if resume_path.exists():
        blob = torch.load(resume_path, map_location=device, weights_only=False)
        target = model.module if hasattr(model, "module") else model
        target.load_state_dict(blob["model"])
        opt.load_state_dict(blob["opt"])
        for group_state in opt.state.values():
            for key, value in list(group_state.items()):
                if torch.is_tensor(value):
                    group_state[key] = value.to(device)
        scaler.load_state_dict(blob["scaler"])
        step = int(blob["step"])
        seen = int(blob["seen"])
        rank = int((config or {}).get("rank", 0))
        cursor_states = blob.get("cursor_states")
        if hasattr(cursor, "load_state") and cursor_states:
            cursor.load_state(cursor_states[rank])
        elif hasattr(cursor, "load_state") and blob.get("cursor_state"):
            cursor.load_state(blob["cursor_state"])
        else:
            cursor.offset = int(blob["data_offset"])
        best = float(blob.get("best", best))
        history = list(blob.get("history", []))
        rng_states = blob.get("rng_states")
        _load_rng(rng_states[rank] if rng_states else blob["rng"])
        print({"resumed": run_id, "step": step, "seen": seen, "offset": cursor.offset, "rank": rank}, flush=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    wall_offset = float(blob["wall_offset"]) if resume_path.exists() and "wall_offset" in blob else 0.0
    t0 = time.perf_counter()
    last_save = time.perf_counter()
    use_amp = dtype is not None and device.type == "cuda"
    model.train()
    step_log: list[dict] = []
    sync_off = getattr(model, "no_sync", None)

    def checkpoint(extra: dict | None = None) -> None:
        local_cursor = cursor.state() if hasattr(cursor, "state") else {"offset": int(cursor.offset)}
        gathered = None
        if gather_fn is not None:
            gathered = gather_fn({"cursor": local_cursor, "rng": _rng_state()})
        if not write_checkpoint:
            return
        raw = model.module if hasattr(model, "module") else model
        payload = {
            "model": raw.state_dict(),
            "opt": opt.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "seen": seen,
            "data_offset": cursor.offset,
            "cursor_state": local_cursor if not hasattr(cursor, "state") else cursor.state(),
            "wall_offset": wall_offset + (time.perf_counter() - t0),
            "best": best,
            "history": history,
            "rng": _rng_state(),
            "run_id": run_id,
            "config": config or {},
            "world_size": int((config or {}).get("world_size", 1)),
            "rank": int((config or {}).get("rank", 0)),
        }
        if gathered is not None:
            payload["cursor_states"] = [item["cursor"] for item in gathered]
            payload["rng_states"] = [item["rng"] for item in gathered]
        if extra:
            payload.update(extra)
        save_resume(resume_path, payload)
        _retain_checkpoints(
            resume_path,
            step=step,
            seen=seen,
            world=int((config or {}).get("world_size", 1)),
            keep_last=keep_last,
            permanent_every_bytes=permanent_every_bytes,
        )
        if mirror_path is not None:
            mirror_path.mkdir(parents=True, exist_ok=True)
            dest = mirror_path / resume_path.name
            tmp = dest.with_suffix(dest.suffix + ".mirror_tmp")
            shutil.copyfile(resume_path, tmp)
            tmp.replace(dest)

    stopped = False
    loss_sum = 0.0
    loss_n = 0
    evaluated_this_step = False
    while seen < target_bytes and (max_steps is None or step < max_steps):
        evaluated_this_step = False
        stop_now = bool(stop_flag is not None and stop_flag.get("stop"))
        if align_fn is not None:
            stop_now = align_fn(stop_now)
        if stop_now and step > 0:
            checkpoint()
            stopped = True
            print({"stopped": run_id, "step": step, "seen": seen}, flush=True)
            break
        t_step = time.perf_counter()
        lr = cosine_lr(step, total_steps, peak_lr, warmup_frac=warmup_frac, min_frac=min_frac)
        for group in opt.param_groups:
            group["lr"] = lr
        opt.zero_grad(set_to_none=True)
        loss_value = 0.0
        for _micro in range(accum):
            windows = [cursor.next_window() for _ in range(batch_size)]
            batch = torch.tensor([list(w) for w in windows], dtype=torch.long, device=device)
            ctx = sync_off() if (sync_off is not None and _micro < accum - 1) else contextlib.nullcontext()
            with ctx:
                with autocast(enabled=use_amp, dtype=dtype if use_amp else torch.float32):
                    loss = loss_fn(model, batch) / accum
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            loss_value += float(loss.detach())
            seen += batch.numel()
        if scaler.is_enabled():
            scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        step += 1
        running_mean = (loss_sum / loss_n) if loss_n else float(loss_value)
        spike = bool(loss_n >= 20 and float(loss_value) > 2.0 * running_mean)
        loss_sum += float(loss_value)
        loss_n += 1
        row = {
            "step": step,
            "loss": loss_value,
            "grad_norm": float(grad_norm),
            "seconds": time.perf_counter() - t_step,
            "seen": seen,
            "lr": lr,
            "spike": spike,
        }
        if record_steps:
            step_log.append(row)
        if spike:
            print(
                {
                    "spike": run_id,
                    "step": step,
                    "loss": round(loss_value, 4),
                    "running_mean": round(running_mean, 4),
                    "grad_norm": round(float(grad_norm), 4),
                },
                flush=True,
            )
        if curve_path is not None and int((config or {}).get("rank", 0)) == 0:
            import json

            curve_path.parent.mkdir(parents=True, exist_ok=True)
            with curve_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"run_id": run_id, "kind": "step", **row}) + "\n")
        if step % 50 == 0:
            print(
                {
                    "run": run_id,
                    "step": step,
                    "seen": seen,
                    "loss": round(loss_value, 4),
                    "grad_norm": round(float(grad_norm), 4),
                    "lr": lr,
                },
                flush=True,
            )
        if eval_fn is not None and step % eval_every_steps == 0:
            raw_model = model.module if hasattr(model, "module") else model
            raw_model._eval_step = step
            evaluated_this_step = True
            metrics = eval_fn(model)
            metrics["step"] = step
            metrics["seen"] = seen
            metrics["wall_seconds"] = wall_offset + (time.perf_counter() - t0)
            if hasattr(cursor, "source_share"):
                metrics["source_share"] = cursor.source_share(reset_window=True)
                print({"source_share": run_id, "seen": seen, **metrics["source_share"]["interval"]["share"]}, flush=True)
            history.append(metrics)
            bpb = float(metrics.get("bpb_all_bytes", metrics.get("bpb", 99.0)))
            if bpb < best:
                best = bpb
            model.train()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print({"eval": run_id, **metrics}, flush=True)
            if curve_path is not None:
                curve_path.parent.mkdir(parents=True, exist_ok=True)
                import json

                with curve_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"run_id": run_id, **metrics}) + "\n")
        due = time.perf_counter() - last_save >= save_every_s
        if align_fn is not None:
            due = align_fn(due)
        if due:
            checkpoint()
            last_save = time.perf_counter()
            print({"saved": str(resume_path), "step": step}, flush=True)
    if eval_fn is not None and not stopped and not evaluated_this_step:
        raw_model = model.module if hasattr(model, "module") else model
        raw_model._eval_step = step
        final = eval_fn(model)
    else:
        final = dict(history[-1]) if history else {}
    wall = time.perf_counter() - t0
    model.train()
    result = {
        "run_id": run_id,
        "bytes_seen": int(seen),
        "optimizer_steps": int(step),
        "wall_seconds": wall,
        "bytes_per_second": seen / max(wall, 1e-9),
        "peak_vram_mb": peak_vram_mb(),
        "parameters": {
            "total": count_parameters(model, non_embedding=False),
            "non_embedding": count_parameters(model, non_embedding=True),
        },
        "best_bpb": best if best < float("inf") else final.get("bpb_all_bytes"),
        "final": final,
        "history": history,
        "checkpoint": str(resume_path),
        "stopped": stopped,
        "step_log": step_log if record_steps else None,
        "adamw": {k: v for k, v in opt_kwargs.items() if k != "lr"},
        "config": {
            "target_bytes": target_bytes,
            "peak_lr": peak_lr,
            "batch_size": batch_size,
            "accum": accum,
            "window": cursor.window,
            "warmup_frac": warmup_frac,
            "min_frac": min_frac,
        },
    }
    if not stopped:
        checkpoint({"result": result})
    return result


def byte_loss(model: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
    inp, tgt = batch[:, :-1], batch[:, 1:]
    return model(inp, targets=tgt)["loss"]


def patch_loss(model: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
    return model(batch)["loss"]


def make_bpe_loss(tokenizer, pad_id: int = 0):
    def loss_fn(model: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
        texts = []
        for row in batch.tolist():
            raw = bytes(int(b) for b in row)
            texts.append(raw.decode("utf-8", errors="replace"))
        encoded = [tokenizer.encode(t).ids or [pad_id] for t in texts]
        width = max(len(ids) for ids in encoded)
        width = max(width, 2)
        tokens = torch.full((len(encoded), width), pad_id, dtype=torch.long, device=batch.device)
        for i, ids in enumerate(encoded):
            tokens[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=batch.device)
        inp, tgt = tokens[:, :-1], tokens[:, 1:]
        return model(inp, targets=tgt)["loss"]

    return loss_fn


def bpb_from_nats(nats: float, n_bytes: int) -> float:
    return (nats / max(n_bytes, 1)) / LN2
