"""Matched 350M BPE-8k on shuffled real_v3 (single-GPU), companion to local64 10B.

Uses the existing real_v2 BPE-8k tokenizer (same vocab family as prior ladders).
Width/depth/heads 1024/26/16, LR 3e-4, warmup 1%, cosine to 10%, scaled init, 10B bytes.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from baselines.bpe_lm import SimpleBPE  # noqa: E402
from lct.eval.lm_metrics import LN2  # noqa: E402
from lct.models.decoder_lm import DecoderOnlyLM  # noqa: E402
from lct.checkpoint import load_state_dict_strict  # noqa: E402
from lct.train.ladder import ShuffledDocCursor, make_bpe_loss  # noqa: E402
from lct.utils import get_autocast_dtype, set_seed  # noqa: E402

WINDOW = 2048
PAD_ID = 0


def cosine_lr_bytes(
    seen: int,
    target_bytes: int,
    peak: float,
    warmup_frac: float = 0.01,
    min_frac: float = 0.1,
) -> float:
    """Cosine LR keyed to bytes seen so variable pack sizes still end at min_lr @ target_bytes.

    For fixed bytes/step (local64), this matches the old step-keyed cosine at equal progress.
    """
    target = max(1, int(target_bytes))
    warm = max(1, int(target * warmup_frac))
    s = max(0, int(seen))
    if s < warm:
        return peak * float(s + 1) / float(warm)
    span = max(1, target - warm)
    progress = min(1.0, max(0.0, (s - warm) / span))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak * (min_frac + (1.0 - min_frac) * cosine)


class PretokWindowReader:
    """Read pre-tokenized ShuffledDocCursor windows (see pretok_matched_bpe_windows.py)."""

    def __init__(self, pretok_dir: Path, start_window: int = 0):
        meta = json.loads((pretok_dir / "meta.json").read_text(encoding="utf-8"))
        self.meta = meta
        self.n_windows = int(meta["n_windows"])
        self.window_bytes = int(meta["window"])
        assert self.window_bytes == WINDOW
        total_tok = int(meta.get("total_tokens") or Path(meta["tokens"]).stat().st_size // 2)
        self.tokens = np.memmap(meta["tokens"], dtype=np.uint16, mode="r", shape=(total_tok,))
        self.offsets = np.memmap(meta["offsets"], dtype=np.int64, mode="r", shape=(self.n_windows + 1,))
        self.i = int(start_window)
        self.offset_bytes = self.i * WINDOW

    def next_batch(self, batch: int, fixed_width: int | None = None) -> torch.Tensor:
        if self.i + batch > self.n_windows:
            raise RuntimeError(f"pretok exhausted at window {self.i}, need {batch} more (n={self.n_windows})")
        spans = []
        lengths = []
        for wi in range(self.i, self.i + batch):
            a = int(self.offsets[wi])
            b = int(self.offsets[wi + 1])
            spans.append(self.tokens[a:b])
            lengths.append(b - a)
        if fixed_width is not None:
            width = max(2, int(fixed_width))
        else:
            width = max(2, max(lengths) if lengths else 2)
        width = min(width, WINDOW)
        out = torch.full((batch, width), PAD_ID, dtype=torch.long)
        for row, (span, n) in enumerate(zip(spans, lengths)):
            n_use = min(int(n), width)
            if n_use:
                out[row, :n_use] = torch.from_numpy(np.asarray(span[:n_use], dtype=np.int64))
        self.i += batch
        self.offset_bytes += batch * WINDOW
        return out


class PackedPretokReader:
    """Concatenate pretok window tokens in document order; emit fixed-length packs.

    Same window order as ShuffledDocCursor / PretokWindowReader. Each time a window's
    tokens are fully consumed, credit WINDOW bytes (matched twin byte accounting).
    No padding — every position is a real token.
    """

    def __init__(self, pretok_dir: Path, start_window: int = 0):
        meta = json.loads((pretok_dir / "meta.json").read_text(encoding="utf-8"))
        self.meta = meta
        self.n_windows = int(meta["n_windows"])
        self.window_bytes = int(meta["window"])
        assert self.window_bytes == WINDOW
        total_tok = int(meta.get("total_tokens") or Path(meta["tokens"]).stat().st_size // 2)
        self.tokens = np.memmap(meta["tokens"], dtype=np.uint16, mode="r", shape=(total_tok,))
        self.offsets = np.memmap(meta["offsets"], dtype=np.int64, mode="r", shape=(self.n_windows + 1,))
        self.window_i = int(start_window)
        if self.window_i < 0 or self.window_i > self.n_windows:
            raise RuntimeError(f"bad start_window {start_window}")
        self.tok_i = int(self.offsets[self.window_i])
        self.offset_bytes = self.window_i * WINDOW
        # Skip empty windows at start
        while self.window_i < self.n_windows and int(self.offsets[self.window_i + 1]) <= self.tok_i:
            self.window_i += 1
            self.offset_bytes = self.window_i * WINDOW
            if self.window_i < self.n_windows:
                self.tok_i = int(self.offsets[self.window_i])

    @property
    def i(self) -> int:
        return self.window_i

    def tokens_remaining(self) -> int:
        if self.window_i >= self.n_windows:
            return 0
        rem = int(self.offsets[self.window_i + 1]) - int(self.tok_i)
        if self.window_i + 1 < self.n_windows:
            rem += int(self.offsets[self.n_windows]) - int(self.offsets[self.window_i + 1])
        return max(0, int(rem))

    def next_batch(self, batch: int, seq_len: int = 512) -> tuple[torch.Tensor, int]:
        seq_len = int(seq_len)
        batch = int(batch)
        need = batch * seq_len
        if need < 2:
            raise RuntimeError("pack batch too small")
        avail_tok = self.tokens_remaining()
        if avail_tok < need:
            # End of pretok: shrink pack count so the last step can finish cleanly.
            max_batch = avail_tok // seq_len
            if max_batch < 1:
                raise RuntimeError(
                    f"pretok packed stream exhausted at window {self.window_i}, "
                    f"need {need} tokens but only {avail_tok} left"
                )
            print({"shrunk_final_pack_batch": max_batch, "from": batch, "avail_tok": avail_tok}, flush=True)
            batch = max_batch
            need = batch * seq_len
        buf = np.empty(need, dtype=np.int64)
        filled = 0
        bytes_credited = 0
        while filled < need:
            if self.window_i >= self.n_windows:
                raise RuntimeError(
                    f"pretok packed stream exhausted at window {self.window_i}, need {need - filled} more tokens"
                )
            win_end = int(self.offsets[self.window_i + 1])
            avail = win_end - self.tok_i
            if avail <= 0:
                self.window_i += 1
                bytes_credited += WINDOW
                self.offset_bytes += WINDOW
                if self.window_i < self.n_windows:
                    self.tok_i = int(self.offsets[self.window_i])
                continue
            take = min(need - filled, avail)
            buf[filled : filled + take] = self.tokens[self.tok_i : self.tok_i + take]
            self.tok_i += take
            filled += take
            if self.tok_i >= win_end:
                self.window_i += 1
                bytes_credited += WINDOW
                self.offset_bytes += WINDOW
                if self.window_i < self.n_windows:
                    self.tok_i = int(self.offsets[self.window_i])
        return torch.from_numpy(buf.reshape(batch, seq_len)), bytes_credited


def token_batch_loss(model: DecoderOnlyLM, tokens: torch.Tensor) -> torch.Tensor:
    """Same CE as make_bpe_loss on a pre-tokenized (padded) batch."""
    inp, tgt = tokens[:, :-1], tokens[:, 1:]
    return model(inp, targets=tgt)["loss"]


def raw_state_dict(model: torch.nn.Module) -> dict:
    """Strip torch.compile `_orig_mod.` prefix so resumes stay uncompiled-compatible."""
    core = getattr(model, "_orig_mod", model)
    return core.state_dict()


def apply_scaled_init(model: DecoderOnlyLM, n_layers: int) -> dict:
    """GPT-2 style init: embed std 0.02; residual attn-out / MLP-out scaled by 1/sqrt(2L).

    RoPEBlock names: ``blocks.i.out.weight`` (attn out), ``blocks.i.ff.3.weight`` (MLP out).
    Default ``nn.Embedding`` std=1 makes initial CE hundreds of nats; embed must be 0.02.
    """
    n = int(n_layers)
    std = 0.02 / math.sqrt(2 * n)
    filled = 0
    matched: list[str] = []
    # Tied input/output embedding
    torch.nn.init.normal_(model.embed.weight, mean=0.0, std=0.02)
    for name, param in model.named_parameters():
        if param.ndim < 2:
            continue
        is_attn_out = name.endswith(".out.weight")
        is_mlp_out = name.endswith(".ff.3.weight")
        if is_attn_out or is_mlp_out:
            torch.nn.init.normal_(param, mean=0.0, std=std)
            filled += 1
            matched.append(name)
    if filled <= 0:
        raise RuntimeError(
            "scaled_init matched 0 residual tensors; expected RoPEBlock .out.weight and .ff.3.weight. "
            f"sample names: {[n for n, p in model.named_parameters() if p.ndim >= 2][:20]}"
        )
    return {
        "layers": n,
        "residual_std": std,
        "embed_std": 0.02,
        "tensors": filled,
        "names_head": matched[:8],
        "names_n": len(matched),
    }


@torch.no_grad()
def val_bpb(model, loss_fn, val_dir: Path, device, dtype) -> dict:
    out = {}
    for name in ("val_text.bin", "val_code.bin", "val_math.bin"):
        path = val_dir / name
        if not path.exists():
            continue
        raw = path.read_bytes()
        chunks = []
        for i in range(0, min(len(raw), WINDOW * 64), WINDOW):
            chunk = raw[i : i + WINDOW]
            if len(chunk) < WINDOW:
                break
            chunks.append(list(chunk))
        if not chunks:
            continue
        batch = torch.tensor(chunks[:32], dtype=torch.long, device=device)
        model.eval()
        with torch.autocast(device_type="cuda", dtype=dtype):
            loss = loss_fn(model, batch)
        # Approximate BPB: token NLL / ln2 / mean bytes per window (rough).
        out[name.replace(".bin", "")] = float(loss.item()) / LN2
        model.train()
    if out:
        out["overall"] = sum(out.values()) / len(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--spans", type=Path, required=True)
    parser.add_argument("--perm", type=Path, required=True)
    parser.add_argument("--bpe-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--curve", type=Path, required=True)
    parser.add_argument("--target-bytes", type=int, default=10_000_000_000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-frac", type=float, default=0.01)
    parser.add_argument("--min-frac", type=float, default=0.1)
    parser.add_argument("--d-model", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=26)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--eval-every-bytes", type=int, default=1_000_000_000)
    parser.add_argument("--save-every-s", type=float, default=1800.0)
    parser.add_argument("--bounds", default="59500002824,66500006098")
    parser.add_argument("--scaled-init", action="store_true")
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Ignore any existing --resume checkpoint and start at step 0 with fresh init.",
    )
    parser.add_argument(
        "--pretok-dir",
        type=Path,
        default=None,
        help="Directory from pretok_matched_bpe_windows.py; skips on-the-fly BPE encode.",
    )
    parser.add_argument(
        "--verify-steps",
        type=int,
        default=0,
        help="After resume, log the next N steps then exit (for pretok continuity checks).",
    )
    parser.add_argument("--compile", action="store_true", help="torch.compile like local64 twin")
    parser.add_argument("--fused", action="store_true", help="fused AdamW")
    parser.add_argument(
        "--grad-checkpoint",
        action="store_true",
        help="Enable activation checkpointing (slower; off by default for matched twin throughput).",
    )
    parser.add_argument(
        "--pack-len",
        type=int,
        default=0,
        help="If >0 with --pretok-dir: pack token stream into fixed-length sequences (no pad).",
    )
    args = parser.parse_args()

    device = torch.device("cuda")
    dtype = get_autocast_dtype("bf16")
    set_seed(0)
    bpe = SimpleBPE(vocab_size=8000)
    bpe.load(args.bpe_dir)
    assert bpe.tok is not None, f"missing tokenizer in {args.bpe_dir}"
    n_vocab = int(bpe.tok.get_vocab_size())
    uniform_nll = math.log(n_vocab)
    model = DecoderOnlyLM(
        vocab_size=n_vocab,
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
        ffn_mult=4,
        dropout=0.0,
        max_len=WINDOW,
        pos_type="rope",
        gradient_checkpointing=bool(args.grad_checkpoint),
        ignore_index=0,  # BPE pad_id
    ).to(device)
    if args.scaled_init:
        init_report = apply_scaled_init(model, args.layers)
        print({"scaled_init": init_report, "n_vocab": n_vocab, "uniform_nll_nats": uniform_nll}, flush=True)
        assert init_report["tensors"] > 0, init_report
    adamw_kwargs = {"lr": args.lr, "betas": (0.9, 0.95), "weight_decay": 0.1}
    if args.fused:
        adamw_kwargs["fused"] = True
    opt = torch.optim.AdamW(model.parameters(), **adamw_kwargs)
    print({"fused_adamw": bool(args.fused), "grad_checkpoint": bool(args.grad_checkpoint)}, flush=True)
    bounds = tuple(int(x) for x in args.bounds.split(","))
    cursor = ShuffledDocCursor(args.train, WINDOW, args.spans, args.perm, rank=0, world_size=1, bounds=bounds)
    loss_fn = make_bpe_loss(bpe.tok, pad_id=0)
    use_pretok = args.pretok_dir is not None
    pretok: PretokWindowReader | None = None
    step = 0
    seen = 0
    cursor_state = None
    if args.from_scratch and args.resume.exists():
        bak = args.resume.with_suffix(args.resume.suffix + ".bak_pre_scratch")
        args.resume.replace(bak)
        print({"from_scratch": True, "moved_old_resume_to": str(bak)}, flush=True)
    if (not args.from_scratch) and args.resume.exists():
        blob = torch.load(args.resume, map_location="cpu", weights_only=False)
        load_state_dict_strict(model, blob["model"], label=f"train_bpe_resume:{args.resume}")
        opt.load_state_dict(blob["opt"])
        step = int(blob["step"])
        seen = int(blob["seen"])
        cursor_state = blob.get("cursor_state")
        if cursor_state and hasattr(cursor, "load_state"):
            cursor.load_state(cursor_state)
        else:
            cursor.offset = int(blob.get("cursor_offset", 0))
        print({"resumed": str(args.resume), "step": step, "seen": seen, "cursor_offset": cursor.offset}, flush=True)
    # Compile after load so state_dict keys stay unprefixed.
    if args.compile:
        model = torch.compile(model)
        print({"compile": True}, flush=True)

    start_window = seen // WINDOW
    use_pack = use_pretok and int(args.pack_len) > 0
    pretok_packed: PackedPretokReader | None = None
    if use_pretok:
        if use_pack:
            pretok_packed = PackedPretokReader(args.pretok_dir, start_window=start_window)
            pretok = None
            print(
                {
                    "pretok_packed": str(args.pretok_dir),
                    "pack_len": int(args.pack_len),
                    "start_window": start_window,
                    "pretok_offset_bytes": pretok_packed.offset_bytes,
                    "cursor_offset": cursor.offset,
                    "cursor_match": pretok_packed.offset_bytes == cursor.offset,
                    "n_windows": pretok_packed.n_windows,
                    "mean_tok_per_window": pretok_packed.meta.get("mean_tok_per_window"),
                },
                flush=True,
            )
            if pretok_packed.offset_bytes != cursor.offset:
                raise RuntimeError(
                    f"pretok/cursor mismatch: pretok_bytes={pretok_packed.offset_bytes} "
                    f"cursor.offset={cursor.offset}"
                )
        else:
            pretok = PretokWindowReader(args.pretok_dir, start_window=start_window)
            print(
                {
                    "pretok": str(args.pretok_dir),
                    "start_window": start_window,
                    "pretok_offset_bytes": pretok.offset_bytes,
                    "cursor_offset": cursor.offset,
                    "cursor_match": pretok.offset_bytes == cursor.offset,
                    "n_windows": pretok.n_windows,
                },
                flush=True,
            )
            if pretok.offset_bytes != cursor.offset:
                raise RuntimeError(
                    f"pretok/cursor mismatch: pretok_bytes={pretok.offset_bytes} cursor.offset={cursor.offset}"
                )

    # LR is keyed to bytes seen (not optimizer step) so packed variable step_bytes
    # still reaches min_lr at the same 10B endpoint as local64's fixed 65536 B/step.
    nominal_bytes_per_step = args.batch * WINDOW  # only used for non-pack path / logging
    print(
        {
            "lr_schedule": "cosine_bytes",
            "warmup_frac": args.warmup_frac,
            "min_frac": args.min_frac,
            "target_bytes": args.target_bytes,
            "pack_len": int(args.pack_len) if use_pack else 0,
        },
        flush=True,
    )
    last_save = time.time()
    args.curve.parent.mkdir(parents=True, exist_ok=True)
    args.resume.parent.mkdir(parents=True, exist_ok=True)
    next_eval = ((seen // args.eval_every_bytes) + 1) * args.eval_every_bytes
    model.train()
    t_loop = time.time()
    steps_this_run = 0
    bytes_this_run = 0
    verify_losses: list[float] = []
    while seen < args.target_bytes:
        step += 1
        steps_this_run += 1
        step_bytes = nominal_bytes_per_step
        if use_pack:
            assert pretok_packed is not None
            try:
                batch_tok, step_bytes = pretok_packed.next_batch(args.batch, seq_len=int(args.pack_len))
            except RuntimeError as exc:
                if "exhausted" not in str(exc):
                    raise
                # Pretok ended with <1 pack left; if already at/near target, finalize.
                print({"pretok_exhausted": True, "seen": seen, "target": args.target_bytes}, flush=True)
                # Within one pack-batch of target (or already past): finalize at ≥ target.
                if seen >= args.target_bytes - int(args.batch) * WINDOW:
                    seen = max(seen, args.target_bytes)
                    break
                raise
            batch_tok = batch_tok.to(device)
            # Lockstep byte cursor: one next_window per credited window.
            n_win = step_bytes // WINDOW
            for _ in range(n_win):
                cursor.next_window()
            if cursor.offset != pretok_packed.offset_bytes:
                raise RuntimeError(
                    f"cursor drift: cursor={cursor.offset} packed={pretok_packed.offset_bytes}"
                )
        elif use_pretok:
            assert pretok is not None
            batch_tok = pretok.next_batch(args.batch, fixed_width=1024).to(device)
            for _ in range(args.batch):
                cursor.next_window()
            step_bytes = nominal_bytes_per_step
        else:
            windows = [list(cursor.next_window()) for _ in range(args.batch)]
            batch_bytes = torch.tensor(windows, dtype=torch.long, device=device)
            batch_tok = None
            step_bytes = nominal_bytes_per_step
        # Use post-step seen for schedule after increment; pre-increment here matches
        # local64's "LR for this step from current progress".
        lr = cosine_lr_bytes(seen, args.target_bytes, args.lr, args.warmup_frac, args.min_frac)
        for group in opt.param_groups:
            group["lr"] = lr
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=dtype):
            if use_pretok or use_pack:
                loss = token_batch_loss(model, batch_tok)
            else:
                loss = loss_fn(model, batch_bytes)
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        seen += step_bytes
        bytes_this_run += step_bytes
        if args.verify_steps and steps_this_run <= args.verify_steps:
            verify_losses.append(float(loss.item()))
        if step == 1 or step % 50 == 0 or (args.verify_steps and steps_this_run <= args.verify_steps):
            elapsed = max(time.time() - t_loop, 1e-6)
            bps = bytes_this_run / elapsed
            row = {
                "step": step,
                "seen": seen,
                "loss": float(loss.item()),
                "lr": lr,
                "MB_s": round(bps / 1e6, 3),
                "bytes_s": int(bps),
                "step_bytes": step_bytes,
                "cursor_offset": cursor.offset,
                "pretok_window": (
                    pretok_packed.i if pretok_packed is not None else (None if pretok is None else pretok.i)
                ),
                "pack_len": int(args.pack_len) if use_pack else 0,
            }
            if step == 1 and seen <= step_bytes:
                row["expected_uniform_nll"] = uniform_nll
                row["ln8192"] = math.log(8192)
                row["init_loss_ok"] = abs(float(loss.item()) - uniform_nll) < 2.5
            print(row, flush=True)
            if step == 1 and seen <= step_bytes and abs(float(loss.item()) - uniform_nll) >= 2.5:
                raise RuntimeError(
                    f"step-1 loss {float(loss.item()):.3f} far from uniform ln(V)={uniform_nll:.3f}; aborting"
                )
        if args.verify_steps and steps_this_run >= args.verify_steps:
            elapsed = max(time.time() - t_loop, 1e-6)
            report = {
                "verify_steps": args.verify_steps,
                "losses": verify_losses,
                "loss_mean": sum(verify_losses) / len(verify_losses),
                "loss_std": float(np.std(verify_losses)),
                "cursor_offset": cursor.offset,
                "seen": seen,
                "step": step,
                "pretok_window": (
                    pretok_packed.i if pretok_packed is not None else (None if pretok is None else pretok.i)
                ),
                "MB_s": round(bytes_this_run / elapsed / 1e6, 3),
                "bytes_s": int(bytes_this_run / elapsed),
                "pack_len": int(args.pack_len) if use_pack else 0,
            }
            print({"VERIFY_DONE": report}, flush=True)
            payload = {
                "model": raw_state_dict(model),
                "opt": opt.state_dict(),
                "step": step,
                "seen": seen,
                "cursor_offset": cursor.offset,
                "cursor_state": cursor.state(),
            }
            tmp = args.resume.with_suffix(".tmp")
            torch.save(payload, tmp)
            tmp.replace(args.resume)
            return
        now = time.time()
        if now - last_save >= args.save_every_s or seen >= args.target_bytes:
            last_save = now
            payload = {
                "model": raw_state_dict(model),
                "opt": opt.state_dict(),
                "step": step,
                "seen": seen,
                "cursor_offset": cursor.offset,
                "cursor_state": cursor.state(),
            }
            tmp = args.resume.with_suffix(".tmp")
            torch.save(payload, tmp)
            tmp.replace(args.resume)
            mark = seen // 2_000_000_000
            if mark > 0:
                held = args.resume.with_name(f"{args.resume.stem}_{mark * 2}b{args.resume.suffix}")
                if not held.exists():
                    torch.save(payload, held)
            elapsed = max(time.time() - t_loop, 1e-6)
            print(
                {
                    "saved": str(args.resume),
                    "step": step,
                    "seen": seen,
                    "MB_s": round(bytes_this_run / elapsed / 1e6, 3),
                    "bytes_s": int(bytes_this_run / elapsed),
                },
                flush=True,
            )
        if seen >= next_eval or seen >= args.target_bytes:
            # Val still uses on-the-fly encode (small).
            row = {"step": step, "seen": seen, "val_bpb": val_bpb(model, loss_fn, args.val_dir, device, dtype)}
            with args.curve.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            print({"eval": row}, flush=True)
            next_eval += args.eval_every_bytes
    # Final save (covers early break when pretok ends at/near target).
    payload = {
        "model": raw_state_dict(model),
        "opt": opt.state_dict(),
        "step": step,
        "seen": seen,
        "cursor_offset": cursor.offset,
        "cursor_state": cursor.state(),
    }
    tmp = args.resume.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(args.resume)
    print({"saved": str(args.resume), "step": step, "seen": seen, "final": True}, flush=True)
    print({"DONE": True, "step": step, "seen": seen}, flush=True)


if __name__ == "__main__":
    main()
