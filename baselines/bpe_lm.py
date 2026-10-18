"""Matched BPE baseline language model (Phase 2B / 2B.1: byte-budgeted streaming)."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lct.data.dataset import (
    ByteWindowIterableDataset,
    build_dataset,
    export_fixed_val_corpus,
)
from lct.eval.lm_metrics import evaluate_bpe_lm_bpb, param_report, peak_vram_mb
from lct.models.decoder_lm import DecoderOnlyLM
from lct.utils import (
    CSVLogger,
    ensure_dir,
    get_autocast_dtype,
    get_device,
    load_config,
    save_json,
    set_seed,
)


class SimpleBPE:
    """HuggingFace tokenizers BPE with byte-level fallback."""

    def __init__(self, vocab_size: int = 8000):
        self.vocab_size = vocab_size
        self.tok = None
        self.byte_mode = False

    def train(self, texts: list[str], save_dir: Path, special_tokens: list[str] | None = None) -> None:
        save_dir.mkdir(parents=True, exist_ok=True)
        try:
            from tokenizers import Tokenizer
            from tokenizers.models import BPE
            from tokenizers.trainers import BpeTrainer
            from tokenizers.pre_tokenizers import ByteLevel
            from tokenizers.decoders import ByteLevel as ByteLevelDecoder

            tokenizer = Tokenizer(BPE(unk_token="<unk>"))
            tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
            tokenizer.decoder = ByteLevelDecoder()
            trainer = BpeTrainer(
                vocab_size=self.vocab_size,
                special_tokens=special_tokens or ["<pad>", "<unk>", "<bos>", "<eos>"],
                initial_alphabet=ByteLevel.alphabet(),
            )
            # Ensure roundtrip samples appear in the train iterator
            from lct.data.bpe_checks import ROUNDTRIP_SAMPLES

            extra = []
            for raw in ROUNDTRIP_SAMPLES:
                try:
                    extra.append(raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw))
                except UnicodeDecodeError:
                    continue
            tokenizer.train_from_iterator(list(texts) + extra, trainer=trainer)
            path = save_dir / "bpe.json"
            tokenizer.save(str(path))
            self.tok = tokenizer
            self.byte_mode = False
        except ImportError:
            self.byte_mode = True
            self.tok = None
            (save_dir / "bpe_byte_mode.txt").write_text("byte_mode", encoding="utf-8")
            return
        except Exception as e:
            # Training failure → byte mode; do not swallow roundtrip AssertionError after success
            if self.tok is not None:
                raise
            self.byte_mode = True
            self.tok = None
            (save_dir / "bpe_byte_mode.txt").write_text(f"byte_mode:{e}", encoding="utf-8")
            return
        from lct.data.bpe_checks import ensure_bytelevel_decoder, assert_bytelevel_roundtrip

        ensure_bytelevel_decoder(self.tok)
        assert_bytelevel_roundtrip(self.tok)

    def load(self, save_dir: Path) -> None:
        path = save_dir / "bpe.json"
        if path.exists():
            from tokenizers import Tokenizer
            from lct.data.bpe_checks import ensure_bytelevel_decoder, assert_bytelevel_roundtrip

            self.tok = Tokenizer.from_file(str(path))
            ensure_bytelevel_decoder(self.tok)
            self.byte_mode = False
            assert_bytelevel_roundtrip(self.tok)
        else:
            self.byte_mode = True

    def encode(self, text: str, max_len: int) -> list[int]:
        ids, _ = self.encode_with_bytes(text, max_len)
        return ids

    def encode_bytes_window(self, raw_bytes: list[int] | bytes, max_len: int) -> tuple[list[int], int]:
        """Encode an exact raw-byte window; n_bytes is always len(window) (pre-pad)."""
        ids, weights, n_scored = self.encode_window_with_weights(raw_bytes, max_len)
        return ids, n_scored

    def encode_window_with_weights(
        self,
        raw_bytes: list[int] | bytes,
        max_len: int,
        byte_mask: list[int] | None = None,
    ) -> tuple[list[int], list[int], int]:
        """Encode window; return (token_ids, per-token scored-byte weights, total scored bytes).

        Weights[i] = number of UTF-8 content bytes covered by token i (mask∩offsets).
        Pad tokens get weight 0.
        """
        if isinstance(raw_bytes, list):
            raw = bytes(raw_bytes)
        else:
            raw = raw_bytes
        if byte_mask is None:
            byte_mask = [1] * len(raw)
        assert len(byte_mask) == len(raw)

        if self.byte_mode or self.tok is None:
            ids = list(raw)[:max_len]
            weights = list(byte_mask[: len(ids)])
            if len(ids) < max_len:
                pad = max_len - len(ids)
                ids = ids + [0] * pad
                weights = weights + [0] * pad
            return ids, weights, int(sum(weights))

        # Decode for HF tokenizer; ByteLevel BPE should round-trip ASCII/UTF-8
        text = raw.decode("utf-8", errors="replace")
        enc = self.tok.encode(text)
        ids = enc.ids[:max_len]
        offsets = list(enc.offsets[: len(ids)]) if hasattr(enc, "offsets") else [(0, 0)] * len(ids)
        weights: list[int] = []
        # Map character offsets → UTF-8 byte indices
        # Build char→byte start map
        char_to_byte = [0]
        for ch in text:
            char_to_byte.append(char_to_byte[-1] + len(ch.encode("utf-8")))
        for i, (c0, c1) in enumerate(offsets):
            c0 = max(0, min(c0, len(char_to_byte) - 1))
            c1 = max(0, min(c1, len(char_to_byte) - 1))
            b0 = char_to_byte[c0]
            b1 = char_to_byte[c1]
            w = sum(byte_mask[b0:b1]) if b1 > b0 else 0
            weights.append(int(w))
        if len(ids) < max_len:
            pad = max_len - len(ids)
            ids = ids + [0] * pad
            weights = weights + [0] * pad
        return ids, weights, int(sum(weights))

    def verify_roundtrip(self, raw: bytes | list[int]) -> bool:
        """Check decode(encode(text)) == text for UTF-8 content (text-level)."""
        if isinstance(raw, list):
            raw = bytes(raw)
        if self.byte_mode or self.tok is None:
            return True
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return False
        enc = self.tok.encode(text)
        decoded = self.tok.decode(enc.ids)
        return decoded == text

    def verify_roundtrip_stats(self, docs: list[tuple[str, str]], limit: int = 200) -> dict:
        """Aggregate roundtrip diagnostics on document texts."""
        n_ok = n_tot = 0
        for text, _ in docs[:limit]:
            n_tot += 1
            if self.byte_mode or self.tok is None:
                n_ok += 1
                continue
            enc = self.tok.encode(text)
            if self.tok.decode(enc.ids) == text:
                n_ok += 1
        return {
            "ok": n_ok,
            "total": n_tot,
            "rate": n_ok / max(n_tot, 1),
            "byte_mode": self.byte_mode,
        }

    def encode_with_bytes(self, text: str, max_len: int) -> tuple[list[int], int]:
        """Return token ids and UTF-8 byte count of the encoded prefix."""
        raw = text.encode("utf-8", errors="replace")
        ids, _, n = self.encode_window_with_weights(raw, max_len)
        return ids, n

    @property
    def n_vocab(self) -> int:
        if self.byte_mode or self.tok is None:
            return 256
        return self.tok.get_vocab_size()


# Back-compat alias
BPELanguageModel = DecoderOnlyLM


class BPEDataset(Dataset):
    """Map-style BPE dataset (val / legacy)."""

    def __init__(self, docs: list[tuple[str, str]], bpe: SimpleBPE, seq_len: int):
        self.docs = docs
        self.bpe = bpe
        self.seq_len = seq_len

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        text, _ = self.docs[idx]
        ids, weights, n_bytes = self.bpe.encode_window_with_weights(
            text.encode("utf-8", errors="replace"), self.seq_len
        )
        return {
            "tokens": torch.tensor(ids, dtype=torch.long),
            "token_byte_weights": torch.tensor(weights, dtype=torch.long),
            "n_bytes": torch.tensor(n_bytes, dtype=torch.long),
        }


class BPEPackedWindowDataset(Dataset):
    """BPE over packed byte windows with per-token byte weights (Phase 2B.1a)."""

    def __init__(self, packed_ds, bpe: SimpleBPE, token_seq_len: int):
        self.packed_ds = packed_ds
        self.bpe = bpe
        self.token_seq_len = token_seq_len

    def __len__(self):
        return len(self.packed_ds)

    def __getitem__(self, idx):
        item = self.packed_ds[idx]
        raw = item["bytes"].tolist()
        # Primary mask includes separator (Phase 2B.1b)
        mask = item.get("byte_mask_all", item.get("byte_mask"))
        mask = mask.tolist() if torch.is_tensor(mask) else list(mask)
        mask_content = item.get("byte_mask_content", item.get("byte_mask"))
        if mask_content is not None:
            mask_content = mask_content.tolist() if torch.is_tensor(mask_content) else list(mask_content)
        ids, weights, n_bytes = self.bpe.encode_window_with_weights(
            raw, self.token_seq_len, byte_mask=mask
        )
        out = {
            "tokens": torch.tensor(ids, dtype=torch.long),
            "token_byte_weights": torch.tensor(weights, dtype=torch.long),
            "n_bytes": torch.tensor(n_bytes, dtype=torch.long),
            "byte_mask": item.get("byte_mask_all", item["byte_mask"]),
        }
        if mask_content is not None:
            _, w_c, n_c = self.bpe.encode_window_with_weights(
                raw, self.token_seq_len, byte_mask=mask_content
            )
            out["token_byte_weights_content"] = torch.tensor(w_c, dtype=torch.long)
            out["n_bytes_content"] = torch.tensor(n_c, dtype=torch.long)
        return out


class BPEByteWindowIterable(IterableDataset):
    """Infinite BPE stream over fixed raw-byte windows (n_bytes == window length)."""

    def __init__(
        self,
        docs: list[tuple[str, str]],
        bpe: SimpleBPE,
        byte_seq_len: int,
        token_seq_len: int,
        seed: int = 42,
        packed: bool = False,
        sentinel: int = 0x1E,
    ):
        super().__init__()
        self.bpe = bpe
        self.token_seq_len = token_seq_len
        self.packed = packed
        if packed:
            from lct.data.packed_stream import PackedWindowIterable

            self.byte_stream = PackedWindowIterable(
                docs, window_bytes=byte_seq_len, sentinel=sentinel, seed=seed
            )
        else:
            self.byte_stream = ByteWindowIterableDataset(docs, seq_len=byte_seq_len, seed=seed)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for item in self.byte_stream:
            raw = item["bytes"].tolist()
            mask = item.get("byte_mask_all", item.get("byte_mask"))
            if mask is not None:
                mask = mask.tolist() if torch.is_tensor(mask) else mask
            ids, weights, n_bytes = self.bpe.encode_window_with_weights(
                raw, self.token_seq_len, byte_mask=mask
            )
            yield {
                "tokens": torch.tensor(ids, dtype=torch.long),
                "token_byte_weights": torch.tensor(weights, dtype=torch.long),
                "n_bytes": torch.tensor(n_bytes, dtype=torch.long),
            }


def _collate(batch):
    out = {
        "tokens": torch.stack([b["tokens"] for b in batch], 0),
        "n_bytes": torch.stack([b["n_bytes"] for b in batch], 0),
    }
    if "token_byte_weights" in batch[0]:
        out["token_byte_weights"] = torch.stack([b["token_byte_weights"] for b in batch], 0)
    return out


class _PackedBinBPEMap(Dataset):
    """Map-style PackedBin windows → BPE tokens with byte weights."""

    def __init__(self, packed_ds, bpe: SimpleBPE, token_seq_len: int):
        self.packed_ds = packed_ds
        self.bpe = bpe
        self.token_seq_len = token_seq_len

    def __len__(self):
        return len(self.packed_ds)

    def __getitem__(self, idx):
        item = self.packed_ds[idx]
        raw = item["bytes"].tolist() if torch.is_tensor(item["bytes"]) else list(item["bytes"])
        ids, weights, n_bytes = self.bpe.encode_window_with_weights(raw, self.token_seq_len)
        return {
            "tokens": torch.tensor(ids, dtype=torch.long),
            "token_byte_weights": torch.tensor(weights, dtype=torch.long),
            "n_bytes": torch.tensor(n_bytes, dtype=torch.long),
        }


class _PackedBinBPEIterable(IterableDataset):
    """Streaming PackedBin → BPE tokens."""

    def __init__(
        self,
        bin_path: Path,
        bpe: SimpleBPE,
        byte_seq_len: int,
        token_seq_len: int,
        seed: int = 42,
    ):
        super().__init__()
        from lct.data.real_corpus import PackedBinIterable

        self.byte_stream = PackedBinIterable(bin_path, window_bytes=byte_seq_len, seed=seed)
        self.bpe = bpe
        self.token_seq_len = token_seq_len

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for item in self.byte_stream:
            raw = item["bytes"].tolist() if torch.is_tensor(item["bytes"]) else list(item["bytes"])
            ids, weights, n_bytes = self.bpe.encode_window_with_weights(raw, self.token_seq_len)
            yield {
                "tokens": torch.tensor(ids, dtype=torch.long),
                "token_byte_weights": torch.tensor(weights, dtype=torch.long),
                "n_bytes": torch.tensor(n_bytes, dtype=torch.long),
            }


def train_bpe_lm(cfg: dict[str, Any]) -> dict[str, Any]:
    set_seed(cfg.get("seed", 42))
    device = get_device(cfg.get("device", "cuda"))
    dtype = get_autocast_dtype(cfg.get("precision", "bf16"))
    run_id = cfg.get("run_id", "lm_bpe_40m")
    exp_dir = ensure_dir(Path(cfg["train"]["experiment_dir"]) / run_id)
    ckpt_dir = ensure_dir(cfg["train"]["checkpoint_dir"])
    save_json({k: v for k, v in cfg.items() if not str(k).startswith("_")}, exp_dir / "config.json")

    if cfg.get("data", {}).get("val_corpus_path"):
        export_fixed_val_corpus(cfg)

    byte_seq_len = cfg["data"].get("seq_len", 512)
    token_seq_len = cfg["data"].get("bpe_seq_len", byte_seq_len)
    use_stream = bool(cfg["train"].get("target_bytes_seen") or cfg["data"].get("streaming_train", False))
    use_packed = bool(cfg["data"].get("packed_stream", False))
    use_packed_bin = cfg.get("data", {}).get("source") == "packed_bin"
    sentinel = int(cfg["data"].get("doc_sentinel_byte", 0x1E))

    bpe = SimpleBPE(vocab_size=cfg["data"].get("bpe_vocab_size", 8000))
    bpe_dir = Path(
        cfg["data"].get("bpe_cache_dir")
        or cfg["data"].get("bpe_dir")
        or cfg["data"].get("bpe_cache_hint")
        or (
            Path(cfg["data"].get("cache_dir", "data/cache"))
            / ("bpe_phase2b1b" if cfg["data"].get("packed_stream") or use_packed_bin else "bpe_phase2b")
        )
    )
    if (bpe_dir / "bpe.json").exists():
        bpe.load(bpe_dir)
    else:
        # Train vocab from docs (packed_bin → sample decoded windows / docs helper)
        if use_packed_bin:
            from lct.data.real_corpus import PackedBinDataset

            root = Path(cfg["data"].get("packed_bin_dir", "data/real_v1"))
            train_bin = root / cfg["data"].get("packed_bin_file", "train.bin")
            sample_ds = PackedBinDataset(train_bin, window_bytes=min(byte_seq_len, 2048), drop_partial_tail=True)
            texts = []
            for i in range(min(len(sample_ds), 8000)):
                raw = bytes(int(x) for x in sample_ds[i]["bytes"].tolist())
                texts.append(raw.decode("utf-8", errors="replace"))
        else:
            train_ds = build_dataset(cfg, "train")
            texts = [t for t, _ in train_ds.docs]
        bpe.train(texts, bpe_dir)

    if use_packed_bin:
        from lct.data.real_corpus import PackedBinDataset, PackedBinIterable

        root = Path(cfg["data"].get("packed_bin_dir", "data/real_v1"))
        train_bin = root / cfg["data"].get("packed_bin_file", "train.bin")
        val_name = cfg["data"].get("val_bin", "val_text.bin")
        val_bin = root / val_name
        if use_stream:
            train_set: Dataset | IterableDataset = _PackedBinBPEIterable(
                train_bin, bpe, byte_seq_len, token_seq_len, seed=cfg.get("seed", 42)
            )
            train_loader = DataLoader(
                train_set,
                batch_size=cfg["train"]["batch_size"],
                shuffle=False,
                drop_last=True,
                collate_fn=_collate,
                num_workers=0,
            )
        else:
            packed_train = PackedBinDataset(train_bin, window_bytes=byte_seq_len, drop_partial_tail=True)
            train_set = _PackedBinBPEMap(packed_train, bpe, token_seq_len)
            train_loader = DataLoader(
                train_set,
                batch_size=cfg["train"]["batch_size"],
                shuffle=True,
                drop_last=True,
                collate_fn=_collate,
                num_workers=0,
            )
        packed_val = PackedBinDataset(val_bin, window_bytes=byte_seq_len, drop_partial_tail=True)
        val_set = _PackedBinBPEMap(packed_val, bpe, token_seq_len)
        val_loader = DataLoader(
            val_set,
            batch_size=cfg["train"]["batch_size"],
            shuffle=False,
            drop_last=False,
            collate_fn=_collate,
            num_workers=0,
        )
    else:
        train_ds = build_dataset(cfg, "train")
        val_ds = build_dataset(cfg, "val")

        if use_stream:
            train_set = BPEByteWindowIterable(
                train_ds.docs,
                bpe,
                byte_seq_len,
                token_seq_len,
                seed=cfg.get("seed", 42),
                packed=use_packed,
                sentinel=sentinel,
            )
            train_loader = DataLoader(
                train_set,
                batch_size=cfg["train"]["batch_size"],
                shuffle=False,
                drop_last=True,
                collate_fn=_collate,
                num_workers=0,
            )
        else:
            train_set = BPEDataset(train_ds.docs, bpe, token_seq_len)
            train_loader = DataLoader(
                train_set,
                batch_size=cfg["train"]["batch_size"],
                shuffle=True,
                drop_last=True,
                collate_fn=_collate,
                num_workers=0,
            )

        # Val: packed windows when enabled (identical byte stream to byte/VQ)
        if use_packed:
            from lct.data.packed_stream import PackedWindowDataset

            packed_val = PackedWindowDataset(
                val_ds.docs, window_bytes=byte_seq_len, sentinel=sentinel, drop_partial_tail=True
            )
            val_set = BPEPackedWindowDataset(packed_val, bpe, token_seq_len)
            val_loader = DataLoader(
                val_set,
                batch_size=cfg["train"]["batch_size"],
                shuffle=False,
                drop_last=False,
                collate_fn=_collate,
                num_workers=0,
            )
        else:
            val_set = BPEDataset(val_ds.docs, bpe, token_seq_len)
            val_loader = DataLoader(
                val_set,
                batch_size=cfg["train"]["batch_size"],
                shuffle=False,
                drop_last=True,
                collate_fn=_collate,
                num_workers=0,
            )

    m = cfg["model"]
    model = DecoderOnlyLM(
        vocab_size=bpe.n_vocab,
        d_model=m["d_model"],
        n_heads=m["n_heads"],
        n_layers=m["n_layers"],
        ffn_mult=m.get("ffn_mult", 4),
        dropout=m.get("dropout", 0.1),
        label_smoothing=m.get("label_smoothing", 0.0),
        max_len=m.get("max_len", 2048),
        ignore_index=0,
        pos_type=m.get("pos_type", "sinusoidal"),
    ).to(device)

    param_info = param_report(model)
    param_info["vocab_size"] = bpe.n_vocab
    save_json(param_info, exp_dir / "param_count.json")

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 0.01),
    )
    scaler = GradScaler(enabled=(dtype == torch.float16))
    logger = CSVLogger(exp_dir / "metrics.csv")

    max_steps = cfg["train"]["max_steps"]
    accum = cfg["train"].get("grad_accum", 1)
    target_bytes = cfg["train"].get("target_bytes_seen")
    count_scored = bool(cfg["train"].get("count_scored_bytes_only", False))
    step = 0
    best = float("inf")
    bytes_seen = 0
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
        tokens = batch["tokens"].to(device)
        nb = int(batch["n_bytes"].sum().item())
        bytes_seen += nb
        scored_bytes += nb  # n_bytes already uses primary mask weights when packed
        inp, tgt = tokens[:, :-1], tokens[:, 1:]
        with autocast(enabled=use_amp, dtype=dtype if use_amp else torch.float32):
            out = model(inp, targets=tgt)
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
            logger.log({
                "step": step,
                "loss": float(out["loss"].detach()),
                "bytes_seen": bytes_seen,
                "scored_bytes": scored_bytes,
                "wall_seconds": time.perf_counter() - t0,
            })
            pbar.set_postfix(loss=round(float(out["loss"].detach()), 3), scored=scored_bytes)

        if step % cfg["train"].get("eval_every", 250) == 0:
            val = evaluate_bpe_lm_bpb(model, val_loader, device, dtype, ignore_index=0)
            logger.log({"step": step, **{f"val_{k}": v for k, v in val.items()}})
            if val["bpb"] < best:
                best = val["bpb"]
                ckpt_name = cfg["train"].get("best_checkpoint_name", f"{run_id}.pt")
                torch.save(
                    {"model": model.state_dict(), "config": cfg, "vocab": bpe.n_vocab, "step": step, "val": val},
                    Path(ckpt_dir) / ckpt_name,
                )
            model.train()

        budget = scored_bytes if count_scored else bytes_seen
        if target_bytes is not None and budget >= int(target_bytes):
            break

    pbar.close()
    wall = time.perf_counter() - t0
    final = evaluate_bpe_lm_bpb(model, val_loader, device, dtype, ignore_index=0)
    last_name = cfg["train"].get("last_checkpoint_name", f"{run_id}_last.pt")
    torch.save(
        {"model": model.state_dict(), "config": cfg, "vocab": bpe.n_vocab, "step": step, "val": final},
        Path(ckpt_dir) / last_name,
    )

    # Packed SHA-256 round-trip verification
    packed_sha = None
    decoded_sha = None
    cache_dir = Path(cfg["data"].get("cache_dir", "data/cache"))
    packed_bin = cache_dir / "phase2b_val_packed.bin"
    man_path = cache_dir / "phase2b_val_manifest.json"
    if man_path.exists():
        import json

        packed_sha = json.loads(man_path.read_text(encoding="utf-8")).get("raw_utf8_sha256")
    if packed_bin.exists() and not bpe.byte_mode:
        from lct.data.bpe_checks import assert_packed_sha_matches

        packed_bytes = packed_bin.read_bytes()
        try:
            sha_info = assert_packed_sha_matches(bpe, packed_bytes, expected_sha256=packed_sha)
            decoded_sha = sha_info["decoded_sha256"]
            packed_sha = sha_info["sha256"]
        except AssertionError as e:
            # Log but do not abort training results write
            decoded_sha = f"FAIL: {e}"

    run_manifest = {
        "packed_stream_sha256": packed_sha,
        "bpe_decoded_packed_sha256": decoded_sha,
        "doc_separator_hex": format(int(cfg.get("data", {}).get("doc_sentinel_byte", 0x1E)), "02x"),
        "train_scored_bytes": int(scored_bytes),
        "train_raw_tensor_bytes": int(bytes_seen),
        "validation_scored_bytes": int(final.get("n_bytes") or 0),
        "context_bytes": int(cfg.get("data", {}).get("seq_len", 1024)),
        "sequence_positions": int(cfg.get("data", {}).get("bpe_seq_len", cfg.get("data", {}).get("seq_len", 1024))),
        "optimizer_steps": int(step),
        "wall_clock_seconds": wall,
        "raw_bytes_per_second": bytes_seen / max(wall, 1e-9),
        "peak_vram_gb": (peak_vram_mb() or 0) / 1024.0,
        "core_lm_parameters": param_info.get("lm_core"),
        "total_parameters": param_info.get("lm_total_all"),
        "count_scored_bytes_only": count_scored,
        "packed_stream": bool(cfg.get("data", {}).get("packed_stream", False)),
        "bpe_vocab": bpe.n_vocab,
    }
    save_json(run_manifest, exp_dir / "run_manifest.json")

    results = {
        "run_id": run_id,
        "kind": "bpe_lm",
        "best_val_bpb": best,
        "final_metrics": final,
        "steps": step,
        "bytes_seen": bytes_seen,
        "scored_bytes": scored_bytes,
        "wall_seconds": wall,
        "bytes_per_sec_train": bytes_seen / max(wall, 1e-9),
        "peak_vram_mb": peak_vram_mb(),
        "params": param_info,
        "run_manifest": run_manifest,
    }
    save_json(results, exp_dir / "results.json")
    return results


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/5060/phase2b_bpe.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    print(train_bpe_lm(cfg))
