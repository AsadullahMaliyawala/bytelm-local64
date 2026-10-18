"""Deterministic packed byte streams for Phase 2B.1a/1b evaluation integrity.

Documents are concatenated with a single reserved sentinel byte (default 0x1E).
Windows are fixed-length, unpadded slices.

Dual masks (Phase 2B.1b):
  - byte_mask / byte_mask_content: 1 = content, 0 = sentinel (secondary BPB)
  - byte_mask_all: 1 on every byte including separator (primary train/BPB)
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils.data import Dataset, IterableDataset

# ASCII Record Separator — not space/newline
DEFAULT_DOC_SENTINEL = 0x1E


@dataclass
class ValManifest:
    validation_document_ids: list[str]
    separator_byte: int
    separator_name: str
    raw_utf8_sha256: str
    packed_length: int
    num_scored_raw_bytes: int  # content-only (excludes sentinel)
    num_all_bytes: int  # includes sentinel
    window_bytes: int
    drop_partial_tail: bool
    n_windows: int
    n_docs: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def doc_id(text: str, domain: str, idx: int) -> str:
    h = hashlib.sha256(f"{domain}:{idx}:{text}".encode("utf-8")).hexdigest()[:16]
    return f"{domain}_{idx}_{h}"


def pack_documents(
    docs: list[tuple[str, str]],
    sentinel: int = DEFAULT_DOC_SENTINEL,
    include_trailing_sentinel: bool = True,
) -> tuple[bytearray, bytearray, bytearray, list[str]]:
    """Pack docs into a continuous byte stream with dual per-byte masks.

    Returns:
        packed: raw bytes including sentinels
        mask_content: 1 = content, 0 = sentinel
        mask_all: 1 on every packed byte (including sentinel)
        doc_ids: deterministic IDs in pack order
    """
    packed = bytearray()
    mask_content = bytearray()
    mask_all = bytearray()
    ids: list[str] = []
    for i, (text, domain) in enumerate(docs):
        raw = text.encode("utf-8", errors="replace")
        ids.append(doc_id(text, domain, i))
        packed.extend(raw)
        mask_content.extend([1] * len(raw))
        mask_all.extend([1] * len(raw))
        if include_trailing_sentinel or i < len(docs) - 1:
            packed.append(sentinel & 0xFF)
            mask_content.append(0)
            mask_all.append(1)
    return packed, mask_content, mask_all, ids


def slice_windows(
    packed: bytearray | bytes,
    mask: bytearray | bytes,
    window_bytes: int,
    drop_partial_tail: bool = True,
    mask_all: bytearray | bytes | None = None,
) -> list[tuple[list[int], list[int], list[int], int]]:
    """Non-overlapping windows.

    Returns list of (bytes, byte_mask_content, byte_mask_all, start_offset).
    If ``mask_all`` is None, byte_mask_all defaults to all-ones over the window
    length (legacy single-mask callers).
    """
    n = len(packed)
    assert len(mask) == n
    if mask_all is None:
        mask_all = bytes([1] * n)
    assert len(mask_all) == n
    out: list[tuple[list[int], list[int], list[int], int]] = []
    for start in range(0, n, window_bytes):
        end = start + window_bytes
        if end > n:
            if drop_partial_tail:
                break
            pad = window_bytes - (n - start)
            chunk = list(packed[start:n]) + [0] * pad
            m = list(mask[start:n]) + [0] * pad
            ma = list(mask_all[start:n]) + [0] * pad
            out.append((chunk, m, ma, start))
            break
        out.append(
            (
                list(packed[start:end]),
                list(mask[start:end]),
                list(mask_all[start:end]),
                start,
            )
        )
    return out


def build_val_manifest(
    docs: list[tuple[str, str]],
    window_bytes: int = 1024,
    sentinel: int = DEFAULT_DOC_SENTINEL,
    drop_partial_tail: bool = True,
) -> tuple[
    ValManifest,
    bytearray,
    bytearray,
    bytearray,
    list[tuple[list[int], list[int], list[int], int]],
]:
    packed, mask_content, mask_all, ids = pack_documents(docs, sentinel=sentinel)
    windows = slice_windows(
        packed, mask_content, window_bytes, drop_partial_tail=drop_partial_tail, mask_all=mask_all
    )
    scored = int(sum(mask_content))
    sha = hashlib.sha256(bytes(packed)).hexdigest()
    manifest = ValManifest(
        validation_document_ids=ids,
        separator_byte=sentinel,
        separator_name="RS_0x1E",
        raw_utf8_sha256=sha,
        packed_length=len(packed),
        num_scored_raw_bytes=scored,
        num_all_bytes=len(packed),
        window_bytes=window_bytes,
        drop_partial_tail=drop_partial_tail,
        n_windows=len(windows),
        n_docs=len(docs),
    )
    return manifest, packed, mask_content, mask_all, windows


def export_val_manifest(
    docs: list[tuple[str, str]],
    out_dir: str | Path,
    window_bytes: int = 1024,
    sentinel: int = DEFAULT_DOC_SENTINEL,
    drop_partial_tail: bool = True,
) -> ValManifest:
    """Write manifest JSON + packed.bin + mask.bin (+ mask_all.bin) for replay."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest, packed, mask_content, mask_all, windows = build_val_manifest(
        docs, window_bytes=window_bytes, sentinel=sentinel, drop_partial_tail=drop_partial_tail
    )
    (out_dir / "phase2b_val_manifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2), encoding="utf-8"
    )
    (out_dir / "phase2b_val_packed.bin").write_bytes(bytes(packed))
    (out_dir / "phase2b_val_mask.bin").write_bytes(bytes(mask_content))
    (out_dir / "phase2b_val_mask_all.bin").write_bytes(bytes(mask_all))
    index = [{"start": s, "n_scored": sum(m), "n_all": sum(ma)} for _, m, ma, s in windows]
    (out_dir / "phase2b_val_windows.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    return manifest


def _window_item(b: list[int], m: list[int], ma: list[int], start: int) -> dict[str, Any]:
    return {
        "bytes": torch.tensor(b, dtype=torch.long),
        "byte_mask": torch.tensor(m, dtype=torch.long),  # content (secondary)
        "byte_mask_content": torch.tensor(m, dtype=torch.long),
        "byte_mask_all": torch.tensor(ma, dtype=torch.long),  # primary (incl. sep)
        "n_bytes": int(sum(ma)),  # primary scored count
        "n_bytes_content": int(sum(m)),
        "start_offset": start,
        "domain": "packed",
        "text": "",
        "structure": torch.zeros(5, dtype=torch.float),
    }


class PackedWindowDataset(Dataset):
    """Map-style packed windows (validation / finite eval)."""

    def __init__(
        self,
        docs: list[tuple[str, str]],
        window_bytes: int = 1024,
        sentinel: int = DEFAULT_DOC_SENTINEL,
        drop_partial_tail: bool = True,
    ):
        self.window_bytes = window_bytes
        self.sentinel = sentinel
        _, _, _, _, windows = build_val_manifest(
            docs, window_bytes=window_bytes, sentinel=sentinel, drop_partial_tail=drop_partial_tail
        )
        self.windows = windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        b, m, ma, start = self.windows[idx]
        return _window_item(b, m, ma, start)


class PackedWindowIterable(IterableDataset):
    """Infinite train stream: shuffle doc order each epoch, pack, emit windows.

    Patch/VQ grouping uses contiguous packed bytes across document separators
    (no per-doc pad-to-patch-boundary).
    """

    def __init__(
        self,
        docs: list[tuple[str, str]],
        window_bytes: int = 1024,
        sentinel: int = DEFAULT_DOC_SENTINEL,
        seed: int = 42,
        drop_partial_tail: bool = True,
    ):
        super().__init__()
        self.docs = docs
        self.window_bytes = window_bytes
        self.sentinel = sentinel
        self.seed = seed
        self.drop_partial_tail = drop_partial_tail

    def __iter__(self) -> Iterator[dict[str, Any]]:
        epoch = 0
        while True:
            rng = random.Random(self.seed + epoch * 9973)
            order = list(range(len(self.docs)))
            rng.shuffle(order)
            ordered = [self.docs[i] for i in order]
            packed, mask_content, mask_all, _ = pack_documents(ordered, sentinel=self.sentinel)
            windows = slice_windows(
                packed,
                mask_content,
                self.window_bytes,
                drop_partial_tail=self.drop_partial_tail,
                mask_all=mask_all,
            )
            for b, m, ma, start in windows:
                yield _window_item(b, m, ma, start)
            epoch += 1


def content_mask_from_legacy_padding(bytes_t: torch.Tensor, pad_value: int = 32) -> torch.Tensor:
    """Heuristic mask for legacy space-padded windows: content = not trailing pad.

    Trailing contiguous pad_value bytes are masked out. Interior pad_value
    (legitimate spaces) remain scored. Used only for padding-ablation diagnostics.
    """
    if bytes_t.dim() == 1:
        t = bytes_t
        mask = torch.ones_like(t, dtype=torch.long)
        i = t.numel() - 1
        while i >= 0 and int(t[i].item()) == pad_value:
            mask[i] = 0
            i -= 1
        return mask
    B, T = bytes_t.shape
    mask = torch.ones(B, T, dtype=torch.long, device=bytes_t.device)
    for b in range(B):
        i = T - 1
        while i >= 0 and int(bytes_t[b, i].item()) == pad_value:
            mask[b, i] = 0
            i -= 1
    return mask
