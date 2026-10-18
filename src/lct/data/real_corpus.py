"""Build and load the real_v1 hashed packed-byte corpus."""

from __future__ import annotations

import ast
import hashlib
import json
import mmap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

SEP = 0x1E
PERMISSIVE = frozenset(
    {
        "mit",
        "apache-2.0",
        "bsd-2-clause",
        "bsd-3-clause",
        "isc",
        "MIT",
        "Apache-2.0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "ISC",
    }
)
MIN_DOC = 200
MAX_DOC = 64 * 1024


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def assign_split(doc_sha: str) -> str:
    """Stable per-document split from sha256 mod 1000: 980 train / 10 val / 10 test."""
    bucket = int(doc_sha[:8], 16) % 1000
    if bucket < 980:
        return "train"
    if bucket < 990:
        return "val"
    return "test"


def clean_text(text: str) -> bytes | None:
    if not text or not isinstance(text, str):
        return None
    try:
        raw = text.encode("utf-8")
        raw.decode("utf-8")
    except Exception:
        return None
    # Strip document separator bytes from content
    raw = raw.replace(bytes([SEP]), b"")
    if len(raw) < MIN_DOC:
        return None
    if len(raw) > MAX_DOC:
        raw = raw[:MAX_DOC]
    return raw


def python_ok(raw: bytes) -> bool:
    try:
        ast.parse(raw.decode("utf-8"))
        return True
    except Exception:
        return False


def normalize_license(lic: Any) -> str:
    if lic is None:
        return ""
    if isinstance(lic, (list, tuple)):
        lic = lic[0] if lic else ""
    s = str(lic).strip().lower()
    # common HF variants
    s = s.replace(" ", "-")
    aliases = {
        "apache2": "apache-2.0",
        "apache-2": "apache-2.0",
        "bsd": "bsd-3-clause",
        "bsd-3": "bsd-3-clause",
        "bsd-2": "bsd-2-clause",
    }
    return aliases.get(s, s)


@dataclass
class DocRecord:
    source: str
    license: str
    path: str
    sha256: str
    split: str
    raw: bytes
    domain: str  # text | code


def stream_fineweb_scored(
    max_bytes: int,
    min_score: float = 3.0,
    config: str = "sample-100BT",
    skip_rows: int = 0,
    counter: dict | None = None,
) -> Iterator[DocRecord]:
    """FineWeb-Edu rows with educational score >= min_score. Streams; does not store the dataset."""
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceFW/fineweb-edu", name=config, split="train", streaming=True)
    seen: set[str] = set()
    total = 0
    for i, row in enumerate(ds):
        if counter is not None:
            counter["row"] = i + 1
        if i < skip_rows:
            if i > 0 and i % 200_000 == 0:
                print({"fineweb_skip": i, "skip_rows": skip_rows}, flush=True)
            continue
        if total >= max_bytes:
            break
        if i % 2000 == 0 and i > 0:
            print({"fineweb_rows": i, "kept_bytes": total, "min_score": min_score}, flush=True)
        score = row.get("score")
        try:
            score_f = float(score)
        except (TypeError, ValueError):
            continue
        if score_f < min_score:
            continue
        text = row.get("text") or ""
        raw = clean_text(text)
        if raw is None:
            continue
        digest = hashlib.sha256(raw).digest()
        if digest in seen:
            continue
        seen.add(digest)
        yield DocRecord(
            source="fineweb-edu",
            license="odc-by-1.0",
            path=str(row.get("id") or row.get("dump") or ""),
            sha256=digest.hex(),
            split=assign_split(digest.hex()),
            raw=raw,
            domain="text",
        )
        total += len(raw)


def stream_fineweb(max_bytes: int) -> Iterator[DocRecord]:
    from datasets import load_dataset

    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )
    seen: set[str] = set()
    total = 0
    for i, row in enumerate(ds):
        if total >= max_bytes:
            break
        if i % 200 == 0 and i > 0:
            print(f"  fineweb rows={i} bytes={total}", flush=True)
        text = row.get("text") or ""
        raw = clean_text(text)
        if raw is None:
            continue
        h = sha256_bytes(raw)
        if h in seen:
            continue
        seen.add(h)
        yield DocRecord(
            source="fineweb-edu",
            license="odc-by",
            path=str(row.get("id") or row.get("dump") or ""),
            sha256=h,
            split=assign_split(h),
            raw=raw,
            domain="text",
        )
        total += len(raw)


def stream_codeparrot(
    language: str,
    max_bytes: int,
    domain: str,
    require_parse: bool = False,
    skip_rows: int = 0,
    counter: dict | None = None,
    stats: dict | None = None,
) -> Iterator[DocRecord]:
    """Stream permissively licensed Python from codeparrot-clean-train (parquet-native).

    ``language`` is kept for API compatibility; this corpus is Python-only.
    Markdown/docs should use FineWeb (see ``write_packed_bin``).
    """
    from datasets import load_dataset

    # github-code-clean uses removed dataset scripts under datasets>=4; use clean-train instead
    ds = load_dataset("codeparrot/codeparrot-clean-train", split="train", streaming=True)
    permissive_l = {p.lower() for p in PERMISSIVE}
    seen: set[str] = set()
    total = 0
    for i, row in enumerate(ds):
        if counter is not None:
            counter["row"] = i + 1
        if i < skip_rows:
            if i > 0 and i % 200_000 == 0:
                print({"code_skip": i, "skip_rows": skip_rows}, flush=True)
            continue
        if total >= max_bytes:
            break
        if i % 500 == 0 and i > 0:
            print(f"  codeparrot rows={i} bytes={total}", flush=True)
        lic = normalize_license(row.get("license"))
        if lic not in permissive_l:
            continue
        text = row.get("content") or row.get("code") or row.get("text") or ""
        raw = clean_text(text)
        if raw is None:
            continue
        if require_parse and not python_ok(raw):
            if stats is not None:
                stats["parse_rejected"] = int(stats.get("parse_rejected", 0)) + 1
            continue
        h = sha256_bytes(raw)
        if h in seen:
            continue
        seen.add(h)
        yield DocRecord(
            source="codeparrot-clean-train",
            license=lic or "unknown",
            path=str(row.get("path") or row.get("repo_name") or ""),
            sha256=h,
            split=assign_split(h),
            raw=raw,
            domain=domain,
        )
        total += len(raw)


def stream_open_web_math(max_bytes: int, skip_rows: int = 0, counter: dict | None = None) -> Iterator[DocRecord]:
    from datasets import load_dataset

    ds = load_dataset("open-web-math/open-web-math", split="train", streaming=True)
    seen: set[str] = set()
    total = 0
    for i, row in enumerate(ds):
        if counter is not None:
            counter["row"] = i + 1
        if i < skip_rows:
            if i > 0 and i % 200_000 == 0:
                print({"math_skip": i, "skip_rows": skip_rows}, flush=True)
            continue
        if total >= max_bytes:
            break
        text = row.get("text") or ""
        raw = clean_text(text)
        if raw is None:
            continue
        h = sha256_bytes(raw)
        if h in seen:
            continue
        seen.add(h)
        yield DocRecord(
            source="open-web-math",
            license="odc-by",
            path=str(row.get("url") or ""),
            sha256=h,
            split=assign_split(h),
            raw=raw,
            domain="text",
        )
        total += len(raw)


def write_packed_bin(
    out_dir: Path,
    target_train_bytes: int = 100_000_000,
) -> dict[str, Any]:
    """Stream sources, assign splits, write .bin + manifest + summary."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    budgets = {
        "text": int(0.55 * target_train_bytes),
        "code": int(0.30 * target_train_bytes),
        "docs": int(0.10 * target_train_bytes),
        "math": int(0.05 * target_train_bytes),
    }
    # Over-collect slightly so val/test get enough after split (train is ~98%)
    over = 1.05

    collectors: dict[str, list[DocRecord]] = {
        "train": [],
        "val_text": [],
        "val_code": [],
        "test_text": [],
        "test_code": [],
    }
    byte_counts = {k: 0 for k in collectors}
    source_bytes: dict[str, int] = {}
    global_seen: set[str] = set()

    def take(stream: Iterator[DocRecord], budget: int, code_domain: bool = False) -> None:
        got = 0
        for doc in stream:
            if doc.sha256 in global_seen:
                continue
            global_seen.add(doc.sha256)
            if doc.split == "train":
                key = "train"
            elif doc.split == "val":
                key = "val_code" if (code_domain or doc.domain == "code") else "val_text"
            else:
                key = "test_code" if (code_domain or doc.domain == "code") else "test_text"
            collectors[key].append(doc)
            byte_counts[key] += len(doc.raw)
            source_bytes[doc.source] = source_bytes.get(doc.source, 0) + len(doc.raw)
            if key == "train":
                got += len(doc.raw)
                if got >= budget:
                    break

    print("Streaming FineWeb-Edu (text 55% + docs 10%)...", flush=True)
    # Single FineWeb pass for text+docs so docs are not duplicates of the text head
    take(
        stream_fineweb(int((budgets["text"] + budgets["docs"]) * over)),
        budgets["text"] + budgets["docs"],
        code_domain=False,
    )
    # Relabel a tail of train fineweb docs as docs source for accounting
    relabeled = 0
    docs_budget = budgets["docs"]
    for doc in reversed(collectors["train"]):
        if doc.source == "fineweb-edu" and relabeled < docs_budget:
            doc.source = "fineweb-edu-docs"
            relabeled += len(doc.raw)
    print(f"  train so far {byte_counts['train']} (docs-relabeled ~{relabeled})", flush=True)
    print("Streaming codeparrot-clean Python...", flush=True)
    take(
        stream_codeparrot("Python", int(budgets["code"] * over), domain="code", require_parse=True),
        budgets["code"],
        code_domain=True,
    )
    print(f"  train so far {byte_counts['train']}", flush=True)
    print("Streaming open-web-math...", flush=True)
    take(stream_open_web_math(int(budgets["math"] * over)), budgets["math"], code_domain=False)
    print(f"  train total {byte_counts['train']}", flush=True)

    # Truncate train to target if overshot
    train_docs = collectors["train"]
    acc = 0
    trimmed: list[DocRecord] = []
    for d in train_docs:
        if acc >= target_train_bytes:
            break
        trimmed.append(d)
        acc += len(d.raw)
    collectors["train"] = trimmed
    byte_counts["train"] = acc

    manifest_path = out_dir / "manifest.jsonl"
    bin_paths = {
        "train": out_dir / "train.bin",
        "val_text": out_dir / "val_text.bin",
        "val_code": out_dir / "val_code.bin",
        "test_text": out_dir / "test_text.bin",
        "test_code": out_dir / "test_code.bin",
    }

    file_hashes: dict[str, str] = {}
    with manifest_path.open("w", encoding="utf-8") as mf:
        for key, path in bin_paths.items():
            offset = 0
            h = hashlib.sha256()
            with path.open("wb") as bf:
                for i, doc in enumerate(collectors[key]):
                    if offset > 0:
                        bf.write(bytes([SEP]))
                        h.update(bytes([SEP]))
                        offset += 1
                    bf.write(doc.raw)
                    h.update(doc.raw)
                    rec = {
                        "id": f"{key}_{i}",
                        "source": doc.source,
                        "license": doc.license,
                        "path": doc.path,
                        "sha256": doc.sha256,
                        "split": key,
                        "offset": offset,
                        "length": len(doc.raw),
                        "domain": doc.domain,
                    }
                    mf.write(json.dumps(rec) + "\n")
                    offset += len(doc.raw)
            file_hashes[path.name] = h.hexdigest()

    summary = {
        "target_train_bytes": target_train_bytes,
        "byte_counts": byte_counts,
        "source_bytes": source_bytes,
        "n_docs": {k: len(v) for k, v in collectors.items()},
        "file_sha256": file_hashes,
        "separator": SEP,
        "mix": {"text": 0.55, "code": 0.30, "docs": 0.10, "math": 0.05},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def write_real_v2(
    out_dir: Path,
    target_train_bytes: int = 3_000_000_000,
    target_val_bytes: int = 5_000_000,
) -> dict[str, Any]:
    """Stream real_v2 to disk. Documents are written immediately and never held in a list."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_share = {"fineweb-edu": 0.70, "codeparrot-clean-train": 0.20, "open-web-math": 0.10}
    val_share = dict(train_share)
    budgets = {k: int(target_train_bytes * v) for k, v in train_share.items()}
    val_budgets = {k: int(target_val_bytes * v) for k, v in val_share.items()}
    # Keep the integer sum exact.
    budgets["fineweb-edu"] += target_train_bytes - sum(budgets.values())
    val_budgets["fineweb-edu"] += target_val_bytes - sum(val_budgets.values())
    files = {
        "train": out_dir / "train.bin",
        "val_text": out_dir / "val_text.bin",
        "val_code": out_dir / "val_code.bin",
        "val_math": out_dir / "val_math.bin",
    }
    val_key = {
        "fineweb-edu": "val_text",
        "codeparrot-clean-train": "val_code",
        "open-web-math": "val_math",
    }
    state_path = out_dir / "build_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        state = {"sources": {}, "done": False}
        for path in files.values():
            path.write_bytes(b"")
        state_path.write_text(json.dumps(state), encoding="utf-8")
    sources = state.setdefault("sources", {})

    def source_state(name: str) -> dict[str, int]:
        row = sources.setdefault(
            name, {"train_docs": 0, "train_bytes": 0, "val_docs": 0, "val_bytes": 0}
        )
        return row

    def flush_state() -> None:
        state_path.write_text(json.dumps(state), encoding="utf-8")

    def consume(name: str, stream: Iterator[DocRecord]) -> None:
        row = source_state(name)
        skip_train = int(row["train_docs"])
        skip_val = int(row["val_docs"])
        seen_train = 0
        seen_val = 0
        seen: set[str] = set()
        train_f = files["train"].open("ab")
        val_f = files[val_key[name]].open("ab")
        n_new = 0
        try:
            for doc in stream:
                if doc.sha256 in seen:
                    continue
                seen.add(doc.sha256)
                if doc.split == "test":
                    continue
                if doc.split == "train":
                    if seen_train < skip_train:
                        seen_train += 1
                        continue
                    if row["train_bytes"] >= budgets[name]:
                        if row["val_bytes"] >= val_budgets[name]:
                            break
                        continue
                    train_f.write(doc.raw)
                    train_f.write(bytes([SEP]))
                    row["train_docs"] += 1
                    row["train_bytes"] += len(doc.raw) + 1
                    seen_train += 1
                elif doc.split == "val":
                    if seen_val < skip_val:
                        seen_val += 1
                        continue
                    if row["val_bytes"] >= val_budgets[name]:
                        if row["train_bytes"] >= budgets[name]:
                            break
                        continue
                    val_f.write(doc.raw)
                    val_f.write(bytes([SEP]))
                    row["val_docs"] += 1
                    row["val_bytes"] += len(doc.raw) + 1
                    seen_val += 1
                n_new += 1
                if n_new % 200 == 0:
                    train_f.flush()
                    val_f.flush()
                    flush_state()
                    print(
                        {
                            "source": name,
                            "train_bytes": row["train_bytes"],
                            "train_budget": budgets[name],
                            "val_bytes": row["val_bytes"],
                            "val_budget": val_budgets[name],
                        },
                        flush=True,
                    )
                if row["train_bytes"] >= budgets[name] and row["val_bytes"] >= val_budgets[name]:
                    break
        finally:
            train_f.close()
            val_f.close()
            flush_state()

    if sources.get("fineweb-edu", {}).get("train_bytes", 0) < budgets["fineweb-edu"] or sources.get(
        "fineweb-edu", {}
    ).get("val_bytes", 0) < val_budgets["fineweb-edu"]:
        print("streaming fineweb-edu", flush=True)
        consume(
            "fineweb-edu",
            stream_fineweb(int(budgets["fineweb-edu"] / 0.9) + val_budgets["fineweb-edu"]),
        )
    if sources.get("codeparrot-clean-train", {}).get("train_bytes", 0) < budgets[
        "codeparrot-clean-train"
    ] or sources.get("codeparrot-clean-train", {}).get("val_bytes", 0) < val_budgets[
        "codeparrot-clean-train"
    ]:
        print("streaming codeparrot-clean", flush=True)
        consume(
            "codeparrot-clean-train",
            stream_codeparrot(
                "Python",
                int(budgets["codeparrot-clean-train"] / 0.9) + val_budgets["codeparrot-clean-train"],
                domain="code",
                require_parse=False,
            ),
        )
    if sources.get("open-web-math", {}).get("train_bytes", 0) < budgets["open-web-math"] or sources.get(
        "open-web-math", {}
    ).get("val_bytes", 0) < val_budgets["open-web-math"]:
        print("streaming open-web-math", flush=True)
        consume(
            "open-web-math",
            stream_open_web_math(int(budgets["open-web-math"] / 0.9) + val_budgets["open-web-math"]),
        )
    state["done"] = True
    flush_state()
    total_train = sum(int(sources[k]["train_bytes"]) for k in train_share)
    manifest = {
        "separator": SEP,
        "target_train_bytes": target_train_bytes,
        "target_val_bytes": target_val_bytes,
        "bytes_per_source": {k: int(sources[k]["train_bytes"]) for k in train_share},
        "val_bytes_per_source": {k: int(sources[k]["val_bytes"]) for k in train_share},
        "doc_counts": {k: {"train": int(sources[k]["train_docs"]), "val": int(sources[k]["val_docs"])} for k in train_share},
        "train_share": {k: (sources[k]["train_bytes"] / total_train if total_train else 0.0) for k in train_share},
        "targets": train_share,
        "files": {k: v.name for k, v in files.items()},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


class PackedBinDataset(Dataset):
    """Deterministic contiguous windows over a packed .bin (eval)."""

    def __init__(self, bin_path: str | Path, window_bytes: int = 2048, drop_partial_tail: bool = True):
        self.path = Path(bin_path)
        self.window = window_bytes
        self.data = np.memmap(self.path, dtype=np.uint8, mode="r")
        n = int(self.data.shape[0])
        if drop_partial_tail:
            self.n_windows = n // window_bytes
        else:
            self.n_windows = max(1, (n + window_bytes - 1) // window_bytes)

    def __len__(self) -> int:
        return max(self.n_windows, 0)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        start = idx * self.window
        chunk = bytes(self.data[start : start + self.window])
        if len(chunk) < self.window:
            chunk = chunk + bytes([32]) * (self.window - len(chunk))
        ids = list(chunk)
        mask_all = torch.ones(self.window, dtype=torch.float)
        content = torch.tensor([0.0 if b == SEP else 1.0 for b in ids], dtype=torch.float)
        return {
            "bytes": torch.tensor(ids, dtype=torch.long),
            "byte_mask_all": mask_all,
            "byte_mask_content": content,
            "byte_mask": mask_all,
            "n_bytes": self.window,
            "n_bytes_content": int(content.sum().item()),
            "domain": "mixed",
            "text": "",
            "structure": torch.zeros(5, dtype=torch.float),
        }


class PackedBinIterable(IterableDataset):
    """Random windows over mmap'd packed .bin (train)."""

    def __init__(
        self,
        bin_path: str | Path,
        window_bytes: int = 2048,
        seed: int = 42,
    ):
        super().__init__()
        self.path = Path(bin_path)
        self.window = window_bytes
        self.seed = seed

    def __iter__(self) -> Iterator[dict[str, Any]]:
        import random

        with self.path.open("rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            n = len(mm)
            if n < self.window:
                raise RuntimeError(f"{self.path} shorter than window {self.window}")
            epoch = 0
            while True:
                rng = random.Random(self.seed + epoch * 9973)
                # Emit enough windows per "epoch" to cover the file once
                n_win = max(1, n // self.window)
                for _ in range(n_win):
                    start = rng.randint(0, n - self.window)
                    chunk = mm[start : start + self.window]
                    ids = list(chunk)
                    mask_all = torch.ones(self.window, dtype=torch.float)
                    content = torch.tensor(
                        [0.0 if b == SEP else 1.0 for b in ids], dtype=torch.float
                    )
                    yield {
                        "bytes": torch.tensor(ids, dtype=torch.long),
                        "byte_mask_all": mask_all,
                        "byte_mask_content": content,
                        "byte_mask": mask_all,
                        "n_bytes": self.window,
                        "n_bytes_content": int(content.sum().item()),
                        "domain": "mixed",
                        "text": "",
                        "structure": torch.zeros(5, dtype=torch.float),
                    }
                epoch += 1


def make_packed_bin_loader(cfg: dict[str, Any], split: str = "train"):
    from torch.utils.data import DataLoader
    from lct.data.dataset import collate_batch

    data_cfg = cfg.get("data", cfg)
    train_cfg = cfg.get("train", {})
    bs = train_cfg.get("batch_size", 4)
    window = int(data_cfg.get("seq_len", 2048))
    root = Path(data_cfg.get("packed_bin_dir", "data/real_v1"))
    if not root.is_absolute():
        root = Path.cwd() / root

    # split name mapping
    bin_name = data_cfg.get("packed_bin_file")
    if bin_name and split == "train":
        path = root / bin_name if not Path(bin_name).is_absolute() else Path(bin_name)
    else:
        mapping = {
            "train": "train.bin",
            "val": data_cfg.get("val_bin", "val_text.bin"),
            "val_text": "val_text.bin",
            "val_code": "val_code.bin",
            "val_math": "val_math.bin",
            "test": data_cfg.get("test_bin", "test_text.bin"),
            "test_text": "test_text.bin",
            "test_code": "test_code.bin",
        }
        path = root / mapping.get(split, "train.bin")

    use_stream = split == "train" and bool(
        train_cfg.get("target_bytes_seen") or data_cfg.get("streaming_train", True)
    )
    if use_stream:
        ds: Dataset | IterableDataset = PackedBinIterable(
            path, window_bytes=window, seed=cfg.get("seed", 42)
        )
        return DataLoader(
            ds, batch_size=bs, shuffle=False, collate_fn=collate_batch, drop_last=True, num_workers=0
        )
    ds = PackedBinDataset(path, window_bytes=window, drop_partial_tail=True)
    return DataLoader(
        ds, batch_size=bs, shuffle=False, collate_fn=collate_batch, drop_last=False, num_workers=0
    )
