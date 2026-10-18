from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset


TEXT_SNIPPETS = [
    "The latent concept tokenizer learns discrete symbols from raw bytes.",
    "Language models compress statistical regularities into next-token predictions.",
    "World models simulate how concepts evolve under interventions and time.",
    "Creativity arises when concepts recombine into novel but coherent structures.",
    "FineWeb samples provide diverse English prose for pretraining experiments.",
    "Vector quantization maps continuous vectors onto a finite codebook.",
    "Attention mechanisms relate distant tokens through learned similarity.",
    "Sample efficiency measures how quickly validation loss falls with data.",
    "Bits per byte is a tokenizer-agnostic compression metric for language models.",
    "Structured reasoning benefits from reusable intermediate representations.",
    "Scientists discovered that discrete latents can act as a shared vocabulary across modalities.",
    "Planning systems imagine futures before selecting actions in the real world.",
    "Compression is intelligence: good representations discard noise and keep structure.",
    "Recursive neural programs compose skills into longer-horizon behaviors.",
    "Counterfactual reasoning asks what would change if a latent were different.",
    "Scale alone is not enough; inductive bias toward concepts may unlock autonomy.",
    "Code and natural language share hierarchical syntax that a codebook can capture.",
    "Entropy regularization keeps rare concepts alive during early training stages.",
    "A breakthrough foundation model must generalize beyond memorized surface patterns.",
    "Downstream agents need internal simulations that remain consistent over many steps.",
]

CODE_SNIPPETS = [
    "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\n",
    "class Stack:\n    def __init__(self):\n        self.items = []\n    def push(self, x):\n        self.items.append(x)\n",
    "for i in range(10):\n    if i % 2 == 0:\n        print(i)\n",
    "try:\n    data = open('f.txt').read()\nexcept OSError as e:\n    print(e)\n",
    "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n",
    "import json\n\ndef load_cfg(path):\n    with open(path) as f:\n        return json.load(f)\n",
    "while True:\n    x = input()\n    if x == 'quit':\n        break\n    print(x.upper())\n",
    "def merge(a, b):\n    i = j = 0\n    out = []\n    while i < len(a) and j < len(b):\n        if a[i] < b[j]:\n            out.append(a[i]); i += 1\n        else:\n            out.append(b[j]); j += 1\n    return out + a[i:] + b[j:]\n",
    "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        if arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n",
    "class Graph:\n    def __init__(self):\n        self.adj = {}\n    def add(self, u, v):\n        self.adj.setdefault(u, []).append(v)\n",
    "def train_step(model, batch, opt):\n    opt.zero_grad()\n    loss = model(batch)\n    loss.backward()\n    opt.step()\n    return float(loss)\n",
    "async def main():\n    tasks = [fetch(u) for u in urls]\n    return await asyncio.gather(*tasks)\n",
]


STRUCTURE_KEYWORDS = {
    "loop": ["for ", "while "],
    "branch": ["if ", "elif ", "else:"],
    "try": ["try:", "except "],
    "def": ["def "],
    "class": ["class "],
}


def text_to_bytes(text: str, seq_len: int) -> list[int]:
    raw = list(text.encode("utf-8", errors="replace"))
    if len(raw) >= seq_len:
        start = random.randint(0, len(raw) - seq_len)
        return raw[start : start + seq_len]
    # pad with spaces (32)
    return raw + [32] * (seq_len - len(raw))


def structure_labels_from_text(text: str, tags: list[str] | None = None) -> list[float]:
    tags = tags or list(STRUCTURE_KEYWORDS.keys())
    out = []
    for tag in tags:
        keys = STRUCTURE_KEYWORDS.get(tag, [])
        out.append(1.0 if any(k in text for k in keys) else 0.0)
    return out


def doc_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


class SyntheticCorpus(Dataset):
    """Offline synthetic mix of text + code for 5060 prototyping."""

    def __init__(
        self,
        n_docs: int,
        seq_len: int = 256,
        text_weight: float = 0.7,
        seed: int = 42,
        split: str = "train",
    ):
        self.seq_len = seq_len
        self.text_weight = text_weight
        rng = random.Random(seed + (0 if split == "train" else 999))
        self.docs: list[tuple[str, str]] = []
        for i in range(n_docs):
            if rng.random() < text_weight:
                base = rng.choice(TEXT_SNIPPETS)
                domain = "text"
                # lightly mutate
                text = (base + " " + rng.choice(TEXT_SNIPPETS))[:2000]
            else:
                base = rng.choice(CODE_SNIPPETS)
                domain = "code"
                text = base + "\n# id=%d\n" % i + rng.choice(CODE_SNIPPETS)
            self.docs.append((text, domain))

    def __len__(self) -> int:
        return len(self.docs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        text, domain = self.docs[idx]
        ids = text_to_bytes(text, self.seq_len)
        return {
            "bytes": torch.tensor(ids, dtype=torch.long),
            "domain": domain,
            "text": text,
            "structure": torch.tensor(structure_labels_from_text(text), dtype=torch.float),
        }


class HFStreamCorpus(IterableDataset):
    """Optional HuggingFace streaming loader (FineWeb / The Stack)."""

    def __init__(self, cfg: dict[str, Any], split: str = "train"):
        self.cfg = cfg
        self.split = split
        self.seq_len = cfg.get("seq_len", 256)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError("Install datasets to use huggingface source") from e

        text_name = self.cfg.get("hf_text", "HuggingFaceFW/fineweb-edu")
        # Fall back gracefully: if download fails, yield nothing and caller should use synthetic
        try:
            ds = load_dataset(text_name, split=self.cfg.get("hf_text_split", "train"), streaming=True)
            for i, row in enumerate(ds):
                if i >= self.cfg.get("max_train_docs" if self.split == "train" else "max_val_docs", 1000):
                    break
                text = row.get("text") or row.get("content") or ""
                if not text:
                    continue
                yield {
                    "bytes": torch.tensor(text_to_bytes(text, self.seq_len), dtype=torch.long),
                    "domain": "text",
                    "text": text[:2000],
                    "structure": torch.tensor(structure_labels_from_text(text), dtype=torch.float),
                }
        except Exception:
            # degrade to synthetic
            syn = SyntheticCorpus(
                n_docs=self.cfg.get("max_train_docs", 1000),
                seq_len=self.seq_len,
                text_weight=self.cfg.get("text_weight", 0.7),
                split=self.split,
            )
            for i in range(len(syn)):
                yield syn[i]


def _load_docs_jsonl(path: Path) -> list[tuple[str, str]]:
    docs: list[tuple[str, str]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            docs.append((d["text"], d.get("domain", "text")))
    return docs


def export_fixed_val_corpus(cfg: dict[str, Any], path: str | Path | None = None) -> Path:
    """Write a deterministic held-out val JSONL once (no train overlap via val seed)."""
    data_cfg = cfg.get("data", cfg)
    out = Path(path or data_cfg.get("val_corpus_path", "data/cache/phase2b_val_corpus.jsonl"))
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        return out
    n = int(data_cfg.get("max_val_docs", 800))
    # Use val split seed (SyntheticCorpus already offsets by +999)
    syn = SyntheticCorpus(
        n_docs=n,
        seq_len=data_cfg.get("seq_len", 256),
        text_weight=data_cfg.get("text_weight", 0.7),
        seed=cfg.get("seed", 42),
        split="val",
    )
    with out.open("w", encoding="utf-8") as f:
        for text, domain in syn.docs:
            f.write(json.dumps({"text": text, "domain": domain}) + "\n")
    return out


def build_dataset(cfg: dict[str, Any], split: str = "train") -> Dataset:
    data_cfg = cfg.get("data", cfg)
    n = data_cfg.get("max_train_docs" if split == "train" else "max_val_docs", 1000)
    source = data_cfg.get("source", "synthetic")
    seq_len = data_cfg.get("seq_len", 256)

    # Fixed held-out corpus for fair Phase 2B.1 comparison
    if split == "val" and data_cfg.get("val_corpus_path"):
        path = Path(data_cfg["val_corpus_path"])
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            export_fixed_val_corpus(cfg, path)
        docs = _load_docs_jsonl(path)
        ds = SyntheticCorpus(n_docs=1, seq_len=seq_len, seed=cfg.get("seed", 42), split="val")
        ds.docs = docs[:n] if n else docs
        return ds

    if source == "huggingface":
        # Materialize a small cache for DataLoader friendliness
        cache_dir = Path(data_cfg.get("cache_dir", "data/cache")) / f"hf_{split}"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / "docs.jsonl"
        if not cache_file.exists():
            docs = []
            for i, item in enumerate(HFStreamCorpus(data_cfg, split)):
                docs.append({"text": item["text"], "domain": item["domain"]})
                if i + 1 >= n:
                    break
            with cache_file.open("w", encoding="utf-8") as f:
                for d in docs:
                    f.write(json.dumps(d) + "\n")
        # wrap as synthetic-like dataset
        loaded = []
        with cache_file.open(encoding="utf-8") as f:
            for line in f:
                loaded.append(json.loads(line))
        ds = SyntheticCorpus(n_docs=1, seq_len=seq_len)
        ds.docs = [(d["text"], d["domain"]) for d in loaded]
        return ds
    return SyntheticCorpus(
        n_docs=n,
        seq_len=seq_len,
        text_weight=data_cfg.get("text_weight", 0.7),
        seed=cfg.get("seed", 42),
        split=split,
    )


class ByteWindowIterableDataset(IterableDataset):
    """Infinite stream of fixed-length raw-byte windows for byte-budgeted training.

    Emits non-overlapping windows of ``seq_len`` bytes from shuffled documents.
    Documents shorter than ``seq_len`` are padded with spaces (32), matching
    ``text_to_bytes``. Epochs reshape indefinitely until the train loop stops.
    """

    def __init__(
        self,
        docs: list[tuple[str, str]],
        seq_len: int = 1024,
        seed: int = 42,
    ):
        super().__init__()
        self.docs = docs
        self.seq_len = seq_len
        self.seed = seed

    def __iter__(self) -> Iterator[dict[str, Any]]:
        epoch = 0
        while True:
            rng = random.Random(self.seed + epoch * 9973)
            order = list(range(len(self.docs)))
            rng.shuffle(order)
            for idx in order:
                text, domain = self.docs[idx]
                raw = list(text.encode("utf-8", errors="replace"))
                if len(raw) < self.seq_len:
                    # one padded window
                    window = raw + [32] * (self.seq_len - len(raw))
                    yield {
                        "bytes": torch.tensor(window, dtype=torch.long),
                        "domain": domain,
                        "text": text,
                        "structure": torch.tensor(structure_labels_from_text(text), dtype=torch.float),
                        "n_bytes": self.seq_len,
                    }
                    continue
                # non-overlapping windows; drop incomplete tail
                n_win = len(raw) // self.seq_len
                for w in range(n_win):
                    start = w * self.seq_len
                    window = raw[start : start + self.seq_len]
                    yield {
                        "bytes": torch.tensor(window, dtype=torch.long),
                        "domain": domain,
                        "text": text,
                        "structure": torch.tensor(structure_labels_from_text(text), dtype=torch.float),
                        "n_bytes": self.seq_len,
                    }
            epoch += 1


def collate_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "bytes": torch.stack([x["bytes"] for x in items]),
        "domain": [x.get("domain", "text") for x in items],
        "text": [x.get("text", "") for x in items],
        "structure": torch.stack(
            [
                x["structure"]
                if torch.is_tensor(x.get("structure"))
                else torch.zeros(5, dtype=torch.float)
                for x in items
            ]
        ),
    }
    if "n_bytes" in items[0]:
        nb = items[0]["n_bytes"]
        if torch.is_tensor(nb):
            out["n_bytes"] = torch.stack([x["n_bytes"] for x in items], 0)
        else:
            out["n_bytes"] = torch.tensor([x["n_bytes"] for x in items], dtype=torch.long)
    if "byte_mask" in items[0]:
        out["byte_mask"] = torch.stack([x["byte_mask"] for x in items], 0)
    if "byte_mask_content" in items[0]:
        out["byte_mask_content"] = torch.stack([x["byte_mask_content"] for x in items], 0)
    if "byte_mask_all" in items[0]:
        out["byte_mask_all"] = torch.stack([x["byte_mask_all"] for x in items], 0)
    if "n_bytes_content" in items[0]:
        out["n_bytes_content"] = torch.tensor(
            [x["n_bytes_content"] for x in items], dtype=torch.long
        )
    return out


def make_loader(
    cfg: dict[str, Any],
    split: str = "train",
    shuffle: bool | None = None,
    streaming: bool | None = None,
) -> DataLoader:
    """Build a DataLoader. Supports packed streams (Phase 2B.1a) and legacy padding."""
    from lct.data.packed_stream import (
        DEFAULT_DOC_SENTINEL,
        PackedWindowDataset,
        PackedWindowIterable,
    )

    data_cfg = cfg.get("data", cfg)
    train_cfg = cfg.get("train", {})
    bs = train_cfg.get("batch_size", 4)
    if shuffle is None:
        shuffle = split == "train"

    if data_cfg.get("source") == "packed_bin":
        from lct.data.real_corpus import make_packed_bin_loader

        return make_packed_bin_loader(cfg, split=split)

    use_packed = bool(data_cfg.get("packed_stream", False))
    window_bytes = int(data_cfg.get("seq_len", data_cfg.get("window_bytes", 1024)))
    sentinel = int(data_cfg.get("doc_sentinel_byte", DEFAULT_DOC_SENTINEL))
    drop_tail = bool(data_cfg.get("drop_partial_tail", True))

    use_stream = streaming if streaming is not None else bool(
        train_cfg.get("target_bytes_seen") or data_cfg.get("streaming_train", False)
    )

    if use_packed:
        if data_cfg.get("val_corpus_path"):
            export_fixed_val_corpus(cfg)
        base = build_dataset(cfg, "train" if split == "train" else "val")
        docs = getattr(base, "docs", None)
        if docs is None:
            raise RuntimeError("Packed stream requires a map dataset with .docs")
        if split == "train" and use_stream:
            ds: Dataset | IterableDataset = PackedWindowIterable(
                docs=docs,
                window_bytes=window_bytes,
                sentinel=sentinel,
                seed=cfg.get("seed", 42),
                drop_partial_tail=drop_tail,
            )
            return DataLoader(
                ds,
                batch_size=bs,
                shuffle=False,
                collate_fn=collate_batch,
                drop_last=True,
                num_workers=0,
            )
        # Val or finite train: map-style packed windows
        ds = PackedWindowDataset(
            docs=docs,
            window_bytes=window_bytes,
            sentinel=sentinel,
            drop_partial_tail=drop_tail,
        )
        return DataLoader(
            ds,
            batch_size=bs,
            shuffle=False,
            collate_fn=collate_batch,
            drop_last=False,
            num_workers=0,
        )

    if split == "train" and use_stream:
        # Ensure val corpus exists first so train docs don't overlap later exports
        if data_cfg.get("val_corpus_path"):
            export_fixed_val_corpus(cfg)
        base = build_dataset(cfg, "train")
        docs = getattr(base, "docs", None)
        if docs is None:
            raise RuntimeError("Streaming train requires a map dataset with .docs")
        ds = ByteWindowIterableDataset(
            docs=docs,
            seq_len=window_bytes,
            seed=cfg.get("seed", 42),
        )
        return DataLoader(
            ds,
            batch_size=bs,
            shuffle=False,
            collate_fn=collate_batch,
            drop_last=True,
            num_workers=0,
        )

    ds = build_dataset(cfg, split)
    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=shuffle,
        collate_fn=collate_batch,
        drop_last=True,
        num_workers=0,
    )


def write_canonical_probes(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    probes = {
        "text": TEXT_SNIPPETS[:20],
        "code": CODE_SNIPPETS[:20],
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(probes, f, indent=2)