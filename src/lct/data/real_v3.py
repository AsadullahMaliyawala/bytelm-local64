"""Stream real_v3 onto disk. Documents are written immediately and never collected in RAM.

Mix by bytes: 85% FineWeb-Edu with score >= 3, 10% permissive AST-valid Python,
5% open-web-math. Validation is about 5 MB. A train document is dropped when its
64-byte MinHash Jaccard against any validation document is at least 0.5.
"""

from __future__ import annotations

import hashlib
import json
import mmap
from pathlib import Path

import numpy as np

from lct.data.real_corpus import (
    SEP,
    stream_codeparrot,
    stream_fineweb_scored,
    stream_open_web_math,
)

SHINGLE = 64
N_HASH = 64
JACCARD = 0.5
CHUNK = 256
PERM_SEED = 0
HUGE = 10**18

_A = np.array(
    [np.uint64(int(0x9E3779B185EBCA87 ^ (i * 0xBF58476D1CE4E5B9)) & 0xFFFFFFFFFFFFFFFF) for i in range(N_HASH)],
    dtype=np.uint64,
)


def doc_shingles(raw: bytes) -> np.ndarray:
    n = len(raw) // SHINGLE
    if n == 0:
        return np.empty(0, dtype=np.uint64)
    arr = np.frombuffer(raw[: n * SHINGLE], dtype=np.uint8).reshape(n, SHINGLE)
    acc = np.full(n, np.uint64(0xCBF29CE484222325), dtype=np.uint64)
    prime = np.uint64(0x100000001B3)
    for col in range(SHINGLE):
        acc = (acc ^ arr[:, col].astype(np.uint64)) * prime
    return acc


def signature(shingles: np.ndarray) -> np.ndarray | None:
    if shingles.size == 0:
        return None
    hv = shingles.astype(np.uint64, copy=False)
    mins = np.empty(N_HASH, dtype=np.uint64)
    for i, coeff in enumerate(_A):
        mins[i] = np.min(np.uint64(coeff) * hv + np.uint64(i))
    return mins


class ValMinHash:
    """Band index so a train document is compared with validation signatures in 64 lookups."""

    def __init__(self) -> None:
        self.bands: list[dict[int, list[int]]] = [dict() for _ in range(N_HASH)]
        self.n = 0

    def add(self, sig: np.ndarray) -> None:
        doc_id = self.n
        self.n += 1
        for i, value in enumerate(sig.tolist()):
            self.bands[i].setdefault(int(value), []).append(doc_id)

    def near(self, sig: np.ndarray, threshold: float = JACCARD) -> bool:
        counts: dict[int, int] = {}
        for i, value in enumerate(sig.tolist()):
            for doc_id in self.bands[i].get(int(value), ()):
                counts[doc_id] = counts.get(doc_id, 0) + 1
                if counts[doc_id] >= threshold * N_HASH:
                    return True
        return False


def chunk_ids(raw: bytes) -> list[int]:
    out = []
    for start in range(0, len(raw) - (CHUNK - 1), CHUNK):
        digest = hashlib.blake2s(raw[start : start + CHUNK], digest_size=8).digest()
        out.append(int.from_bytes(digest, "little"))
    return out


def iter_docs(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("rb") as handle:
        mapped = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        n = len(mapped)
        start = 0
        while start < n:
            rel = mapped.find(bytes([SEP]), start)
            end = n if rel < 0 else rel
            if end > start:
                yield start, bytes(mapped[start:end])
            if rel < 0:
                break
            start = rel + 1
        mapped.close()


def score_histogram(n_rows: int, config: str, out_path: Path) -> dict:
    from datasets import load_dataset

    edges = [i * 0.5 for i in range(12)]
    counts = [0 for _ in range(len(edges) - 1)]
    missing = 0
    ge3 = 0
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name=config, split="train", streaming=True)
    seen = 0
    for row in ds:
        if seen >= n_rows:
            break
        seen += 1
        try:
            score = float(row.get("score"))
        except (TypeError, ValueError):
            missing += 1
            continue
        if score >= 3.0:
            ge3 += 1
        placed = False
        for i in range(len(edges) - 1):
            if edges[i] <= score < edges[i + 1]:
                counts[i] += 1
                placed = True
                break
        if not placed and score >= edges[-1]:
            counts[-1] += 1
        if seen % 20_000 == 0:
            print({"histogram_rows": seen, "fraction_ge_3": ge3 / seen}, flush=True)
    payload = {
        "dataset": "HuggingFaceFW/fineweb-edu",
        "config": config,
        "n_rows": seen,
        "edges": edges,
        "counts": counts,
        "missing_score": missing,
        "n_score_ge_3": ge3,
        "fraction_score_ge_3": (ge3 / seen) if seen else 0.0,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print({"wrote_histogram": str(out_path), **payload}, flush=True)
    return payload


def _empty_source() -> dict:
    return {
        "val_bytes": 0,
        "val_docs": 0,
        "val_done": False,
        "train_bytes": 0,
        "train_docs": 0,
        "train_rows": 0,
        "train_done": False,
        "exact_dup": 0,
        "near_dup": 0,
        "chunk_dup": 0,
        "parse_rejected": 0,
        "exhausted": False,
    }


def write_real_v3(
    out_dir: Path,
    target_train_bytes: int = 70_000_000_000,
    target_val_bytes: int = 5_000_000,
    fineweb_config: str = "sample-100BT",
    min_score: float = 3.0,
    histogram_rows: int = 200_000,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shares = {
        "fineweb-edu": 0.85,
        "codeparrot-clean-train": 0.10,
        "open-web-math": 0.05,
    }
    budgets = {name: int(target_train_bytes * share) for name, share in shares.items()}
    budgets["fineweb-edu"] += target_train_bytes - sum(budgets.values())
    val_budgets = {name: int(target_val_bytes * share) for name, share in shares.items()}
    val_budgets["fineweb-edu"] += target_val_bytes - sum(val_budgets.values())
    val_names = {
        "fineweb-edu": "val_text",
        "codeparrot-clean-train": "val_code",
        "open-web-math": "val_math",
    }
    files = {"train": out_dir / "train.bin"}
    for key in val_names.values():
        files[key] = out_dir / f"{key}.bin"
    state_path = out_dir / "build_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        state = {"sources": {name: _empty_source() for name in shares}, "done": False, "shortfall": None}
        for path in files.values():
            path.write_bytes(b"")
    sources = state.setdefault("sources", {})
    for name in shares:
        sources.setdefault(name, _empty_source())

    def flush() -> None:
        state_path.write_text(json.dumps(state), encoding="utf-8")

    hist_path = out_dir / "fineweb_score_histogram.json"
    if not hist_path.exists():
        score_histogram(histogram_rows, fineweb_config, hist_path)
    else:
        print({"histogram": "exists", "path": str(hist_path)}, flush=True)

    def open_stream(name: str, skip_rows: int, counter: dict, stats: dict):
        if name == "fineweb-edu":
            return stream_fineweb_scored(
                HUGE, min_score=min_score, config=fineweb_config, skip_rows=skip_rows, counter=counter
            )
        if name == "codeparrot-clean-train":
            return stream_codeparrot(
                "Python",
                HUGE,
                domain="code",
                require_parse=True,
                skip_rows=skip_rows,
                counter=counter,
                stats=stats,
            )
        return stream_open_web_math(HUGE, skip_rows=skip_rows, counter=counter)

    for name, val_key in val_names.items():
        row = sources[name]
        if row.get("val_done"):
            continue
        files[val_key].write_bytes(b"")
        row["val_bytes"] = 0
        row["val_docs"] = 0
        counter: dict = {"row": 0}
        print({"val_pass": name, "budget": val_budgets[name]}, flush=True)
        with files[val_key].open("ab") as handle:
            for doc in open_stream(name, 0, counter, row):
                if doc.split != "val":
                    continue
                handle.write(doc.raw)
                handle.write(bytes([SEP]))
                row["val_docs"] += 1
                row["val_bytes"] += len(doc.raw) + 1
                if row["val_bytes"] >= val_budgets[name]:
                    break
        row["val_done"] = True
        flush()
        print({"val_done": name, "val_bytes": row["val_bytes"], "val_docs": row["val_docs"]}, flush=True)

    index = ValMinHash()
    val_hashes: set[bytes] = set()
    for val_key in val_names.values():
        for _start, raw in iter_docs(files[val_key]):
            val_hashes.add(hashlib.sha256(raw).digest())
            sig = signature(doc_shingles(raw))
            if sig is not None:
                index.add(sig)
    print({"val_docs_indexed": index.n, "val_hashes": len(val_hashes)}, flush=True)

    train_hashes: set[bytes] = set()
    for _start, raw in iter_docs(files["train"]):
        train_hashes.add(hashlib.sha256(raw).digest())
    chunk_seen: set[int] = set()
    code_row = sources["codeparrot-clean-train"]
    if int(code_row.get("train_rows", 0)) > 0:
        code_start = int(state.get("code_start", 0))
        for start, raw in iter_docs(files["train"]):
            if start < code_start:
                continue
            chunk_seen.update(chunk_ids(raw))
        print({"resumed_chunk_hashes": len(chunk_seen)}, flush=True)

    for name in shares:
        row = sources[name]
        if row.get("train_done"):
            continue
        if name == "codeparrot-clean-train" and "code_start" not in state:
            state["code_start"] = files["train"].stat().st_size
            flush()
        counter = {"row": 0}
        skip = int(row.get("train_rows", 0))
        print({"train_pass": name, "skip_rows": skip, "budget": budgets[name], "have": row["train_bytes"]}, flush=True)
        n_new = 0
        with files["train"].open("ab") as handle:
            for doc in open_stream(name, skip, counter, row):
                row["train_rows"] = int(counter["row"])
                if doc.split != "train":
                    continue
                digest = hashlib.sha256(doc.raw).digest()
                if digest in val_hashes or digest in train_hashes:
                    row["exact_dup"] += 1
                    continue
                if name == "codeparrot-clean-train":
                    chunks = chunk_ids(doc.raw)
                    if any(item in chunk_seen for item in chunks):
                        row["chunk_dup"] += 1
                        continue
                sig = signature(doc_shingles(doc.raw))
                if sig is not None and index.near(sig):
                    row["near_dup"] += 1
                    continue
                handle.write(doc.raw)
                handle.write(bytes([SEP]))
                train_hashes.add(digest)
                if name == "codeparrot-clean-train":
                    chunk_seen.update(chunk_ids(doc.raw))
                row["train_docs"] += 1
                row["train_bytes"] += len(doc.raw) + 1
                n_new += 1
                if row["train_bytes"] >= budgets[name]:
                    row["train_done"] = True
                    break
                if n_new % 50 == 0:
                    handle.flush()
                    flush()
                    print(
                        {
                            "source": name,
                            "train_bytes": row["train_bytes"],
                            "budget": budgets[name],
                            "docs": row["train_docs"],
                            "exact_dup": row["exact_dup"],
                            "near_dup": row["near_dup"],
                            "chunk_dup": row["chunk_dup"],
                        },
                        flush=True,
                    )
            else:
                row["exhausted"] = True
        if row["train_bytes"] >= budgets[name]:
            row["train_done"] = True
        flush()
        print({"train_status": name, "done": row["train_done"], "exhausted": row.get("exhausted"), "bytes": row["train_bytes"]}, flush=True)
        if row.get("exhausted") and not row["train_done"]:
            state["shortfall"] = {
                "source": name,
                "have_bytes": row["train_bytes"],
                "budget": budgets[name],
            }
            flush()
            break

    short = state.get("shortfall")
    all_done = all(sources[name].get("train_done") for name in shares)
    if all_done and not short:
        spans_list = [(start, len(raw)) for start, raw in iter_docs(files["train"])]
        spans = np.asarray(spans_list, dtype=np.int64)
        perm = np.random.default_rng(PERM_SEED).permutation(len(spans)).astype(np.int64)
        np.save(out_dir / "doc_spans.npy", spans)
        np.save(out_dir / "doc_perm_seed0.npy", perm)
        state["n_docs"] = int(len(spans))
        state["perm_seed"] = PERM_SEED
        state["done"] = True
        fineweb_end = int(sources["fineweb-edu"]["train_bytes"])
        code_end = fineweb_end + int(sources["codeparrot-clean-train"]["train_bytes"])
        state["bounds"] = [fineweb_end, code_end]
    flush()
    total_train = sum(int(sources[name]["train_bytes"]) for name in shares)
    manifest = {
        "separator": SEP,
        "target_train_bytes": target_train_bytes,
        "target_val_bytes": target_val_bytes,
        "min_score": min_score,
        "fineweb_config": fineweb_config,
        "perm_seed": PERM_SEED if state.get("done") else None,
        "n_docs": state.get("n_docs"),
        "bounds": state.get("bounds"),
        "bytes_per_source": {name: int(sources[name]["train_bytes"]) for name in shares},
        "val_bytes_per_source": {name: int(sources[name]["val_bytes"]) for name in shares},
        "doc_counts": {
            name: {"train": int(sources[name]["train_docs"]), "val": int(sources[name]["val_docs"])} for name in shares
        },
        "train_share": {name: (sources[name]["train_bytes"] / total_train if total_train else 0.0) for name in shares},
        "targets": shares,
        "dedup": {
            name: {
                "exact_dup": int(sources[name]["exact_dup"]),
                "near_dup": int(sources[name]["near_dup"]),
                "chunk_dup": int(sources[name]["chunk_dup"]),
                "parse_rejected": int(sources[name].get("parse_rejected", 0)),
            }
            for name in shares
        },
        "licenses": {
            "fineweb-edu": "ODC-By 1.0",
            "codeparrot-clean-train": "MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause, ISC",
            "open-web-math": "ODC-By 1.0 (Hugging Face dataset card open-web-math/open-web-math)",
        },
        "minhash": {"shingle": SHINGLE, "stride": SHINGLE, "n_hash": N_HASH, "jaccard": JACCARD, "drop": "train"},
        "shortfall": state.get("shortfall"),
        "done": bool(state.get("done")),
        "val_train_hash_overlap": 0,
    }
    overlap = 0
    train_set = train_hashes
    for digest in val_hashes:
        if digest in train_set:
            overlap += 1
    manifest["val_train_hash_overlap"] = overlap
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print({"wrote_manifest": str(out_dir / "manifest.json"), "done": manifest["done"], "overlap": overlap}, flush=True)
    return manifest
