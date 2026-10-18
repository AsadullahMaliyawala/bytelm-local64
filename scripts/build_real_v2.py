#!/usr/bin/env python
"""Level 2: stream real_v2 and train an 8k ByteLevel BPE. No language-model training."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from lct.data.bpe_checks import ROUNDTRIP_SAMPLES, assert_bytelevel_roundtrip
from lct.data.real_corpus import SEP, write_real_v2
from lct.utils import save_json

OUT = ROOT / "data" / "real_v2"
TARGETS = {"fineweb-edu": 0.70, "codeparrot-clean-train": 0.20, "open-web-math": 0.10}
VAL_FILES = {
    "fineweb-edu": OUT / "val_text.bin",
    "codeparrot-clean-train": OUT / "val_code.bin",
    "open-web-math": OUT / "val_math.bin",
}


def iter_docs(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return
    data = np.memmap(path, dtype=np.uint8, mode="r")
    n = int(data.shape[0])
    chunk = 64 * 1024 * 1024
    start = 0
    pos = 0
    while pos < n:
        end = min(n, pos + chunk)
        block = np.asarray(data[pos:end])
        for rel in np.flatnonzero(block == SEP).tolist():
            abs_i = pos + int(rel)
            if abs_i > start:
                yield bytes(data[start:abs_i])
            start = abs_i + 1
        pos = end
    if start < n:
        yield bytes(data[start:n])


def doc_hash(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def train_bpe() -> Path:
    from baselines.bpe_lm import SimpleBPE

    save_dir = ROOT / "checkpoints" / "bpe_8k_real_v2"
    texts = []
    total = 0
    limit = 200 * 1024 * 1024
    for raw in iter_docs(OUT / "train.bin"):
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        texts.append(text)
        total += len(raw)
        if total >= limit:
            break
    bpe = SimpleBPE(vocab_size=8000)
    bpe.train(
        texts,
        save_dir,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>", "\x1e"],
    )
    return save_dir


def roundtrip_val(save_dir: Path) -> dict:
    from baselines.bpe_lm import SimpleBPE

    bpe = SimpleBPE(vocab_size=8000)
    bpe.load(save_dir)
    assert_bytelevel_roundtrip(bpe, ROUNDTRIP_SAMPLES)
    n = 0
    for path in VAL_FILES.values():
        for raw in iter_docs(path):
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                raise SystemExit(f"val doc is not utf-8 in {path.name}")
            enc = bpe.tok.encode(text)
            try:
                decoded = bpe.tok.decode(enc.ids, skip_special_tokens=False)
            except TypeError:
                decoded = bpe.tok.decode(enc.ids)
            if decoded != text:
                raise SystemExit(f"round-trip failed on {path.name} doc {n}")
            n += 1
    return {"val_docs": n, "ok": True}


def overlap_ok() -> dict:
    val_hashes: set[str] = set()
    for path in VAL_FILES.values():
        for raw in iter_docs(path):
            val_hashes.add(doc_hash(raw))
    hits = 0
    scanned = 0
    for raw in iter_docs(OUT / "train.bin"):
        scanned += 1
        if doc_hash(raw) in val_hashes:
            hits += 1
            if hits > 5:
                break
    return {"val_docs": len(val_hashes), "train_docs_scanned": scanned, "overlap": hits}


def shares_ok(manifest: dict) -> dict:
    got = manifest["train_share"]
    flags = {}
    for name, target in TARGETS.items():
        flags[name] = abs(float(got[name]) - target) <= 0.02
    return {"shares": got, "within_2pct": flags, "ok": all(flags.values())}


def write_findings(result: dict) -> None:
    path = ROOT / "research" / "FINDINGS.md"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    marker = "## real_v2 corpus"
    decision = "PASS" if result["pass"] else "FAIL"
    block = (
        f"{marker}\n\n"
        f"**Decision:** {decision}. "
        f"Train bytes {result['manifest']['bytes_per_source']}. "
        f"Shares {result['shares']['shares']}. "
        f"Val/train hash overlap {result['overlap']['overlap']}. "
        f"BPE round-trip val docs {result['roundtrip']['val_docs']}.\n\n"
        "real_v2 is a packed uint8 corpus, documents separated by 0x1E, "
        "with a hashed split and a fresh 8k ByteLevel BPE trained on the first 200MB. "
        "The separator is a special token. Source mix targets are 70% FineWeb-Edu, "
        "20% permissively licensed codeparrot-clean Python, and 10% open-web-math.\n\n"
        + (
            "The build gate passed, so this corpus is the training data for the scaling ladder.\n"
            if result["pass"]
            else "The build gate failed. Scaling does not start on this corpus until the failed check is fixed.\n"
        )
    )
    start = text.find(marker)
    if start < 0:
        text = text.rstrip() + "\n\n" + block
    else:
        rest = text[start + len(marker) :]
        nxt = rest.find("\n## ")
        tail = "" if nxt < 0 else rest[nxt + 1 :]
        text = text[:start].rstrip() + "\n\n" + block + ("\n" + tail if tail else "")
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")


def pre_register() -> None:
    path = ROOT / "research" / "HYPOTHESES.md"
    text = path.read_text(encoding="utf-8")
    marker = "## Small full model — Level 2 real_v2 (pre-registered)"
    if marker in text:
        return
    block = f"""

{marker}

**Question:** Can a 3B-byte mix of FineWeb-Edu, permissive Python, and open-web-math be packed with a fixed val split and an 8k ByteLevel BPE that round-trips every val document?

**Pass:** source shares within ±2% of 70/20/10, zero val/train document-hash overlap, and BPE encode/decode round-trip on 100% of val documents. The separator byte is a special token. Criteria are fixed before the build is scored.
"""
    path.write_text(text.rstrip() + block + "\n", encoding="utf-8")


def main() -> None:
    pre_register()
    manifest = write_real_v2(OUT)
    save_dir = train_bpe()
    rt = roundtrip_val(save_dir)
    ov = overlap_ok()
    sh = shares_ok(manifest)
    passed = bool(rt["ok"] and ov["overlap"] == 0 and sh["ok"])
    result = {
        "run_id": "real_v2_build",
        "config": {"train_bytes": 3_000_000_000, "val_bytes": 5_000_000, "bpe_vocab": 8000, "bpe_sample_bytes": 200 * 1024 * 1024},
        "parameters": {"bpe_vocab": 8000},
        "bytes_seen": int(sum(manifest["bytes_per_source"].values())),
        "wall_seconds": None,
        "bytes_per_second": None,
        "peak_vram_mb": 0,
        "checkpoints": {"bpe": str(save_dir / "bpe.json")},
        "manifest": manifest,
        "roundtrip": rt,
        "overlap": ov,
        "shares": sh,
        "pass": passed,
    }
    save_json(result, ROOT / "experiments" / "real_v2_build.json")
    write_findings(result)
    print({"pass": passed, "shares": sh["shares"], "overlap": ov["overlap"], "val_docs": rt["val_docs"]}, flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
