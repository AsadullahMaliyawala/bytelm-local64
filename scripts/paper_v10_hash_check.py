#!/usr/bin/env python3
"""D7: compare on-disk checkpoint hashes to FROZEN_RESULTS.md / paper_v10_numbers.json.

350M twins: full-file SHA-256 + model state_dict tensor SHA-256 (both must match).
120M twins: first 32MB + size (legacy freeze recipe).
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
FROZEN = (ROOT / "research" / "FROZEN_RESULTS.md").read_text(encoding="utf-8")
NUMS = json.loads((ROOT / "research" / "paper_v10_numbers.json").read_text(encoding="utf-8"))


def sha_full(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_head_size(path: Path, nbytes: int = 32 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        left = nbytes
        while left > 0:
            chunk = f.read(min(1 << 20, left))
            if not chunk:
                break
            h.update(chunk)
            left -= len(chunk)
    h.update(str(path.stat().st_size).encode())
    return h.hexdigest()


def _tensor_raw_bytes(t: torch.Tensor) -> bytes:
    contig = t.detach().to("cpu").contiguous()
    if contig.numel() == 0:
        return b""
    return contig.reshape(-1).view(torch.uint8).numpy().tobytes()


def sha_model_tensors(path: Path) -> str:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    sd = blob.get("model")
    if not isinstance(sd, dict):
        raise RuntimeError(f"no model state_dict in {path}")
    h = hashlib.sha256()
    for k in sorted(sd.keys()):
        t = sd[k]
        if not torch.is_tensor(t):
            continue
        h.update(k.encode("utf-8"))
        h.update(str(t.dtype).encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(_tensor_raw_bytes(t))
    del blob
    return h.hexdigest()


def parse_350m_expected() -> dict[str, dict[str, str]]:
    """Parse full-file + model-tensor hashes from the 350M checkpoint table."""
    out: dict[str, dict[str, str]] = {}
    for line in FROZEN.splitlines():
        hashes = re.findall(r"`([0-9a-f]{64})`", line)
        if len(hashes) < 2:
            continue
        if "matched350m_local64.pt" in line:
            out["local64_350m"] = {"full": hashes[0], "model": hashes[1]}
        elif "matched350m_bpe.pt" in line:
            out["bpe_350m"] = {"full": hashes[0], "model": hashes[1]}
    # Prefer paper_v10_numbers.json locks when present (must agree with FROZEN)
    locks = NUMS.get("locks", {})
    for key, name in (
        ("matched350m_local64", "local64_350m"),
        ("matched350m_bpe", "bpe_350m"),
    ):
        rec = locks.get(key) or {}
        if rec.get("sha256_full") and rec.get("sha256_model_tensors"):
            if name in out:
                assert out[name]["full"] == rec["sha256_full"], (name, "full mismatch FROZEN vs numbers")
                assert out[name]["model"] == rec["sha256_model_tensors"], (name, "model mismatch FROZEN vs numbers")
            out[name] = {"full": rec["sha256_full"], "model": rec["sha256_model_tensors"]}
    return out


def parse_120m_expected() -> dict[str, str]:
    out: dict[str, str] = {}
    for line in FROZEN.splitlines():
        m = re.search(r"`([0-9a-f]{64})`", line)
        if not m:
            continue
        if "ladder120_shuf_local64_s0.pt" in line:
            out["local64_120m"] = m.group(1)
        elif "ladder120_shuf_bpe_s0.pt" in line:
            out["bpe_120m"] = m.group(1)
    return out


exp350 = parse_350m_expected()
exp120 = parse_120m_expected()

wanted350 = {
    "local64_350m": "${LCT_DATA}/checkpoints/matched350m_local64/matched350m_local64.pt",
    "bpe_350m": "${LCT_DATA}/checkpoints/matched350m_bpe/matched350m_bpe.pt",
}
wanted120 = {
    "local64_120m": "${LCT_DATA}/checkpoints/ladder120_shuf/ladder120_shuf_local64_s0.pt",
    "bpe_120m": "${LCT_DATA}/checkpoints/ladder120_shuf/ladder120_shuf_bpe_s0.pt",
}

rows = []
for name, path in wanted350.items():
    p = Path(path)
    exp = exp350.get(name)
    if not p.exists():
        rows.append({"name": name, "path": path, "verdict": "FAIL", "reason": "MISSING_FILE"})
        print(f"FAIL {name}: MISSING_FILE", flush=True)
        continue
    if not exp:
        rows.append({"name": name, "path": path, "verdict": "FAIL", "reason": "NO_EXPECTED"})
        print(f"FAIL {name}: NO_EXPECTED", flush=True)
        continue
    got_full = sha_full(p)
    got_model = sha_model_tensors(p)
    match_full = got_full == exp["full"]
    match_model = got_model == exp["model"]
    match = match_full and match_model
    verdict = "PASS" if match else "FAIL"
    rows.append(
        {
            "name": name,
            "path": path,
            "got_full": got_full,
            "expected_full": exp["full"],
            "match_full": match_full,
            "got_model": got_model,
            "expected_model": exp["model"],
            "match_model": match_model,
            "match": match,
            "verdict": verdict,
        }
    )
    print(
        f"{verdict} {name}: full={got_full[:16]}… "
        f"({'ok' if match_full else 'BAD'}) model={got_model[:16]}… "
        f"({'ok' if match_model else 'BAD'})",
        flush=True,
    )

for name, path in wanted120.items():
    p = Path(path)
    exp = exp120.get(name)
    if not p.exists():
        rows.append({"name": name, "path": path, "verdict": "FAIL", "reason": "MISSING_FILE"})
        print(f"FAIL {name}: MISSING_FILE", flush=True)
        continue
    got = sha_head_size(p)
    match = bool(exp) and got == exp
    verdict = "PASS" if match else "FAIL"
    rows.append(
        {
            "name": name,
            "path": path,
            "got": got,
            "expected": exp,
            "match": match,
            "verdict": verdict,
            "recipe": "first32MB+size",
        }
    )
    print(f"{verdict} {name}: got={got[:16]}… expected={(exp or 'NONE')[:16]}…", flush=True)

out = {
    "rows": rows,
    "all_match": all(r.get("verdict") == "PASS" for r in rows),
    "D7": "PASS" if all(r.get("verdict") == "PASS" for r in rows) else "FAIL",
    "note": "350M: full-file + model-tensor; 120M: first32MB+size",
}
Path("${LCT_DATA}/experiments/paper_v10").mkdir(parents=True, exist_ok=True)
Path("${LCT_DATA}/experiments/paper_v10/hash_check.json").write_text(json.dumps(out, indent=2))
(ROOT / "research" / "hash_check_v10.json").write_text(json.dumps(out, indent=2))
print(json.dumps({"D7": out["D7"], "per_file": {r["name"]: r["verdict"] for r in rows}}, indent=2), flush=True)
