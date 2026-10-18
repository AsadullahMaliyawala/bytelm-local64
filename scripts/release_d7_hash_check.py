#!/usr/bin/env python3
"""Release D7: full-file + model-tensor SHA-256 vs SHA256SUMS / FROZEN locks."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch


def sha_full(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
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


def parse_sums(path: Path) -> dict[str, dict[str, str]]:
    """Parse SHA256SUMS with lines: FULL <hash>  path  and TENSOR <hash>  path"""
    out: dict[str, dict[str, str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^(FULL|TENSOR)\s+([0-9a-f]{64})\s+(.+)$", line)
        if not m:
            continue
        kind, hx, name = m.group(1), m.group(2), m.group(3).strip()
        out.setdefault(name, {})[kind.lower()] = hx
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights-dir", type=Path, required=True)
    p.add_argument("--sums", type=Path, default=None)
    p.add_argument("--out", type=Path, default=Path("d7_release_verify.json"))
    args = p.parse_args()
    sums_path = args.sums or (args.weights_dir / "SHA256SUMS")
    expected = parse_sums(sums_path)
    rows = []
    ok = True
    for name in ("matched350m_local64.pt", "matched350m_bpe.pt"):
        path = args.weights_dir / name
        exp = expected.get(name) or expected.get(f"./{name}") or {}
        if not path.exists():
            rows.append({"name": name, "verdict": "FAIL", "reason": "MISSING"})
            ok = False
            print(f"FAIL {name}: MISSING", flush=True)
            continue
        got_full = sha_full(path)
        got_model = sha_model_tensors(path)
        match_full = got_full == exp.get("full")
        match_model = got_model == exp.get("tensor")
        match = match_full and match_model
        ok = ok and match
        verdict = "PASS" if match else "FAIL"
        rows.append(
            {
                "name": name,
                "verdict": verdict,
                "got_full": got_full,
                "expected_full": exp.get("full"),
                "match_full": match_full,
                "got_model": got_model,
                "expected_model": exp.get("tensor"),
                "match_model": match_model,
            }
        )
        print(
            f"{verdict} {name}: full={'ok' if match_full else 'BAD'} model={'ok' if match_model else 'BAD'}",
            flush=True,
        )
    out = {"D7": "PASS" if ok else "FAIL", "rows": rows}
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"D7": out["D7"]}, indent=2), flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
