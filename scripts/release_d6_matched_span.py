#!/usr/bin/env python3
"""Release D6: matched-span overall/text/code/math BPB vs frozen Table numbers.

Paths via env or flags (no machine-specific defaults):
  --ckpt-local64  --ckpt-bpe  --val-dir  --bpe-dir  [--device cuda|cpu] [--batch-docs N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts")]

import paper_v4_recompute as v4  # noqa: E402
import paper_v8_matched_span as ms  # noqa: E402
from matched_bpe_load import load_matched_bpe  # noqa: E402

FILES = {"text": "val_text.bin", "code": "val_code.bin", "math": "val_math.bin"}
DOMAINS = ("text", "code", "math")
EXPECTED = {
    "overall": {"local64": 0.9633564469210495, "bpe": 0.915476456185929},
    "text": {"local64": 0.9906911761998325, "bpe": 0.9447898423888139},
    "code": {"local64": 0.6563408883676186, "bpe": 0.6003194192987635},
    "math": {"local64": 1.102334142783805, "bpe": 1.0388251924006202},
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-local64", type=Path, required=True)
    p.add_argument("--ckpt-bpe", type=Path, required=True)
    p.add_argument("--val-dir", type=Path, required=True)
    p.add_argument("--bpe-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, default=Path("d6_release_verify.json"))
    args = p.parse_args()
    device = torch.device(args.device)
    loc = ms.load_local(args.ckpt_local64, device)
    bpe_m, bpe, step, seen = load_matched_bpe(args.ckpt_bpe, device, bpe_dir=args.bpe_dir)
    tok = bpe.tok
    size_rows = {}
    for name, fname in FILES.items():
        rows = []
        raw = (args.val_dir / fname).read_bytes()
        for i, doc in enumerate(v4.docs_of(raw)):
            nats_b, bytes_b, _p, wins = ms.score_bpe_windows(bpe_m, tok, doc, device)
            nats_l, bytes_l = ms.score_local_matched(loc, doc, wins, device)
            if bytes_l != bytes_b:
                raise RuntimeError(f"mismatch {name} {i}")
            rows.append({"nats_l": nats_l, "bytes_l": bytes_l, "nats_b": nats_b, "bytes_b": bytes_b})
            if (i + 1) % 200 == 0:
                print({"domain": name, "doc": i + 1}, flush=True)
        size_rows[name] = rows
        print({"domain": name, "docs": len(rows)}, flush=True)
    stats = {name: v4.bootstrap_delta(size_rows[name]) for name in DOMAINS}
    stats["overall"] = v4.bootstrap_delta([r for name in DOMAINS for r in size_rows[name]])
    report = {"protocol": "matched_span_min512tok_2048B", "step_bpe": step, "seen_bpe": seen, "domains": {}}
    ok = True
    for name in list(DOMAINS) + ["overall"]:
        bl = stats[name]["bpb_local64"]
        bb = stats[name]["bpb_bpe"]
        el, eb = EXPECTED[name]["local64"], EXPECTED[name]["bpe"]
        match4 = round(bl, 4) == round(el, 4) and round(bb, 4) == round(eb, 4)
        ok = ok and match4
        report["domains"][name] = {
            "bpb_local64": bl,
            "bpb_bpe": bb,
            "expected_local64": el,
            "expected_bpe": eb,
            "match_4dp": match4,
        }
        print({"domain": name, "local64": round(bl, 4), "bpe": round(bb, 4), "match_4dp": match4}, flush=True)
    report["PASS"] = ok
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print({"D6": "PASS" if ok else "FAIL", "out": str(args.out)}, flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
