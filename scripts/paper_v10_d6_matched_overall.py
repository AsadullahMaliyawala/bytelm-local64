#!/usr/bin/env python3
"""D6: re-score 350M matched-span overall BPB only (both twins). Short eval."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts")]

import paper_v4_recompute as v4  # noqa: E402
import paper_v8_matched_span as ms  # noqa: E402
from matched_bpe_load import load_matched_bpe  # noqa: E402

OUT = Path("${LCT_DATA}/experiments/paper_v10")
RESEARCH = ROOT / "research"
VAL = Path("${LCT_DATA}/data/real_v3")
FILES = {"text": "val_text.bin", "code": "val_code.bin", "math": "val_math.bin"}
DOMAINS = ("text", "code", "math")
LOC = Path("${LCT_DATA}/checkpoints/matched350m_local64/matched350m_local64.pt")
BPE = Path("${LCT_DATA}/checkpoints/matched350m_bpe/matched350m_bpe.pt")
EXPECTED = {
    "overall": {"local64": 0.9633564469210495, "bpe": 0.915476456185929},
    "text": {"local64": 0.9906911761998325, "bpe": 0.9447898423888139},
    "code": {"local64": 0.6563408883676186, "bpe": 0.6003194192987635},
    "math": {"local64": 1.102334142783805, "bpe": 1.0388251924006202},
}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    print({"device": str(device), "mem_before": torch.cuda.mem_get_info() if device.type == "cuda" else None}, flush=True)
    loc = ms.load_local(LOC, device)
    bpe_m, bpe, step, seen = load_matched_bpe(BPE, device)
    tok = bpe.tok
    size_rows = {}
    for name, fname in FILES.items():
        rows = []
        raw = (VAL / fname).read_bytes()
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
            "delta": stats[name]["diff_bpe_minus_local64"],
        }
        print({"domain": name, "local64": bl, "bpe": bb, "match_4dp": match4}, flush=True)
    report["PASS"] = ok
    out = OUT / "d6_matched_span_rerun.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (RESEARCH / "d6_matched_span_rerun.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print({"D6": "PASS" if ok else "FAIL", "out": str(out)}, flush=True)


if __name__ == "__main__":
    main()
