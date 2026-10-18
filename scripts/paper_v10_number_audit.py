#!/usr/bin/env python3
"""D1: extract numbers from main.tex tables/abstract and map to research JSON."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
NUMS = json.loads((ROOT / "research" / "paper_v10_numbers.json").read_text(encoding="utf-8"))
MS = json.loads((ROOT / "research" / "slide_350m_matched_span.json").read_text(encoding="utf-8"))
CURVE = json.loads((ROOT / "research" / "curve_350m_matched_span.json").read_text(encoding="utf-8"))
HOLM = json.loads((ROOT / "research" / "downstream_holm.json").read_text(encoding="utf-8"))

# Flatten known floats (rounded forms accepted)
known: dict[str, str] = {}

def add(val: float, src: str, places: tuple[int, ...] = (1, 2, 3, 4)):
    for p in places:
        s = f"{val:.{p}f}"
        known[s] = src
        # also without trailing zeros issues
        known[s.rstrip("0").rstrip(".") if "." in s else s] = src

for dom in ("overall", "text", "code", "math"):
    p = MS["paired"][dom]
    add(p["bpb_local64"], f"slide_350m_matched_span.json:{dom}.bpb_local64")
    add(p["bpb_bpe"], f"slide_350m_matched_span.json:{dom}.bpb_bpe")
    add(p["diff_bpe_minus_local64"], f"slide_350m_matched_span.json:{dom}.delta")
    add(p["diff_lo"], f"slide_350m_matched_span.json:{dom}.lo")
    add(p["diff_hi"], f"slide_350m_matched_span.json:{dom}.hi")
    rel = NUMS["matched_span"]["relative_pct_bpe_minus_local_over_bpe"][dom]
    add(rel, f"paper_v10_numbers.json:rel:{dom}", places=(1,))
    known[f"{rel:.1f}"] = f"paper_v10_numbers.json:rel:{dom}"

# relative as printed -5.2 etc
for s, src in [("-5.2", "rel overall"), ("-4.9", "rel text"), ("-9.3", "rel code"), ("-6.1", "rel math")]:
    known[s] = src

# curve gaps
for tag, pt in [(p["tag"], p["point"]["overall"]["diff"]) for p in CURVE["points"]]:
    add(pt, f"curve_350m_matched_span.json:{tag}")
known["-0.113"] = "curve 2b rounded"
known["-0.048"] = "curve 10b rounded"

# downstream
for k, st in HOLM["stats"].items():
    add(st["local64_acc"], f"downstream_holm.json:{k}.local")
    add(st["bpe_acc"], f"downstream_holm.json:{k}.bpe")

# abstract percents
known["53.7"] = "charbench FT *100"
known["40.0"] = "charbench FT *100"
known["16.5"] = "fact200 *100"
known["7.0"] = "fact200 *100"

# params millions
known["351.3"] = "paper_v2_meta_flops local total"
known["335.6"] = "paper_v2_meta_flops bpe total"
known["350.8"] = "paper_v2_meta_flops local nonemb"
known["327.4"] = "paper_v2_meta_flops backbone/nonemb"

# Extract abstract + tabular numbers
abs_m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", TEX, re.S)
tabs = re.findall(r"\\begin\{tabular\}.*?\\end\{tabular\}", TEX, re.S)
chunks = [abs_m.group(1) if abs_m else ""] + tabs
num_re = re.compile(r"(?<![A-Za-z\\])([+-]?\d+\.\d+|[+-]?\d{4,}|\d+\.\d+)")

rows = []
unmapped = []
for ci, chunk in enumerate(chunks):
    for m in num_re.finditer(chunk):
        raw = m.group(1)
        # skip years-like 2016 in cites if in abstract? keep
        src = known.get(raw) or known.get(raw.lstrip("+"))
        # try 4dp round trip
        if src is None:
            try:
                v = float(raw)
                for k, s in known.items():
                    try:
                        if abs(float(k) - v) < 5e-5:
                            src = s
                            break
                    except ValueError:
                        pass
            except ValueError:
                pass
        where = "abstract" if ci == 0 else f"table_{ci}"
        rec = {"where": where, "value": raw, "source": src or "UNMAPPED"}
        rows.append(rec)
        if src is None and re.search(r"\.", raw):
            # ignore integers like 26, 1024, 512 used as hyperparams — still list
            unmapped.append(rec)

out_md = ["# Number audit (D1)", "", f"Extracted floats in abstract+tables: {len(rows)}", f"Unmapped floats: {len([r for r in unmapped if r['source']=='UNMAPPED'])}", ""]
out_md.append("| where | value | source |")
out_md.append("|---|---|---|")
for r in rows:
    flag = " **FAIL**" if r["source"] == "UNMAPPED" and "." in r["value"] else ""
    out_md.append(f"| {r['where']} | `{r['value']}` | {r['source']}{flag} |")

(ROOT / "research" / "number_audit.md").write_text("\n".join(out_md), encoding="utf-8")
(ROOT / "research" / "number_audit.json").write_text(json.dumps({"rows": rows, "unmapped_floats": [r for r in rows if r["source"]=="UNMAPPED" and "." in r["value"]]}, indent=2), encoding="utf-8")
print({"n": len(rows), "unmapped_floats": len([r for r in rows if r["source"]=="UNMAPPED" and "." in r["value"]])})
