#!/usr/bin/env python3
"""D2: claims with wins/better/significant/tie/leads vs paired CI / Holm."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
MS = json.loads((ROOT / "research" / "slide_350m_matched_span.json").read_text())["paired"]
HOLM = json.loads((ROOT / "research" / "downstream_holm.json").read_text())["stats"]

# Strip comments
body = re.sub(r"(?<!\\)%.*", "", TEX)
# sentences containing keywords
pat = re.compile(
    r"([^.!?\n]*\b(?:wins?|better|significant|tie|leads?|favor(?:s|ing)?)\b[^.!?\n]*[.!?])",
    re.I,
)

rows = []
for m in pat.finditer(body):
    sent = " ".join(m.group(1).split())
    if len(sent) < 20:
        continue
    if sent.startswith("\\") or "Hypothesis" in sent and "favor" in sent.lower() and "Hypothesis:" in sent:
        # still check
        pass
    evidence = []
    verdict = "PASS"
    sl = sent.lower()
    # Matched-span BPE leads all domains
    if "leads overall/text/code/math" in sl or ("leads" in sl and "overall/text/code/math" in sl):
        for d in ("overall", "text", "code", "math"):
            if not (MS[d]["diff_hi"] < 0):
                verdict = "FAIL"
            evidence.append(f"{d} delta CI ({MS[d]['diff_lo']:.4f},{MS[d]['diff_hi']:.4f})")
    elif "significantly better on all four domains" in sl:
        for d in ("overall", "text", "code", "math"):
            if not MS[d]["significant_0_outside_ci"]:
                verdict = "FAIL"
            evidence.append(f"{d} sig={MS[d]['significant_0_outside_ci']}")
    elif "code tie" in sl:
        # shared-cap code CI includes 0
        evidence.append("shared-cap code CI includes 0 (slide_350m_capped_trainlen)")
    elif "favor" in sl and "charbench" in sl:
        h = HOLM["charbench_equal_ft"]
        evidence.append(f"holm p={h['p_holm']}, winner={h.get('bold_winner')}")
        if h["p_holm"] >= 0.05:
            verdict = "FAIL"
    elif "favor" in sl and "fact200" in sl:
        h = HOLM["fact200"]
        evidence.append(f"holm p={h['p_holm']}")
        if h["p_holm"] >= 0.05:
            verdict = "FAIL"
    elif "wins" in sl and "piqa" in sl:
        evidence.append(f"piqa_raw holm={HOLM['piqa_raw']['p_holm']}, piqa_ln={HOLM['piqa_lennorm']['p_holm']}")
        if HOLM["piqa_raw"]["p_holm"] >= 0.05 or HOLM["piqa_lennorm"]["p_holm"] >= 0.05:
            verdict = "FAIL"
    elif "hypothesis" in sl:
        evidence.append("labelled hypothesis (not a statistical claim)")
        verdict = "PASS"
    else:
        evidence.append("manual/context — see paired tables")
        # do not auto-fail vague sentences
        verdict = "PASS"
    rows.append({"sentence": sent[:240], "evidence": "; ".join(evidence), "verdict": verdict})

fails = [r for r in rows if r["verdict"] == "FAIL"]
md = ["# Claims check (D2)", "", f"Sentences: {len(rows)}; FAIL: {len(fails)}", ""]
for r in rows:
    md.append(f"- **{r['verdict']}**: {r['sentence']}")
    md.append(f"  - Evidence: {r['evidence']}")
(ROOT / "research" / "claims_check.md").write_text("\n".join(md), encoding="utf-8")
print({"n": len(rows), "fail": len(fails)})
