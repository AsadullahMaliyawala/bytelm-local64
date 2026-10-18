"""Score the four full benchmark splits and the locked fact200 set on one checkpoint.

HellaSwag validation 10,042, PIQA validation 1,838, ARC-Easy validation (the
whole split), LAMBADA test 5,153. Each row keeps a Wilson 95% interval.
Run this on the final 50B checkpoint after training has stopped.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lct.train.instruct_v7 import load_local64_checkpoint, sample_prompt  # noqa: E402

COUNTS = {"hellaswag": 10042, "piqa": 1838, "arc_easy": 0, "lambada": 5153}
PROMPTS = (
    "The capital of France is",
    "In one sentence, explain gravity.",
    "1 + 1 =",
    "Once upon a time",
    "def add(a, b):\n    return",
)


def main() -> None:
    path = Path(sys.argv[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, step, seen = load_local64_checkpoint(path, device)
    spec = importlib.util.spec_from_file_location("eval_suite", ROOT / "scripts" / "eval_suite.py")
    suite = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(suite)
    report = {"checkpoint": str(path), "step": step, "seen_per_rank": seen, "benchmarks": {}, "prompts": []}
    for name, count in COUNTS.items():
        row = suite.run_benchmark(model, "patch", name, n_examples=count)
        report["benchmarks"][name] = row
        print({"benchmark": name, "row": row}, flush=True)
    fact_spec = importlib.util.spec_from_file_location("score_fact200", ROOT / "scripts" / "score_fact200.py")
    fact_mod = importlib.util.module_from_spec(fact_spec)
    fact_spec.loader.exec_module(fact_mod)
    fact = fact_mod.score_model(model, device)
    report["fact200"] = {
        "n": fact["n"],
        "correct": fact["correct"],
        "accuracy": fact["accuracy"],
        "categories": fact["categories"],
    }
    print({"fact200": report["fact200"]}, flush=True)
    for prompt in PROMPTS:
        sample = sample_prompt(model, prompt, device, temperature=0.6, top_p=0.9, max_new=128)
        report["prompts"].append(
            {"prompt": prompt, "stopped": sample["stopped"], "raw": repr(sample["raw"]), "text": sample["text"]}
        )
    out = path.with_suffix(".full_benchmarks.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print({"wrote": str(out)}, flush=True)


if __name__ == "__main__":
    main()
