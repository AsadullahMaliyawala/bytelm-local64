"""Full benchmarks + fact200 for the matched BPE-8k twin (kind=bpe in eval_suite)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from matched_bpe_load import greedy_bpe_generate, load_matched_bpe  # noqa: E402

COUNTS = {"hellaswag": 10042, "piqa": 1838, "arc_easy": 0, "lambada": 5153}


def score_fact200_bpe(model, bpe, device) -> dict:
    fact_spec = importlib.util.spec_from_file_location("score_fact200", ROOT / "scripts" / "score_fact200.py")
    fact_mod = importlib.util.module_from_spec(fact_spec)
    assert fact_spec.loader is not None
    fact_spec.loader.exec_module(fact_mod)
    rows = fact_mod.verify_lock()
    correct = 0
    answers = []
    by_cat: dict[str, list[int]] = {}
    for row in rows:
        prompt = fact_mod.prompt_for(row["question"])
        pred = greedy_bpe_generate(model, bpe, prompt, device, max_new=200).strip()
        hit = fact_mod.alias_hit(pred, row["aliases"])
        correct += int(hit)
        cat = row.get("category", "unknown")
        by_cat.setdefault(cat, []).append(int(hit))
        answers.append(
            {"id": row.get("id"), "category": cat, "question": row["question"], "pred": pred, "correct": bool(hit)}
        )
        if len(answers) == 1 or len(answers) % 20 == 0:
            print({"fact200_scored": len(answers), "correct": correct, "last": pred[:120]}, flush=True)
    categories = {
        c: {"n": len(v), "correct": sum(v), "accuracy": sum(v) / len(v)} for c, v in sorted(by_cat.items())
    }
    return {
        "n": len(rows),
        "correct": correct,
        "accuracy": correct / max(len(rows), 1),
        "categories": categories,
        "answers": answers,
        "kind": "bpe",
    }


def main() -> None:
    ckpt = Path(sys.argv[1])
    bpe_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "checkpoints" / "bpe_8k_real_v2"
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else ckpt.with_suffix(".full_benchmarks.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, bpe, step, seen = load_matched_bpe(ckpt, device, bpe_dir)
    assert bpe.tok is not None

    suite_spec = importlib.util.spec_from_file_location("eval_suite", ROOT / "scripts" / "eval_suite.py")
    suite = importlib.util.module_from_spec(suite_spec)
    assert suite_spec.loader is not None
    suite_spec.loader.exec_module(suite)

    report = {
        "checkpoint": str(ckpt),
        "kind": "bpe",
        "step": step,
        "seen": seen,
        "bpe_dir": str(bpe_dir),
        "benchmarks": {},
    }
    for name, count in COUNTS.items():
        row = suite.run_benchmark(model, "bpe", name, tok=bpe.tok, n_examples=count)
        report["benchmarks"][name] = row
        print({"benchmark": name, "row": {k: row[k] for k in row if k != "details"}}, flush=True)

    fact = score_fact200_bpe(model, bpe, device)
    report["fact200"] = {k: fact[k] for k in fact if k != "answers"}
    print({"fact200": report["fact200"]}, flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    # also drop answers beside for audit
    (out.parent / (out.stem + "_fact200_answers.json")).write_text(
        json.dumps(fact, indent=2), encoding="utf-8"
    )
    print({"wrote": str(out)}, flush=True)


if __name__ == "__main__":
    main()
