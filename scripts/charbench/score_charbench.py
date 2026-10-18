"""Few-shot CharBench scoring for local64 checkpoints and Hugging Face causal LMs."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from format_charbench import eval_prefix, few_shot_prompt  # noqa: E402


def normalize_answer(text: str) -> str:
    text = text.strip()
    text = text.splitlines()[0].strip() if text else ""
    # Drop trailing period / quotes common in chatty models.
    text = text.strip().strip("\"'`").rstrip(".")
    return text


def match(pred: str, gold: str) -> bool:
    p = normalize_answer(pred)
    g = normalize_answer(gold)
    if not p:
        return False
    if p.casefold() == g.casefold():
        return True
    # Allow leading "A: " and answers embedded as first token.
    p2 = re.sub(r"^(a|answer)\s*[:\-]\s*", "", p, flags=re.I).strip()
    if p2.casefold() == g.casefold():
        return True
    # Numeric equality
    if g.isdigit() and re.search(rf"(?<!\d){re.escape(g)}(?!\d)", p):
        return True
    if g.casefold() in {"yes", "no"} and p.casefold().startswith(g.casefold()):
        return True
    # Single-character gold: first non-space char
    if len(g) == 1 and p[:1] == g:
        return True
    return False


def format_prompt(shots: list[dict], question: str) -> str:
    return few_shot_prompt(shots, question)


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def score_local64(ckpt: Path, data_dir: Path, chat: bool, max_new: int, *, few_shot: bool = True) -> dict:
    from lct.train.instruct_v7 import SEP, load_local64_checkpoint, prompt_bytes

    device = torch.device("cuda")
    model, step, seen = load_local64_checkpoint(ckpt, device)
    few = json.loads((data_dir / "few_shot.json").read_text(encoding="utf-8"))
    test = load_rows(data_dir / "test.jsonl")

    def generate(prefix: bytes) -> str:
        # Raw greedy: no chat prepare_logits / sentence bias / repeat cuts.
        generated = bytearray()
        for _ in range(max_new):
            tokens = torch.tensor([[b for b in (prefix + bytes(generated))]], dtype=torch.long, device=device)
            with torch.no_grad():
                if hasattr(model, "_next_byte_logits"):
                    logits = model._next_byte_logits(tokens)
                else:
                    logits = model._byte_logits(tokens)
            byte = int(torch.argmax(logits[0].detach().float().cpu()).item())
            if byte in (10, SEP):
                break
            generated.append(byte)
        return generated.decode("utf-8", errors="ignore")

    details = []
    by_task: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(test, start=1):
        shots = few[row["task"]]
        question = row["question"]
        if few_shot:
            user = format_prompt(shots, question)
        else:
            user = eval_prefix(question)
        if chat:
            prefix = prompt_bytes([], user)
        else:
            prefix = user.encode("utf-8")
        pred = generate(prefix)
        hit = match(pred, row["answer"])
        by_task[row["task"]].append(int(hit))
        details.append(
            {
                "task": row["task"],
                "question": question,
                "gold": row["answer"],
                "pred": pred,
                "correct": hit,
            }
        )
        if index % 50 == 0 or index == len(test):
            print({"scored": index, "of": len(test), "correct": sum(d["correct"] for d in details)}, flush=True)
    per_task = {task: {"n": len(v), "correct": sum(v), "accuracy": sum(v) / len(v)} for task, v in sorted(by_task.items())}
    overall = sum(d["correct"] for d in details)
    return {
        "model": str(ckpt),
        "kind": "local64_chat" if chat else "local64_base",
        "step": step,
        "seen": seen,
        "n": len(details),
        "correct": overall,
        "accuracy": overall / max(len(details), 1),
        "per_task": per_task,
        "details": details,
    }


def score_bpe(ckpt: Path, bpe_dir: Path, data_dir: Path, max_new: int, *, few_shot: bool = True) -> dict:
    sys.path.insert(0, str(ROOT / "scripts"))
    from matched_bpe_load import greedy_bpe_generate, load_matched_bpe

    device = torch.device("cuda")
    model, bpe, step, seen = load_matched_bpe(ckpt, device, bpe_dir)
    few = json.loads((data_dir / "few_shot.json").read_text(encoding="utf-8"))
    test = load_rows(data_dir / "test.jsonl")
    details = []
    by_task: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(test, start=1):
        if few_shot:
            prompt = format_prompt(few[row["task"]], row["question"])
        else:
            prompt = eval_prefix(row["question"])
        pred = greedy_bpe_generate(model, bpe, prompt, device, max_new=max_new)
        hit = match(pred, row["answer"])
        by_task[row["task"]].append(int(hit))
        details.append(
            {"task": row["task"], "question": row["question"], "gold": row["answer"], "pred": pred, "correct": hit}
        )
        if index % 50 == 0 or index == len(test):
            print({"scored": index, "of": len(test), "correct": sum(d["correct"] for d in details)}, flush=True)
    per_task = {task: {"n": len(v), "correct": sum(v), "accuracy": sum(v) / len(v)} for task, v in sorted(by_task.items())}
    overall = sum(d["correct"] for d in details)
    return {
        "model": str(ckpt),
        "kind": "bpe_8k",
        "bpe_dir": str(bpe_dir),
        "step": step,
        "seen": seen,
        "n": len(details),
        "correct": overall,
        "accuracy": overall / max(len(details), 1),
        "per_task": per_task,
        "details": details,
    }


def score_hf(model_id: str, data_dir: Path, max_new: int, *, few_shot: bool = True) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    few = json.loads((data_dir / "few_shot.json").read_text(encoding="utf-8"))
    test = load_rows(data_dir / "test.jsonl")
    details = []
    by_task: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(test, start=1):
        if few_shot:
            prompt = format_prompt(few[row["task"]], row["question"])
        else:
            prompt = eval_prefix(row["question"])
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        gen = out[0, inputs["input_ids"].shape[1] :]
        pred = tok.decode(gen, skip_special_tokens=True)
        hit = match(pred, row["answer"])
        by_task[row["task"]].append(int(hit))
        details.append({"task": row["task"], "question": row["question"], "gold": row["answer"], "pred": pred, "correct": hit})
        if index % 50 == 0 or index == len(test):
            print({"model": model_id, "scored": index, "of": len(test), "correct": sum(d["correct"] for d in details)}, flush=True)
    per_task = {task: {"n": len(v), "correct": sum(v), "accuracy": sum(v) / len(v)} for task, v in sorted(by_task.items())}
    overall = sum(d["correct"] for d in details)
    return {
        "model": model_id,
        "kind": "hf",
        "n": len(details),
        "correct": overall,
        "accuracy": overall / max(len(details), 1),
        "per_task": per_task,
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--local64", type=Path, default=None)
    parser.add_argument("--bpe", type=Path, default=None)
    parser.add_argument("--bpe-dir", type=Path, default=ROOT / "checkpoints" / "bpe_8k_real_v2")
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--hf", type=str, default=None)
    parser.add_argument("--max-new", type=int, default=64)
    args = parser.parse_args()
    if args.local64 is not None:
        report = score_local64(args.local64, args.data, args.chat, args.max_new)
    elif args.bpe is not None:
        report = score_bpe(args.bpe, args.bpe_dir, args.data, args.max_new)
    elif args.hf is not None:
        report = score_hf(args.hf, args.data, args.max_new)
    else:
        raise SystemExit("need --local64, --bpe, or --hf")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    slim = {k: report[k] for k in report if k != "details"}
    print(json.dumps(slim, indent=2), flush=True)
    print({"wrote": str(args.out)}, flush=True)


if __name__ == "__main__":
    main()
