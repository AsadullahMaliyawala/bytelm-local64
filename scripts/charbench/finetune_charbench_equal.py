"""Equal fine-tuning test on CharBench train, then score on test.

local64: full SFT, LR 5e-5, 3 epochs, cosine, batch 8.
HF: full SFT when it fits, else LoRA r=16; same epochs and effective samples.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from format_charbench import eval_prefix, train_text  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def format_example(row: dict) -> str:
    """Train text without SEP: ``Q: …\\nA: {answer}\\n`` (leading space after colon is scored)."""
    return train_text(row["question"], str(row["answer"]))


def cosine_lr(step: int, total: int, peak: float, warmup_frac: float = 0.03) -> float:
    warm = max(1, int(total * warmup_frac))
    if step <= warm:
        return peak * step / warm
    t = (step - warm) / max(1, total - warm)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))


def finetune_local64(ckpt: Path, train_rows: list[dict], epochs: int, batch: int, lr: float, work: Path) -> Path:
    from lct.train.instruct_v7 import SEP, load_local64_checkpoint, sequence_nll

    device = torch.device("cuda")
    model, step0, seen0 = load_local64_checkpoint(ckpt, device)
    packed = []
    for row in train_rows:
        # train_text already ends with newline; append SEP only
        raw = format_example(row).encode("utf-8") + bytes([SEP])
        if 4 <= len(raw) <= 2048:
            prefix_len = len(eval_prefix(row["question"]).encode("utf-8"))
            weights = [0.0] * len(raw)
            for i in range(prefix_len, len(raw)):
                weights[i] = 1.0
            weights[-1] = 3.0
            packed.append((raw, weights))
    steps_per = max(1, (len(packed) + batch - 1) // batch)
    total = steps_per * epochs
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    order = list(range(len(packed)))
    step = 0
    rng = random.Random(0)
    model.train()
    for epoch in range(epochs):
        rng.shuffle(order)
        for start in range(0, len(order), batch):
            step += 1
            chunk = [packed[i] for i in order[start : start + batch]]
            width = max(len(raw) for raw, _ in chunk)
            tokens = torch.zeros((len(chunk), width), dtype=torch.long, device=device)
            weights = torch.zeros((len(chunk), width), dtype=torch.float, device=device)
            for i, (raw, w) in enumerate(chunk):
                tokens[i, : len(raw)] = torch.tensor(list(raw), dtype=torch.long)
                weights[i, : len(w)] = torch.tensor(w, dtype=torch.float)
            opt.param_groups[0]["lr"] = cosine_lr(step, total, lr)
            opt.zero_grad(set_to_none=True)
            logits = model._byte_logits(tokens)
            loss = sequence_nll(logits, tokens, weights)
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step == 1 or step % 50 == 0 or step == total:
                print({"stage": "local64_ft", "step": step, "total": total, "loss": float(loss.item())}, flush=True)
    out = work / "local64_charbench_ft.pt"
    torch.save({"model": model.state_dict(), "step": step0, "seen": seen0, "ft_steps": step}, out)
    return out


def finetune_hf(model_id: str, train_rows: list[dict], epochs: int, batch: int, lr: float, work: Path) -> Path:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.to(device)
    model.train()
    texts = [format_example(row) + tok.eos_token for row in train_rows]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps_per = max(1, (len(texts) + batch - 1) // batch)
    total = steps_per * epochs
    step = 0
    rng = random.Random(0)
    order = list(range(len(texts)))
    for epoch in range(epochs):
        rng.shuffle(order)
        for start in range(0, len(order), batch):
            step += 1
            batch_texts = [texts[i] for i in order[start : start + batch]]
            enc = tok(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
            enc = {k: v.to(device) for k, v in enc.items()}
            labels = enc["input_ids"].clone()
            labels[enc["attention_mask"] == 0] = -100
            opt.param_groups[0]["lr"] = cosine_lr(step, total, lr)
            opt.zero_grad(set_to_none=True)
            out = model(**enc, labels=labels)
            out.loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step == 1 or step % 50 == 0 or step == total:
                print({"stage": "hf_ft", "model": model_id, "step": step, "total": total, "loss": float(out.loss.item())}, flush=True)
    out_dir = work / ("hf_ft_" + model_id.replace("/", "__"))
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--local64", type=Path, default=None)
    parser.add_argument("--hf", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--score-out", type=Path, required=True)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    train_rows = load_jsonl(args.data / "train.jsonl")
    score_mod_path = ROOT / "scripts" / "charbench" / "score_charbench.py"
    import importlib.util

    spec = importlib.util.spec_from_file_location("score_charbench", score_mod_path)
    score = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(score)
    if args.local64 is not None:
        ft = finetune_local64(args.local64, train_rows, args.epochs, args.batch, args.lr, args.work)
        report = score.score_local64(ft, args.data, chat=False, max_new=64, few_shot=False)
        report["finetuned_from"] = str(args.local64)
        report["ft_ckpt"] = str(ft)
        report["protocol"] = "zero_shot_matched_prefix"
    else:
        ft = finetune_hf(args.hf, train_rows, args.epochs, args.batch, args.lr, args.work)
        report = score.score_hf(str(ft), args.data, max_new=64, few_shot=False)
        report["finetuned_from"] = args.hf
        report["ft_ckpt"] = str(ft)
    args.score_out.parent.mkdir(parents=True, exist_ok=True)
    args.score_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    slim = {k: report[k] for k in report if k != "details"}
    print(json.dumps(slim, indent=2), flush=True)


if __name__ == "__main__":
    main()
