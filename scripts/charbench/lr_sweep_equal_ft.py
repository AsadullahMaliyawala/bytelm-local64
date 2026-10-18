"""Equal-FT LR sweep on CharBench with val selection (never tune on test).

For each model × LR in {1e-5, 5e-5, 2e-4}: train on train_ft.jsonl, score val.jsonl,
then score test.jsonl. Choose LR by val accuracy; report Wilson intervals on test
at the chosen LR plus the full sweep table.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from format_charbench import eval_prefix, train_text  # noqa: E402
from wilson import wilson_interval, with_wilson  # noqa: E402

LRS = (1e-5, 5e-5, 2e-4)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def cosine_lr(step: int, total: int, peak: float, warmup_frac: float = 0.03) -> float:
    warm = max(1, int(total * warmup_frac))
    if step <= warm:
        return peak * step / warm
    t = (step - warm) / max(1, total - warm)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))


def score_local64_rows(model, device, rows: list[dict], max_new: int = 64) -> dict:
    from lct.train.instruct_v7 import SEP

    correct = 0
    by: dict[str, list[int]] = defaultdict(list)
    details = []
    import re

    def match(pred: str, gold: str) -> bool:
        p = pred.strip().splitlines()[0].strip().strip("\"'`").rstrip(".") if pred.strip() else ""
        g = gold.strip().strip("\"'`").rstrip(".")
        if not p:
            return False
        if p.casefold() == g.casefold():
            return True
        p2 = re.sub(r"^(a|answer)\s*[:\-]\s*", "", p, flags=re.I).strip()
        if p2.casefold() == g.casefold():
            return True
        if g.isdigit() and re.search(rf"(?<!\d){re.escape(g)}(?!\d)", p):
            return True
        if g.casefold() in {"yes", "no"} and p.casefold().startswith(g.casefold()):
            return True
        if len(g) == 1 and p[:1] == g:
            return True
        return False

    for row in rows:
        prefix = eval_prefix(row["question"]).encode("utf-8")
        generated = bytearray()
        for _ in range(max_new):
            tokens = torch.tensor([[b for b in (prefix + bytes(generated))]], dtype=torch.long, device=device)
            with torch.no_grad():
                logits = model._next_byte_logits(tokens)[0].detach().float().cpu()
            byte = int(torch.argmax(logits).item())
            if byte in (10, SEP):
                break
            generated.append(byte)
        pred = generated.decode("utf-8", errors="ignore")
        hit = match(pred, row["answer"])
        correct += int(hit)
        by[row["task"]].append(int(hit))
        details.append({"task": row["task"], "gold": row["answer"], "pred": pred, "correct": hit})
    per_task = {t: {"n": len(v), "correct": sum(v), "accuracy": sum(v) / len(v)} for t, v in sorted(by.items())}
    return {
        "n": len(rows),
        "correct": correct,
        "accuracy": correct / max(len(rows), 1),
        "per_task": with_wilson(per_task),
        "overall_wilson": wilson_interval(correct, len(rows)),
        "details": details,
    }


def finetune_local64(ckpt: Path, train_rows: list[dict], epochs: int, batch: int, lr: float, work: Path) -> Path:
    from lct.train.instruct_v7 import SEP, load_local64_checkpoint, sequence_nll

    device = torch.device("cuda")
    model, step0, seen0 = load_local64_checkpoint(ckpt, device)
    packed = []
    for row in train_rows:
        raw = train_text(row["question"], str(row["answer"])).encode("utf-8") + bytes([SEP])
        if not (4 <= len(raw) <= 2048):
            continue
        plen = len(eval_prefix(row["question"]).encode("utf-8"))
        w = [0.0] * len(raw)
        for i in range(plen, len(raw)):
            w[i] = 1.0
        w[-1] = 3.0
        packed.append((raw, w))
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
            tokens = torch.full((len(chunk), width), 32, dtype=torch.long, device=device)
            weights = torch.zeros((len(chunk), width), dtype=torch.float, device=device)
            for i, (raw, w) in enumerate(chunk):
                tokens[i, : len(raw)] = torch.tensor(list(raw), dtype=torch.long)
                weights[i, : len(w)] = torch.tensor(w, dtype=torch.float)
            opt.param_groups[0]["lr"] = cosine_lr(step, total, lr)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model._byte_logits(tokens)
                loss = sequence_nll(logits, tokens, weights)
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step == 1 or step % 50 == 0 or step == total:
                print({"stage": "local64_ft", "lr": lr, "step": step, "total": total, "loss": float(loss.item())}, flush=True)
    out = work / f"local64_ft_lr{lr:.0e}.pt"
    torch.save({"model": model.state_dict(), "step": step0, "seen": seen0, "ft_steps": step, "lr": lr}, out)
    return out


def score_bpe_rows(model, bpe, device, rows: list[dict], max_new: int = 64) -> dict:
    sys.path.insert(0, str(ROOT / "scripts"))
    from matched_bpe_load import greedy_bpe_generate

    import importlib.util

    spec = importlib.util.spec_from_file_location("score_charbench", ROOT / "scripts" / "charbench" / "score_charbench.py")
    score = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(score)

    correct = 0
    by: dict[str, list[int]] = defaultdict(list)
    details = []
    for row in rows:
        prompt = eval_prefix(row["question"])
        pred = greedy_bpe_generate(model, bpe, prompt, device, max_new=max_new)
        hit = score.match(pred, row["answer"])
        correct += int(hit)
        by[row["task"]].append(int(hit))
        details.append({"task": row["task"], "gold": row["answer"], "pred": pred, "correct": hit})
    per_task = {t: {"n": len(v), "correct": sum(v), "accuracy": sum(v) / len(v)} for t, v in sorted(by.items())}
    return {
        "n": len(rows),
        "correct": correct,
        "accuracy": correct / max(len(rows), 1),
        "per_task": with_wilson(per_task),
        "overall_wilson": wilson_interval(correct, len(rows)),
        "details": details,
    }


def finetune_bpe(
    ckpt: Path,
    bpe_dir: Path,
    train_rows: list[dict],
    epochs: int,
    batch: int,
    lr: float,
    work: Path,
) -> Path:
    sys.path.insert(0, str(ROOT / "scripts"))
    from matched_bpe_load import load_matched_bpe

    device = torch.device("cuda")
    model, bpe, step0, seen0 = load_matched_bpe(ckpt, device, bpe_dir)
    assert bpe.tok is not None
    model.train()
    examples = []
    for row in train_rows:
        prefix = eval_prefix(row["question"])
        full = train_text(row["question"], str(row["answer"])) + "\n"
        pref_ids = list(bpe.tok.encode(prefix).ids)
        full_ids = list(bpe.tok.encode(full).ids)
        if len(full_ids) < 4 or len(full_ids) > 1024:
            continue
        labels = list(full_ids)
        for i in range(min(len(pref_ids), len(labels))):
            labels[i] = -100
        examples.append({"input_ids": full_ids, "labels": labels})
    steps_per = max(1, (len(examples) + batch - 1) // batch)
    total = steps_per * epochs
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    order = list(range(len(examples)))
    step = 0
    rng = random.Random(0)
    for epoch in range(epochs):
        rng.shuffle(order)
        for start in range(0, len(order), batch):
            step += 1
            batch_ex = [examples[i] for i in order[start : start + batch]]
            width = max(len(x["input_ids"]) for x in batch_ex)
            input_ids = torch.zeros((len(batch_ex), width), dtype=torch.long, device=device)
            labels_t = torch.full((len(batch_ex), width), -100, dtype=torch.long, device=device)
            for i, ex in enumerate(batch_ex):
                n = len(ex["input_ids"])
                input_ids[i, :n] = torch.tensor(ex["input_ids"], dtype=torch.long, device=device)
                labels_t[i, :n] = torch.tensor(ex["labels"], dtype=torch.long, device=device)
            opt.param_groups[0]["lr"] = cosine_lr(step, total, lr)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(input_ids[:, :-1])["logits"]
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels_t[:, 1:].reshape(-1),
                    ignore_index=-100,
                )

            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step == 1 or step % 50 == 0 or step == total:
                print({"stage": "bpe_ft", "lr": lr, "step": step, "total": total, "loss": float(loss.item())}, flush=True)
    out = work / f"bpe_ft_lr{lr:.0e}.pt"
    torch.save({"model": model.state_dict(), "step": step0, "seen": seen0, "ft_steps": step, "lr": lr}, out)
    return out


def finetune_and_score_hf(model_id: str, train_rows, val_rows, test_rows, epochs, batch, lr, work: Path) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import importlib.util

    spec = importlib.util.spec_from_file_location("score_charbench", ROOT / "scripts" / "charbench" / "score_charbench.py")
    score = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(score)

    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.to(device)
    model.train()
    examples = []
    for row in train_rows:
        prefix = eval_prefix(row["question"])
        full = train_text(row["question"], str(row["answer"])) + (tok.eos_token or "")
        pref_ids = tok(prefix, add_special_tokens=False)["input_ids"]
        full_ids = tok(full, add_special_tokens=False)["input_ids"]
        labels = list(full_ids)
        for i in range(min(len(pref_ids), len(labels))):
            labels[i] = -100
        examples.append({"input_ids": full_ids, "labels": labels})
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps_per = max(1, (len(examples) + batch - 1) // batch)
    total = steps_per * epochs
    step = 0
    rng = random.Random(0)
    order = list(range(len(examples)))
    for epoch in range(epochs):
        rng.shuffle(order)
        for start in range(0, len(order), batch):
            step += 1
            batch_ex = [examples[i] for i in order[start : start + batch]]
            width = max(len(x["input_ids"]) for x in batch_ex)
            input_ids = torch.full((len(batch_ex), width), tok.pad_token_id, dtype=torch.long, device=device)
            labels_t = torch.full((len(batch_ex), width), -100, dtype=torch.long, device=device)
            attn = torch.zeros((len(batch_ex), width), dtype=torch.long, device=device)
            for i, ex in enumerate(batch_ex):
                n = len(ex["input_ids"])
                input_ids[i, :n] = torch.tensor(ex["input_ids"], dtype=torch.long, device=device)
                labels_t[i, :n] = torch.tensor(ex["labels"], dtype=torch.long, device=device)
                attn[i, :n] = 1
            opt.param_groups[0]["lr"] = cosine_lr(step, total, lr)
            opt.zero_grad(set_to_none=True)
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels_t)
            out.loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step == 1 or step % 50 == 0 or step == total:
                print({"stage": "hf_ft", "model": model_id, "lr": lr, "step": step, "total": total, "loss": float(out.loss.item())}, flush=True)
    out_dir = work / ("hf_ft_" + model_id.replace("/", "__") + f"_lr{lr:.0e}")
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)

    def score_rows(rows):
        details = []
        correct = 0
        by: dict[str, list[int]] = defaultdict(list)
        model.eval()
        for row in rows:
            prompt = eval_prefix(row["question"])
            inputs = tok(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False, pad_token_id=tok.eos_token_id)
            pred = tok.decode(out_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            hit = score.match(pred, row["answer"])
            correct += int(hit)
            by[row["task"]].append(int(hit))
            details.append({"task": row["task"], "gold": row["answer"], "pred": pred, "correct": hit})
        per_task = {t: {"n": len(v), "correct": sum(v), "accuracy": sum(v) / len(v)} for t, v in sorted(by.items())}
        return {
            "n": len(rows),
            "correct": correct,
            "accuracy": correct / max(len(rows), 1),
            "per_task": with_wilson(per_task),
            "overall_wilson": wilson_interval(correct, len(rows)),
        }

    return {"val": score_rows(val_rows), "test": score_rows(test_rows), "ft_ckpt": str(out_dir)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--local64", type=Path, default=None)
    parser.add_argument("--bpe", type=Path, default=None)
    parser.add_argument("--bpe-dir", type=Path, default=ROOT / "checkpoints" / "bpe_8k_real_v2")
    parser.add_argument("--hf", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lrs", type=str, default="1e-5,5e-5,2e-4")
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)

    train_path = args.data / "train_ft.jsonl"
    if not train_path.exists():
        train_path = args.data / "train.jsonl"
    train_rows = load_jsonl(train_path)
    val_rows = load_jsonl(args.data / "val.jsonl")
    test_rows = load_jsonl(args.data / "test.jsonl")
    lrs = [float(x) for x in args.lrs.split(",")]

    sweep = []
    if args.local64 is not None:
        name = "local64"
    elif args.bpe is not None:
        name = "bpe_8k"
    else:
        name = args.hf
    for lr in lrs:
        print({"model": name, "lr": lr, "stage": "start"}, flush=True)
        work = args.work / f"lr_{lr:.0e}"
        work.mkdir(parents=True, exist_ok=True)
        if args.local64 is not None:
            from lct.train.instruct_v7 import load_local64_checkpoint

            ft = finetune_local64(args.local64, train_rows, args.epochs, args.batch, lr, work)
            device = torch.device("cuda")
            model, _, _ = load_local64_checkpoint(ft, device)
            val = score_local64_rows(model, device, val_rows)
            test = score_local64_rows(model, device, test_rows)
            row = {"model": name, "lr": lr, "val": {k: val[k] for k in val if k != "details"}, "test": {k: test[k] for k in test if k != "details"}, "ft_ckpt": str(ft)}
        elif args.bpe is not None:
            sys.path.insert(0, str(ROOT / "scripts"))
            from matched_bpe_load import load_matched_bpe

            ft = finetune_bpe(args.bpe, args.bpe_dir, train_rows, args.epochs, args.batch, lr, work)
            device = torch.device("cuda")
            model, bpe, _, _ = load_matched_bpe(ft, device, args.bpe_dir)
            val = score_bpe_rows(model, bpe, device, val_rows)
            test = score_bpe_rows(model, bpe, device, test_rows)
            row = {
                "model": name,
                "lr": lr,
                "val": {k: val[k] for k in val if k != "details"},
                "test": {k: test[k] for k in test if k != "details"},
                "ft_ckpt": str(ft),
            }
        else:
            scored = finetune_and_score_hf(args.hf, train_rows, val_rows, test_rows, args.epochs, args.batch, lr, work)
            row = {"model": name, "lr": lr, "val": scored["val"], "test": scored["test"], "ft_ckpt": scored["ft_ckpt"]}
        sweep.append(row)
        (work / "result.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        print({"model": name, "lr": lr, "val_acc": row["val"]["accuracy"], "test_acc": row["test"]["accuracy"]}, flush=True)

    best = max(sweep, key=lambda r: r["val"]["accuracy"])
    report = {
        "model": name,
        "lrs": lrs,
        "chosen_lr": best["lr"],
        "selection": "max val accuracy",
        "chosen": best,
        "sweep": sweep,
        "train_file": str(train_path),
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "n_test": len(test_rows),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in report if k != "sweep"}, indent=2), flush=True)
    print({"wrote": str(args.out), "chosen_lr": best["lr"], "val": best["val"]["accuracy"], "test": best["test"]["accuracy"]}, flush=True)


if __name__ == "__main__":
    main()
