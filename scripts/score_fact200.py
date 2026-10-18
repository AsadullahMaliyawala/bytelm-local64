"""Score data/eval/fact200.jsonl with newline few-shot and greedy decoding.

Five examples, no spider prompt, no 0x1E. Stop at the first newline.
A row is correct when the gold string or an alias appears as a whole token.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lct.models.continuous_pool4 import ContinuousPool4LM  # noqa: E402
from lct.train.instruct_v7 import prepare_logits  # noqa: E402

DATA = ROOT / "data" / "eval" / "fact200.jsonl"
LOCK = ROOT / "data" / "eval" / "fact200.lock.json"
EXPECT_BYTES = 1_414_599_680

EXAMPLES = (
    ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
    ("What color is the sky on a clear day?", "The sky is blue."),
    ("What is 3 + 4?", "3 + 4 = 7."),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci painted the Mona Lisa."),
    ("What do bees make?", "Bees make honey."),
)


def verify_lock() -> list[dict]:
    raw = DATA.read_bytes()
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    digest = hashlib.sha256(raw).hexdigest()
    if digest != lock["sha256"] or lock["n"] != 200:
        raise SystemExit(f"fact200 hash lock failed: {digest}")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if len(rows) != 200:
        raise SystemExit(f"fact200 has {len(rows)} rows")
    return rows


def prompt_for(question: str) -> str:
    blocks = [f"Q: {question}\nA: {answer}" for question, answer in EXAMPLES]
    blocks.append(f"Q: {question}\nA:")
    text = "\n\n".join(blocks)
    if "\x1e" in text:
        raise SystemExit("0x1E is not allowed in a fact200 prompt")
    return text


def alias_hit(answer: str, aliases: list[str]) -> bool:
    folded = answer.casefold()
    for alias in aliases:
        if alias.isalpha() and len(alias) <= 2:
            if re.search(rf"(?<![\w]){re.escape(alias)}(?![\w])", answer):
                return True
            continue
        needle = re.escape(alias.casefold())
        if re.search(rf"(?<![\w]){needle}(?![\w])", folded):
            return True
    return False


def greedy_answer(model, device, question: str) -> str:
    prefix = prompt_for(question).encode("utf-8")
    generated = bytearray()
    for _ in range(200):
        tokens = torch.tensor([list(prefix + bytes(generated))], dtype=torch.long, device=device)
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model._next_byte_logits(tokens)[0]
        logits = prepare_logits(logits.detach().float().cpu(), bytes(generated), False, code_like=False)
        byte = int(torch.argmax(logits).item())
        if byte == 10:
            break
        generated.append(byte)
    return generated.decode("utf-8", errors="ignore").strip()


def score_model(model, device) -> dict:
    was_training = model.training
    model.eval()
    rows = verify_lock()
    by_cat: dict[str, dict[str, int]] = {}
    details = []
    correct = 0
    for row in rows:
        answer = greedy_answer(model, device, row["question"])
        hit = alias_hit(answer, row["aliases"])
        correct += int(hit)
        bucket = by_cat.setdefault(row["category"], {"n": 0, "correct": 0})
        bucket["n"] += 1
        bucket["correct"] += int(hit)
        details.append({"id": row["id"], "category": row["category"], "correct": hit, "answer": answer})
        if bucket["n"] == 1 or (len(details) % 20 == 0):
            print({"scored": len(details), "correct": correct, "last": answer[:120]}, flush=True)
    categories = {}
    for name, bucket in by_cat.items():
        categories[name] = {
            "n": bucket["n"],
            "correct": bucket["correct"],
            "accuracy": bucket["correct"] / bucket["n"],
        }
    if was_training:
        model.train()
    return {
        "n": 200,
        "correct": correct,
        "accuracy": correct / 200,
        "categories": categories,
        "answers": details,
    }


def load_bin(bin_path: Path, meta_path: Path, device: torch.device) -> ContinuousPool4LM:
    if bin_path.stat().st_size != EXPECT_BYTES:
        raise SystemExit(f"{bin_path.name} is {bin_path.stat().st_size} bytes, expected {EXPECT_BYTES}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    state = {}
    with bin_path.open("rb") as handle:
        for row in meta["tensors"]:
            handle.seek(int(row["offset"]))
            raw = handle.read(int(row["numel"]) * 4)
            flat = torch.frombuffer(bytearray(raw), dtype=torch.float32).clone()
            state[row["name"]] = flat.reshape(row["size"])
    model = ContinuousPool4LM(
        {
            "d_model": 1024,
            "n_heads": 16,
            "n_layers": 26,
            "ffn_mult": 4,
            "dropout": 0.0,
            "pool_factor": 4,
            "vocab_size": 256,
            "max_len": 2048,
            "pos_type": "rope",
            "gradient_checkpointing": False,
            "local_window": 64,
        }
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(f"state mismatch missing={len(missing)} unexpected={len(unexpected)}")
    return model.to(device).eval()


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    jobs = [
        (148475, 9_730_457_600, "main_50b_s148475"),
        (193446, 12_677_677_056, "main_50b_s193446"),
    ]
    summary = []
    for step, seen, stem in jobs:
        print({"load": stem, "step": step, "seen_per_rank": seen}, flush=True)
        model = load_bin(ROOT / "checkpoints" / f"{stem}_model.bin", ROOT / "checkpoints" / f"{stem}_model.json", device)
        report = score_model(model, device)
        report["step"] = step
        report["seen_per_rank"] = seen
        out = ROOT / "experiments" / f"fact200_{stem}.json"
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        slim = {key: report[key] for key in ("step", "seen_per_rank", "n", "correct", "accuracy", "categories")}
        summary.append(slim)
        print(json.dumps(slim), flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    (ROOT / "experiments" / "fact200_compare.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
