"""Build CharBench: locked character-manipulation train/test/few-shot splits.

Prior chat text for \"charbench as specified earlier\" was not recoverable from
transcripts; this locks a concrete suite suited to byte-level vs BPE claims.

Tasks (answers are short strings / numbers / yes|no):
  first_char, last_char, nth_char, count_char, reverse, length,
  swap_case, delete_char, insert_char, replace_char, sort_chars, is_palindrome

Sizes (seed 0): 5 few-shot + 200 train + 50 test per task.

Disjointness: few-shot+train words are drawn from a train lexicon / train random
stream; test words from a disjoint test lexicon / test random stream. Question
strings are also globally unique.
"""

from __future__ import annotations

import argparse
import json
import random
import string
from pathlib import Path

TASKS = (
    "first_char",
    "last_char",
    "nth_char",
    "count_char",
    "reverse",
    "length",
    "swap_case",
    "delete_char",
    "insert_char",
    "replace_char",
    "sort_chars",
    "is_palindrome",
)

WORDS = [
    "apple", "banana", "orange", "grape", "melon", "peach", "berry", "mango",
    "lemon", "cherry", "kiwi", "plum", "table", "chair", "house", "water",
    "earth", "flame", "storm", "cloud", "river", "ocean", "stone", "glass",
    "bread", "sugar", "honey", "spice", "music", "piano", "violin", "drum",
    "happy", "quiet", "brave", "swift", "green", "blue", "white", "black",
    "robot", "pixel", "byte", "token", "model", "layer", "query", "value",
    "alpha", "beta", "gamma", "delta", "sigma", "omega", "prime", "logic",
    "stack", "queue", "graph", "tree", "leaf", "root", "node", "edge",
    "train", "plane", "ship", "truck", "bike", "road", "path", "gate",
    "light", "dark", "warm", "cold", "soft", "hard", "open", "close",
    "north", "south", "east", "west", "moon", "star", "sun", "sky",
    "code", "script", "build", "fix", "test", "check", "score", "bench",
    "input", "output", "error", "debug", "trace", "cache", "index", "hash",
]


def _partition_words(seed: int) -> tuple[list[str], list[str]]:
    words = WORDS[:]
    rng = random.Random(seed ^ 0xC0FFEE)
    rng.shuffle(words)
    mid = len(words) // 2
    return words[:mid], words[mid:]


def _word(rng: random.Random, lexicon: list[str], *, reserved: set[str], tag: str) -> str:
    """Sample a word never in ``reserved``; random alnums get a split tag prefix."""
    for _ in range(10_000):
        if rng.random() < 0.35:
            n = rng.randint(4, 8)
            # Tag keeps train/test random streams in disjoint string spaces.
            base = tag + "".join(rng.choice(string.ascii_lowercase) for _ in range(n))
        else:
            base = rng.choice(lexicon)
        if rng.random() < 0.25:
            chars = list(base)
            for i in range(len(chars)):
                if rng.random() < 0.4:
                    chars[i] = chars[i].upper()
            base = "".join(chars)
        if base not in reserved:
            reserved.add(base)
            return base
    raise RuntimeError("could not sample disjoint word")


def make_item(task: str, rng: random.Random, lexicon: list[str], reserved: set[str], tag: str) -> dict:
    w = _word(rng, lexicon, reserved=reserved, tag=tag)
    if task == "first_char":
        q, a = f'What is the first character of "{w}"?', w[0]
    elif task == "last_char":
        q, a = f'What is the last character of "{w}"?', w[-1]
    elif task == "nth_char":
        n = rng.randint(1, len(w))
        q, a = f'What is character number {n} of "{w}"? (1-based)', w[n - 1]
    elif task == "count_char":
        ch = rng.choice(list({c.lower() for c in w}))
        count = sum(1 for c in w if c.lower() == ch)
        q, a = f'How many times does the letter "{ch}" appear in "{w}"? (case-insensitive)', str(count)
    elif task == "reverse":
        q, a = f'Reverse the characters of "{w}".', w[::-1]
    elif task == "length":
        q, a = f'How many characters are in "{w}"?', str(len(w))
    elif task == "swap_case":
        q, a = f'Swap the case of every letter in "{w}".', w.swapcase()
    elif task == "delete_char":
        idx = rng.randint(0, len(w) - 1)
        q, a = f'Delete character number {idx + 1} from "{w}" and write the result.', w[:idx] + w[idx + 1 :]
    elif task == "insert_char":
        idx = rng.randint(0, len(w))
        ch = rng.choice(string.ascii_lowercase)
        q, a = f'Insert "{ch}" at position {idx + 1} in "{w}" (1-based, before that index).', w[:idx] + ch + w[idx:]
    elif task == "replace_char":
        idx = rng.randint(0, len(w) - 1)
        ch = rng.choice(string.ascii_lowercase)
        q, a = f'Replace character number {idx + 1} in "{w}" with "{ch}".', w[:idx] + ch + w[idx + 1 :]
    elif task == "sort_chars":
        q, a = (
            f'Sort the characters of "{w}" in alphabetical order '
            f"(case-insensitive, keep original case of each char by stable lower-key sort).",
            "".join(sorted(w, key=lambda c: (c.lower(), c))),
        )
    elif task == "is_palindrome":
        if rng.random() < 0.5:
            half = _word(rng, lexicon, reserved=reserved, tag=tag)[: rng.randint(2, 4)]
            w = half + half[::-1]
            if rng.random() < 0.5 and len(half) > 1:
                w = half + half[-2::-1]
            reserved.add(w)
        ans = "yes" if w == w[::-1] else "no"
        q, a = f'Is "{w}" a palindrome? Answer yes or no.', ans
    else:
        raise ValueError(task)
    return {"task": task, "question": q, "answer": a, "word": w}


def build_split(rng: random.Random, n_shot: int, n_train: int, n_test: int, seed: int) -> dict:
    train_lex, test_lex = _partition_words(seed)
    shots: dict[str, list] = {}
    train: list[dict] = []
    test: list[dict] = []
    seen_q: set[str] = set()
    reserved_train: set[str] = set()
    reserved_test: set[str] = set()
    for task in TASKS:
        train_bucket: list[dict] = []
        while len(train_bucket) < n_shot + n_train:
            item = make_item(task, rng, train_lex, reserved_train, tag="tr")
            if item["question"] in seen_q:
                continue
            seen_q.add(item["question"])
            train_bucket.append(item)
        test_bucket: list[dict] = []
        while len(test_bucket) < n_test:
            item = make_item(task, rng, test_lex, reserved_test, tag="te")
            if item["question"] in seen_q or item["word"] in reserved_train:
                continue
            seen_q.add(item["question"])
            test_bucket.append(item)
        shots[task] = train_bucket[:n_shot]
        train.extend(train_bucket[n_shot:])
        test.extend(test_bucket)
    # Hard assert disjoint words
    tw = {r["word"] for r in train} | {r["word"] for rows in shots.values() for r in rows}
    ew = {r["word"] for r in test}
    assert not (tw & ew), f"word leakage {sorted(tw & ew)[:10]}"
    rng.shuffle(train)
    rng.shuffle(test)
    return {
        "tasks": list(TASKS),
        "few_shot": shots,
        "train": train,
        "test": test,
        "train_lexicon_n": len(train_lex),
        "test_lexicon_n": len(test_lex),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-shot", type=int, default=5)
    parser.add_argument("--n-train", type=int, default=200)
    parser.add_argument("--n-test", type=int, default=50)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    data = build_split(rng, args.n_shot, args.n_train, args.n_test, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "few_shot.json").write_text(json.dumps(data["few_shot"], indent=2), encoding="utf-8")
    with (args.out / "train.jsonl").open("w", encoding="utf-8") as handle:
        for row in data["train"]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (args.out / "test.jsonl").open("w", encoding="utf-8") as handle:
        for row in data["test"]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "seed": args.seed,
        "n_shot": args.n_shot,
        "n_train_per_task": args.n_train,
        "n_test_per_task": args.n_test,
        "tasks": data["tasks"],
        "train_rows": len(data["train"]),
        "test_rows": len(data["test"]),
        "disjoint_lexicons": True,
        "train_lexicon_n": data["train_lexicon_n"],
        "test_lexicon_n": data["test_lexicon_n"],
        "note": "Train/few vs test use disjoint word lexicons and tagged random strings.",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
