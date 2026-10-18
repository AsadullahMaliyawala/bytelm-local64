#!/usr/bin/env python
"""Level 4 eval harness for a patch-4, byte, or BPE checkpoint.

Teacher-forced log-likelihood on multiple choice, LAMBADA exact match,
per-domain bits/byte, and the shared story protocol. The patch-4 leak check
runs before any benchmark number is written.

The first 500 rows of each benchmark, in dataset order, are the fixed slice.
This script does not record the Level 4 sanity gate; that waits until the
scaling ladder's PASS/FAIL is in FINDINGS.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from lct.models.continuous_pool4 import ContinuousPool4LM
from lct.models.decoder_lm import DecoderOnlyLM
from lct.utils import save_json

N_EXAMPLES = 500
BENCHMARKS = ("hellaswag", "piqa", "arc_easy", "blimp", "lambada")


def tiny_patch(device: torch.device) -> ContinuousPool4LM:
    model = ContinuousPool4LM(
        {
            "d_model": 32,
            "n_heads": 4,
            "n_layers": 1,
            "ffn_mult": 4,
            "dropout": 0.0,
            "pool_factor": 4,
            "vocab_size": 256,
            "max_len": 128,
            "pos_type": "rope",
        }
    ).to(device)
    model.eval()
    return model


def leak_check(model: ContinuousPool4LM, t: int = 40) -> dict:
    """Changing bytes after t must not change log-probs at or before t."""
    torch.manual_seed(0)
    device = next(model.parameters()).device
    x = torch.randint(0, 256, (4, 64), device=device)
    y = x.clone()
    y[:, t + 1 :] = (y[:, t + 1 :] + 11) % 256
    with torch.no_grad():
        a = model.byte_logprobs(x)
        b = model.byte_logprobs(y)
    delta = (a[:, : t + 1] - b[:, : t + 1]).abs()
    return {"pass": bool(torch.allclose(a[:, : t + 1], b[:, : t + 1], atol=1e-4)), "max_abs": float(delta.max())}


def _align_patch(raw: bytes, pad_to_patch: bool) -> bytes:
    """Optional trailing spaces. Default scoring does not add them."""
    if not pad_to_patch:
        return raw
    pad = (4 - (len(raw) % 4)) % 4
    return raw + b" " * pad


def continuation_logprob_patch(model, context: bytes, continuation: bytes, pad_to_patch: bool = False) -> tuple[float, int]:
    raw = _align_patch(context + continuation, pad_to_patch)
    device = next(model.parameters()).device
    x = torch.tensor([list(raw)], dtype=torch.long, device=device)
    lp = model.byte_logprobs(x)[0]
    start = len(context)
    end = start + len(continuation)
    if start >= end or start >= lp.numel():
        return 0.0, len(continuation)
    end = min(end, int(lp.numel()))
    return float(lp[start:end].sum().item()), len(continuation)


def patch_exact_match(model, context: bytes, continuation: bytes, pad_to_patch: bool = False) -> bool:
    """Teacher-forced argmax over the continuation bytes equals the gold bytes."""
    if not continuation:
        return False
    raw = _align_patch(context + continuation, pad_to_patch)
    device = next(model.parameters()).device
    x = torch.tensor([list(raw)], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model._byte_logits(x)
        pred = logits.argmax(dim=-1)[0]
    gold_end = len(context) + len(continuation)
    return bool(torch.equal(pred[len(context) : gold_end], x[0, len(context) : gold_end]))


def char_byte_starts(text: str) -> list[int]:
    starts = [0]
    for ch in text:
        starts.append(starts[-1] + len(ch.encode("utf-8")))
    return starts


def continuation_token_mask(text: str, context: str, offsets: list[tuple[int, int]]) -> list[bool]:
    """True when a token's byte start is at or after the context."""
    starts = char_byte_starts(text)
    ctx_bytes = len(context.encode("utf-8"))
    mask = []
    for a, b in offsets:
        if b <= a:
            mask.append(False)
            continue
        byte_start = starts[a] if a < len(starts) else starts[-1]
        mask.append(byte_start >= ctx_bytes)
    return mask


def continuation_logprob_bpe(model, tok, context: str, continuation: str) -> dict:
    """Sum log-probs of jointly tokenized continuation tokens, attributed by byte offset."""
    full = context + continuation
    enc = tok.encode(full)
    mask = continuation_token_mask(full, context, list(enc.offsets))
    bos_id = tok.token_to_id("<bos>") if hasattr(tok, "token_to_id") else None
    ids = list(enc.ids)
    prefix = [bos_id] if bos_id is not None else []
    seq = prefix + ids
    device = next(model.parameters()).device
    if len(seq) < 2:
        return {"logprob": 0.0, "n_bytes": len(continuation.encode("utf-8")), "scored_ids": [], "joint_ids": ids}
    x = torch.tensor([seq], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(x[:, :-1])["logits"].float()
        logp = torch.log_softmax(logits, dim=-1)
        tok_lp = logp.gather(-1, x[:, 1:].unsqueeze(-1)).squeeze(-1)[0]
        pred = logits.argmax(dim=-1)[0]
    total = 0.0
    scored = []
    exact = True
    scored_any = False
    for i, keep in enumerate(mask):
        if not keep:
            continue
        pos = i + (1 if prefix else 0)
        if pos == 0 or pos - 1 >= tok_lp.numel():
            continue
        total += float(tok_lp[pos - 1].item())
        scored.append(ids[i])
        scored_any = True
        if int(pred[pos - 1].item()) != ids[i]:
            exact = False
    return {
        "logprob": total,
        "n_bytes": len(continuation.encode("utf-8")),
        "scored_ids": scored,
        "joint_ids": ids,
        "exact_match": bool(exact and scored_any),
    }


def wilson_ci(k: int, n: int, z: float = 1.96) -> list[float | None]:
    if n <= 0:
        return [None, None]
    p = k / n
    z2 = z * z
    den = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / den
    margin = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / den
    return [max(0.0, center - margin), min(1.0, center + margin)]


def sentence_logprob_patch(model, text: str, pad_to_patch: bool = False) -> tuple[float, int]:
    """Total log-probability of the sentence bytes, conditioned on a leading 0x1E.

    Label comparisons use this total, not an empty-context continuation.
    """
    body = text.encode("utf-8")
    raw = _align_patch(bytes([0x1E]) + body, pad_to_patch)
    device = next(model.parameters()).device
    x = torch.tensor([list(raw)], dtype=torch.long, device=device)
    lp = model.byte_logprobs(x)[0]
    end = min(1 + len(body), int(lp.numel()))
    if end <= 1:
        return 0.0, len(body)
    return float(lp[1:end].sum().item()), len(body)


def sentence_logprob_bpe(model, tok, text: str) -> tuple[float, int]:
    """Sum the NLL of every token in the full sentence.

    A leading 0x1E token is context only, matching real_v2 document starts, so
    the first sentence token is scored too.
    """
    sep = list(tok.encode("\x1e").ids)
    ids = list(tok.encode(text).ids)
    full = list(tok.encode("\x1e" + text).ids)
    if full[: len(sep)] == sep and full[len(sep) :] == ids:
        seq = full
    else:
        seq = sep + ids
    n_bytes = len(text.encode("utf-8"))
    if len(seq) < 2 or not ids:
        return 0.0, n_bytes
    device = next(model.parameters()).device
    x = torch.tensor([seq], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(x[:, :-1])["logits"].float()
        logp = torch.log_softmax(logits, dim=-1)
        tok_lp = logp.gather(-1, x[:, 1:].unsqueeze(-1)).squeeze(-1)[0]
    start = len(sep) - 1
    return float(tok_lp[start:].sum().item()), n_bytes


def lambada_bpe_pieces(tok, context: str, word: str) -> dict:
    """Target ids are the joint encoding minus the prefix encoding.

    Context must not end with a space. Target is ``" word"`` (leading space on
    the gold word stays in the target so ByteLevel ``Ġ`` attaches correctly).
    Encoding ``prefix + " "`` alone would peel that space into its own token.
    """
    bare = context.rstrip(" ")
    target = word if word.startswith(" ") else (" " + word)
    full_ids = list(tok.encode(bare + target).ids)
    ctx_ids = list(tok.encode(bare).ids)
    prefix_ok = full_ids[: len(ctx_ids)] == ctx_ids
    target_ids = full_ids[len(ctx_ids) :] if prefix_ok else []
    return {
        "bare_context": bare,
        "target_str": target,
        "context_ids": ctx_ids,
        "full_ids": full_ids,
        "target_ids": target_ids,
        "target_tokens": [tok.id_to_token(i) for i in target_ids],
        "decoded_target": tok.decode(target_ids) if target_ids else "",
        "prefix_ok": prefix_ok,
    }


def lambada_bpe_score(model, tok, context: str, word: str) -> dict:
    pieces = lambada_bpe_pieces(tok, context, word)
    target_ids = pieces["target_ids"]
    if not pieces["prefix_ok"] or not target_ids:
        return {**pieces, "logprob": 0.0, "exact_match": False}
    seq = pieces["context_ids"] + target_ids
    device = next(model.parameters()).device
    x = torch.tensor([seq], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(x[:, :-1])["logits"].float()
        logp = torch.log_softmax(logits, dim=-1)
        tok_lp = logp.gather(-1, x[:, 1:].unsqueeze(-1)).squeeze(-1)[0]
        pred = logits.argmax(dim=-1)[0]
    ctx_len = len(pieces["context_ids"])
    total = 0.0
    exact = True
    for k, tid in enumerate(target_ids):
        pos = ctx_len + k - 1
        if pos < 0 or pos >= tok_lp.numel():
            exact = False
            continue
        total += float(tok_lp[pos].item())
        if int(pred[pos].item()) != tid:
            exact = False
    return {**pieces, "logprob": total, "exact_match": exact}


def score_choices(logprob_of, context: str, choices: list[str]) -> tuple[int, int]:
    totals = []
    norms = []
    for choice in choices:
        total, n_bytes = logprob_of(context, choice)
        totals.append(total)
        norms.append(total / max(n_bytes, 1))
    return int(max(range(len(totals)), key=lambda i: totals[i])), int(max(range(len(norms)), key=lambda i: norms[i]))


def load_tokenizer(path: Path):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(path))


def load_model(path: Path, kind: str, device: torch.device, tokenizer=None):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    cfg = blob.get("config", {}) if isinstance(blob, dict) else {}
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    if kind == "patch":
        local_window = cfg.get("local_window") if isinstance(cfg, dict) else None
        local_d_model = cfg.get("local_d_model") if isinstance(cfg, dict) else None
        local_n_heads = cfg.get("local_n_heads") if isinstance(cfg, dict) else None
        if "d_model" not in cfg and isinstance(cfg.get("size"), dict):
            spec = cfg["size"]
            cfg = {
                "d_model": spec["d_model"],
                "n_heads": spec["n_heads"],
                "n_layers": spec["layers"],
                "ffn_mult": 4,
                "dropout": 0.0,
                "pool_factor": 4,
                "vocab_size": 256,
                "max_len": 2048,
                "pos_type": "rope",
            }
            if local_window:
                cfg["local_window"] = local_window
            if local_d_model:
                cfg["local_d_model"] = local_d_model
                cfg["local_n_heads"] = local_n_heads
        merged = cfg if "d_model" in cfg else cfg.get("model", cfg)
        if isinstance(cfg, dict) and cfg.get("local_window") and "local_window" not in merged:
            merged = {**merged, "local_window": cfg["local_window"]}
        model = ContinuousPool4LM(merged)
    else:
        spec = cfg.get("size", {})
        vocab = 256 if kind == "byte" else (tokenizer.get_vocab_size() if tokenizer is not None else cfg.get("vocab_size"))
        if vocab is None:
            raise SystemExit("BPE eval needs --tokenizer")
        model = DecoderOnlyLM(
            vocab_size=int(vocab),
            d_model=int(spec.get("d_model", cfg.get("d_model", 384))),
            n_heads=int(spec.get("n_heads", cfg.get("n_heads", 6))),
            n_layers=int(spec.get("layers", cfg.get("n_layers", 6))),
            ffn_mult=4,
            dropout=0.0,
            max_len=2048,
            pos_type="rope",
            ignore_index=0 if kind == "bpe" else -100,
        )
    from lct.checkpoint import load_state_dict_strict

    load_state_dict_strict(model, state, label=f"eval_suite:{kind}:{path}")
    model.to(device).eval()
    return model


def make_logprob_fn(model, kind: str, tok=None, pad_to_patch: bool = False):
    if kind == "patch":
        def logprob_of(context: str, choice: str) -> tuple[float, int]:
            return continuation_logprob_patch(
                model, context.encode("utf-8"), choice.encode("utf-8"), pad_to_patch=pad_to_patch
            )

        return logprob_of
    if tok is None:
        raise SystemExit("BPE scoring needs a tokenizer")

    def logprob_of(context: str, choice: str) -> tuple[float, int]:
        detail = continuation_logprob_bpe(model, tok, context, choice)
        return detail["logprob"], detail["n_bytes"]

    return logprob_of


def load_mc(name: str, blimp_all: bool = False):
    from datasets import get_dataset_config_names, load_dataset

    if name == "hellaswag":
        ds = load_dataset("Rowan/hellaswag", split="validation", streaming=True, trust_remote_code=True)

        def row(ex):
            return ex["ctx"], list(ex["endings"]), int(ex["label"])

        return ds, row
    if name == "piqa":
        ds = load_dataset("ybisk/piqa", split="validation", streaming=True, trust_remote_code=True)

        def row(ex):
            return ex["goal"], [ex["sol1"], ex["sol2"]], int(ex["label"])

        return ds, row
    if name == "arc_easy":
        ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="validation", streaming=True, trust_remote_code=True)

        def row(ex):
            labels = ex["choices"]["label"]
            texts = ex["choices"]["text"]
            key = ex["answerKey"]
            label = labels.index(key) if key in labels else 0
            return ex["question"], list(texts), label

        return ds, row
    if name == "lambada":
        ds = load_dataset("EleutherAI/lambada_openai", split="test", streaming=True, trust_remote_code=True)

        def row(ex):
            # Context without trailing space; target is " word" (leading space included).
            parts = ex["text"].rsplit(" ", 1)
            if len(parts) < 2 or not parts[1]:
                return None
            return parts[0], " " + parts[1], 0

        return ds, row
    if name == "blimp":
        configs = (
            list(get_dataset_config_names("nyu-mll/blimp"))
            if blimp_all
            else ["anaphor_gender_agreement"]
        )

        def _iter():
            for cfg in configs:
                ds = load_dataset("nyu-mll/blimp", cfg, split="train", streaming=True, trust_remote_code=True)
                for ex in ds:
                    row_ex = dict(ex)
                    row_ex["_config"] = cfg
                    yield row_ex

        def row(ex):
            return ex.get("_config", ""), [ex["sentence_good"], ex["sentence_bad"]], 0

        return _iter(), row
    raise SystemExit(f"unknown benchmark {name}")


def run_benchmark(
    model, kind: str, name: str, tok=None, n_examples: int = N_EXAMPLES, blimp_all: bool = False, pad_to_patch: bool = False
) -> dict:
    ds, parse = load_mc(name, blimp_all=blimp_all and name == "blimp")
    logprob_of = make_logprob_fn(model, kind, tok, pad_to_patch=pad_to_patch)
    correct = 0
    correct_norm = 0
    correct_sum = 0
    exact = 0
    n = 0
    target_lp = 0.0
    target_bytes = 0
    chance_sum = 0.0
    per_config: dict[str, list[int]] = {}
    for ex in ds:
        parsed = parse(ex)
        if parsed is None:
            continue
        ctx, choices, label = parsed
        if name == "lambada":
            # load_mc returns (ctx, word_str, label) — word_str is a str, not a list.
            # Using choices[0] on a str would keep only the first character (a prior bug).
            raw_target = choices if isinstance(choices, str) else choices[0]
            target = raw_target if raw_target.startswith(" ") else (" " + raw_target)
            bare_ctx = ctx.rstrip(" ")
            target_bytes_utf8 = target.encode("utf-8")
            if kind == "patch":
                hit = patch_exact_match(
                    model, bare_ctx.encode("utf-8"), target_bytes_utf8, pad_to_patch=pad_to_patch
                )
                lp, _nbytes = continuation_logprob_patch(
                    model, bare_ctx.encode("utf-8"), target_bytes_utf8, pad_to_patch=pad_to_patch
                )
            else:
                detail = lambada_bpe_score(model, tok, bare_ctx, target)
                hit = bool(detail["exact_match"])
                lp = float(detail["logprob"])
            exact += int(hit)
            correct += int(hit)
            correct_norm += int(hit)
            chance_sum += 0.0
            target_lp += lp
            target_bytes += len(target_bytes_utf8)
        elif name == "blimp":
            # Full-sentence scores. The decision is the mean, not the sum:
            # on anaphor_gender_agreement the good sentence is longer in bytes
            # 88/200 times and has more BPE tokens 138/200 times, so a sum
            # prefers the bad sentence even at random init (~0.22).
            good, bad = choices
            if kind == "patch":
                good_lp, good_n = sentence_logprob_patch(model, good, pad_to_patch=pad_to_patch)
                bad_lp, bad_n = sentence_logprob_patch(model, bad, pad_to_patch=pad_to_patch)
                good_mean = good_lp / max(good_n, 1)
                bad_mean = bad_lp / max(bad_n, 1)
            elif kind == "bpe":
                good_lp, good_bytes = sentence_logprob_bpe(model, tok, good)
                bad_lp, bad_bytes = sentence_logprob_bpe(model, tok, bad)
                good_mean = good_lp / max(len(tok.encode(good).ids), 1)
                bad_mean = bad_lp / max(len(tok.encode(bad).ids), 1)
            else:
                good_lp, good_n = logprob_of("", good)
                bad_lp, bad_n = logprob_of("", bad)
                good_mean = good_lp / max(good_n, 1)
                bad_mean = bad_lp / max(bad_n, 1)
            pred = 0 if good_mean >= bad_mean else 1
            pred_sum = 0 if good_lp >= bad_lp else 1
            hit = int(pred == label)
            correct += hit
            correct_norm += hit
            correct_sum += int(pred_sum == label)
            chance_sum += 0.5
            bucket = per_config.setdefault(str(ctx), [0, 0])
            bucket[0] += hit
            bucket[1] += 1
        else:
            pred, pred_n = score_choices(logprob_of, ctx, choices)
            correct += int(pred == label)
            correct_norm += int(pred_n == label)
            chance_sum += 1.0 / max(len(choices), 1)
        n += 1
        if n % 250 == 0:
            print({"benchmark": name, "n": n, "accuracy": correct / n}, flush=True)
        if n_examples > 0 and n >= n_examples:
            break
    out = {
        "n": n,
        "n_correct": correct,
        "accuracy": correct / max(n, 1),
        "accuracy_ci95_wilson": wilson_ci(correct, n),
        "n_correct_byte_normalized": correct_norm,
        "accuracy_byte_normalized": correct_norm / max(n, 1),
        "accuracy_byte_normalized_ci95_wilson": wilson_ci(correct_norm, n),
        "chance": chance_sum / max(n, 1),
        "lambada_exact_match": (exact / max(n, 1)) if name == "lambada" else None,
        "lambada_target_logprob_total": target_lp if name == "lambada" else None,
        "lambada_target_bytes": target_bytes if name == "lambada" else None,
        "lambada_target_logprob_per_byte": (target_lp / target_bytes) if name == "lambada" and target_bytes else None,
        "accuracy_total_logprob": (correct_sum / max(n, 1)) if name == "blimp" else None,
    }
    if per_config:
        out["per_config"] = {
            cfg: {"n": counts[1], "accuracy": counts[0] / max(counts[1], 1), "accuracy_ci95_wilson": wilson_ci(counts[0], counts[1])}
            for cfg, counts in per_config.items()
        }
    return out


def per_domain_bpb(model, kind: str, data_dir: Path, device: torch.device, window: int = 2048, max_batches: int = 10**9, tok=None) -> dict:
    from torch.utils.data import DataLoader

    from lct.data.dataset import collate_batch
    from lct.data.real_corpus import PackedBinDataset
    from lct.eval.lm_metrics import evaluate_bpe_lm_bpb, evaluate_byte_lm_bpb
    from lct.train.continuous_pool4 import evaluate_continuous_pool4
    from lct.train.ladder import bpb_from_nats

    out = {}
    total_nats = 0.0
    total_bytes = 0
    names = ("text", "code", "math")
    files = {"text": "val_text.bin", "code": "val_code.bin", "math": "val_math.bin"}
    for domain in names:
        path = data_dir / files[domain]
        if not path.exists():
            path = data_dir / f"{domain}.bin"
        ds = PackedBinDataset(path, window_bytes=window)
        loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_batch)
        if kind == "patch":
            metrics = evaluate_continuous_pool4(model, loader, device, None, max_batches=max_batches)
            bpb = float(metrics["bpb_all_bytes"])
            n_bytes = int(metrics["n_bytes"])
        elif kind == "byte":
            metrics = evaluate_byte_lm_bpb(model, loader, device, None, max_batches=max_batches, mask_key="byte_mask_all")
            bpb = float(metrics["bpb"])
            n_bytes = int(metrics["n_bytes"])
        else:
            if tok is None:
                raise SystemExit("BPE domain BPB needs a tokenizer")
            batches = []
            seen = 0
            for batch in loader:
                row = batch["bytes"][0].tolist()
                text = bytes(int(b) for b in row).decode("utf-8", errors="replace")
                enc = tok.encode(text)
                ids = enc.ids or [0]
                weights = []
                for a, b in enc.offsets:
                    piece = text[a:b] if b > a else ""
                    weights.append(len(piece.encode("utf-8")))
                if len(weights) != len(ids):
                    weights = [1] * len(ids)
                batches.append(
                    {
                        "tokens": torch.tensor([ids], dtype=torch.long),
                        "token_byte_weights": torch.tensor([weights], dtype=torch.float),
                    }
                )
                seen += 1
                if seen >= max_batches:
                    break
            metrics = evaluate_bpe_lm_bpb(model, batches, device, max_batches=max_batches, ignore_index=0)
            bpb = float(metrics["bpb"])
            n_bytes = int(metrics["n_bytes"])
        out[domain] = {"bpb": bpb, "n_bytes": n_bytes}
        total_nats += bpb * n_bytes * torch.log(torch.tensor(2.0)).item()
        total_bytes += n_bytes
        if hasattr(ds, "data"):
            del ds.data
        del loader, ds
    out["bpb_all_bytes"] = bpb_from_nats(total_nats, total_bytes) if total_bytes else None
    import gc

    gc.collect()
    return out


def story_metrics(rows: list[dict], hash_path: Path, word_path: Path) -> dict:
    from lct.eval.story_protocol import local_scores

    import numpy as np

    hashes = np.load(hash_path)
    word_ids = json.loads(word_path.read_text(encoding="utf-8"))
    return local_scores(rows, hashes, word_ids)


def comparison_table(results: list[dict]) -> str:
    names = []
    for result in results:
        for name in result.get("benchmarks", {}):
            if name not in names:
                names.append(name)
    header = "| Run | " + " | ".join(names) + " | text BPB | code BPB | math BPB |"
    sep = "|---|" + "|".join(["---:"] * len(names)) + "|---:|---:|---:|"
    lines = [header, sep]
    for result in results:
        cells = []
        benches = result.get("benchmarks", {})
        for name in names:
            row = benches.get(name, {})
            acc = row.get("accuracy")
            if acc is None:
                cells.append("")
            elif name == "lambada" and row.get("lambada_target_logprob_per_byte") is not None:
                cells.append(f"{acc:.3f} (lp/byte {row['lambada_target_logprob_per_byte']:.4f})")
            elif name in ("hellaswag", "arc_easy") and row.get("accuracy_byte_normalized") is not None:
                cells.append(f"{acc:.3f} (len {row['accuracy_byte_normalized']:.3f})")
            else:
                cells.append(f"{acc:.3f}")
        domains = result.get("domains") or {}
        def cell(domain):
            val = domains.get(domain, {})
            bpb = val.get("bpb") if isinstance(val, dict) else None
            return "" if bpb is None else f"{bpb:.3f}"

        lines.append("| " + result.get("run_id", "") + " | " + " | ".join(cells) + f" | {cell('text')} | {cell('code')} | {cell('math')} |")
    return "\n".join(lines) + "\n"


def self_test(out_path: Path | None = None) -> dict:
    """CPU checks: leak, choice scoring, BPE byte offsets, domain BPB, story metrics."""
    device = torch.device("cpu")
    model = tiny_patch(device)
    leak = leak_check(model)
    if not leak["pass"]:
        raise SystemExit(f"patch-4 leak check failed: {leak}")
    pred, pred_n = score_choices(
        lambda ctx, choice: continuation_logprob_patch(model, ctx.encode("utf-8"), choice.encode("utf-8")),
        "Once ",
        ["upon", "zzzz"],
    )
    if pred not in (0, 1) or pred_n not in (0, 1):
        raise SystemExit("choice scoring returned an invalid index")
    exact = patch_exact_match(model, b"Once ", b"upon")
    if not isinstance(exact, bool):
        raise SystemExit("exact match did not return a bool")

    from tempfile import TemporaryDirectory

    import numpy as np

    from lct.eval.story_protocol import build_copy_index

    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        data = tmp_path / "bins"
        data.mkdir()
        payload = (b"the cat sat on the mat and then ran home today\x1e" * 4)[:128]
        for name in ("text", "code", "math"):
            (data / f"{name}.bin").write_bytes(payload)
        domains = per_domain_bpb(model, "patch", data, device, window=32, max_batches=1)
        for name in ("text", "code", "math"):
            if domains[name]["n_bytes"] <= 0 or domains[name]["bpb"] != domains[name]["bpb"]:
                raise SystemExit(f"domain BPB missing for {name}")
        train = tmp_path / "train.bin"
        train.write_bytes(b"once upon a time there was a small cat and a dog\x1e")
        hashes, words = build_copy_index(train, tmp_path / "h.npy", tmp_path / "w.json")
        rows = [
            {"method": "tiny", "prompt_i": 0, "sample_i": 0, "continuation": "the cat sat on the mat and then ran home"},
            {"method": "tiny", "prompt_i": 0, "sample_i": 1, "continuation": "zzzz qqqq notaword sat on the mat and then ran"},
        ]
        stories = story_metrics(rows, tmp_path / "h.npy", tmp_path / "w.json")
        if stories["tiny"]["mean_broken_words"] <= 0:
            raise SystemExit("story suite did not count a broken word")
        del hashes, words

    bpe_check = {"skipped": True}
    tok_path = ROOT / "checkpoints" / "bpe_8k_real_v2" / "bpe.json"
    if tok_path.exists():
        tok = load_tokenizer(tok_path)
        bpe_model = DecoderOnlyLM(
            vocab_size=tok.get_vocab_size(),
            d_model=32,
            n_heads=4,
            n_layers=1,
            dropout=0.0,
            max_len=128,
            pos_type="rope",
            ignore_index=0,
        ).to(device).eval()
        detail = continuation_logprob_bpe(bpe_model, tok, "Hello", " world")
        if detail["n_bytes"] != len(b" world"):
            raise SystemExit("BPE normalizer is not the continuation byte length")
        if not detail["scored_ids"]:
            raise SystemExit("BPE scoring dropped every continuation token")
        mask = continuation_token_mask("Hello world", "Hello", list(tok.encode("Hello world").offsets))
        if not any(mask) or all(mask):
            raise SystemExit(f"BPE byte-offset mask did not separate context and continuation: {mask}")
        if any(i not in detail["joint_ids"] for i in detail["scored_ids"]):
            raise SystemExit("scored BPE ids were not taken from the joint encoding")
        bpe_check = {"skipped": False, "logprob": detail["logprob"], "n_bytes": detail["n_bytes"], "n_scored_tokens": len(detail["scored_ids"])}

    result = {
        "run_id": "cpu_selftest",
        "device": "cpu",
        "leak_check": leak,
        "choice_index": pred,
        "choice_index_normalized": pred_n,
        "patch_exact_match_is_bool": True,
        "domains": domains,
        "stories": stories,
        "bpe_offsets": bpe_check,
        "benchmarks": {},
        "pass": True,
    }
    result["comparison_table"] = comparison_table([result])
    dest = out_path or (ROOT / "experiments" / "eval_cpu_selftest.json")
    save_json(result, dest)
    return result


def lambada_handcheck(n: int = 10, tok_path: Path | None = None) -> dict:
    from datasets import load_dataset

    tok = load_tokenizer(tok_path or (ROOT / "checkpoints" / "bpe_8k_real_v2" / "bpe.json"))
    ds = load_dataset("EleutherAI/lambada_openai", split="test", streaming=True, trust_remote_code=True)
    rows = []
    for ex in ds:
        parts = ex["text"].rsplit(" ", 1)
        if len(parts) < 2 or not parts[1]:
            continue
        context, word = parts[0] + " ", parts[1]
        pieces = lambada_bpe_pieces(tok, context, word)
        rows.append(
            {
                "text": ex["text"],
                "context": context,
                "word": word,
                "context_ids": pieces["context_ids"],
                "full_ids": pieces["full_ids"],
                "target_ids": pieces["target_ids"],
                "target_tokens": pieces["target_tokens"],
                "decoded_target": pieces["decoded_target"],
                "prefix_ok": pieces["prefix_ok"],
            }
        )
        if len(rows) >= n:
            break
    out = {"n": len(rows), "prefix_ok": sum(1 for r in rows if r["prefix_ok"]), "examples": rows}
    save_json(out, ROOT / "experiments" / "lambada_bpe_handcheck.json")
    return out


def _random_kind(kind: str, tok, device: torch.device, seed: int):
    torch.manual_seed(seed)
    if kind == "patch":
        return ContinuousPool4LM(
            {
                "d_model": 64,
                "n_heads": 4,
                "n_layers": 2,
                "ffn_mult": 4,
                "dropout": 0.0,
                "pool_factor": 4,
                "vocab_size": 256,
                "max_len": 512,
                "pos_type": "rope",
                "local_window": 64,
            }
        ).to(device).eval()
    return DecoderOnlyLM(
        vocab_size=tok.get_vocab_size(),
        d_model=64,
        n_heads=4,
        n_layers=2,
        ffn_mult=4,
        dropout=0.0,
        max_len=512,
        pos_type="rope",
        ignore_index=0,
    ).to(device).eval()


def random_init_blimp(n: int = 200, device: str = "cpu", seeds: tuple[int, ...] = (0, 1, 2, 3, 4)) -> dict:
    """Mean full-sentence score of random-init models, averaged over seeds, near 0.50.

    One seed swings by a lot: the pairs differ by a few bytes, so an untrained
    head picks a side. The sum (not the mean) stays near 0.22 because the good
    sentence is usually longer. The check is the average of five seeds.
    """
    dev = torch.device(device)
    tok = load_tokenizer(ROOT / "checkpoints" / "bpe_8k_real_v2" / "bpe.json")
    out: dict = {"n": n, "seeds": list(seeds), "score": "mean full-sentence logprob (bytes for patch, tokens for BPE)"}
    for kind in ("patch", "bpe"):
        accs = []
        for seed in seeds:
            model = _random_kind(kind, tok, dev, seed)
            stats = run_benchmark(model, kind, "blimp", tok if kind == "bpe" else None, n_examples=n, blimp_all=False)
            accs.append(float(stats["accuracy"]))
            print({"random_init": kind, "seed": seed, "accuracy": accs[-1]}, flush=True)
            del model
        mean = sum(accs) / len(accs)
        out[kind] = {"per_seed": accs, "mean": mean, "near_chance": abs(mean - 0.5) <= 0.10}
        if abs(mean - 0.5) > 0.10:
            save_json(out, ROOT / "experiments" / "blimp_random_init.json")
            raise SystemExit(
                f"random-init {kind} mean BLiMP accuracy {mean:.3f} is not about 0.50; not rerunning the full set"
            )
    save_json(out, ROOT / "experiments" / "blimp_random_init.json")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--lambada-handcheck", action="store_true")
    ap.add_argument("--random-init-blimp", action="store_true")
    ap.add_argument("--blimp-all", action="store_true")
    ap.add_argument("--checkpoint")
    ap.add_argument("--kind", choices=("patch", "byte", "bpe"))
    ap.add_argument("--run-id")
    ap.add_argument("--tokenizer", default=str(ROOT / "checkpoints" / "bpe_8k_real_v2" / "bpe.json"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--benchmarks", default=",".join(BENCHMARKS))
    ap.add_argument("--domains", action="store_true")
    ap.add_argument("--stories", default="")
    ap.add_argument("--limit", type=int, default=N_EXAMPLES)
    ap.add_argument("--compare", nargs="*", default=[])
    args = ap.parse_args()
    if args.compare:
        rows = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.compare]
        table = comparison_table(rows)
        dest = ROOT / "experiments" / "eval_comparison.md"
        dest.write_text(table, encoding="utf-8")
        print(table, flush=True)
        return
    if args.lambada_handcheck:
        out = lambada_handcheck()
        print({"n": out["n"], "prefix_ok": out["prefix_ok"]}, flush=True)
        return
    if args.random_init_blimp:
        random_init_blimp(n=200, device=args.device)
        return
    if args.self_test:
        result = self_test()
        print({"leak_check": result["leak_check"], "pass": result["pass"]}, flush=True)
        return
    if not args.checkpoint or not args.kind or not args.run_id:
        raise SystemExit("pass --self-test or --checkpoint, --kind, and --run-id")
    device = torch.device(args.device)
    tok = None
    if args.kind == "bpe":
        tok = load_tokenizer(Path(args.tokenizer))
    model = load_model(Path(args.checkpoint), args.kind, device, tok)
    leak = {"pass": True, "max_abs": None}
    if args.kind == "patch":
        leak = leak_check(model)
        if not leak["pass"]:
            save_json({"run_id": args.run_id, "leak_check": leak, "pass": False}, ROOT / "experiments" / f"eval_{args.run_id}.json")
            raise SystemExit("leak check failed; benchmarks not reported")
    benches = {}
    names = [n for n in args.benchmarks.split(",") if n]
    for name in names:
        print({"benchmark": name}, flush=True)
        benches[name] = run_benchmark(
            model, args.kind, name, tok, n_examples=args.limit, blimp_all=args.blimp_all
        )
    domains = None
    if args.domains:
        domains = per_domain_bpb(model, args.kind, ROOT / "data" / "real_v2", device, tok=tok)
    stories = None
    if args.stories:
        stories = story_metrics(
            [json.loads(line) for line in Path(args.stories).read_text(encoding="utf-8").splitlines() if line.strip()],
            ROOT / "data" / "tinystories" / "train_10gram_hashes.npy",
            ROOT / "data" / "tinystories" / "train_word_ids.json",
        )
    result = {
        "run_id": args.run_id,
        "checkpoint": args.checkpoint,
        "kind": args.kind,
        "n_examples": args.limit,
        "leak_check": leak,
        "benchmarks": benches,
        "domains": domains,
        "stories": stories,
        "pass": bool(leak["pass"]),
    }
    save_json(result, ROOT / "experiments" / f"eval_{args.run_id}.json")
    print(comparison_table([result]), flush=True)


if __name__ == "__main__":
    main()
