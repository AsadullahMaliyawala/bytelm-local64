"""Phase 2B.2 generation / validity smoke checks."""

from __future__ import annotations

import ast
import re
from typing import Any

import torch


def bytes_to_text(byte_ids: list[int] | torch.Tensor) -> tuple[str, bool]:
    """Decode byte ids to text; return (text, utf8_valid)."""
    if torch.is_tensor(byte_ids):
        raw = bytes(int(x) % 256 for x in byte_ids.view(-1).tolist())
    else:
        raw = bytes(int(x) % 256 for x in byte_ids)
    try:
        text = raw.decode("utf-8")
        return text, True
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="replace"), False


def max_run_length(ids: list[int]) -> int:
    if not ids:
        return 0
    best = cur = 1
    for i in range(1, len(ids)):
        if ids[i] == ids[i - 1]:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best


def length_before_invalid_utf8(byte_ids: list[int]) -> int:
    """Longest prefix that is valid UTF-8."""
    raw = bytes(int(x) % 256 for x in byte_ids)
    for n in range(len(raw), -1, -1):
        try:
            raw[:n].decode("utf-8")
            return n
        except UnicodeDecodeError:
            continue
    return 0


def python_parse_ok(text: str) -> bool:
    try:
        ast.parse(text)
        return True
    except SyntaxError:
        return False


@torch.no_grad()
def generate_byte_lm(
    model,
    prefix: torch.Tensor,
    max_new: int = 128,
    temperature: float = 1.0,
) -> torch.Tensor:
    return model.generate(prefix, max_new=max_new, temperature=temperature)


@torch.no_grad()
def generate_bpe_lm(
    model,
    prefix_tokens: torch.Tensor,
    max_new: int = 64,
    temperature: float = 1.0,
) -> torch.Tensor:
    return model.generate(prefix_tokens, max_new=max_new, temperature=temperature)


@torch.no_grad()
def generate_vq_lm(
    concept_lm,
    prefix_bytes: torch.Tensor,
    max_new_concepts: int = 32,
    temperature: float = 1.0,
) -> dict[str, Any]:
    """Sample concepts from LM then decode to bytes; also check fixed-span decode."""
    concepts = concept_lm.tokenizer.tokenize(prefix_bytes)
    gen = concept_lm.lm.generate(concepts, max_new=max_new_concepts, temperature=temperature)
    logits = concept_lm.decode_concepts_to_bytes(gen)
    byte_ids = logits.argmax(dim=-1)
    p = max(concept_lm.tokenizer.pool_factor, 1)
    # Sanity: each concept covers exactly p bytes after upsample
    expected_len = gen.size(1) * p
    span_ok = byte_ids.size(1) == expected_len or byte_ids.size(1) >= expected_len - p
    return {
        "bytes": byte_ids,
        "concepts": gen,
        "pool_factor": p,
        "span_len_ok": bool(span_ok),
        "n_concepts": int(gen.size(1)),
        "n_bytes": int(byte_ids.size(1)),
    }


def score_generations(
    samples: list[dict[str, Any]],
    domain: str = "text",
) -> dict[str, float]:
    """Aggregate validity metrics over generated samples.

    Each sample dict needs key ``bytes`` (1D/2D long tensor or list) and optional
    ``span_len_ok`` for VQ.
    """
    n = len(samples)
    if n == 0:
        return {}
    utf8_ok = 0
    parse_ok = 0
    collapse = 0
    len_valid = 0.0
    span_ok = 0
    span_n = 0
    for s in samples:
        b = s["bytes"]
        if torch.is_tensor(b):
            ids = b.view(-1).tolist()
        else:
            ids = list(b)
        text, valid = bytes_to_text(ids)
        utf8_ok += int(valid)
        len_valid += length_before_invalid_utf8(ids)
        if max_run_length(ids) >= max(32, len(ids) // 4):
            collapse += 1
        if domain == "code" and python_parse_ok(text):
            parse_ok += 1
        if "span_len_ok" in s:
            span_n += 1
            span_ok += int(bool(s["span_len_ok"]))
    out = {
        "n": n,
        "utf8_valid_rate": utf8_ok / n,
        "avg_len_before_invalid": len_valid / n,
        "repetition_collapse_rate": collapse / n,
    }
    if domain == "code":
        out["python_parse_rate"] = parse_ok / n
    if span_n:
        out["vq_span_ok_rate"] = span_ok / span_n
    return out


def exact_continuation_match(generated: str, target: str, min_len: int = 8) -> bool:
    """Whether generated text starts with / contains the held-out target continuation."""
    g = generated.strip()
    t = target.strip()
    if len(t) < min_len:
        return False
    return g.startswith(t) or t in g[: max(len(t) * 2, 64)]


_TEMPLATE_RE = re.compile(r"def\s+\w+\s*\(|class\s+\w+|The latent concept")
