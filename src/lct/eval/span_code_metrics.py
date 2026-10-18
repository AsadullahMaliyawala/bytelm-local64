"""Concept-code stability metrics for fixed-stride VQ patches."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any


def _entropy(counts: Counter | dict) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / total
        h -= p * math.log(p + 1e-30)
    return h


def contingency_metrics(
    pairs: list[tuple[bytes, int]],
    min_span_count: int = 5,
) -> dict[str, Any]:
    """Compute H(C), H(C|S), I(S;C), purity stats from (span, code) pairs."""
    span_code: dict[bytes, Counter] = defaultdict(Counter)
    code_span: dict[int, Counter] = defaultdict(Counter)
    code_counts: Counter = Counter()
    span_counts: Counter = Counter()
    for span, code in pairs:
        span_code[span][code] += 1
        code_span[code][span] += 1
        code_counts[code] += 1
        span_counts[span] += 1

    n = max(len(pairs), 1)
    h_c = _entropy(code_counts)
    h_s = _entropy(span_counts)

    # Conditional H(C|S) over all spans (frequency-weighted)
    h_c_given_s = 0.0
    span_purities = []
    for span, cnts in span_code.items():
        w = span_counts[span] / n
        h_c_given_s += w * _entropy(cnts)
        top = max(cnts.values()) / max(sum(cnts.values()), 1)
        span_purities.append(top)

    # Frequency-filtered subset
    filtered_pairs = [(s, c) for s, c in pairs if span_counts[s] >= min_span_count]
    filt_span_code: dict[bytes, Counter] = defaultdict(Counter)
    for s, c in filtered_pairs:
        filt_span_code[s][c] += 1
    n_f = max(len(filtered_pairs), 1)
    h_c_given_s_filt = 0.0
    filt_span_counts = Counter(s for s, _ in filtered_pairs)
    for span, cnts in filt_span_code.items():
        w = filt_span_counts[span] / n_f
        h_c_given_s_filt += w * _entropy(cnts)

    mi = h_c - h_c_given_s
    code_purities = []
    for code, cnts in code_span.items():
        top = max(cnts.values()) / max(sum(cnts.values()), 1)
        code_purities.append(top)

    # Pairwise repeat agreement: for spans with ≥2 occurrences, P(same code)
    agreements = []
    for span, cnts in span_code.items():
        tot = sum(cnts.values())
        if tot < 2:
            continue
        # probability two random draws share a code
        agree = sum(c * (c - 1) for c in cnts.values()) / (tot * (tot - 1))
        agreements.append(agree)

    return {
        "n_pairs": len(pairs),
        "n_unique_spans": len(span_counts),
        "n_unique_codes": len(code_counts),
        "H_C": h_c,
        "H_S": h_s,
        "H_C_given_S": h_c_given_s,
        "I_span_code": mi,
        "H_C_given_S_filtered": h_c_given_s_filt,
        "min_span_count_filter": min_span_count,
        "n_filtered_pairs": len(filtered_pairs),
        "mean_span_to_code_purity": sum(span_purities) / max(len(span_purities), 1),
        "mean_code_to_span_purity": sum(code_purities) / max(len(code_purities), 1),
        "mean_pairwise_repeat_agreement": sum(agreements) / max(len(agreements), 1),
    }


def probe_span_stats(
    pairs: list[tuple[bytes, int]],
    probes: list[bytes],
) -> dict[str, Any]:
    out = {}
    for span in probes:
        codes = [c for s, c in pairs if s == span]
        cnt = Counter(codes)
        out[span.decode("utf-8", errors="replace")] = {
            "n": len(codes),
            "n_unique_codes": len(cnt),
            "top_code": cnt.most_common(1)[0] if cnt else None,
            "entropy": _entropy(cnt),
            "top5": cnt.most_common(5),
        }
    return out
