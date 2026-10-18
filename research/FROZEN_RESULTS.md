# Frozen results — matched twins + scale

**Frozen at:** 2026-10-13T23:05:00Z  
**Corrected:** 2026-10-14. 350M BPE validation BPB now uses target-token UTF-8 bytes as the denominator. The previous full-window figures are listed under Superseded. Other rows are unchanged.  
**Re-frozen (hashes only):** 2026-10-18. 350M checkpoint identity hashes upgraded to full-file SHA-256 + model `state_dict` tensor SHA-256 (mtime-independent). Prior “first 32MB + size” values for the 350M twins were actually first-32MB+size+mtime from `audit_matched_eval_bugs.py`; on-disk bytes were unchanged (mtimes 2026-10-11 / 2026-10-13, before the freeze). See `research/d7_resolve.json`.  
**Rule:** Do not change these numbers unless a new bug is found. Paper table may cite this file.

No git commit available in this workspace copy. Eval-code fingerprint (SHA-256 of key scorer files at freeze):

| File | SHA-256 |
|---|---|
| `scripts/eval_suite.py` | `44b2d781dc852adbde7f934f03c2e04c8551c8a7e490eb6e4d9b5842a6e6b0c2` |
| `scripts/eval_matched_bpe_benchmarks.py` | `0026c4f5777c28732368c7f021a3f557065fcbd137430e6bd0e6a0f1a05d86de` |
| `scripts/eval_full_benchmarks.py` | `c8db4ef876a70ada95d44c2787892c0a2e57a2073912061186bcde8ee9c3d51b` |
| `scripts/charbench/score_charbench.py` | `07c8e2a75ccdfb23a51b40e72ae0ed4a9071b3b4150d7b0afd0fd73d3b050029` |
| `scripts/score_fact200.py` | `35945ac93597e18e7bd6303cb41fc540f0620008d3d2343a3a01a22444938fa6` |
| `src/lct/checkpoint.py` | `629e905d5faf52e2e81bfaa70eae4c7985d7439d69c0d9cd3378bde9dd61c90f` |

---

## 1. Scorer audit (BPE) — PASS

Artifact: `research/scorer_audit_bpe.json`  
Verdict: **no task used the LAMBADA bug path** (`choices[0]` on a string / trailing space forced onto context).

| Task | Path | Buggy? |
|---|---|---|
| HellaSwag | `score_choices` + `continuation_logprob_bpe` on **list** of full ending strings | no |
| PIQA | same | no |
| ARC-Easy | same | no |
| fact200 | `greedy_bpe_generate` + `alias_hit` | no |
| CharBench few-shot / equal-FT | `greedy_bpe_generate` + string match (FT reuses same scorer) | no |
| LAMBADA | fixed earlier (bare context; target=`" word"`; full-string target) | was buggy; fixed |

### Example dumps (2 each; abbreviated)

**HellaSwag** — ctx tokens `A Ġman Ġis Ġsitting Ġon Ġa Ġroof . Ġhe`; gold choice scored as full token sequence e.g. `is Ġusing Ġw rap …` with logprob over all choice tokens (not first char).

**PIQA / ARC-Easy** — `choices_are_list=true`; each choice is a full string; score = sum logprob of jointly tokenized continuation tokens.

**fact200** — generative; prompt ends `…\nA:`; pred tokens from greedy decode; score = alias match (not MC).

**CharBench** — generative few-shot; gold e.g. `"E"` → token `E`; pred e.g. `" r"` → `Ġr`; exact/normalized match.

No reruns required for these tasks (rows stay as previously reported).

---

## 2. Scale table — val BPB (bootstrap 95% CI)

**Formula (both sizes):** `BPB = Σ nats / (ln(2) · scored_bytes)`.

### 120M BPE formula check (real_v2, training val)

| | Value |
|---|---:|
| Recomputed BPB overall | **1.234010** [1.2235–1.2450] |
| Locked gate (`ladder120_shuf_gate.json`) | **1.234010** |
| Bits/token if wrong formula | **3.809** |
| **Verdict** | **Correct** (matches gate; ≢ bits/token) |

120M real_v2 (gate-locked, with new bootstrap CIs):

| Domain | local64 | BPE | Gap % `(BPE−loc)/BPE` |
|---|---|---|---|
| overall | 1.2167 [1.2059–1.2278] | 1.2340 [1.2235–1.2450] | +1.40% |
| text | 1.3021 [1.2930–1.3104] | 1.3135 [1.3045–1.3222] | +0.87% |
| code | 0.8107 [0.7916–0.8306] | 0.8560 [0.8365–0.8756] | +5.29% |
| math | 1.4348 [1.4075–1.4637] | 1.4353 [1.4077–1.4645] | +0.03% |

### Same-val scale (real_v3) — 120M shuffled vs 350M @ 10B

| Domain | 120M local64 | 120M BPE | 120M gap% | 350M local64 | 350M BPE | 350M gap% |
|---|---|---|---|---|---|---|
| overall | 1.2480 [1.2375–1.2575] | 1.2582 [1.2481–1.2673] | +0.81% | 1.0011 [0.9917–1.0090] | 0.9774 [0.9703–0.9846] | −2.42% |
| text | 1.2935 [1.2855–1.3017] | 1.3011 [1.2930–1.3088] | +0.58% | 1.0291 [1.0215–1.0362] | 0.9845 [0.9773–0.9915] | −4.53% |
| code | 0.7844 [0.7550–0.8123] | 0.8209 [0.7916–0.8484] | +4.45% | 0.6692 [0.6437–0.6919] | 0.8433 [0.8160–0.8709] | +20.64% |
| math | 1.3912 [1.3517–1.4290] | 1.3924 [1.3539–1.4306] | +0.08% | 1.1745 [1.1356–1.2119] | 1.1111 [1.0743–1.1488] | −5.71% |

Gap % = `(BPE − local64) / BPE × 100`. Positive ⇒ local64 lower BPB (better).  
Artifact: `research/scale_120m_350m_bpb.json`.

---

## 3. Frozen 350M @ 10B twin numbers

### Checkpoints

| Twin | Path | SHA-256 full file | SHA-256 model tensors | step | seen |
|---|---|---|---|---:|---:|
| local64 | `${LCT_DATA}/checkpoints/matched350m_local64/matched350m_local64.pt` | `5080ed9b27bf491465b335f8dff6ceea200ec52d1e7fe4bcbd08a9cb35d0bd0a` | `33522128c54a8b2660b8075faaf3a4753ef786d31e5890700919751b84d5a350` | 152588 | 10000007168 |
| BPE-8k | `${LCT_DATA}/checkpoints/matched350m_bpe/matched350m_bpe.pt` | `98f4a77c181b40a88fe7a10ff7aa7617a6a7026bd196e05cd9e52cb4349973e6` | `53986bd88bc89d6b784c3e4a7e235d6580eae95e890ad4373449fcf289187a09` | 145794 | 10000000000 |

Superseded identity hashes (do not use for D7): local64 `bcb7bd2b…` / BPE `b93b3217…` were first-32MB+size+**mtime** (audit recipe), not content-only.

BPE tokenizer: vocab **8000**, vocab sha256 `0548fe573b36f758…`, `bpe.json` sha256 `0f35d25e9af75283…`.

### Val BPB (real_v3, bootstrap over 2048-B windows)

> **Paper v10 note:** these 2048-B window rows remain frozen as appendix-only
> (“uncapped: BPE beyond trained length on code/math; not comparable”).
> The paper’s **primary** 350M protocol is **matched-span** document sliding
> (BPE `min(512 tok, 2048 B)`; local64 same start bytes).
> Matched-span overall 0.9634 / 0.9155 (Δ −0.0479); relative gaps −5.2/−4.9/−9.3/−6.1%.
> Artifact: `research/slide_350m_matched_span.json` /
> `research/curve_350m_matched_span.json`.
> Shared-cap / own-context remain appendix diagnostics
> (`research/slide_350m_capped_trainlen.json`, `research/slide_350m_bpe512_local2048.json`).
> The paper is 350M-only in the main text; 120M is appendix + `research/diagnosis_120m.md`.

| Domain | local64 | BPE |
|---|---|---|
| overall | 1.0011 [0.9917–1.0090] | 0.9774 [0.9703–0.9846] |
| text | 1.0291 [1.0215–1.0362] | 0.9845 [0.9773–0.9915] |
| code | 0.6692 [0.6437–0.6919] | 0.8433 [0.8160–0.8709] |
| math | 1.1745 [1.1356–1.2119] | 1.1111 [1.0743–1.1488] |

BPE bytes/token on val: **3.415**; mean token NLL **2.318** nats; implied BPB from train loss 2.40 ≈ **1.014**.

### Benchmarks (95% Wilson)

| Metric | local64 | BPE | Notes |
|---|---|---|---|
| HellaSwag | 0.2846 [0.2759–0.2935] | 0.2846 [0.2759–0.2935] | agreement 86.64%; distinct ckpts |
| PIQA | 0.5887 [0.5660–0.6110] | 0.6143 [0.5918–0.6362] | scorer audit PASS |
| ARC-Easy | 0.4456 [0.4053–0.4866] | 0.4895 [0.4486–0.5304] | scorer audit PASS |
| LAMBADA | 0.2360 [0.2246–0.2478] | 0.2401 [0.2286–0.2519] | **fixed** (was 0.293 / 0.093) |
| fact200 | 0.0700 [0.0422–0.1141] | 0.1650 [0.1200–0.2227] | scorer audit PASS |
| CharBench few-shot | 0.2083 [0.1777–0.2426] | 0.1783 [0.1498–0.2110] | scorer audit PASS |
| CharBench equal-FT | 0.5367 [0.4967–0.5762] | 0.4000 [0.3616–0.4397] | chosen LR 5e-5 both |

### Superseded (bug) — do not cite

| Metric | Old | Why |
|---|---|---|
| BPE val BPB overall | 3.1979 | token-mean CE/ln2 (bits/token), not BPB |
| BPE val BPB overall | 0.9792 [0.9712–0.9861] | full re-encoded window as the denominator; corrected value is 0.9774 [0.9703–0.9846] |
| BPE val BPB text | 0.9861 [0.9790–0.9930] | same denominator bug; corrected 0.9845 [0.9773–0.9915]; Δ vs local64 −0.0446 (gap −4.53%, was −4.36%) |
| BPE val BPB math | 1.1212 [1.0846–1.1567] | same denominator bug; corrected 1.1111 [1.0743–1.1488]; Δ vs local64 −0.0634 (gap −5.71%, was −4.76%) |
| BPE val BPB code | 0.8432 [0.8150–0.8706] | same denominator bug; corrected 0.8433 [0.8160–0.8709] |
| LAMBADA BPE | 0.0930 | `choices[0]` on str = first char only |
| LAMBADA local64 | 0.2926 | same first-char bug (+ space on context) |

### Setup notes (frozen)

- BPE ~114k B/step from ~0.84B→2.0B (`batch=64`), then ~66k B/step to 10B (`batch=37`).
- LR cosine keyed to bytes for both twins.
- Checkpoint load strips `_orig_mod.`/`module.`; local64 stripped 305, 0 missing/unexpected; BPE 0/0/0.

---

## 4. 120M shuffled checkpoints

| Twin | Path | SHA-256 head |
|---|---|---|
| local64 | `${LCT_DATA}/checkpoints/ladder120_shuf/ladder120_shuf_local64_s0.pt` | `ffd39dff10487c3725706f24276006278c4918470d96fe05af1ed9330a69e23c` |
| BPE | `${LCT_DATA}/checkpoints/ladder120_shuf/ladder120_shuf_bpe_s0.pt` | `b3058cd8e7a30ba9cb7442f689948a340383cc58e10ab42d3abfd1ef1edf3d04` |

---

## Source artifacts

- `research/matched_eval_audit_summary.json` — 350M BPB/HS/LAMBADA audit  
- `research/scorer_audit_bpe.json` — BPE scorer path audit  
- `research/scale_120m_350m_bpb.json` — scale + formula check  
- `research/matched_350m_comparison.md` — audited twin table  
- `experiments/ladder120_shuf_gate.json` — locked 120M shuffled gate  
