# bytelm-local64

Tokenizer-free byte-level language model (**local64**: 4-byte patches + 64-byte
sliding local decoder) with an **8k byte-level BPE** matched baseline, trained
on the same $10^{10}$ raw bytes (real_v3). Primary likelihood protocol:
**matched-span** document sliding.

**Paper:** arXiv link coming

```bibtex
@article{bytelm-local64,
  title   = {Byte Patches versus {BPE} at Matched Data:
             A Matched-Twin Comparison at {350M} Parameters},
  author  = {Asadullah Maliyawala},
  year    = {2026},
  note    = {Preprint}
}
```

**Licence:** [Apache License 2.0](LICENSE) for **code** and released model
checkpoints. Training / validation **data** keep their upstream licences
(see [Dataset attributions](#dataset-attributions) and `NOTICE`).

## Weights

| Twin | Hugging Face | Checkpoint file |
|---|---|---|
| local64 350M | [bytelm-local64-350m](https://huggingface.co/Asadullah1569/bytelm-local64-350m) | `matched350m_local64.pt` |
| BPE-8k 350M | [bytelm-bpe8k-350m-baseline](https://huggingface.co/Asadullah1569/bytelm-bpe8k-350m-baseline) | `matched350m_bpe.pt` |

Download **both** `.pt` files into `./weights/`. The 8k BPE vocab
`tokenizer/bpe.json` is included in this repository. Weights are on Hugging
Face (links above).

Frozen checkpoint hashes (also in `research/SHA256SUMS` / `research/FROZEN_RESULTS.md`):

| Twin | Full-file SHA-256 | Model-tensor SHA-256 |
|---|---|---|
| local64 | `5080ed9b27bf491465b335f8dff6ceea200ec52d1e7fe4bcbd08a9cb35d0bd0a` | `33522128c54a8b2660b8075faaf3a4753ef786d31e5890700919751b84d5a350` |
| BPE-8k | `98f4a77c181b40a88fe7a10ff7aa7617a6a7026bd196e05cd9e52cb4349973e6` | `53986bd88bc89d6b784c3e4a7e235d6580eae95e890ad4373449fcf289187a09` |

Suite locks: fact200 `545a7dcde7c65b03fb22790359eb20196732a64a55a39480d3e4de7d600a1a16`;
CharBench `becda801323a53aa4e558e26b50544356532f487e3717d693a450fe0209a02a8`
(see `locks/`).

## Validation data

Frozen real_v3 validation bins are **not** in this git tree. Download them from
Hugging Face ([bytelm-local64-val](https://huggingface.co/datasets/Asadullah1569/bytelm-local64-val))
into `./data/real_v3/`:

| File | Bytes | SHA-256 |
|---|---:|---|
| `data/real_v3/val_text.bin` | 4251080 | `07a3e3f93dbd15405ffabeca4b4eee65fdf24b2a9a63047ce0b78e73d1b85614` |
| `data/real_v3/val_code.bin` | 503270 | `efe3f0901fa02edfd07f1761922c85e82d8a387e10fd1c529eb8bef6fc7bae3e` |
| `data/real_v3/val_math.bin` | 275881 | `add690396108cb0173f9d39086e0466bb2d065d0966f82e7dfb3f0ed9e3c4264` |

Verify checkpoints **and** val bins:

```bash
python scripts/release_d7_hash_check.py \
  --ckpt-dir ./weights \
  --val-dir ./data/real_v3 \
  --sums ./research/SHA256SUMS
```

## Environment

- Python >= 3.11
- PyTorch >= 2.1 with **CUDA** (recommended). CPU works but is much slower.
- ~**12–16 GB** GPU memory for the matched-span re-score (both twins loaded).
- Matched-span D6 on a single modern GPU is typically on the order of
  **~20–40 minutes**.

## Reproduce Table 3 (matched-span BPB)

One copy-paste block from a **clean clone** (after placing the two `.pt` files
in `./weights/` and the three val bins in `./data/real_v3/`):

```bash
git clone https://github.com/AsadullahMaliyawala/bytelm-local64.git
cd bytelm-local64
python -m pip install -e .

# download matched350m_local64.pt and matched350m_bpe.pt into ./weights/
# (links in the Weights table above)
# download val_text.bin, val_code.bin, val_math.bin into ./data/real_v3/
# (https://huggingface.co/datasets/Asadullah1569/bytelm-local64-val)

python scripts/release_d7_hash_check.py \
  --ckpt-dir ./weights \
  --val-dir ./data/real_v3 \
  --sums ./research/SHA256SUMS
python scripts/release_d6_matched_span.py \
  --ckpt-local64 ./weights/matched350m_local64.pt \
  --ckpt-bpe ./weights/matched350m_bpe.pt \
  --bpe-dir ./tokenizer \
  --val-dir ./data/real_v3 \
  --device cuda \
  --out d6_release_verify.json
```

Expected **4-decimal** BPB (local64 / BPE):

| Domain | local64 | BPE |
|---|---:|---:|
| overall | 0.9634 | 0.9155 |
| text | 0.9907 | 0.9448 |
| code | 0.6563 | 0.6003 |
| math | 1.1023 | 1.0388 |

## Rebuilding the validation data

Recipe (same as training data build; **does not** bit-reproduce the frozen val
bins without extra locked inputs):

```bash
python scripts/build_real_v3.py \
  --out ./data/real_v3 \
  --target-train-bytes 70000000000 \
  --target-val-bytes 5000000 \
  --fineweb-config sample-100BT \
  --min-score 3.0
```

Sources / versions:

| Share | Upstream | Config / notes |
|---:|---|---|
| 85% | FineWeb-Edu (`HuggingFaceFW/fineweb-edu`) | `sample-100BT`, min classifier score 3.0 |
| 10% | codeparrot-clean (`codeparrot/codeparrot-clean`) | train split |
| 5% | OpenWebMath (`open-web-math/open-web-math`) | — |

Document permutation for training: `doc_perm_seed0.npy` is written with
**numpy seed 0** (`perm_seed: 0` in `manifest.json`) after packing `train.bin`.

**What is needed for a bit-identical rebuild:** the script streams live Hugging
Face datasets. Without a pinned Hugging Face dataset revision and the frozen
build-state / intermediate counters from the original pack
(build_state.json, full train.bin, full manifest.json — not shipped),
a fresh run is **not guaranteed** to match the SHA-256 table above. For exact
Table 3 reproduction, download the frozen bins from
[bytelm-local64-val](https://huggingface.co/datasets/Asadullah1569/bytelm-local64-val)
into `./data/real_v3/` and verify their hashes. The full `train.bin` (~70 GB)
is **not** shipped.

## Training the twins (reference)

- local64: `scripts/train_local64_ddp.py` (DDP; 26/1024/16, pool-4, local window 64, target $10^{10}$ B)
- BPE-8k: `scripts/train_bpe_matched350m.py` (pack length 512; same byte budget)
- Tokenizer: `baselines/bpe_lm.py` / `scripts/train_tokenizer.py` on real_v2; released vocab is `./tokenizer/bpe.json` (8k)

## Eval scripts included

- Matched-span: `scripts/paper_v8_matched_span.py`, `scripts/release_d6_matched_span.py`
- Shared-cap / own-context: `scripts/paper_v4_slide_cap.py`, `scripts/paper_v5_capped_curve.py`
- Downstream: `scripts/eval_suite.py`, `scripts/eval_full_benchmarks.py`, `scripts/eval_matched_bpe_benchmarks.py`
- CharBench: `scripts/charbench/`
- fact200: `scripts/build_fact200.py`, `scripts/score_fact200.py`, `data/eval/fact200.jsonl`, `data/eval/fact200.lock.json`
- Checks: `scripts/paper_v10_claims_check.py`, `scripts/paper_v10_d6_matched_overall.py`, `scripts/paper_v10_hash_check.py`, `scripts/paper_v10_number_audit.py`, `scripts/release_d7_hash_check.py`

## Known limitations

- Under the primary **matched-span** protocol the 8k-BPE twin has lower BPB in
  all domains (overall relative gap **−5.2%**).
- Downstream: **local64** wins CharBench equal fine-tuning; **BPE** wins
  fact200 and PIQA / ARC-Easy length-normalized (Holm $\alpha=0.05$).
- **HellaSwag** is near chance and **uninformative** at this size.
- local64 was trained at **2048-byte** contexts and scored at about **900 bytes**
  on code under matched-span (in range, possibly disadvantaged). Neither
  matched-span nor own-context is perfectly neutral on code.
- The **120M** twin comparison is **excluded** from main conclusions because the
  old BPE twin was defective (see paper appendix / `research/`).
- Results are from a **single seed** at **350M**.

## Dataset attributions

Full training corpora are **not** redistributed. Frozen validation bins are
hosted separately on Hugging Face
([bytelm-local64-val](https://huggingface.co/datasets/Asadullah1569/bytelm-local64-val))
and keep their **upstream** licences (Apache-2.0 applies to code only):

| Source | Licence |
|---|---|
| FineWeb-Edu | ODC-By 1.0; also Common Crawl Terms of Use |
| codeparrot-clean | Mixed permissive per-file (Apache-2.0, MIT, BSD, Unlicense, ISC, CC0, …) |
| OpenWebMath | ODC-By 1.0; also Common Crawl Terms of Use |

The validation text is raw web data and may contain personal information
(including email addresses). Removal requests:
https://github.com/AsadullahMaliyawala/bytelm-local64/issues

See also `NOTICE` and `research/FROZEN_RESULTS.md`.
