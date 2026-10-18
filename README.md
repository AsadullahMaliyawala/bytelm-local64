# Matched twins release (local64 vs BPE-8k @ 350M)

Byte-patch Transformer with 64-byte sliding local decoder (**local64**) vs matched
byte-level 8k BPE decoder (**BPE-8k**), trained on the same $10^{10}$ raw bytes
(real_v3). Primary likelihood protocol: **matched-span** document sliding.

Licence: Apache License 2.0 (see `LICENSE`).

## Environment

- Python >= 3.11
- PyTorch >= 2.1 with CUDA recommended for eval (CPU works, slow)
- `pip install -e .` from this directory (see `pyproject.toml` / `requirements.txt`)

Optional data (not shipped): real_v3 validation bins under a directory you pass as
`--val-dir` (`val_text.bin`, `val_code.bin`, `val_math.bin`). Build with
`scripts/build_real_v3.py`.

Weights live in the companion `release_weights/` package (checkpoints + tokenizer).

## Hash table (350M checkpoints)

| Twin | Full-file SHA-256 | Model-tensor SHA-256 |
|---|---|---|
| local64 | `5080ed9b27bf491465b335f8dff6ceea200ec52d1e7fe4bcbd08a9cb35d0bd0a` | `33522128c54a8b2660b8075faaf3a4753ef786d31e5890700919751b84d5a350` |
| BPE-8k | `98f4a77c181b40a88fe7a10ff7aa7617a6a7026bd196e05cd9e52cb4349973e6` | `53986bd88bc89d6b784c3e4a7e235d6580eae95e890ad4373449fcf289187a09` |

Suite locks: fact200 `545a7dcde7c65b03fb22790359eb20196732a64a55a39480d3e4de7d600a1a16`; CharBench
`becda801323a53aa4e558e26b50544356532f487e3717d693a450fe0209a02a8` (see `locks/`).

Frozen numbers: `research/FROZEN_RESULTS.md`.

## Reproduce Table 3 (matched-span BPB) from checkpoints

Assuming `release_code/` is the cwd and weights/tokenizer are in `../release_weights/`:

```bash
export LCT_ROOT="$PWD"
pip install -e .

python scripts/release_d6_matched_span.py \
  --ckpt-local64 ../release_weights/matched350m_local64.pt \
  --ckpt-bpe ../release_weights/matched350m_bpe.pt \
  --bpe-dir ../release_weights/tokenizer \
  --val-dir /path/to/real_v3 \
  --device cuda \
  --out d6_release_verify.json
```

Expected (4 decimals): overall **0.9634 / 0.9155**, text **0.9907 / 0.9448**,
code **0.6563 / 0.6003**, math **1.1023 / 1.0388** (local64 / BPE).

Verify checkpoint hashes:

```bash
python scripts/release_d7_hash_check.py \
  --weights-dir ../release_weights \
  --out d7_release_verify.json
```

## Training the twins (reference)

- local64: `scripts/train_local64_ddp.py` (DDP; 26/1024/16, pool-4, local window 64, target $10^{10}$ B)
- BPE-8k: `scripts/train_bpe_matched350m.py` (pack length 512; same byte budget)
- Tokenizer: train with `baselines/bpe_lm.py` / `scripts/train_tokenizer.py` on real_v2; released vocab is `release_weights/tokenizer/bpe.json` (8k)

## Eval scripts included

- Matched-span: `scripts/paper_v8_matched_span.py`, `scripts/release_d6_matched_span.py`
- Shared-cap / own-context: `scripts/paper_v4_slide_cap.py`, `scripts/paper_v5_capped_curve.py`
- Downstream: `scripts/eval_suite.py`, `scripts/eval_full_benchmarks.py`, `scripts/eval_matched_bpe_benchmarks.py`
- CharBench: `scripts/charbench/`
- fact200: `scripts/build_fact200.py`, `scripts/score_fact200.py`, `data/eval/fact200.*`
- Checks: `scripts/paper_v10_*.py`, `scripts/release_d7_hash_check.py`

## Upload

See `UPLOAD.md` (commands only; do not run until logged in).
