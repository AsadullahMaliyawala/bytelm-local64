# Architecture Spec — ContinuousPool4LM (local64) & DecoderOnlyLM (BPE-8k) @ 350M

**Scope:** Code-backed description of the matched 350M twins.  
**Configs:** `d_model=1024`, `n_layers=26`, `n_heads=16` for both (`scripts/train_local64_ddp.py` `build`; `scripts/train_bpe_matched350m.py` defaults / docstring).  
**Rule:** Every factual claim cites `path` + class/function. Shapes marked **UNKNOWN** if not fixed by code.

---

## 1. ContinuousPool4LM (local64) — modules in forward order

Entry class: `ContinuousPool4LM` in `src/lct/models/continuous_pool4.py`.  
350M local64 build: `scripts/train_local64_ddp.py` → `build()` sets `pool_factor=4`, `pos_type="rope"`, `local_window=64`, `ffn_mult=4`, `dropout=0.0`, `vocab_size=256`, `max_len=2048`.

### 1.1 Patch-4 encoder (byte embed inside encoder)

Class: `LocalPatchEncoder` (`continuous_pool4.py`).

| Step | What | Cite |
|------|------|------|
| Group bytes | `encode_patches`: trim to multiple of `pool_factor`, reshape `(B, T) → (B·Tc, p)` with `p=4` | `ContinuousPool4LM.encode_patches` |
| Byte embedding | `nn.Embedding(vocab_size=256, d_model)` | `LocalPatchEncoder.__init__` → `self.embed` |
| Within-patch PE | sinusoidal `PositionalEncoding(d_model, max_len=pool_factor)` | `LocalPatchEncoder.__init__` → `self.pos`; `lct.models.quantizer.PositionalEncoding` |
| MLP + mean | `LayerNorm` → `Linear(D,2D)→GELU→Linear(2D,D)` → `.mean(dim=1)` | `LocalPatchEncoder.forward` |
| Output | one continuous vector per patch `(B, Tc, D)` | `encode_patches` return |

No VQ: module docstring of `continuous_pool4.py` states “Continuous pool-4 patch LM control **(no VQ)**.”

### 1.2 Global Transformer

| Piece | Value @ 350M local64 | Cite |
|-------|----------------------|------|
| Input LN | `self.input_ln = LayerNorm(D)` before blocks | `ContinuousPool4LM.__init__` |
| Position | `pos_type=="rope"` → no absolute PE module; RoPE inside each block | `ContinuousPool4LM.__init__`, `patch_transformer` |
| Depth / width / heads | 26 / 1024 / 16 | `train_local64_ddp.build` args; paper twins |
| Block | `RoPEBlock(d_model, n_heads, ffn_mult=4, dropout)` | `ContinuousPool4LM.__init__`; `lct.models.rope_backbone.RoPEBlock` |
| Attention | Pre-LN → `Linear(D,3D)` QKV (bias=False) → RoPE → causal SDPA → `Linear(D,D)` out (bias=False) | `RoPEBlock.forward` |
| MLP | `ffn_mult=4` → hidden `4096`; `Linear→GELU→Dropout→Linear→Dropout` | `RoPEBlock.__init__` |
| Norm | Pre-LN (`ln1`, `ln2`) per block; final `self.ln` after stack | `RoPEBlock`; `ContinuousPool4LM.patch_transformer` |
| Causal | `F.scaled_dot_product_attention(..., is_causal=True)` | `RoPEBlock.forward` |

### 1.3 How patch vectors condition the local decoder

Path used when `local_window > 0`: `ContinuousPool4LM._byte_logits`.

1. Encode complete patches only: `past = (length // p) * p`, then `emb = encode_patches(x[:, :past])`, `h_seq = patch_transformer(emb)`.
2. Build `cond_p` of length `past//p` (or +1 if trailing partial patch): **zeros on patch index 0**; `cond_p[:, 1:, :] = h_seq[:, :-1, :]` when length is a multiple of `p` (shift so patch \(t\) sees global state of patches \(< t\) only).
3. Expand to bytes: `cond_bytes = cond_p.repeat_interleave(p, dim=1)[:, :length]`.
4. Local decode: `byte_decoder.decode(cond_bytes, x)`.

Docstring on `_byte_logits`: “Patch t is conditioned on patches before t.”  
Class docstring: “Target patch embedding is never fed into its own prediction (no future leakage).”

Inside `SlidingWindowByteDecoder.decode`: `fuse(cat([tok, cond], dim=-1))` with `Linear(2·width, width)` — each byte position gets the (shifted) patch conditioning vector for that byte’s patch (`SlidingWindowByteDecoder.forward` / `decode`).

### 1.4 Local64 decoder (2 layers, 64-byte window)

Class: `SlidingWindowByteDecoder` (`continuous_pool4.py`), constructed in `ContinuousPool4LM.__init__` when `local_window` is set:

- `n_layers=2` (hard-coded in `__init__`)
- `window=self.local_window` (=64 from train script)
- `ffn_mult=2` inside `SlidingWindowBlock`
- `n_heads`: same as global when `local_d_model == d_model` (default; 350M does not set `local_d_model`)
- `local_d_model` defaults to `d_model` → `self.down = Identity()`

| Piece | Behavior | Cite |
|-------|----------|------|
| Attention mask | Query \(i\) attends keys in \([i-\mathrm{window}+1, i]\) (causal + window) | `SlidingWindowAttention.forward`; `sliding_causal_mask` |
| Cross-patch | “Context crosses patch boundaries” | `SlidingWindowByteDecoder` class docstring |
| Stack depth | Two stacked windows can carry signal up to `2*window-1` positions back | same docstring |
| Blocks | `SlidingWindowBlock`: Pre-LN → window attn → residual → Pre-LN → MLP (`ffn_mult=2`) | `SlidingWindowBlock` |
| Local PE | sinusoidal `PositionalEncoding(width, max_len=max_len)` with `max_len=WINDOW+256` | `SlidingWindowByteDecoder.__init__`; train `max_len=2048` |

### 1.5 One-byte input shift

`SlidingWindowByteDecoder.decode`:

```text
prev = target_bytes[:, :-1]
tok  = cat([start, byte_embed(prev)], dim=1)   # length T
```

Position \(i\) sees byte \(i-1\) (or learned `start` at \(i=0\)), then predicts byte \(i\).  
Docstring: “The input at position i is byte i-1, so a layer cannot see the byte it predicts or any later byte.”

### 1.6 Output head

`self.head = nn.Linear(width, vocab_size)` after `LayerNorm` (`SlidingWindowByteDecoder.__init__` / `decode`).  
Vocab 256. Untied from embeddings (separate `byte_embed` and encoder `embed`).

Training loss wrapper: `ContinuousPool4LM.forward` → CE over `logits` vs bytes (`F.cross_entropy`); ladder uses `patch_loss` → `model(batch)["loss"]` (`lct.train.ladder.patch_loss`).

---

## 2. DecoderOnlyLM (BPE-8k) @ 350M

Class: `DecoderOnlyLM` in `src/lct/models/decoder_lm.py`.  
Built in `scripts/train_bpe_matched350m.py` `main()`:

| Hyperparam | Value | Cite |
|------------|-------|------|
| `vocab_size` | `bpe.tok.get_vocab_size()` (asserted 8k family; script loads `SimpleBPE(vocab_size=8000)`) | `train_bpe_matched350m.py` |
| `d_model` / `n_layers` / `n_heads` | 1024 / 26 / 16 (argparse defaults) | same |
| `ffn_mult` | 4 | same |
| `dropout` | 0.0 | same |
| `pos_type` | `"rope"` | same |
| `max_len` | `WINDOW` (=2048) | same |
| `ignore_index` | 0 (pad) | same |

Forward order (`DecoderOnlyLM.forward`):

1. `embed`: `(B, T_tok) → (B, T_tok, D)` — tied to `head.weight` (`self.head.weight = self.embed.weight`).
2. RoPE stack: same `RoPEBlock` recipe as local64 global (`ffn_mult=4`, causal SDPA).
3. Final `LayerNorm` → `head` → logits `(B, T_tok, V)`.
4. Optional CE vs `targets` with `ignore_index`.

Teacher-forced shift for training: `inp, tgt = tokens[:, :-1], tokens[:, 1:]` (`token_batch_loss` / `make_bpe_loss` in `ladder.py`).

---

## 3. Tensor shapes — 2048-byte sequence @ 350M

Conventions: `B` = batch, `D=1024`, `H=16`, `Dh=64`, `p=4`, `Tc=T/p`, `V_byte=256`, `V_bpe=8000`.

### 3.1 local64 (`ContinuousPool4LM`, `local_window=64`)

Verified with a smoke forward (`encode_patches` / `patch_transformer` / `_byte_logits`) for `T=2048`:

| Tensor | Shape | Cite |
|--------|-------|------|
| `bytes_x` | `(B, 2048)` | input to `forward` / `_byte_logits` |
| Reshaped patches | `(B·512, 4)` | `encode_patches` |
| After `LocalPatchEncoder` (flat) | `(B·512, 1024)` | `LocalPatchEncoder.forward` |
| `patch_emb` / `h_seq` | `(B, 512, 1024)` | `encode_patches`, `patch_transformer` |
| `cond_p` (full multiple of 4) | `(B, 512, 1024)` | `_byte_logits` local_window branch |
| `cond_bytes` | `(B, 2048, 1024)` | `repeat_interleave(p)` then `[:, :length]` |
| Local `tok` (shifted) | `(B, 2048, 1024)` | `SlidingWindowByteDecoder.decode` |
| Fuse concat | `(B, 2048, 2048)` → `(B, 2048, 1024)` | `fuse` |
| Q/K/V per local layer | `(B, 16, 2048, 64)` | `SlidingWindowAttention.forward` |
| Logits | `(B, 2048, 256)` | `decode` → `head` |

Global RoPE Q/K/V: `(B, 16, 512, 64)` over patch length 512 (`RoPEBlock.forward`).

### 3.2 BPE (`DecoderOnlyLM`) for a 2048-**byte** window

| Tensor | Shape | Notes | Cite |
|--------|-------|-------|------|
| Raw bytes window | `(B, 2048)` | cursor `WINDOW=2048` | `train_bpe_matched350m.WINDOW` |
| Token ids for that window | `(B, T_tok)` with **T_tok data-dependent** | **UNKNOWN** fixed length; val mean bytes/token ≈ 3.415 ⇒ rough \(T_\mathrm{tok}\approx 600\), not hard-coded | `research/matched_350m_comparison.md`; `paper_v2_meta_flops.json` `bytes_per_token_for_conversion` |
| 350M **train pack** (when `--pack-len 512`) | `(B, 512)` tokens | `PretokPackedReader.next_batch(..., seq_len=512)` default; paper table “512 tok” | `train_bpe_matched350m.py`; `paper/main.tex` tab:configs |
| Embed / hidden / logits (train pack) | `(B, 511, 1024)` inputs if targets shifted from 512; logits `(B, 511, 8000)` | shift in `token_batch_loss` | `DecoderOnlyLM.forward`, `token_batch_loss` |
| FLOPs meta assuming `seq=2048` tokens | attention length 2048 | **counting assumption**, not the 350M train pack | `scripts/paper_v2_count_meta.py` `bpe_forward_flops_per_token(..., seq=WINDOW)` |

---

## 4. Initialization (scaled residual)

### 4.1 local64 — `apply_scaled_residual_init`

Function: `apply_scaled_residual_init` in `src/lct/models/continuous_pool4.py`.  
Enabled by `--scaled-init` in `scripts/train_local64_ddp.py`.

Formula (docstring + body):

\[
\mathrm{std} = \frac{0.02}{\sqrt{2 \cdot n_{\mathrm{layers\ of\ that\ stack}}}}
\]

| Stack | Layers | std @ 350M | What is filled |
|-------|--------|------------|----------------|
| Global `model.blocks` (`RoPEBlock`) | 26 | \(0.02/\sqrt{52} \approx 0.0027735\) | `blk.out.weight`, `blk.ff[3].weight` (+ zero `ff[3].bias` if present) |
| Local `byte_decoder.blocks` (`SlidingWindowBlock`) | 2 | \(0.02/\sqrt{4} = 0.01\) | `attn.out_proj.weight` (+ bias 0), `ff[3].weight` (+ bias 0) |

Verified numerically via `apply_scaled_residual_init` report: `global_std≈0.0027735`, `local_std=0.01`.

**Not** set by this function: embeddings, QKV, fuse, heads, patch encoder MLP, LayerNorms (remain PyTorch defaults unless other code runs).

### 4.2 BPE — `apply_scaled_init`

Function: `apply_scaled_init` in `scripts/train_bpe_matched350m.py`.

| Item | Value | Cite |
|------|-------|------|
| Embed | `Normal(0, 0.02)` on `model.embed.weight` (tied head) | `apply_scaled_init` |
| Residual attn-out / MLP-out | `std = 0.02 / sqrt(2L)` with `L=n_layers` → **≈0.0027735** for L=26 | same; matches names `*.out.weight`, `*.ff.3.weight` |

---

## 5. Loss and BPB

### 5.1 Constants

`LN2 = math.log(2)` in `src/lct/eval/lm_metrics.py`.

### 5.2 local64 training loss

- `ContinuousPool4LM.forward`: mean CE over all scored byte positions (`logits` vs `x`), optional `ignore_index=-100` via mask.
- Ladder: `patch_loss(model, batch) = model(batch)["loss"]` (`ladder.py`).

### 5.3 local64 BPB (eval)

`evaluate_continuous_pool4` (`src/lct/train/continuous_pool4.py`):

1. `nats, n_bytes = model.byte_nll(x, mask)` — sum of \(-\log p(x_i)\) under teacher forcing (`ContinuousPool4LM.byte_nll`).
2. \(\mathrm{BPB} = (\mathrm{nats} / n_{\mathrm{bytes}}) / \ln 2\).

Also `bpb_from_nats(nats, n_bytes) = (nats / n_bytes) / LN2` (`ladder.py`).

Frozen formula statement: `research/FROZEN_RESULTS.md` §2 — `BPB = Σ nats / (ln(2) · scored_bytes)`.

### 5.4 BPE training loss

- `make_bpe_loss` / `token_batch_loss`: CE on next-token targets; pad id ignored (`ignore_index=0`).
- Train script’s online `val_bpb()` divides **token** CE by `LN2` only — comment calls it “Approximate BPB” (`train_bpe_matched350m.val_bpb`). **Not** the frozen byte-normalized metric.

### 5.5 BPE byte-normalized BPB (canonical eval)

`evaluate_bpe_lm_bpb` (`lm_metrics.py`):

\[
\mathrm{BPB} = \frac{\sum \mathrm{token\ NLL\ nats}}{\ln 2 \cdot \sum \mathrm{scored\ UTF\text{-}8\ bytes}}
\]

Uses `token_byte_weights` when present (denominator = byte weights of scored target tokens; each token counted once in the numerator).  
Paper / audit: `research/matched_350m_comparison.md` — BPE was fixed from token CE/ln2 to nats/(ln2·bytes); `FROZEN_RESULTS.md` corrected denominator to target-token UTF-8 bytes.

---

## 6. Parameter breakdown @ 350M

Source of totals: `research/paper_v2_meta_flops.json` (`350m_local64`, `350m_bpe`), produced by `scripts/paper_v2_count_meta.py` via `lct.utils.count_parameters`.  
Module splits: recomputed from the same constructors (must sum to JSON totals).

### 6.1 local64 — ContinuousPool4LM

| Module | Parameters | Cite |
|--------|------------|------|
| **Total** | **351,286,528** | `paper_v2_meta_flops.json` `350m_local64.params_total` |
| **Non-embedding** | **350,762,240** | same `params_non_embedding`; `count_parameters(..., non_embedding=True)` skips names containing `embed` |
| `blocks` (26× `RoPEBlock`) | 327,395,328 | `param_report` / named_parameters; 12,592,128 × 26 |
| `byte_decoder` | 19,425,536 | top-level sum |
| └ `blocks` (2× sliding) | 16,799,744 | submodule |
| └ `fuse` | 2,098,176 | `Linear(2048,1024)` |
| └ `head` | 262,400 | `Linear(1024,256)` + bias |
| └ `byte_embed` | 262,144 | `Embedding(256,1024)` |
| └ `ln` + `start` | 2,048 + 1,024 | |
| `patch_encoder` | 4,461,568 | |
| └ `ff` | 4,197,376 | `D→2D→D` |
| └ `embed` | 262,144 | |
| └ `ln` | 2,048 | |
| `input_ln` + final `ln` | 2,048 + 2,048 | |

`param_report(self, backbone_attr="blocks")` (`rope_backbone.param_report`): `backbone_params=327395328`, `embed_params=0` (only checks attrs named `embed`/`tok_embed`/`bos` on the root — patch/local embeds live under children).

### 6.2 BPE — DecoderOnlyLM

| Module | Parameters | Cite |
|--------|------------|------|
| **Total** | **335,589,376** | `paper_v2_meta_flops.json` `350m_bpe.params_total` |
| **Non-embedding** | **327,397,376** | same |
| `blocks` (26× `RoPEBlock`) | 327,395,328 | identical backbone width/depth to local64 global |
| `embed` (tied head) | 8,192,000 | `8000×1024`; `head_params_untied=0` because weights tied |
| final `ln` | 2,048 | |

---

## 7. FLOPs derivation (worked example)

Counting rules (shared): one multiply and one add each count as one FLOP; exclude LN / softmax / GELU. Train ≈ **3×** forward.  
Cites: `local_decoder_flops_per_byte`, `train_local64_ddp.train_flops_per_byte`, `paper_v2_count_meta.py`, `research/paper_v2_meta_flops.json`.

### 7.1 local64 @ 350M (`d=1024`, `L=26`, byte window 2048)

**Global (per byte), patch sequence `seq = 2048/4 = 512`:**

\[
\mathrm{per\_layer} = 8d^{2} + 4d\cdot\mathrm{seq} + 16d^{2}
= 8\cdot1024^{2} + 4\cdot1024\cdot512 + 16\cdot1024^{2}
= 27{,}262{,}976
\]

\[
\mathrm{global\_fwd/byte} = L \cdot \mathrm{per\_layer} / 4
= 26 \cdot 27{,}262{,}976 / 4
= 177{,}209{,}344
\]

(`/4` amortizes patch-level compute across 4 bytes — `train_flops_per_byte`.)

**Local decoder (per byte)** — `local_decoder_flops_per_byte(d, window=64, n_layers=2, ffn_mult=2)`:

\[
\mathrm{per\_layer} = 8d^{2} + 4d\cdot 64 + 4\cdot 2\cdot d^{2} = 17{,}039{,}360
\]
\[
\mathrm{fuse}=4d^{2},\quad \mathrm{head}=2d\cdot 256
\]
\[
\mathrm{local\_fwd/byte} = 2\cdot 17{,}039{,}360 + 4{,}194{,}304 + 524{,}288 = 38{,}797{,}312
\]

**Train / infer:**

\[
\mathrm{train\ FLOPs/byte} = 3\cdot(177{,}209{,}344 + 38{,}797{,}312) = 648{,}019{,}968
\]
\[
\mathrm{infer\ FLOPs/byte} = \mathrm{train}/3 = 216{,}006{,}656
\]

Matches `paper_v2_meta_flops.json` → `350m_local64`.

### 7.2 BPE @ 350M

Per-layer (same formula), then × layers; **token** FLOPs then ÷ bytes/token.

**As stored in `paper_v2_meta_flops.json` (seq=2048 tokens):**

\[
\mathrm{per\_layer} = 8d^{2} + 4d\cdot 2048 + 16d^{2},\quad
\mathrm{fwd/token} = 26\cdot\mathrm{per\_layer} = 872{,}415{,}232
\]
\[
\mathrm{train/token} = 3\times = 2{,}617{,}245{,}696
\]
\[
\mathrm{train/byte} = \mathrm{train/token} / 3.415 \approx 766{,}396{,}982
\]

(`bytes_per_token_for_conversion: 3.415` from frozen val; method string in JSON.)

**Paper appendix note (`paper/main.tex` app:compute):** 350M BPE **trains** at `seq=512` tokens; recounting at 512 yields ≈ \(6.19\times 10^{8}\) train FLOPs/byte (ratio BPE/local64 ≈ 0.955). The JSON/meta script still documents the seq=2048 estimate — do not conflate the two without saying which.

---

## 8. Pseudocode (~30 lines) — forward + BPB

```text
# ----- local64 ContinuousPool4LM (pool=4, local_window=64) -----
function forward_local64(bytes_x):           # (B, T), T%4==0 in train
  patches = reshape(bytes_x, (B*T/4, 4))
  # LocalPatchEncoder: embed → sin PE → LN → FF → mean
  emb = PatchEncode(patches).view(B, T/4, D) # continuous, no VQ
  h = InputLN(emb)
  for blk in RoPEBlocks: h = blk(h)          # causal over T/4 patches
  h = FinalLN(h)
  cond_p[:,0]=0; cond_p[:,1:]=h[:,:-1]       # past patches only
  cond = repeat_each(cond_p, 4)              # (B,T,D)
  tok = cat(start, ByteEmbed(bytes_x[:,:-1]))# one-byte input shift
  z = Fuse(cat(tok, cond)) + SinPE
  for _ in 1..2: z = SlidingWindowBlock(z, window=64)  # crosses patches
  logits = Head(LN(z))                       # (B,T,256)
  loss = mean CE(logits, bytes_x)
  return loss

function bpb_local64(bytes_x, mask=None):
  nats, n = byte_nll(bytes_x, mask)          # sum -log p(x_i)
  return (nats / max(n,1)) / ln(2)

# ----- BPE-8k DecoderOnlyLM -----
function forward_bpe(tokens):                # (B, L) ids, pad=0
  inp, tgt = tokens[:,:-1], tokens[:,1:]
  h = Embed(inp)                             # tied to Head
  for blk in RoPEBlocks: h = blk(h)          # causal over tokens
  logits = Head(LN(h))                       # (B,L-1,V)
  loss = CE(logits, tgt, ignore_index=pad)
  return loss

function bpb_bpe(tokens, token_byte_weights):
  # evaluate_bpe_lm_bpb: sum token NLL once; divide by UTF-8 bytes of targets
  nats = sum(-log p(tgt_i) for non-pad i)
  bytes = sum(token_byte_weights on those targets)
  return (nats / max(bytes,1)) / ln(2)
```

---

## 9. Verified differences from MEGABYTE — **only what this code shows**

This repo does **not** implement the MEGABYTE paper model. Differences below are facts about **this** codebase relative to the in-repo “within-patch” decoder path and explicit “no VQ” continuous design.

| Fact in code | Cite |
|--------------|------|
| Patch latents are **continuous** (mean-pooled encoder outputs); **no** vector-quantizer in `ContinuousPool4LM` | `continuous_pool4.py` module docstring; `LocalPatchEncoder`; no codebook module on the class |
| local64 uses `SlidingWindowByteDecoder` with `window=64` whose docstring states attention **crosses patch boundaries** | `ContinuousPool4LM.__init__` (`local_window`); `SlidingWindowByteDecoder` |
| Alternate path `local_window==0` uses `LocalARByteDecoder`: causal attention over **`pool_factor` (=4) positions only**, one patch at a time | `LocalARByteDecoder.forward` + `causal_mask(p)`; `ContinuousPool4LM.__init__` else-branch |
| Global conditioning for byte \(t\) uses **strictly earlier** patch states (zero for first patch) | `_byte_logits` |
| Local stack is **2 layers**, MLP **`ffn_mult=2`**, while global uses **`ffn_mult=4`** | `SlidingWindowByteDecoder` / `SlidingWindowBlock`; `RoPEBlock` via `ffn_mult` from cfg |
| Global position encoding for local64 train is **RoPE**; local byte decoder uses **sinusoidal** `PositionalEncoding` | `build` `pos_type="rope"`; `SlidingWindowByteDecoder.pos` |

**Not evidenced by code (do not treat as verified here):** MEGABYTE’s original patch sizes, parameter counts, discrete vs continuous claim in the paper, training recipe, or any numerical MEGABYTE vs local64 comparison. Related-work prose in `paper/main.tex` is narrative, not executable.

---

## 10. Safe to state / Do not state

### Safe to state

- Both 350M twins use **26 layers, width 1024, 16 heads**, RoPE global/backbone blocks, `ffn_mult=4` on that backbone (`train_local64_ddp.build`, `train_bpe_matched350m`, `RoPEBlock`).
- local64: **patch-4 continuous encoder → causal patch Transformer → 2-layer sliding-window byte decoder (W=64) with one-byte shift**; CE/BPB over bytes (`ContinuousPool4LM`, eval helpers).
- BPE-8k: **decoder-only LM, tied embed/head, vocab 8000 family**, next-token CE; byte BPB = token NLL sum / (ln2 · target UTF-8 bytes) when using `evaluate_bpe_lm_bpb` + weights.
- Param totals **351.3M / 335.6M** (total) and **350.8M / 327.4M** (non-emb) as in `paper_v2_meta_flops.json`.
- local64 train FLOPs/byte **648,019,968** and infer **216,006,656** under the matmul rules above.
- Scaled residual init \(\mathrm{std}=0.02/\sqrt{2L}\) on attn-out and MLP-out; local64 also scales the **2-layer** local stack separately; BPE also sets embed std **0.02**.
- Twin comparison is **data-/optimizer-matched**, not FLOPs-matched (`paper/main.tex` §Question; JSON method notes).

### Do not state

- That local64 **is** MEGABYTE, or that every MEGABYTE paper detail is reproduced here (no MEGABYTE implementation in repo).
- That BPE train-script `val_bpb()` is the frozen byte-normalized BPB (it is token CE/ln2 approximate).
- That `paper_v2_meta_flops.json` BPE FLOPs/byte (**seq=2048**) equals the **512-token train-pack** recount in the paper appendix without labeling the assumption.
- Exact BPE **token** length for an arbitrary 2048-byte string (**UNKNOWN** / data-dependent); do not invent a fixed `T_tok`.
- That non-embedding counts exclude all embedding-like params under nested names the same way for both models without citing `count_parameters`’s `"embed" in name` rule (local64 drops `*.embed*` / `*byte_embed*`; BPE drops `embed`).
- End-to-end FLOPs including LN/softmax/GELU, or claiming exact hardware FLOPs from these estimates.
- That the local decoder uses RoPE (it does not; sinusoidal PE).
- VQ / discrete codes in the 350M local64 twin (explicitly continuous / no VQ).
- Sharing of byte embeddings between patch encoder and local decoder (separate modules).

---

## 11. Key file index

| Path | Role |
|------|------|
| `src/lct/models/continuous_pool4.py` | `ContinuousPool4LM`, encoders/decoders, init, local FLOPs |
| `src/lct/models/decoder_lm.py` | `DecoderOnlyLM` |
| `src/lct/models/rope_backbone.py` | `RoPEBlock`, `param_report` |
| `src/lct/models/quantizer.py` | `PositionalEncoding`, `TransformerBlock`, `causal_mask` (sinusoidal / MHA helpers) |
| `scripts/train_local64_ddp.py` | local64 350M build, FLOPs helper, scaled init flag |
| `scripts/train_bpe_matched350m.py` | BPE-8k 350M build, scaled init, pack-len training |
| `scripts/paper_v2_count_meta.py` | Regenerates `research/paper_v2_meta_flops.json` |
| `research/paper_v2_meta_flops.json` | Locked param + FLOPs table |
| `research/matched_350m_comparison.md` | Twin metrics / BPB audit notes |
| `research/FROZEN_RESULTS.md` | Frozen BPB formula and numbers |
| `src/lct/train/continuous_pool4.py` | `evaluate_continuous_pool4` BPB |
| `src/lct/eval/lm_metrics.py` | `LN2`, `evaluate_bpe_lm_bpb` |
| `src/lct/train/ladder.py` | `patch_loss`, `make_bpe_loss`, `bpb_from_nats` |
