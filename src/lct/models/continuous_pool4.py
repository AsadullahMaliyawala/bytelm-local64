"""Continuous pool-4 patch LM control (no VQ).

Local 4-byte patch encoder → continuous embeddings → causal patch transformer
→ local autoregressive 4-byte decoder. Target patch embedding is never fed into
its own prediction (no future leakage).
"""

from __future__ import annotations

import string
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# Space, newline, and ASCII punctuation. Word-start decoding uses a higher
# temperature on the byte that follows one of these.
WORD_BOUNDARY_BYTES = frozenset(string.punctuation.encode("ascii")) | {0x20, 0x0A}


def byte_sample_probs(logits: torch.Tensor, temperature: float, min_p: float = 0.0) -> torch.Tensor:
    """Softmax with optional min-p. Tokens below ``min_p * max(prob)`` are dropped."""
    probs = torch.softmax(logits.float() / max(float(temperature), 1e-5), dim=-1)
    if min_p and float(min_p) > 0:
        thresh = float(min_p) * probs.amax(dim=-1, keepdim=True)
        probs = torch.where(probs < thresh, torch.zeros_like(probs), probs)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return probs

from lct.models.quantizer import PositionalEncoding, TransformerBlock, causal_mask
from lct.models.rope_backbone import RoPEBlock, param_report


class LocalPatchEncoder(nn.Module):
    """Encode one 4-byte patch independently (no cross-patch attention)."""

    def __init__(self, d_model: int = 512, pool_factor: int = 4, vocab_size: int = 256):
        super().__init__()
        self.pool_factor = pool_factor
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len=pool_factor)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.ln = nn.LayerNorm(d_model)

    def forward(self, patch_bytes: torch.Tensor) -> torch.Tensor:
        """patch_bytes: (B, p) → (B, D)."""
        h = self.pos(self.embed(patch_bytes))
        h = self.ff(self.ln(h)).mean(dim=1)
        return h


class LocalARByteDecoder(nn.Module):
    """Predict p bytes autoregressively from a patch hidden state."""

    def __init__(
        self,
        d_model: int = 512,
        pool_factor: int = 4,
        vocab_size: int = 256,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pool_factor = pool_factor
        self.vocab_size = vocab_size
        self.start = nn.Parameter(torch.zeros(1, 1, d_model))
        self.byte_embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len=pool_factor + 1)
        self.fuse = nn.Linear(d_model * 2, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, ffn_mult=2, dropout=dropout) for _ in range(2)]
        )
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, patch_h: torch.Tensor, target_bytes: torch.Tensor) -> torch.Tensor:
        """
        patch_h: (B, D) conditioning from causal patch LM (past only)
        target_bytes: (B, p) teacher-forced targets
        returns logits (B, p, V)
        """
        bsz, p = target_bytes.shape
        prev = target_bytes[:, :-1]
        tok = torch.cat(
            [self.start.expand(bsz, 1, -1), self.byte_embed(prev)],
            dim=1,
        )
        cond = patch_h.unsqueeze(1).expand(-1, p, -1)
        h = self.fuse(torch.cat([tok, cond], dim=-1))
        h = self.pos(h)
        attn = causal_mask(p, h.device)
        for blk in self.blocks:
            h = blk(h, attn_mask=attn)
        return self.head(self.ln(h))


def sliding_causal_mask(length: int, window: int, device: torch.device) -> torch.Tensor:
    """True means masked. Position i may attend to the ``window`` keys ending at i."""
    positions = torch.arange(length, device=device)
    diff = positions[None, :] - positions[:, None]
    return (diff > 0) | (diff < -(int(window) - 1))


class SlidingWindowAttention(nn.Module):
    """Windowed causal attention with the same parameters as ``nn.MultiheadAttention``.

    Query i attends only to keys in ``[i - window + 1, i]``. The mask is applied
    inside ``scaled_dot_product_attention`` with no Python loop over positions.
    """

    def __init__(self, d_model: int, n_heads: int, window: int, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window = int(window)
        self.dropout = float(dropout)
        self.in_proj_weight = nn.Parameter(torch.empty(3 * d_model, d_model))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * d_model))
        self.out_proj = nn.Linear(d_model, d_model)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.zeros_(self.in_proj_bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, length, _ = x.shape
        heads = self.n_heads
        dim = self.head_dim
        width = self.window
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(bsz, length, heads, dim).transpose(1, 2)
        k = k.view(bsz, length, heads, dim).transpose(1, 2)
        v = v.view(bsz, length, heads, dim).transpose(1, 2)
        pos = torch.arange(length, device=x.device)
        diff = pos[None, :] - pos[:, None]
        # True means this key may be attended. Opposite of MultiheadAttention's mask.
        allow = ~((diff > 0) | (diff < -(width - 1)))
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=allow,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(bsz, length, self.d_model)
        return self.out_proj(out)


class SlidingWindowBlock(nn.Module):
    """Same layout as ``TransformerBlock``, with windowed attention."""

    def __init__(self, d_model: int, n_heads: int, window: int, ffn_mult: int = 2, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = SlidingWindowAttention(d_model, n_heads, window, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        hidden = d_model * ffn_mult
        self.ff = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        a = self.attn(h)
        x = x + self.dropout(a)
        x = x + self.ff(self.ln2(x))
        return x


class SlidingWindowByteDecoder(nn.Module):
    """2-layer causal byte transformer. Each layer attends over ``window`` positions.

    The input at position i is byte i-1, so a layer cannot see the byte it predicts
    or any later byte. Two stacked windows can carry a byte from up to
    ``2 * window - 1`` positions back. Context crosses patch boundaries. Every byte
    in a patch is conditioned on that patch's global state, which the caller must
    build from earlier patches only.
    """

    def __init__(
        self,
        d_model: int = 512,
        pool_factor: int = 4,
        vocab_size: int = 256,
        n_heads: int = 8,
        dropout: float = 0.0,
        window: int = 64,
        n_layers: int = 2,
        max_len: int = 4096,
        local_d_model: int | None = None,
    ):
        super().__init__()
        width = int(local_d_model or d_model)
        if width % n_heads != 0:
            raise ValueError(f"local width {width} is not divisible by {n_heads} heads")
        self.pool_factor = pool_factor
        self.vocab_size = vocab_size
        self.window = int(window)
        self.global_d_model = int(d_model)
        self.local_d_model = width
        self.down = nn.Identity() if width == d_model else nn.Linear(d_model, width)
        self.start = nn.Parameter(torch.zeros(1, 1, width))
        self.byte_embed = nn.Embedding(vocab_size, width)
        self.pos = PositionalEncoding(width, max_len=max_len)
        self.fuse = nn.Linear(width * 2, width)
        self.blocks = nn.ModuleList(
            [
                SlidingWindowBlock(width, n_heads, window=self.window, ffn_mult=2, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        self.ln = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab_size)

    def decode(self, cond_bytes: torch.Tensor, target_bytes: torch.Tensor) -> torch.Tensor:
        """cond_bytes (B, T, D), target_bytes (B, T) → logits (B, T, V)."""
        bsz, length = target_bytes.shape
        cond_bytes = self.down(cond_bytes)
        prev = target_bytes[:, :-1]
        embedded = self.byte_embed(prev)
        start = self.start.to(dtype=embedded.dtype).expand(bsz, 1, -1)
        tok = torch.cat([start, embedded], dim=1)
        cond = cond_bytes.to(dtype=tok.dtype)
        h = self.fuse(torch.cat([tok, cond], dim=-1))
        h = self.pos(h)
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.ln(h))

    def forward(self, patch_h: torch.Tensor, target_bytes: torch.Tensor) -> torch.Tensor:
        """patch_h (B, Tc, D) is repeated across the bytes of each patch."""
        cond = patch_h.repeat_interleave(self.pool_factor, dim=1)
        if cond.size(1) != target_bytes.size(1):
            raise RuntimeError(
                f"patch state covers {cond.size(1)} bytes, targets have {target_bytes.size(1)}"
            )
        return self.decode(cond, target_bytes)


def apply_scaled_residual_init(model: nn.Module) -> dict[str, float]:
    """GPT-2 residual scaling for attention-out and MLP-out only.

    Each stack uses std = 0.02 / sqrt(2 * n_layers of that stack).
    The global blocks and the local byte decoder are scaled separately.
    """
    import math

    report: dict[str, float] = {}
    blocks = getattr(model, "blocks", None)
    if blocks is not None and len(blocks) > 0 and hasattr(blocks[0], "out"):
        std = 0.02 / math.sqrt(2 * len(blocks))
        report["global_std"] = std
        report["global_layers"] = float(len(blocks))
        for blk in blocks:
            nn.init.normal_(blk.out.weight, mean=0.0, std=std)
            mlp_out = blk.ff[3]
            nn.init.normal_(mlp_out.weight, mean=0.0, std=std)
            if getattr(mlp_out, "bias", None) is not None:
                nn.init.zeros_(mlp_out.bias)
    decoder = getattr(model, "byte_decoder", None)
    local_blocks = getattr(decoder, "blocks", None) if decoder is not None else None
    if local_blocks is not None and len(local_blocks) > 0 and hasattr(local_blocks[0], "attn"):
        std = 0.02 / math.sqrt(2 * len(local_blocks))
        report["local_std"] = std
        report["local_layers"] = float(len(local_blocks))
        for blk in local_blocks:
            nn.init.normal_(blk.attn.out_proj.weight, mean=0.0, std=std)
            if blk.attn.out_proj.bias is not None:
                nn.init.zeros_(blk.attn.out_proj.bias)
            mlp_out = blk.ff[3]
            nn.init.normal_(mlp_out.weight, mean=0.0, std=std)
            if getattr(mlp_out, "bias", None) is not None:
                nn.init.zeros_(mlp_out.bias)
    return report


def local_decoder_flops_per_byte(
    d_model: int,
    window: float,
    n_layers: int = 2,
    ffn_mult: int = 2,
    vocab_size: int = 256,
) -> float:
    """Forward FLOPs for one byte of the local decoder.

    One multiply and one add each count as one FLOP. LayerNorm, softmax, and
    GELU are omitted. ``window`` is the number of keys each query attends to.
    """
    d = int(d_model)
    per_layer = (8 * d * d) + (4 * d * float(window)) + (4 * int(ffn_mult) * d * d)
    fuse = 4 * d * d
    head = 2 * d * int(vocab_size)
    return float(int(n_layers) * per_layer + fuse + head)


class ContinuousPool4LM(nn.Module):
    """Leakage-free continuous pool-4 language model."""

    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        m = cfg if "d_model" in cfg else cfg.get("model", cfg)
        self.pool_factor = int(m.get("pool_factor", 4))
        self.d_model = m["d_model"]
        dropout = m.get("dropout", 0.1)
        self.pos_type = str(m.get("pos_type", "sinusoidal")).lower()
        self.patch_encoder = LocalPatchEncoder(
            d_model=m["d_model"], pool_factor=self.pool_factor, vocab_size=m.get("vocab_size", 256)
        )
        self.input_ln = nn.LayerNorm(m["d_model"])
        if self.pos_type == "rope":
            self.pos = None
            self.blocks = nn.ModuleList(
                [
                    RoPEBlock(m["d_model"], m["n_heads"], m.get("ffn_mult", 4), dropout)
                    for _ in range(m["n_layers"])
                ]
            )
        else:
            self.pos = PositionalEncoding(m["d_model"], max_len=m.get("max_len", 2048))
            self.blocks = nn.ModuleList(
                [
                    TransformerBlock(m["d_model"], m["n_heads"], m.get("ffn_mult", 4), dropout)
                    for _ in range(m["n_layers"])
                ]
            )
        self.ln = nn.LayerNorm(m["d_model"])
        self.dropout = nn.Dropout(dropout)
        self.gradient_checkpointing = bool(m.get("gradient_checkpointing", False))
        self.local_window = int(m.get("local_window", 0) or 0)
        if self.local_window:
            local_d = int(m.get("local_d_model") or m["d_model"])
            local_heads = int(m["local_n_heads"]) if local_d != int(m["d_model"]) else int(m["n_heads"])
            self.byte_decoder = SlidingWindowByteDecoder(
                d_model=m["d_model"],
                pool_factor=self.pool_factor,
                vocab_size=m.get("vocab_size", 256),
                n_heads=local_heads,
                dropout=dropout,
                window=self.local_window,
                n_layers=2,
                max_len=int(m.get("max_len", 2048)) + 256,
                local_d_model=local_d,
            )
        else:
            self.byte_decoder = LocalARByteDecoder(
                d_model=m["d_model"],
                pool_factor=self.pool_factor,
                vocab_size=m.get("vocab_size", 256),
                n_heads=m["n_heads"],
                dropout=dropout,
            )

    def param_report(self) -> dict:
        return param_report(self, backbone_attr="blocks")

    def encode_patches(self, bytes_x: torch.Tensor) -> torch.Tensor:
        p = self.pool_factor
        bsz, t = bytes_x.shape
        usable = (t // p) * p
        x = bytes_x[:, :usable]
        tc = usable // p
        patches = x.reshape(bsz * tc, p)
        emb = self.patch_encoder(patches)  # (B*Tc, D)
        return emb.view(bsz, tc, -1)

    def patch_transformer(self, patch_emb: torch.Tensor) -> torch.Tensor:
        h = self.input_ln(patch_emb)
        if self.pos_type == "rope":
            h = self.dropout(h)
            for blk in self.blocks:
                if self.gradient_checkpointing and self.training:
                    h = torch.utils.checkpoint.checkpoint(blk, h, use_reentrant=False)
                else:
                    h = blk(h)
        else:
            h = self.dropout(self.pos(h))
            attn = causal_mask(h.size(1), h.device)
            for blk in self.blocks:
                if self.gradient_checkpointing and self.training:
                    h = torch.utils.checkpoint.checkpoint(blk, h, attn, use_reentrant=False)
                else:
                    h = blk(h, attn_mask=attn)
        return self.ln(h)

    def _byte_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Teacher-forced logits (B, T, V). Patch t is conditioned on patches before t.

        A local-window model scores a partial last patch from the local decoder.
        That tail is not padded, and it is not dropped.
        """
        p = self.pool_factor
        bsz, length = x.shape
        if self.local_window:
            past = (length // p) * p
            if past == 0:
                cond_p = torch.zeros(bsz, 1, self.d_model, device=x.device, dtype=self.byte_decoder.start.dtype)
            else:
                emb = self.encode_patches(x[:, :past])
                h_seq = self.patch_transformer(emb)
                extra = 1 if length > past else 0
                cond_p = torch.zeros(
                    bsz, past // p + extra, self.d_model, device=x.device, dtype=h_seq.dtype
                )
                cond_p[:, 1:, :] = h_seq if extra else h_seq[:, :-1, :]
            cond_bytes = cond_p.repeat_interleave(p, dim=1)[:, :length]
            return self.byte_decoder.decode(cond_bytes, x)
        emb = self.encode_patches(x)
        bsz, tc, _ = emb.shape
        h_seq = self.patch_transformer(emb)
        start = torch.zeros(bsz, 1, self.d_model, device=x.device, dtype=emb.dtype)
        cond = torch.cat([start, h_seq[:, :-1, :]], dim=1)
        flat = self.byte_decoder(
            cond.reshape(bsz * tc, self.d_model),
            x.reshape(bsz * tc, p),
        )
        return flat.reshape(bsz, tc * p, -1)

    def forward(self, bytes_x: torch.Tensor, byte_mask: torch.Tensor | None = None) -> dict:
        """Next-patch prediction using past patch embeddings only."""
        p = self.pool_factor
        length = int(bytes_x.size(1))
        if self.local_window and length % p != 0:
            if length == 0:
                z = bytes_x.new_zeros(())
                return {"loss": z, "n_scored": 0}
            x = bytes_x
            mask = byte_mask
            n_patches = length // p
        else:
            usable = (length // p) * p
            if usable < p:
                z = bytes_x.new_zeros(())
                return {"loss": z, "n_scored": 0}
            x = bytes_x[:, :usable]
            mask = byte_mask[:, :usable] if byte_mask is not None else None
            n_patches = usable // p
        logits = self._byte_logits(x)
        if mask is not None:
            tgt = x.masked_fill(mask == 0, -100)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tgt.reshape(-1),
                ignore_index=-100,
            )
            n_scored = int((mask != 0).sum().item())
        else:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), x.reshape(-1))
            n_scored = int(x.numel())
        return {"loss": loss, "n_scored": n_scored, "n_patches": n_patches}

    @torch.no_grad()
    def byte_nll(
        self, bytes_x: torch.Tensor, byte_mask: torch.Tensor | None = None
    ) -> tuple[float, int]:
        p = self.pool_factor
        usable = (bytes_x.size(1) // p) * p
        if usable < p:
            return 0.0, 0
        x = bytes_x[:, :usable]
        mask = byte_mask[:, :usable] if byte_mask is not None else None
        logits = self._byte_logits(x)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        nll = -log_probs.gather(-1, x.unsqueeze(-1)).squeeze(-1)
        if mask is not None:
            m = mask.float()
            return float((nll * m).sum().item()), int(m.sum().item())
        return float(nll.sum().item()), int(nll.numel())

    @torch.no_grad()
    def byte_logprobs(self, bytes_x: torch.Tensor) -> torch.Tensor:
        """Per-byte log-probability under teacher forcing. Shape (B, T).

        Local-window models return one score per input byte, including a partial
        last patch. Other decoders still require a multiple of the pool.
        """
        p = self.pool_factor
        if self.local_window:
            x = bytes_x
        else:
            x = bytes_x[:, : (bytes_x.size(1) // p) * p]
        if x.size(1) == 0:
            return bytes_x.new_zeros(bytes_x.size(0), 0)
        logits = self._byte_logits(x)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        chosen = log_probs.gather(-1, x.unsqueeze(-1)).squeeze(-1)
        return chosen

    def _next_byte_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        """Logits for the byte that would follow ``tokens``. Shape (B, V)."""
        p = self.pool_factor
        bsz, length = tokens.shape
        device = tokens.device
        dummy = torch.zeros(bsz, 1, dtype=torch.long, device=device)
        seq = torch.cat([tokens, dummy], dim=1)
        patch_index = length // p
        past = patch_index * p
        if past == 0:
            cond_p = torch.zeros(bsz, 1, self.d_model, device=device, dtype=self.byte_decoder.start.dtype)
        else:
            emb = self.encode_patches(tokens[:, :past])
            h_seq = self.patch_transformer(emb)
            cond_p = torch.zeros(bsz, patch_index + 1, self.d_model, device=device, dtype=h_seq.dtype)
            cond_p[:, 1:, :] = h_seq
        cond_bytes = cond_p.repeat_interleave(p, dim=1)[:, : length + 1]
        logits = self.byte_decoder.decode(cond_bytes, seq)
        return logits[:, -1, :]

    @torch.no_grad()
    def _generate_window(
        self,
        prefix: torch.Tensor,
        max_new_bytes: int,
        temperature: float,
        greedy: bool,
        stop_byte: int | None,
        min_p: float,
    ) -> torch.Tensor:
        tokens = prefix.clone()
        # decode() scores one extra dummy byte, and the absolute position table
        # is finite. Keep the end of the prompt so a long holdout still fits.
        pe_len = int(self.byte_decoder.pos.pe.size(1))
        room = max(1, pe_len - 1 - int(max_new_bytes))
        if tokens.size(1) > room:
            tokens = tokens[:, -room:]
        prefix_len = tokens.size(1)
        for _ in range(max_new_bytes):
            if tokens.size(1) - prefix_len >= max_new_bytes:
                break
            logits = self._next_byte_logits(tokens)
            if greedy:
                nxt = logits.argmax(dim=-1)
            else:
                probs = byte_sample_probs(logits, temperature, min_p)
                nxt = torch.multinomial(probs, num_samples=1).squeeze(-1)
            tokens = torch.cat([tokens, nxt.unsqueeze(1)], dim=1)
            if stop_byte is not None and int(nxt[0].item()) == int(stop_byte):
                break
        return tokens

    @torch.no_grad()
    def _joint_patch_bytes(
        self,
        cond: torch.Tensor,
        width: int,
        temperature: float,
        stop_byte: int | None,
    ) -> list[int]:
        """Beam-search one patch, then sample a beam with probability ∝ exp(log p / T)."""
        p = self.pool_factor
        device = cond.device
        seq = torch.zeros(1, p, dtype=torch.long, device=device)
        logp = torch.zeros(1, device=device)
        length = torch.zeros(1, dtype=torch.long, device=device)
        done = torch.zeros(1, dtype=torch.bool, device=device)
        for step in range(p):
            active = ~done
            if not bool(active.any()):
                break
            seq_a = seq[active]
            cond_b = cond.expand(seq_a.size(0), -1)
            logits = self.byte_decoder(cond_b, seq_a)[:, step, :]
            step_lp = F.log_softmax(logits.float(), dim=-1)
            total = logp[active][:, None] + step_lp
            k = min(width, int(total.numel()))
            vals, flat = torch.topk(total.reshape(-1), k)
            parent = torch.div(flat, step_lp.size(-1), rounding_mode="floor")
            byte = flat % step_lp.size(-1)
            active_idx = active.nonzero(as_tuple=False).squeeze(1)
            parent_full = active_idx[parent]
            new_seq = seq[parent_full].clone()
            new_seq[:, step] = byte
            new_len = torch.full((k,), step + 1, device=device, dtype=torch.long)
            if stop_byte is None:
                new_done = torch.zeros(k, dtype=torch.bool, device=device)
            else:
                new_done = byte == int(stop_byte)
            if bool(done.any()):
                new_seq = torch.cat([new_seq, seq[done]], dim=0)
                vals = torch.cat([vals, logp[done]], dim=0)
                new_len = torch.cat([new_len, length[done]], dim=0)
                new_done = torch.cat([new_done, torch.ones(int(done.sum()), dtype=torch.bool, device=device)], dim=0)
            if new_seq.size(0) > width:
                keep = torch.topk(vals, width).indices
                new_seq, vals, new_len, new_done = (
                    new_seq[keep],
                    vals[keep],
                    new_len[keep],
                    new_done[keep],
                )
            seq, logp, length, done = new_seq, vals, new_len, new_done
        probs = torch.softmax(logp / max(float(temperature), 1e-5), dim=0)
        choice = int(torch.multinomial(probs, num_samples=1).item())
        n = int(length[choice].item())
        return [int(x) for x in seq[choice, :n].tolist()]

    @torch.no_grad()
    def generate(
        self,
        prefix: torch.Tensor,
        max_new_bytes: int = 128,
        temperature: float = 1.0,
        greedy: bool = False,
        stop_byte: int | None = None,
        latent_sigma: float = 0.0,
        h_std: torch.Tensor | None = None,
        min_p: float = 0.0,
        word_boundary_temp: tuple[float, float] | None = None,
        joint_beam: int = 0,
    ) -> torch.Tensor:
        """Autoregressive byte generation in pool_factor chunks from a byte prefix.

        ``latent_sigma`` adds noise to the next-patch hidden state only when the
        last byte is a sentence boundary (``.``, ``!``, ``?``, or newline).
        ``h_std`` is a per-dimension scale measured on validation hidden states.
        ``min_p`` drops bytes whose probability is below ``min_p`` times the max.
        ``word_boundary_temp`` is ``(boundary, interior)`` temperatures.
        ``joint_beam`` samples a whole patch from a width-N beam of the local decoder.
        The sliding-window decoder samples one byte at a time and ignores joint beam and latent noise.
        """
        if self.local_window:
            return self._generate_window(
                prefix,
                max_new_bytes=max_new_bytes,
                temperature=temperature,
                greedy=greedy,
                stop_byte=stop_byte,
                min_p=min_p,
            )
        p = self.pool_factor
        tokens = prefix.clone()
        # Align to patch boundary
        usable = (tokens.size(1) // p) * p
        if usable < p:
            pad = torch.full((tokens.size(0), p - usable), 32, dtype=tokens.dtype, device=tokens.device)
            tokens = torch.cat([tokens, pad], dim=1)
            usable = p
        tokens = tokens[:, :usable]
        prefix_len = tokens.size(1)
        n_new = (max_new_bytes // p) * p
        for _ in range(max(1, n_new // p)):
            if tokens.size(1) - prefix_len >= n_new:
                break
            emb = self.encode_patches(tokens)
            h_seq = self.patch_transformer(emb)
            cond = h_seq[:, -1, :]  # (B, D) — last patch hidden conditions next patch
            if latent_sigma and h_std is not None:
                last = int(tokens[0, -1].item())
                if last in (46, 33, 63, 10):
                    scale = h_std.to(device=cond.device, dtype=cond.dtype)
                    if scale.ndim == 1:
                        scale = scale.view(1, -1)
                    cond = cond + float(latent_sigma) * scale * torch.randn_like(cond)
            if joint_beam and int(joint_beam) > 0 and not greedy:
                prev_bytes = self._joint_patch_bytes(
                    cond[:1],
                    width=int(joint_beam),
                    temperature=temperature,
                    stop_byte=stop_byte,
                )
                remain = n_new - (tokens.size(1) - prefix_len)
                prev_bytes = prev_bytes[: max(remain, 0)]
                if not prev_bytes:
                    break
                new_patch = torch.tensor([prev_bytes], dtype=tokens.dtype, device=tokens.device)
            else:
                # Sample p bytes with local AR decoder
                bsz = tokens.size(0)
                sampled = []
                prev_bytes = []
                for bi in range(p):
                    tgt_so_far = torch.zeros(bsz, p, dtype=torch.long, device=tokens.device)
                    for j, b in enumerate(prev_bytes):
                        tgt_so_far[:, j] = b
                    logits = self.byte_decoder(cond, tgt_so_far)  # (B, p, V)
                    prev_b = prev_bytes[-1] if prev_bytes else int(tokens[0, -1].item())
                    temp = temperature
                    if word_boundary_temp is not None:
                        hi, lo = word_boundary_temp
                        temp = hi if prev_b in WORD_BOUNDARY_BYTES else lo
                    if greedy:
                        nxt = logits[:, bi, :].argmax(dim=-1)
                    else:
                        probs = byte_sample_probs(logits[:, bi, :], temp, min_p)
                        nxt = torch.multinomial(probs, num_samples=1).squeeze(-1)
                    bval = int(nxt[0].item())
                    prev_bytes.append(bval)
                    sampled.append(nxt.unsqueeze(1))
                    if stop_byte is not None and bval == stop_byte:
                        break
                    if tokens.size(1) - prefix_len + len(prev_bytes) >= n_new:
                        break
                new_patch = torch.cat(sampled, dim=1)
            tokens = torch.cat([tokens, new_patch], dim=1)
            if stop_byte is not None and stop_byte in prev_bytes:
                break
        return tokens
