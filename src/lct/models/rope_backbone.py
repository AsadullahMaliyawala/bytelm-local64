"""RoPE transformer backbone for real_v1 8L x 640 bakeoff."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q,k: (B, H, T, Dh); cos/sin: (1, 1, T, Dh)
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_len: int = 4096, base: float = 10000.0):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv, persistent=False)
        self.max_len = max_len
        self._build_cache(max_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        self.max_len = seq_len

    def forward(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_len:
            self._build_cache(seq_len)
        return self.cos_cached[:, :, :seq_len].to(device), self.sin_cached[:, :, :seq_len].to(device)


class RoPEBlock(nn.Module):
    """Pre-LN causal attention with RoPE + SDPA."""

    def __init__(self, d_model: int, n_heads: int, ffn_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.ln2 = nn.LayerNorm(d_model)
        hidden = d_model * ffn_mult
        self.ff = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = dropout
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # attn_mask unused for SDPA is_causal; kept for API compat
        b, t, d = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).view(b, t, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, Dh)
        q, k, v = qkv[0], qkv[1], qkv[2]
        cos, sin = self.rope(t, x.device)
        q, k = apply_rope(q, k, cos, sin)
        drop_p = self.dropout if self.training else 0.0
        a = F.scaled_dot_product_attention(q, k, v, dropout_p=drop_p, is_causal=True)
        a = a.transpose(1, 2).contiguous().view(b, t, d)
        x = x + self.out(a)
        x = x + self.ff(self.ln2(x))
        return x


def build_backbone(
    d_model: int = 640,
    n_layers: int = 8,
    n_heads: int = 10,
    ffn_mult: int = 4,
    dropout: float = 0.0,
) -> nn.ModuleList:
    """ffn_mult=4 → hidden 2560 when d_model=640."""
    return nn.ModuleList(
        [RoPEBlock(d_model, n_heads, ffn_mult=ffn_mult, dropout=dropout) for _ in range(n_layers)]
    )


def count_backbone_params(blocks: nn.ModuleList) -> int:
    return sum(p.numel() for p in blocks.parameters())


def param_report(model: nn.Module, backbone_attr: str = "blocks") -> dict:
    total = sum(p.numel() for p in model.parameters())
    backbone = 0
    if hasattr(model, backbone_attr):
        backbone = sum(p.numel() for p in getattr(model, backbone_attr).parameters())
    embed = 0
    for name in ("embed", "tok_embed", "bos"):
        if hasattr(model, name):
            mod = getattr(model, name)
            if isinstance(mod, nn.Module):
                embed += sum(p.numel() for p in mod.parameters())
    head = 0
    if hasattr(model, "head") and isinstance(model.head, nn.Module):
        # avoid double-count if tied
        head = sum(p.numel() for p in model.head.parameters())
        if hasattr(model, "embed") and getattr(model.head, "weight", None) is getattr(
            getattr(model, "embed", None), "weight", object()
        ):
            head = 0
    return {
        "total_params": total,
        "backbone_params": backbone,
        "embed_params": embed,
        "head_params_untied": head,
        "non_backbone_params": total - backbone,
    }
