from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """VQ bottleneck with optional EMA codebook updates + usage entropy."""

    def __init__(
        self,
        codebook_size: int,
        dim: int,
        commitment_beta: float = 0.25,
        use_ema: bool = True,
        ema_decay: float = 0.99,
        entropy_reg: float = 0.0,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        self.commitment_beta = commitment_beta
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.entropy_reg = entropy_reg

        self.codebook = nn.Embedding(codebook_size, dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / codebook_size, 1.0 / codebook_size)

        if use_ema:
            self.register_buffer("ema_cluster_size", torch.zeros(codebook_size))
            self.register_buffer("ema_embed_avg", self.codebook.weight.data.clone())

        self.register_buffer("usage_counts", torch.zeros(codebook_size))
        self._last_indices: Optional[torch.Tensor] = None

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Args:
            z: (B, T, D) continuous encoder outputs
        Returns:
            z_q: quantized (B, T, D) with STE
            indices: (B, T) long
            losses: dict with vq_loss, commitment, entropy, perplexity
        """
        b, t, d = z.shape
        flat = z.reshape(-1, d)

        # distances: (N, K)
        distances = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(1)
        )
        indices = distances.argmin(dim=1)
        z_q = self.codebook(indices).view(b, t, d)

        # losses
        commitment = F.mse_loss(z, z_q.detach())
        if self.use_ema:
            codebook_loss = z.new_zeros(())
            if self.training:
                self._ema_update(flat, indices)
        else:
            codebook_loss = F.mse_loss(z_q, z.detach())

        # STE
        z_q_st = z + (z_q - z).detach()

        # usage stats
        with torch.no_grad():
            ones = torch.bincount(indices, minlength=self.codebook_size).float()
            self.usage_counts += ones.to(self.usage_counts.device)
            probs = ones / ones.sum().clamp_min(1.0)
            entropy = -(probs * (probs + 1e-10).log()).sum()
            perplexity = entropy.exp()
            dead_ratio = (ones == 0).float().mean()

        entropy_loss = z.new_zeros(())
        if self.entropy_reg > 0:
            # encourage uniform usage: maximize entropy => minimize -entropy
            soft = F.softmax(-distances, dim=-1)
            avg = soft.mean(0)
            entropy_loss = self.entropy_reg * (avg * (avg + 1e-10).log()).sum()

        vq_loss = codebook_loss + self.commitment_beta * commitment + entropy_loss
        self._last_indices = indices.view(b, t)

        metrics = {
            "vq_loss": vq_loss.detach(),
            "commitment": commitment.detach(),
            "codebook_loss": codebook_loss.detach() if isinstance(codebook_loss, torch.Tensor) else torch.tensor(0.0),
            "entropy": entropy.detach(),
            "perplexity": perplexity.detach(),
            "dead_ratio": dead_ratio.detach(),
        }
        return z_q_st, indices.view(b, t), {"loss": vq_loss, **metrics}

    @torch.no_grad()
    def _ema_update(self, flat: torch.Tensor, indices: torch.Tensor) -> None:
        flat32 = flat.float()
        ones = torch.zeros(self.codebook_size, device=flat.device, dtype=torch.float32)
        ones.scatter_add_(0, indices, torch.ones(indices.shape, device=flat.device, dtype=torch.float32))
        self.ema_cluster_size.mul_(self.ema_decay).add_(ones, alpha=1 - self.ema_decay)

        embed_sum = torch.zeros(self.codebook_size, self.dim, device=flat.device, dtype=torch.float32)
        embed_sum.index_add_(0, indices, flat32)
        self.ema_embed_avg.mul_(self.ema_decay).add_(embed_sum, alpha=1 - self.ema_decay)

        n = self.ema_cluster_size.sum()
        cluster_size = (
            (self.ema_cluster_size + 1e-5) / (n + self.codebook_size * 1e-5) * n
        )
        embed_normalized = self.ema_embed_avg / cluster_size.unsqueeze(1)
        self.codebook.weight.data.copy_(embed_normalized)

    def reset_usage(self) -> None:
        self.usage_counts.zero_()

    def dead_code_mask(self, threshold: float = 1.0) -> torch.Tensor:
        return self.usage_counts < threshold

    @torch.no_grad()
    def revive_dead_codes(self, z: torch.Tensor, threshold: float = 1.0) -> int:
        """Reinitialize unused codes near random live encoder vectors (anti-collapse)."""
        dead = self.dead_code_mask(threshold)
        n_dead = int(dead.sum().item())
        if n_dead == 0:
            return 0
        flat = z.reshape(-1, self.dim)
        if flat.size(0) == 0:
            return 0
        # sample replacement vectors from current batch
        idx = torch.randint(0, flat.size(0), (n_dead,), device=flat.device)
        replacements = flat[idx] + 0.01 * torch.randn_like(flat[idx])
        self.codebook.weight.data[dead] = replacements
        if self.use_ema:
            self.ema_embed_avg[dead] = replacements
            self.ema_cluster_size[dead] = 1.0
        self.usage_counts[dead] = 0
        return n_dead


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """Add sinusoidal PE; ``start_pos`` offsets absolute indices (carried context)."""
        t = x.size(1)
        return x + self.pe[:, start_pos : start_pos + t]


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
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

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + self.dropout(a)
        x = x + self.ff(self.ln2(x))
        return x


def causal_mask(t: int, device: torch.device) -> torch.Tensor:
    # True = masked out for MultiheadAttention
    return torch.triu(torch.ones(t, t, device=device, dtype=torch.bool), diagonal=1)