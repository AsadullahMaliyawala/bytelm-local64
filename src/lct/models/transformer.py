from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from lct.models.quantizer import PositionalEncoding, TransformerBlock, causal_mask


class ByteEncoder(nn.Module):
    """Maps byte sequences to continuous latent vectors."""

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 384,
        n_heads: int = 6,
        n_layers: int = 4,
        ffn_mult: int = 4,
        dropout: float = 0.1,
        max_len: int = 2048,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, ffn_mult, dropout) for _ in range(n_layers)]
        )
        self.ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T) byte ids
        h = self.dropout(self.pos(self.embed(x)))
        for blk in self.blocks:
            h = blk(h)
        return self.ln(h)


class ByteDecoder(nn.Module):
    """Reconstructs bytes from quantized latents (optionally causal)."""

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 384,
        n_heads: int = 6,
        n_layers: int = 4,
        ffn_mult: int = 4,
        dropout: float = 0.1,
        max_len: int = 2048,
        causal: bool = True,
    ):
        super().__init__()
        self.causal = causal
        self.proj = nn.Linear(d_model, d_model)
        self.pos = PositionalEncoding(d_model, max_len)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, ffn_mult, dropout) for _ in range(n_layers)]
        )
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        # z_q: (B, T, D) → logits (B, T, V)
        h = self.dropout(self.pos(self.proj(z_q)))
        mask = causal_mask(h.size(1), h.device) if self.causal else None
        for blk in self.blocks:
            h = blk(h, attn_mask=mask)
        return self.head(self.ln(h))


class ConceptTransformer(nn.Module):
    """Causal LM over discrete concept token ids."""

    def __init__(
        self,
        codebook_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        ffn_mult: int = 4,
        dropout: float = 0.1,
        max_len: int = 2048,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.label_smoothing = label_smoothing
        self.embed = nn.Embedding(codebook_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, ffn_mult, dropout) for _ in range(n_layers)]
        )
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, codebook_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        # weight tying
        self.head.weight = self.embed.weight

    def forward(
        self,
        z: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        z: (B, T) concept ids
        targets: (B, T) next-concept targets (same length, typically shifted)
        """
        h = self.dropout(self.pos(self.embed(z)))
        if attn_mask is None:
            attn_mask = causal_mask(h.size(1), h.device)
        for blk in self.blocks:
            h = blk(h, attn_mask=attn_mask)
        h = self.ln(h)
        logits = self.head(h)
        out = {"logits": logits, "hidden": h}
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, self.codebook_size),
                targets.reshape(-1),
                label_smoothing=self.label_smoothing,
                ignore_index=-100,
            )
            out["loss"] = loss
        return out

    @torch.no_grad()
    def generate(self, prefix: torch.Tensor, max_new: int = 64, temperature: float = 1.0) -> torch.Tensor:
        tokens = prefix
        for _ in range(max_new):
            out = self.forward(tokens)
            logits = out["logits"][:, -1, :] / max(temperature, 1e-5)
            probs = torch.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_tok], dim=1)
        return tokens


class WorldModelHeads(nn.Module):
    """Auxiliary heads: MCM, skip-ahead, structure probes."""

    def __init__(
        self,
        d_model: int,
        codebook_size: int,
        structure_tags: list[str] | None = None,
        skip_deltas: list[int] | None = None,
    ):
        super().__init__()
        self.structure_tags = structure_tags or ["loop", "branch", "try", "def", "class"]
        self.skip_deltas = skip_deltas or [5, 10, 20]
        self.mcm_head = nn.Linear(d_model, codebook_size)
        self.skip_heads = nn.ModuleDict(
            {str(d): nn.Linear(d_model, codebook_size) for d in self.skip_deltas}
        )
        self.structure_head = nn.Linear(d_model, len(self.structure_tags))

    def forward(self, hidden: torch.Tensor) -> dict:
        return {
            "mcm_logits": self.mcm_head(hidden),
            "skip_logits": {d: self.skip_heads[d](hidden) for d in self.skip_heads},
            "structure_logits": self.structure_head(hidden),
        }