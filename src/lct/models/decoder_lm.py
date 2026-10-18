"""Shared decoder-only language model (~30–40M with 8L×512; ~75M with 8L×640 RoPE).

Used by byte LM, BPE LM, and VQ concept LM so core capacity is matched.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from lct.models.quantizer import PositionalEncoding, TransformerBlock, causal_mask
from lct.models.rope_backbone import RoPEBlock, param_report


class DecoderOnlyLM(nn.Module):
    """Causal transformer LM with tied input/output embeddings."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 8,
        ffn_mult: int = 4,
        dropout: float = 0.1,
        max_len: int = 2048,
        label_smoothing: float = 0.0,
        ignore_index: int = -100,
        pos_type: str = "sinusoidal",
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.codebook_size = vocab_size  # alias for ConceptLM / WorldModelLM
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
        self.pos_type = pos_type
        self.embed = nn.Embedding(vocab_size, d_model)
        if pos_type == "rope":
            self.pos = None
            self.blocks = nn.ModuleList(
                [RoPEBlock(d_model, n_heads, ffn_mult, dropout) for _ in range(n_layers)]
            )
        else:
            self.pos = PositionalEncoding(d_model, max_len)
            self.blocks = nn.ModuleList(
                [TransformerBlock(d_model, n_heads, ffn_mult, dropout) for _ in range(n_layers)]
            )
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.head.weight = self.embed.weight
        self.gradient_checkpointing = gradient_checkpointing

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        start_pos: int = 0,
    ) -> dict:
        """
        x: (B, T) token ids
        targets: (B, T) next-token targets (same length, typically shifted)
        start_pos: absolute PE offset so chunked teacher-forcing matches a full span
        """
        h = self.embed(x)
        if self.pos_type == "rope":
            h = self.dropout(h)
            for blk in self.blocks:
                if self.gradient_checkpointing and self.training:
                    h = torch.utils.checkpoint.checkpoint(blk, h, use_reentrant=False)
                else:
                    h = blk(h)
        else:
            h = self.dropout(self.pos(h, start_pos=start_pos))
            if attn_mask is None:
                attn_mask = causal_mask(h.size(1), h.device)
            for blk in self.blocks:
                h = blk(h, attn_mask=attn_mask)
        h = self.ln(h)
        logits = self.head(h)
        out: dict = {"logits": logits, "hidden": h}
        if targets is not None:
            out["loss"] = nn.functional.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
                label_smoothing=self.label_smoothing,
                ignore_index=self.ignore_index,
            )
        return out

    def param_report(self) -> dict:
        return param_report(self, backbone_attr="blocks")

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
