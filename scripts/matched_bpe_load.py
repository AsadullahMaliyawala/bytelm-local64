"""Load matched 350M BPE-8k twin checkpoints (DecoderOnlyLM, no config in .pt)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from baselines.bpe_lm import SimpleBPE  # noqa: E402
from lct.checkpoint import load_state_dict_strict  # noqa: E402
from lct.models.decoder_lm import DecoderOnlyLM  # noqa: E402

DEFAULT_BPE_DIR = ROOT / "checkpoints" / "bpe_8k_real_v2"


def load_matched_bpe(
    ckpt: Path,
    device: torch.device,
    bpe_dir: Path | None = None,
    *,
    d_model: int = 1024,
    layers: int = 26,
    heads: int = 16,
) -> tuple[DecoderOnlyLM, SimpleBPE, int, int]:
    bpe = SimpleBPE(vocab_size=8000)
    bpe.load(Path(bpe_dir or DEFAULT_BPE_DIR))
    assert bpe.tok is not None
    n_vocab = int(bpe.tok.get_vocab_size())
    model = DecoderOnlyLM(
        vocab_size=n_vocab,
        d_model=d_model,
        n_heads=heads,
        n_layers=layers,
        ffn_mult=4,
        dropout=0.0,
        max_len=2048,
        pos_type="rope",
        ignore_index=0,
    )
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    load_state_dict_strict(model, state, label=f"matched_bpe:{ckpt}")
    model.to(device).eval()
    step = int(blob.get("step", 0)) if isinstance(blob, dict) else 0
    seen = int(blob.get("seen", 0)) if isinstance(blob, dict) else 0
    return model, bpe, step, seen


@torch.no_grad()
def greedy_bpe_generate(
    model: DecoderOnlyLM,
    bpe: SimpleBPE,
    prompt: str,
    device: torch.device,
    max_new: int = 64,
) -> str:
    """Greedy token decode; stop when decoded continuation contains a newline."""
    assert bpe.tok is not None
    ids = list(bpe.tok.encode(prompt).ids)
    if len(ids) < 1:
        ids = [1]
    tokens = torch.tensor([ids], dtype=torch.long, device=device)
    prefix_len = len(prompt)
    for _ in range(max_new):
        out = model(tokens)
        nxt = int(out["logits"][0, -1].argmax().item())
        if nxt == 0:
            break
        tokens = torch.cat([tokens, torch.tensor([[nxt]], device=device)], dim=1)
        text = bpe.tok.decode(tokens[0].tolist())
        cont = text[prefix_len:] if len(text) >= prefix_len else text
        if "\n" in cont:
            return cont.split("\n", 1)[0]
    text = bpe.tok.decode(tokens[0].tolist())
    cont = text[prefix_len:] if len(text) >= prefix_len else text
    return cont.split("\n", 1)[0]
