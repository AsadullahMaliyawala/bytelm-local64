"""BPE ByteLevel round-trip and packed-stream SHA-256 checks (Phase 2B.1b).

Failing tokenizer artifact (pre-fix): checkpoints/bpe without ByteLevel decoder
decoded U+0120 instead of spaces — 0% text round-trip. Always attach
ByteLevelDecoder on load and run these tests on save/load.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence


ROUNDTRIP_SAMPLES: list[bytes] = [
    b"The quick brown fox jumps over the lazy dog.",
    b"def merge(a, b):\n    return a + b\n",
    b"multiple   spaces\tand\ttabs",
    b"line1\nline2\r\nline3\n",
    "café résumé 日本語".encode("utf-8"),
    b"doc\x1e next",  # document separator
    b"<|doc|> special looking text",
    b"",
]


def ensure_bytelevel_decoder(tokenizer) -> None:
    """Attach ByteLevel decoder if missing (fixes silent U+0120 decode bug)."""
    if tokenizer is None:
        return
    try:
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    except ImportError:
        return
    if getattr(tokenizer, "decoder", None) is None:
        tokenizer.decoder = ByteLevelDecoder()


def assert_bytelevel_roundtrip(tokenizer, samples: Sequence[bytes] | None = None) -> dict[str, Any]:
    """Assert decode(encode(raw)) == raw for each sample (UTF-8 text / bytes).

    ``tokenizer`` may be a HuggingFace ``tokenizers.Tokenizer`` or an object
    with ``.tok`` (SimpleBPE). Returns stats; raises AssertionError on failure.
    """
    samples = list(samples) if samples is not None else list(ROUNDTRIP_SAMPLES)
    tok = tokenizer.tok if hasattr(tokenizer, "tok") else tokenizer
    if tok is None:
        return {"ok": True, "total": 0, "skipped": "byte_mode_or_none"}
    ensure_bytelevel_decoder(tok)
    failures: list[dict[str, Any]] = []
    for raw in samples:
        if isinstance(raw, str):
            raw_b = raw.encode("utf-8")
            text = raw
        else:
            raw_b = bytes(raw)
            try:
                text = raw_b.decode("utf-8")
            except UnicodeDecodeError:
                text = raw_b.decode("utf-8", errors="surrogateescape")
        enc = tok.encode(text)
        try:
            decoded = tok.decode(enc.ids, skip_special_tokens=False)
        except TypeError:
            decoded = tok.decode(enc.ids)
        if decoded != text:
            failures.append(
                {
                    "raw_hex": raw_b[:64].hex(),
                    "expected": text[:80],
                    "got": decoded[:80],
                    "expected_repr": repr(text[:40]),
                    "got_repr": repr(decoded[:40]),
                }
            )
    assert not failures, f"BPE ByteLevel roundtrip failed: {failures[:3]}"
    return {"ok": True, "total": len(samples), "failures": 0}


# Back-compat alias (do not name test_* — pytest would collect this module)
test_bytelevel_roundtrip = assert_bytelevel_roundtrip


def packed_stream_decode_sha256(
    tokenizer,
    packed: bytes,
    *,
    chunk_chars: int = 4096,
) -> str:
    """Decode the full packed stream via BPE encode→decode and return SHA-256.

    Encodes UTF-8 text of the packed bytes (errors=replace) in one shot when
    small, else chunk by documents on 0x1E. Stronger check: compare to
    hashlib.sha256(packed).hexdigest() when round-trip is perfect.
    """
    tok = tokenizer.tok if hasattr(tokenizer, "tok") else tokenizer
    ensure_bytelevel_decoder(tok)
    text = packed.decode("utf-8", errors="replace")
    enc = tok.encode(text)
    decoded = tok.decode(enc.ids)
    return hashlib.sha256(decoded.encode("utf-8")).hexdigest()


def assert_packed_sha_matches(
    tokenizer,
    packed: bytes,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Require decoded packed SHA-256 == original packed SHA-256."""
    orig = expected_sha256 or hashlib.sha256(packed).hexdigest()
    got = packed_stream_decode_sha256(tokenizer, packed)
    assert got == orig, (
        f"BPE packed-stream SHA mismatch: decoded={got[:16]}… "
        f"original={orig[:16]}… (ByteLevel decoder / encoding bug)"
    )
    return {"ok": True, "sha256": orig, "decoded_sha256": got}
