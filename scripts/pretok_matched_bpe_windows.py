"""Pre-tokenize the matched BPE twin's exact 10B-byte window stream.

Uses the same ShuffledDocCursor (seed-0 perm, same bounds) as train_bpe_matched350m.py.
Writes per-window BPE ids so training can skip on-the-fly tokenization.

Output (under --out):
  byte_windows.bin  — n_windows * 2048 raw bytes
  tokens.u16        — concatenated uint16 token ids (jagged)
  offsets.i64       — int64[n_windows+1] into tokens.u16
  meta.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from baselines.bpe_lm import SimpleBPE  # noqa: E402
from lct.train.ladder import ShuffledDocCursor  # noqa: E402

WINDOW = 2048
PAD_ID = 0


def dump_byte_windows(
    train: Path,
    spans: Path,
    perm: Path,
    bounds: tuple[int, int],
    target_bytes: int,
    out_bytes: Path,
) -> dict:
    n_windows = target_bytes // WINDOW
    out_bytes.parent.mkdir(parents=True, exist_ok=True)
    mm = np.memmap(out_bytes, dtype=np.uint8, mode="w+", shape=(n_windows, WINDOW))
    cursor = ShuffledDocCursor(train, WINDOW, spans, perm, rank=0, world_size=1, bounds=bounds)
    t0 = time.time()
    for i in range(n_windows):
        win = cursor.next_window()
        assert len(win) == WINDOW
        mm[i] = np.frombuffer(win, dtype=np.uint8)
        if (i + 1) % 10000 == 0 or i + 1 == n_windows:
            mm.flush()
            rate = (i + 1) * WINDOW / max(time.time() - t0, 1e-6) / 1e6
            print({"dump_windows": i + 1, "of": n_windows, "MB_s": round(rate, 2), "cursor_offset": cursor.offset}, flush=True)
    mm.flush()
    # Integrity: a truncated/partial dump leaves trailing zero windows (seen in the wild).
    nonzero_tail = int(np.any(mm[-1] != 0))
    nonzero_mid = int(np.any(mm[n_windows // 2] != 0))
    nonzero_head = int(np.any(mm[0] != 0))
    if not (nonzero_head and nonzero_mid and nonzero_tail):
        raise RuntimeError(
            f"byte_windows dump looks empty/partial: nonzero head/mid/tail="
            f"{nonzero_head}/{nonzero_mid}/{nonzero_tail}"
        )
    state = cursor.state()
    cursor.close()
    del mm
    return {
        "n_windows": n_windows,
        "window": WINDOW,
        "target_bytes": n_windows * WINDOW,
        "final_cursor_state": state,
        "dump_seconds": time.time() - t0,
        "dump_ok": True,
        "nonzero_head_mid_tail": [nonzero_head, nonzero_mid, nonzero_tail],
    }


_WORKER_BPE = None


def _init_worker(bpe_dir: str) -> None:
    global _WORKER_BPE
    bpe = SimpleBPE(vocab_size=8000)
    bpe.load(Path(bpe_dir))
    _WORKER_BPE = bpe


def _tokenize_chunk(args: tuple) -> tuple[int, int, list[list[int]]]:
    start, end, bytes_path, n_windows, bpe_dir = args
    global _WORKER_BPE
    if _WORKER_BPE is None:
        _init_worker(bpe_dir)
    tok = _WORKER_BPE.tok
    mm = np.memmap(bytes_path, dtype=np.uint8, mode="r", shape=(n_windows, WINDOW))
    rows: list[list[int]] = []
    for wi in range(start, end):
        raw = bytes(mm[wi].tolist())
        text = raw.decode("utf-8", errors="replace")
        ids = tok.encode(text).ids or [PAD_ID]
        rows.append([int(x) for x in ids])
    return start, end, rows


def tokenize_parallel(
    bytes_path: Path,
    n_windows: int,
    bpe_dir: Path,
    out_tokens: Path,
    out_offsets: Path,
    workers: int,
    chunk_windows: int,
) -> dict:
    # Collect jagged rows then write once (avoids huge padded NFS allocate).
    # For 4.88M windows, store rows in a list of arrays per chunk on disk as temp parts.
    part_dir = out_tokens.parent / "_tok_parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for start in range(0, n_windows, chunk_windows):
        end = min(n_windows, start + chunk_windows)
        jobs.append((start, end, str(bytes_path), n_windows, str(bpe_dir)))
    t0 = time.time()
    done = 0
    max_seen = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(str(bpe_dir),)) as ex:
        futs = {ex.submit(_tokenize_chunk, j): j[0] for j in jobs}
        for fut in as_completed(futs):
            start, end, rows = fut.result()
            part = part_dir / f"part_{start:08d}.npz"
            # object array of lists — compact enough for chunks of 4096
            np.save(part, np.array(rows, dtype=object), allow_pickle=True)
            max_seen = max(max_seen, max((len(r) for r in rows), default=0))
            done += len(rows)
            print(
                {
                    "tokenize_windows": done,
                    "of": n_windows,
                    "max_tok_seen": max_seen,
                    "seconds": round(time.time() - t0, 1),
                },
                flush=True,
            )

    # Merge parts in order into jagged tokens + offsets
    offsets = np.zeros(n_windows + 1, dtype=np.int64)
    # first pass: lengths
    pos = 0
    part_files = sorted(part_dir.glob("part_*.npy"))
    lengths = np.zeros(n_windows, dtype=np.int32)
    cursor = 0
    for part in part_files:
        rows = np.load(part, allow_pickle=True)
        for row in rows:
            lengths[cursor] = len(row)
            cursor += 1
    total_tok = int(lengths.sum())
    tokens = np.memmap(out_tokens, dtype=np.uint16, mode="w+", shape=(total_tok,))
    offsets_mm = np.memmap(out_offsets, dtype=np.int64, mode="w+", shape=(n_windows + 1,))
    offsets_mm[0] = 0
    for i in range(n_windows):
        offsets_mm[i + 1] = offsets_mm[i] + int(lengths[i])
    # second pass: fill tokens
    cursor = 0
    write_at = 0
    for part in part_files:
        rows = np.load(part, allow_pickle=True)
        for row in rows:
            n = len(row)
            if n:
                tokens[write_at : write_at + n] = np.asarray(row, dtype=np.uint16)
            write_at += n
            cursor += 1
        part.unlink()
    tokens.flush()
    offsets_mm.flush()
    try:
        part_dir.rmdir()
    except OSError:
        pass
    return {
        "tokenize_seconds": time.time() - t0,
        "max_tok_seen": max_seen,
        "total_tokens": total_tok,
        "mean_tok_per_window": total_tok / max(n_windows, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--spans", type=Path, required=True)
    parser.add_argument("--perm", type=Path, required=True)
    parser.add_argument("--bpe-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target-bytes", type=int, default=10_000_000_000)
    parser.add_argument("--bounds", default="59500002824,66500006098")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) - 2))
    parser.add_argument("--chunk-windows", type=int, default=4096)
    parser.add_argument("--skip-dump", action="store_true")
    args = parser.parse_args()

    bounds = tuple(int(x) for x in args.bounds.split(","))
    args.out.mkdir(parents=True, exist_ok=True)
    bytes_path = args.out / "byte_windows.bin"
    tokens_path = args.out / "tokens.u16"
    offsets_path = args.out / "offsets.i64"
    meta_path = args.out / "meta.json"

    n_windows = args.target_bytes // WINDOW
    if args.skip_dump and bytes_path.exists() and bytes_path.stat().st_size == n_windows * WINDOW:
        mm = np.memmap(bytes_path, dtype=np.uint8, mode="r", shape=(n_windows, WINDOW))
        ok = bool(np.any(mm[0] != 0) and np.any(mm[n_windows // 2] != 0) and np.any(mm[-1] != 0))
        del mm
        if not ok:
            print({"reuse_bytes_rejected": str(bytes_path), "reason": "zero_head_mid_or_tail"}, flush=True)
            dump_meta = dump_byte_windows(args.train, args.spans, args.perm, bounds, args.target_bytes, bytes_path)
        else:
            dump_meta = {
                "n_windows": n_windows,
                "window": WINDOW,
                "target_bytes": n_windows * WINDOW,
                "skipped_dump": True,
                "dump_ok": True,
            }
            print({"reuse_bytes": str(bytes_path), "nbytes": bytes_path.stat().st_size}, flush=True)
    else:
        dump_meta = dump_byte_windows(args.train, args.spans, args.perm, bounds, args.target_bytes, bytes_path)

    tok_meta = tokenize_parallel(
        bytes_path,
        dump_meta["n_windows"],
        args.bpe_dir,
        tokens_path,
        offsets_path,
        args.workers,
        args.chunk_windows,
    )
    meta = {
        **dump_meta,
        **tok_meta,
        "pad_id": PAD_ID,
        "format": "jagged_u16",
        "bpe_dir": str(args.bpe_dir),
        "bounds": list(bounds),
        "train": str(args.train),
        "spans": str(args.spans),
        "perm": str(args.perm),
        "tokens": str(tokens_path),
        "offsets": str(offsets_path),
        "byte_windows": str(bytes_path),
        "note": "Window i tokens = tokens[offsets[i]:offsets[i+1]]; same encode as make_bpe_loss.",
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print({"DONE": True, "meta": str(meta_path), **{k: tok_meta[k] for k in tok_meta}}, flush=True)


if __name__ == "__main__":
    main()
