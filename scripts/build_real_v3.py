#!/usr/bin/env python
"""Pack real_v3. Default output is the NAS training volume."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lct.data.real_v3 import write_real_v3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("${LCT_DATA}/data/real_v3"))
    parser.add_argument("--target-train-bytes", type=int, default=70_000_000_000)
    parser.add_argument("--target-val-bytes", type=int, default=5_000_000)
    parser.add_argument("--fineweb-config", default="sample-100BT")
    parser.add_argument("--min-score", type=float, default=3.0)
    parser.add_argument("--histogram-rows", type=int, default=200_000)
    args = parser.parse_args()
    write_real_v3(
        args.out,
        target_train_bytes=args.target_train_bytes,
        target_val_bytes=args.target_val_bytes,
        fineweb_config=args.fineweb_config,
        min_score=args.min_score,
        histogram_rows=args.histogram_rows,
    )


if __name__ == "__main__":
    main()
