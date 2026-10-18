#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lct.train.tokenizer import train_tokenizer
from lct.utils import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/5060/tokenizer_tiny.yaml")
    args = ap.parse_args()
    cfg = load_config(ROOT / args.config if not Path(args.config).is_absolute() else args.config)
    print(train_tokenizer(cfg))


if __name__ == "__main__":
    main()