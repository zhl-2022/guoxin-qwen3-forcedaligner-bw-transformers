#!/usr/bin/env python3
"""Transparent canonical task entry; does not synthesize evaluation results."""
import os
from pathlib import Path
import sys

entry = Path(__file__).resolve().parent / 'tasks/qwen3-forcedaligner-bw-transformers/entry.py'
try:
    os.execv(sys.executable, [sys.executable, str(entry), *sys.argv[1:]])
except OSError as exc:
    print(f'Cannot execute canonical entry: {exc}', file=sys.stderr)
    raise SystemExit(2)
