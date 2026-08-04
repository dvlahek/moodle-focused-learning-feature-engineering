#!/usr/bin/env python3
"""Create compact figures from saved evaluation CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.figures import create_figures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_figures(args.results_dir)
    print(f"Figures written to {args.results_dir.resolve()}")

