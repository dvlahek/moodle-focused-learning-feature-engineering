"""Run regression, classification, cross-year, permutation, and importance analyses."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluation import EvaluationConfig, run_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quick", action="store_true", help="Use smaller models and fewer permutations for a smoke test")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = EvaluationConfig.quick() if args.quick else EvaluationConfig.paper()
    run_evaluation(args.dataset, args.output_dir, config)
    print(f"Evaluation outputs written to {args.output_dir.resolve()}")

