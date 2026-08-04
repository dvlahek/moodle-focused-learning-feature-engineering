"""Build the interpretable feature dataset from Moodle-style event logs."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from src.feature_engineering import build_combined_dataset


def parse_exam(value: str) -> datetime:
    return datetime.fromisoformat(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-2023", type=Path, required=True)
    parser.add_argument("--grades-2023", type=Path, required=True)
    parser.add_argument("--exam-2023", type=parse_exam, required=True)
    parser.add_argument("--logs-2024", type=Path, required=True)
    parser.add_argument("--grades-2024", type=Path, required=True)
    parser.add_argument("--exam-2024", type=parse_exam, required=True)
    parser.add_argument("--weather", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output = build_combined_dataset(
        logs_by_year={2023: args.logs_2023, 2024: args.logs_2024},
        grades_by_year={2023: args.grades_2023, 2024: args.grades_2024},
        exams_by_year={2023: args.exam_2023, 2024: args.exam_2024},
        weather_path=args.weather,
        output_path=args.output,
    )
    print(f"Feature dataset written to {args.output.resolve()} ({len(output)} rows, {len(output.columns)} columns)")

