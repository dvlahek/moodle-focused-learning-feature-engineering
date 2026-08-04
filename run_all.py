"""Generate synthetic inputs and execute the complete supplementary workflow."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path


EXAMS = {
    2023: datetime(2023, 12, 13, 18, 0),
    2024: datetime(2024, 12, 3, 16, 0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Run a fast smoke test with smaller models")
    parser.add_argument("--students-per-year", type=int, default=60)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--data-dir", type=Path, default=Path("data_dummy"))
    parser.add_argument("--results-dir", type=Path, default=Path("results_dummy"))
    return parser.parse_args()


def main() -> None:
    try:
        from generate_dummy_data import generate_dataset
        from src.evaluation import EvaluationConfig, run_evaluation
        from src.feature_engineering import build_combined_dataset
        from src.figures import create_figures
    except ModuleNotFoundError as error:
        missing = error.name or "a required package"
        raise SystemExit(
            f"Missing dependency: {missing}. Install the package requirements first with "
            "`python -m pip install -r requirements.txt`."
        ) from error

    args = parse_args()
    generate_dataset(args.data_dir, args.students_per_year, args.seed)
    feature_path = args.data_dir / "dummy_feature_dataset.csv"
    build_combined_dataset(
        logs_by_year={year: args.data_dir / f"moodle_logs_{year}.csv" for year in EXAMS},
        grades_by_year={year: args.data_dir / f"grades_{year}.csv" for year in EXAMS},
        exams_by_year=EXAMS,
        weather_path=args.data_dir / "weather.csv",
        output_path=feature_path,
    )
    config = EvaluationConfig.quick() if args.quick else EvaluationConfig.paper()
    run_evaluation(feature_path, args.results_dir, config)
    create_figures(args.results_dir)
    print("Complete synthetic-data workflow finished successfully.")
    print(f"Feature dataset: {feature_path.resolve()}")
    print(f"Results directory: {args.results_dir.resolve()}")


if __name__ == "__main__":
    main()
