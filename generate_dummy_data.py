"""Generate privacy-safe synthetic inputs for the supplementary pipeline.

The generated records are artificial. They contain no real student identifiers,
Moodle records, grades, or environmental observations.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


EXAMS = {
    2023: datetime(2023, 12, 13, 18, 0),
    2024: datetime(2024, 12, 3, 16, 0),
}


def random_timestamp(rng: np.random.Generator, day: datetime, hour_center: float) -> datetime:
    hour = int(np.clip(round(rng.normal(hour_center, 3.2)), 0, 23))
    minute = int(rng.integers(0, 60))
    second = int(rng.integers(0, 60))
    return day.replace(hour=hour, minute=minute, second=second)


def generate_weather(rng: np.random.Generator, years: list[int]) -> pd.DataFrame:
    rows = []
    for year in years:
        exam = EXAMS[year]
        for offset in range(65, -1, -1):
            date = (exam - timedelta(days=offset)).date()
            seasonal = np.sin(offset / 9.0)
            tavg = 7.0 + 3.0 * seasonal + rng.normal(0, 1.8)
            prcp = max(0.0, rng.gamma(1.3, 1.7) - 1.2)
            rows.append(
                {
                    "date": date.isoformat(),
                    "pm10": round(max(4.0, rng.normal(24 + prcp, 6)), 2),
                    "pm2_5": round(max(2.0, rng.normal(13 + 0.5 * prcp, 4)), 2),
                    "tavg": round(tavg, 2),
                    "tmin": round(tavg - rng.uniform(2.0, 5.0), 2),
                    "tmax": round(tavg + rng.uniform(2.0, 5.0), 2),
                    "prcp": round(prcp, 2),
                    "wdir": round(float(rng.uniform(0, 360)), 2),
                    "wspd": round(max(0.2, rng.normal(10, 3)), 2),
                }
            )
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def generate_student_records(
    rng: np.random.Generator,
    student_id: str,
    year: int,
    latent_engagement: float,
) -> tuple[list[dict], dict]:
    exam = EXAMS[year]
    preparation_lead = int(np.clip(round(rng.normal(9 - 2.2 * latent_engagement, 3)), 3, 20))
    preparation_start = exam - timedelta(days=preparation_lead)
    total_events = int(np.clip(round(48 + 15 * latent_engagement + rng.normal(0, 9)), 22, 105))
    rows: list[dict] = []

    # Guarantee at least one pre-assessment learning-material event per synthetic
    # student so the focused-window algorithm has a well-defined input.
    guaranteed_material_time = random_timestamp(rng, preparation_start, hour_center=16)
    rows.append(
        {
            "student_id": student_id,
            "timestamp": guaranteed_material_time.strftime("%Y-%m-%d %H:%M:%S"),
            "component": "File",
            "context": "Learning material 1.pdf",
            "event_name": "Course module viewed",
        }
    )

    for event_index in range(total_events):
        focused = rng.random() < (0.48 + 0.18 * (latent_engagement > 0))
        if focused:
            day_offset = int(rng.integers(0, max(1, preparation_lead)))
            day = preparation_start + timedelta(days=day_offset)
        else:
            day = exam - timedelta(days=int(rng.integers(preparation_lead, 56)))

        component_choice = rng.choice(
            ["File", "Assignment", "Quiz", "Forum"],
            p=[0.56, 0.18, 0.18, 0.08],
        )
        timestamp = random_timestamp(rng, day, hour_center=15.5 + 1.2 * latent_engagement)
        if timestamp >= exam:
            timestamp = exam - timedelta(minutes=int(rng.integers(10, 240)))

        if component_choice == "File":
            material_number = int(rng.integers(1, 9))
            context = f"Learning material {material_number}.pdf"
            event_name = "Course module viewed"
        elif component_choice == "Assignment":
            assignment_number = int(rng.integers(1, 4))
            context = f"Assignment: {assignment_number}"
            event_name = "Submission status viewed"
        elif component_choice == "Quiz":
            context = f"Quiz: {int(rng.integers(1, 4))}"
            event_name = "Quiz attempt viewed"
        else:
            context = "Course forum"
            event_name = "Discussion viewed"

        rows.append(
            {
                "student_id": student_id,
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "component": component_choice,
                "context": context,
                "event_name": event_name,
            }
        )

    # Add explicit assignment submissions for well-defined completion-time features.
    for assignment_number in range(1, 4):
        first_day = exam - timedelta(days=int(rng.integers(10, 40)))
        submit_day = first_day + timedelta(days=int(np.clip(rng.normal(5 - latent_engagement, 2), 1, 12)))
        submit_day = min(submit_day, exam - timedelta(days=1))
        rows.extend(
            [
                {
                    "student_id": student_id,
                    "timestamp": random_timestamp(rng, first_day, 16).strftime("%Y-%m-%d %H:%M:%S"),
                    "component": "Assignment",
                    "context": f"Assignment: {assignment_number}",
                    "event_name": "Submission status viewed",
                },
                {
                    "student_id": student_id,
                    "timestamp": random_timestamp(rng, submit_day, 18).strftime("%Y-%m-%d %H:%M:%S"),
                    "component": "Assignment",
                    "context": f"Assignment: {assignment_number}",
                    "event_name": "Assignment submitted",
                },
            ]
        )

    year_effect = 0.7 if year == 2024 else -0.2
    score = np.clip(12.0 + 2.2 * latent_engagement + 0.025 * total_events + year_effect + rng.normal(0, 3.2), 0, 25)
    grade = {"student_id": student_id, "kolokvij1": round(float(score), 2)}
    return rows, grade


def generate_dataset(output_dir: Path, students_per_year: int, seed: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    weather = generate_weather(rng, sorted(EXAMS))
    weather.to_csv(output_dir / "weather.csv", index=False, encoding="utf-8-sig")

    totals = {}
    for year in sorted(EXAMS):
        logs: list[dict] = []
        grades: list[dict] = []
        for index in range(1, students_per_year + 1):
            student_id = f"SYN{year}_{index:03d}"
            latent = float(rng.normal())
            student_logs, student_grade = generate_student_records(rng, student_id, year, latent)
            logs.extend(student_logs)
            grades.append(student_grade)

        log_frame = pd.DataFrame(logs).sort_values(["student_id", "timestamp"])
        grade_frame = pd.DataFrame(grades)
        log_frame.to_csv(output_dir / f"moodle_logs_{year}.csv", index=False, encoding="utf-8-sig")
        grade_frame.to_csv(output_dir / f"grades_{year}.csv", index=False, encoding="utf-8-sig")
        totals[str(year)] = {"students": len(grade_frame), "log_records": len(log_frame)}

    manifest = {
        "synthetic": True,
        "contains_real_student_data": False,
        "seed": seed,
        "students_per_year": students_per_year,
        "exam_datetimes": {str(year): value.isoformat(sep=" ") for year, value in EXAMS.items()},
        "totals": totals,
        "privacy_notice": "These artificial data contain no real student, grade, Moodle, or environmental records.",
        "results_notice": "Outputs derived from these data are execution checks and are not study findings.",
    }
    (output_dir / "dummy_data_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data_dummy"))
    parser.add_argument("--students-per-year", type=int, default=60)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_dataset(args.output_dir, args.students_per_year, args.seed)
    print(f"Synthetic data written to {args.output_dir.resolve()}")
