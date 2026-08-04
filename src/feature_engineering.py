from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .common import ROLLING_WINDOW_DAYS, SESSION_GAP_MINUTES, WEATHER_COLUMNS, coerce_numeric_frame, read_csv


LOG_COLUMN_ALIASES = {
    "Puno ime": "student_id",
    "name": "student_id",
    "Time": "timestamp",
    "Komponenta": "component",
    "Kontekst": "context",
    "Naziv": "event_name",
}

GRADE_COLUMN_ALIASES = {
    "name": "student_id",
    "Puno ime": "student_id",
    "Kolokvij 1. (12,5/25)": "kolokvij1",
    "Kolokvij 1": "kolokvij1",
    "kolokvij 1": "kolokvij1",
    "kolokvij_1": "kolokvij1",
    "K1": "kolokvij1",
}

DAY_PARTS = {
    "Night": (0, 6),
    "Morning": (6, 12),
    "Afternoon": (12, 18),
    "Evening": (18, 24),
}


def normalize_logs(frame: pd.DataFrame) -> pd.DataFrame:
    logs = frame.rename(columns=LOG_COLUMN_ALIASES).copy()
    required = ["student_id", "timestamp", "component", "context", "event_name"]
    missing = [column for column in required if column not in logs.columns]
    if missing:
        raise ValueError(f"Log file is missing required columns: {missing}")
    logs["timestamp"] = pd.to_datetime(logs["timestamp"], errors="coerce", format="mixed", dayfirst=True)
    logs = logs.dropna(subset=["student_id", "timestamp"])
    for column in ("student_id", "component", "context", "event_name"):
        logs[column] = logs[column].fillna("").astype(str).str.strip()
    return logs.sort_values(["student_id", "timestamp"]).reset_index(drop=True)


def normalize_grades(frame: pd.DataFrame) -> pd.DataFrame:
    grades = frame.rename(columns=GRADE_COLUMN_ALIASES).copy()
    required = ["student_id", "kolokvij1"]
    missing = [column for column in required if column not in grades.columns]
    if missing:
        raise ValueError(f"Grade file is missing required columns: {missing}")
    grades["student_id"] = grades["student_id"].astype(str).str.strip()
    grades["kolokvij1"] = pd.to_numeric(
        grades["kolokvij1"].astype(str).str.replace(",", ".", regex=False), errors="coerce"
    )
    return grades.dropna(subset=["student_id", "kolokvij1"])[required].drop_duplicates("student_id")


def split_sessions(timestamps: list[pd.Timestamp]) -> list[list[pd.Timestamp]]:
    ordered = sorted(pd.to_datetime(timestamps))
    if not ordered:
        return []
    sessions = [[ordered[0]]]
    maximum_gap = pd.Timedelta(minutes=SESSION_GAP_MINUTES)
    for timestamp in ordered[1:]:
        if timestamp - sessions[-1][-1] <= maximum_gap:
            sessions[-1].append(timestamp)
        else:
            sessions.append([timestamp])
    return sessions


def day_part_counts(timestamps: list[pd.Timestamp]) -> dict[str, int]:
    counts = {name: 0 for name in DAY_PARTS}
    for timestamp in pd.to_datetime(timestamps):
        for name, (start, end) in DAY_PARTS.items():
            if start <= timestamp.hour < end:
                counts[name] += 1
                break
    return counts


def session_day_part_counts(sessions: list[list[pd.Timestamp]]) -> dict[str, int]:
    return day_part_counts([session[0] for session in sessions if session])


def session_gap_statistics(sessions: list[list[pd.Timestamp]]) -> tuple[float, float, float]:
    if len(sessions) < 2:
        return 0.0, 0.0, 0.0
    gaps = [round((sessions[index][0] - sessions[index - 1][-1]).total_seconds() / 3600, 1) for index in range(1, len(sessions))]
    values, counts = np.unique(gaps, return_counts=True)
    mode = float(values[np.argmax(counts)])
    return float(np.mean(gaps)), float(np.median(gaps)), mode


def material_events(student_logs: pd.DataFrame) -> pd.DataFrame:
    component = student_logs["component"].str.lower()
    context = student_logs["context"].str.lower()
    component_match = component.str.contains(r"file|resource|page|datoteka", regex=True, na=False)
    context_match = context.str.contains(r"pdf|learning material|course material|datoteka", regex=True, na=False)
    return student_logs.loc[component_match | context_match].copy()


def estimate_focused_start(timestamps: pd.Series, exam: datetime) -> tuple[pd.Timestamp, pd.Timestamp, int]:
    times = pd.Series(pd.to_datetime(timestamps, errors="coerce")).dropna().sort_values().reset_index(drop=True)
    times = times[times < exam].reset_index(drop=True)
    if times.empty:
        raise ValueError("Cannot estimate a focused-learning window without pre-assessment material activity")

    duration = pd.Timedelta(days=ROLLING_WINDOW_DAYS)
    candidates = []
    for start in times:
        count = int(((times >= start) & (times < start + duration)).sum())
        candidates.append((count, start))
    best_count, best_window_start = max(candidates, key=lambda item: (item[0], -item[1].value))
    selected = times[(times >= best_window_start) & (times < best_window_start + duration)]
    daily_counts = selected.groupby(selected.dt.date).size()
    highest_count = daily_counts.max()
    highest_day = min(day for day, count in daily_counts.items() if count == highest_count)
    focused_start = times[times.dt.date == highest_day].min()
    return pd.Timestamp(focused_start), pd.Timestamp(best_window_start), int(best_count)


def homework_features(student_logs: pd.DataFrame) -> tuple[float, int]:
    assignment_mask = student_logs["component"].str.lower().str.contains(r"assignment|zada", regex=True, na=False)
    assignments = student_logs.loc[assignment_mask]
    total_hours = 0.0
    status_views_after_submission = 0
    for _, group in assignments.groupby("context"):
        ordered = group.sort_values("timestamp")
        if ordered.empty:
            continue
        submitted = ordered[ordered["event_name"].str.lower().str.contains(r"submitted|predana", regex=True, na=False)]
        submit_time = submitted["timestamp"].iloc[0] if not submitted.empty else ordered["timestamp"].iloc[-1]
        total_hours += max(0.0, (submit_time - ordered["timestamp"].iloc[0]).total_seconds() / 3600)
        after = ordered[ordered["timestamp"] > submit_time]
        status_views_after_submission += int(
            after["event_name"].str.lower().str.contains(r"status viewed|stanje predane", regex=True, na=False).sum()
        )
    return float(total_hours), int(status_views_after_submission)


def extract_behavioral_features(student_logs: pd.DataFrame, exam: datetime) -> dict[str, float]:
    before_exam = student_logs[student_logs["timestamp"] < exam].copy()
    materials = material_events(before_exam)
    if materials.empty:
        raise ValueError("No learning-material events")

    focused_start, best_window_start, best_window_count = estimate_focused_start(materials["timestamp"], exam)
    all_times = before_exam["timestamp"].tolist()
    material_times = materials["timestamp"].tolist()
    focused_times = [timestamp for timestamp in material_times if focused_start <= timestamp < exam]
    sessions = split_sessions(all_times)
    session_parts = session_day_part_counts(sessions)
    activity_parts = day_part_counts(all_times)
    focused_parts = day_part_counts(focused_times)
    mean_gap, median_gap, mode_gap = session_gap_statistics(sessions)
    days_before = max((pd.Timestamp(exam) - focused_start).total_seconds() / 86400, 1 / 24)
    hours_before = days_before * 24
    all_activity = len(all_times)
    focused_activity = len(focused_times)
    weekend_total = int(sum(timestamp.weekday() >= 5 for timestamp in all_times))
    homework_time, homework_count = homework_features(before_exam)

    quiz_mask = before_exam["component"].str.lower().str.contains(r"quiz|test", regex=True, na=False)
    quizzes = before_exam.loc[quiz_mask]
    quiz_total = len(quizzes)
    quiz_focused = int((quizzes["timestamp"] >= focused_start).sum()) if quiz_total else 0
    quiz_earliest_h = float((pd.Timestamp(exam) - quizzes["timestamp"].min()).total_seconds() / 3600) if quiz_total else 0.0

    row: dict[str, float] = {
        "all_activity_sum": float(all_activity),
        "session_Sum": float(len(sessions)),
        "mean_periodicity_h": mean_gap,
        "median_periodicity_h": median_gap,
        "mode_periodicity_h": mode_gap,
        "start_learning_before_exam_d": days_before,
        "start_learning_before_exam_h": hours_before,
        "activities_during_learning": float(focused_activity),
        "best_learning_window_activity_count": float(best_window_count),
        "weekend_total": float(weekend_total),
        "homework_time_total": homework_time,
        "homework_count_total": float(homework_count),
        "attempt_quizes_data_start_learning_date_count": float(quiz_focused),
        "time_diff_earliest_quiz_attempt_h": quiz_earliest_h,
        "count_attempts_quiz": float(quiz_total),
        "focused_start_timestamp": focused_start,
        "best_window_start_timestamp": best_window_start,
    }
    for part in DAY_PARTS:
        row[f"activity_{part}"] = float(activity_parts[part])
        row[f"session_{part}"] = float(session_parts[part])
        row[f"activity_start_learning_before_exam_{part}"] = float(focused_parts[part])

    row["activity_per_learning_day"] = focused_activity / days_before
    row["activity_per_session"] = focused_activity / max(len(sessions), 1)
    row["regularity_inverse_mean_periodicity"] = 1.0 / (mean_gap + 1.0)
    row["regularity_inverse_median_periodicity"] = 1.0 / (median_gap + 1.0)
    row["learning_activity_ratio"] = focused_activity / max(all_activity, 1)
    return row


def load_weather(path: str | Path) -> pd.DataFrame:
    weather = read_csv(path)
    if "date" not in weather.columns:
        raise ValueError("Weather file must contain a 'date' column")
    weather["date"] = pd.to_datetime(weather["date"], errors="coerce").dt.normalize()
    weather = weather.dropna(subset=["date"])
    for column in WEATHER_COLUMNS:
        if column in weather.columns:
            weather[column] = pd.to_numeric(weather[column], errors="coerce")
    return weather.sort_values("date")


def add_environment_and_interactions(row: dict, weather: pd.DataFrame, exam: datetime) -> dict:
    start = pd.Timestamp(row["focused_start_timestamp"]).normalize()
    end = pd.Timestamp(exam).normalize()
    interval = weather[(weather["date"] >= start) & (weather["date"] <= end)]
    for column in WEATHER_COLUMNS:
        if column not in weather.columns:
            continue
        mean_value = float(interval[column].mean()) if not interval.empty else np.nan
        weather_name = f"learning_window_avg_{column}"
        row[weather_name] = mean_value
        row[f"activities_x_{weather_name}"] = row["activities_during_learning"] * mean_value
        row[f"activity_per_day_x_{weather_name}"] = row["activity_per_learning_day"] * mean_value
        row[f"sessions_x_{weather_name}"] = row["session_Sum"] * mean_value
        row[f"regularity_x_{weather_name}"] = row["regularity_inverse_mean_periodicity"] * mean_value
    return row


def build_year_dataset(
    log_path: str | Path,
    grade_path: str | Path,
    weather: pd.DataFrame,
    exam: datetime,
    year: int,
) -> pd.DataFrame:
    logs = normalize_logs(read_csv(log_path))
    grades = normalize_grades(read_csv(grade_path))
    rows = []
    for student_id, group in logs.groupby("student_id"):
        try:
            features = extract_behavioral_features(group, exam)
        except ValueError:
            continue
        features["student_id"] = student_id
        features["year"] = int(year)
        rows.append(add_environment_and_interactions(features, weather, exam))

    frame = pd.DataFrame(rows)
    frame = frame.merge(grades, on="student_id", how="inner", validate="one_to_one")
    frame = frame.drop(columns=["focused_start_timestamp", "best_window_start_timestamp"], errors="ignore")
    frame = coerce_numeric_frame(frame)
    numeric = [column for column in frame.select_dtypes(include=[np.number]).columns if column not in ("year", "kolokvij1")]
    constant = [column for column in numeric if frame[column].nunique(dropna=False) <= 1]
    return frame.drop(columns=constant, errors="ignore")


def build_combined_dataset(
    *,
    logs_by_year: dict[int, str | Path],
    grades_by_year: dict[int, str | Path],
    exams_by_year: dict[int, datetime],
    weather_path: str | Path,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    years = sorted(logs_by_year)
    if set(years) != set(grades_by_year) or set(years) != set(exams_by_year):
        raise ValueError("Logs, grades, and exam mappings must contain the same years")
    weather = load_weather(weather_path)
    frames = [
        build_year_dataset(logs_by_year[year], grades_by_year[year], weather, exams_by_year[year], year)
        for year in years
    ]
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.sort_values(["year", "student_id"]).reset_index(drop=True)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(output_path, index=False, encoding="utf-8-sig")
    return combined
