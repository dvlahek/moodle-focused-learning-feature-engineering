from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_STATE = 42
PASS_THRESHOLD = 12.0
SESSION_GAP_MINUTES = 50
ROLLING_WINDOW_DAYS = 5

WEATHER_COLUMNS = [
    "pm10",
    "pm2_5",
    "tavg",
    "tmin",
    "tmax",
    "prcp",
    "wdir",
    "wspd",
]

INTERACTION_PREFIXES = (
    "activities_x_learning_window_avg_",
    "activity_per_day_x_learning_window_avg_",
    "sessions_x_learning_window_avg_",
    "regularity_x_learning_window_avg_",
)


def read_csv(path: str | Path) -> pd.DataFrame:
    """Read a CSV using common encodings used by Moodle exports."""
    path = Path(path)
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp1250", "windows-1252", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise ValueError(f"Could not decode {path}: {last_error}")


def coerce_numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert object columns to numeric when possible and replace invalid values."""
    result = frame.copy()
    for column in result.columns:
        if result[column].dtype == "object":
            converted = (
                result[column]
                .astype(str)
                .str.replace(",", ".", regex=False)
                .str.replace(" ", "", regex=False)
            )
            numeric = pd.to_numeric(converted, errors="coerce")
            if numeric.notna().any():
                result[column] = numeric
    return result.replace([np.inf, -np.inf], np.nan)


def weather_columns(columns) -> list[str]:
    return [column for column in columns if column.startswith("learning_window_avg_")]


def interaction_columns(columns) -> list[str]:
    return [column for column in columns if column.startswith(INTERACTION_PREFIXES)]


def feature_sets(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    weather = weather_columns(frame.columns)
    interactions = interaction_columns(frame.columns)
    return {
        "Behavior only": frame.drop(columns=weather + interactions, errors="ignore"),
        "Behavior + weather": frame.drop(columns=interactions, errors="ignore"),
        "Behavior + weather + interactions": frame.copy(),
    }


def clean_feature_matrix(
    frame: pd.DataFrame,
    *,
    correlation_threshold: float = 0.95,
) -> tuple[pd.DataFrame, list[str]]:
    """Remove nonnumeric, constant, and highly correlated columns."""
    clean = coerce_numeric_frame(frame)
    clean = clean.select_dtypes(include=[np.number]).fillna(0.0)
    clean = clean.loc[:, clean.nunique(dropna=False) > 1]
    removed: list[str] = []
    if clean.shape[1] > 1:
        correlation = clean.corr().abs()
        upper = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool))
        removed = [column for column in upper.columns if (upper[column] > correlation_threshold).any()]
        clean = clean.drop(columns=removed, errors="ignore")
    return clean, removed

