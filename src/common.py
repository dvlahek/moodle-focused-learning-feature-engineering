from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_STATE = 42
PASS_THRESHOLD = 12.0
SESSION_GAP_MINUTES = 50
ROLLING_WINDOW_DAYS = 5
CORRELATION_THRESHOLD = 0.95

# Environmental variables used by the study pipeline. Columns that are absent
# from a particular input file are simply skipped during feature construction.
WEATHER_COLUMNS = [
    "pm10",
    "pm2_5",
    "tavg",
    "tmin",
    "tmax",
    "prcp",
    "snow",
    "wdir",
    "wspd",
    "wpgt",
    "pres",
]

INTERACTION_PREFIXES = (
    "activities_x_learning_window_avg_",
    "activity_per_day_x_learning_window_avg_",
    "sessions_x_learning_window_avg_",
    "regularity_x_learning_window_avg_",
)


@dataclass(frozen=True)
class FeaturePruner:
    """Training-fitted zero-variance and correlation pruner."""

    selected_columns: tuple[str, ...]
    removed_constant: tuple[str, ...]
    removed_correlated: tuple[str, ...]
    correlation_threshold: float = CORRELATION_THRESHOLD


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
    """Convert object columns to numeric when possible and normalize invalid values."""
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


def _numeric_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    clean = coerce_numeric_frame(frame)
    return (
        clean.select_dtypes(include=[np.number])
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )


def fit_feature_pruner(
    frame: pd.DataFrame,
    *,
    correlation_threshold: float = CORRELATION_THRESHOLD,
) -> FeaturePruner:
    """Fit feature pruning using one training partition only.

    The returned object stores the selected columns. The same selection can then
    be applied to the held-out partition with :func:`transform_feature_matrix`.
    This avoids using test-fold information during preprocessing.
    """
    clean = _numeric_matrix(frame)
    constant = [
        column for column in clean.columns
        if clean[column].nunique(dropna=False) <= 1
    ]
    clean = clean.drop(columns=constant, errors="ignore")

    correlated: list[str] = []
    if clean.shape[1] > 1:
        correlation = clean.corr().abs()
        upper = correlation.where(
            np.triu(np.ones(correlation.shape), k=1).astype(bool)
        )
        correlated = [
            column for column in upper.columns
            if (upper[column] > correlation_threshold).any()
        ]

    selected = [column for column in clean.columns if column not in correlated]
    return FeaturePruner(
        selected_columns=tuple(selected),
        removed_constant=tuple(constant),
        removed_correlated=tuple(correlated),
        correlation_threshold=float(correlation_threshold),
    )


def transform_feature_matrix(frame: pd.DataFrame, pruner: FeaturePruner) -> pd.DataFrame:
    """Apply a training-fitted pruner to any partition."""
    clean = _numeric_matrix(frame)
    return clean.reindex(columns=list(pruner.selected_columns), fill_value=0.0)


def clean_feature_matrix(
    frame: pd.DataFrame,
    *,
    correlation_threshold: float = CORRELATION_THRESHOLD,
) -> tuple[pd.DataFrame, list[str]]:
    """Compatibility helper for descriptive, non-CV use only.

    For validation/evaluation, use ``fit_feature_pruner`` on the training data
    and ``transform_feature_matrix`` on both training and held-out data.
    """
    pruner = fit_feature_pruner(frame, correlation_threshold=correlation_threshold)
    clean = transform_feature_matrix(frame, pruner)
    removed = list(pruner.removed_constant) + list(pruner.removed_correlated)
    return clean, removed
