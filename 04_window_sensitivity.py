"""Evaluate focused-learning-window sensitivity for k = 3, 5, 7, and 10 days."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import balanced_accuracy_score, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

from src.common import RANDOM_STATE, feature_sets, fit_feature_pruner, transform_feature_matrix
from src.evaluation import EvaluationConfig, load_dataset, prepare_features
from src.feature_engineering import build_combined_dataset


DEFAULT_WINDOWS = (3, 5, 7, 10)


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
    parser.add_argument("--output-dir", type=Path, default=Path("window_sensitivity"))
    parser.add_argument("--windows", type=int, nargs="+", default=list(DEFAULT_WINDOWS))
    parser.add_argument("--quick", action="store_true", help="Use 80 RF trees for a fast smoke test")
    return parser.parse_args()


def mean_ci(values: list[float]) -> tuple[float, float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    if len(values) <= 1:
        return mean, std, mean, mean
    low, high = stats.t.interval(0.95, df=len(values) - 1, loc=mean, scale=stats.sem(values))
    return mean, std, float(low), float(high)


def evaluate_one_window(frame: pd.DataFrame, k: int, trees: int) -> tuple[dict, list[dict]]:
    features, y_reg, y_clf = prepare_features(frame)
    full = feature_sets(features)["Behavior + weather + interactions"]

    reg_splits = list(KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE).split(full, y_reg))
    clf_splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE).split(full, y_clf))

    r2_values: list[float] = []
    rmse_values: list[float] = []
    auc_values: list[float] = []
    bal_values: list[float] = []
    importance_rows: list[dict] = []

    for fold, (train_idx, test_idx) in enumerate(reg_splits, 1):
        pruner = fit_feature_pruner(full.iloc[train_idx])
        x_train = transform_feature_matrix(full.iloc[train_idx], pruner)
        x_test = transform_feature_matrix(full.iloc[test_idx], pruner)
        model = RandomForestRegressor(
            n_estimators=trees,
            max_depth=10,
            min_samples_leaf=4,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(x_train, y_reg.iloc[train_idx])
        prediction = model.predict(x_test)
        r2_values.append(float(r2_score(y_reg.iloc[test_idx], prediction)))
        rmse_values.append(float(np.sqrt(mean_squared_error(y_reg.iloc[test_idx], prediction))))
        for feature, importance in zip(pruner.selected_columns, model.feature_importances_):
            importance_rows.append(
                {"k_days": k, "Task": "regression", "Fold": fold, "Feature": feature, "Importance": float(importance)}
            )

    for fold, (train_idx, test_idx) in enumerate(clf_splits, 1):
        pruner = fit_feature_pruner(full.iloc[train_idx])
        x_train = transform_feature_matrix(full.iloc[train_idx], pruner)
        x_test = transform_feature_matrix(full.iloc[test_idx], pruner)
        model = RandomForestClassifier(
            n_estimators=trees,
            max_depth=10,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(x_train, y_clf.iloc[train_idx])
        prediction = model.predict(x_test)
        probability = model.predict_proba(x_test)[:, 1]
        auc_values.append(float(roc_auc_score(y_clf.iloc[test_idx], probability)))
        bal_values.append(float(balanced_accuracy_score(y_clf.iloc[test_idx], prediction)))
        for feature, importance in zip(pruner.selected_columns, model.feature_importances_):
            importance_rows.append(
                {"k_days": k, "Task": "classification", "Fold": fold, "Feature": feature, "Importance": float(importance)}
            )

    r2_mean, r2_std, r2_low, r2_high = mean_ci(r2_values)
    rmse_mean, rmse_std, _, _ = mean_ci(rmse_values)
    auc_mean, auc_std, auc_low, auc_high = mean_ci(auc_values)
    bal_mean, bal_std, _, _ = mean_ci(bal_values)

    result = {
        "k_days": k,
        "N_students": len(frame),
        "R2_mean": r2_mean,
        "R2_std": r2_std,
        "R2_CI95_low": r2_low,
        "R2_CI95_high": r2_high,
        "RMSE_mean": rmse_mean,
        "RMSE_std": rmse_std,
        "ROC_AUC_mean": auc_mean,
        "ROC_AUC_std": auc_std,
        "ROC_AUC_CI95_low": auc_low,
        "ROC_AUC_CI95_high": auc_high,
        "Balanced_Accuracy_mean": bal_mean,
        "Balanced_Accuracy_std": bal_std,
    }
    return result, importance_rows


def aggregate_rankings(importance: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (k, task), group in importance.groupby(["k_days", "Task"]):
        features = sorted(group["Feature"].unique())
        matrix = pd.DataFrame(0.0, index=sorted(group["Fold"].unique()), columns=features)
        for _, row in group.iterrows():
            matrix.loc[row["Fold"], row["Feature"]] = row["Importance"]
        means = matrix.mean(axis=0).sort_values(ascending=False)
        for rank, (feature, value) in enumerate(means.items(), 1):
            rows.append(
                {"k_days": int(k), "Task": task, "Rank": rank, "Feature": feature, "Mean_RF_importance": float(value)}
            )
    return pd.DataFrame(rows)


def ranking_stability(rankings: pd.DataFrame, reference_k: int = 5, top_n: int = 10) -> pd.DataFrame:
    rows: list[dict] = []
    for task in sorted(rankings["Task"].unique()):
        reference = rankings[rankings["Task"].eq(task) & rankings["k_days"].eq(reference_k)]
        for k in sorted(rankings["k_days"].unique()):
            if k == reference_k:
                continue
            current = rankings[rankings["Task"].eq(task) & rankings["k_days"].eq(k)]
            common = reference[["Feature", "Rank"]].merge(
                current[["Feature", "Rank"]], on="Feature", suffixes=("_ref", "_k")
            )
            rho, p_value = stats.spearmanr(common["Rank_ref"], common["Rank_k"])
            top_ref = set(reference.nsmallest(top_n, "Rank")["Feature"])
            top_k = set(current.nsmallest(top_n, "Rank")["Feature"])
            shared = len(top_ref & top_k)
            union = len(top_ref | top_k)
            rows.append(
                {
                    "Task": task,
                    "Reference_k": reference_k,
                    "Compared_k": int(k),
                    "N_common_features": len(common),
                    "Spearman_rank_rho": float(rho),
                    "Spearman_rank_p": float(p_value),
                    f"Top{top_n}_shared_features": shared,
                    f"Top{top_n}_overlap_fraction": shared / top_n,
                    f"Top{top_n}_Jaccard": shared / union if union else np.nan,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trees = 80 if args.quick else 800

    performance_rows: list[dict] = []
    importance_rows: list[dict] = []

    for k in args.windows:
        raw = build_combined_dataset(
            logs_by_year={2023: args.logs_2023, 2024: args.logs_2024},
            grades_by_year={2023: args.grades_2023, 2024: args.grades_2024},
            exams_by_year={2023: args.exam_2023, 2024: args.exam_2024},
            weather_path=args.weather,
            window_days=int(k),
            output_path=None,
        )
        raw["pass_fail"] = (pd.to_numeric(raw["kolokvij1"], errors="coerce") >= 12).astype(int)
        result, importance = evaluate_one_window(raw, int(k), trees)
        performance_rows.append(result)
        importance_rows.extend(importance)
        print(f"Finished k={k}: N={result['N_students']}, R2={result['R2_mean']:.3f}, AUC={result['ROC_AUC_mean']:.3f}")

    performance = pd.DataFrame(performance_rows).sort_values("k_days")
    importance = pd.DataFrame(importance_rows)
    rankings = aggregate_rankings(importance)
    stability = ranking_stability(rankings, reference_k=5, top_n=10)

    performance.to_csv(args.output_dir / "window_sensitivity_performance.csv", index=False)
    rankings.to_csv(args.output_dir / "window_feature_ranking.csv", index=False)
    stability.to_csv(args.output_dir / "window_rank_stability.csv", index=False)
    print(f"Sensitivity outputs written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
