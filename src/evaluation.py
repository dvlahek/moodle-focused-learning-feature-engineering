from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from xgboost import XGBClassifier, XGBRegressor

from .common import (
    CORRELATION_THRESHOLD,
    PASS_THRESHOLD,
    RANDOM_STATE,
    feature_sets,
    fit_feature_pruner,
    interaction_columns,
    read_csv,
    transform_feature_matrix,
    weather_columns,
)


@dataclass(frozen=True)
class EvaluationConfig:
    random_state: int = RANDOM_STATE
    folds: int = 5
    correlation_threshold: float = CORRELATION_THRESHOLD
    rf_estimators: int = 800
    xgb_reg_estimators: int = 800
    xgb_clf_estimators: int = 500
    weather_permutations: int = 300
    importance_repeats: int = 30
    quick_mode: bool = False

    @classmethod
    def paper(cls) -> "EvaluationConfig":
        return cls()

    @classmethod
    def quick(cls) -> "EvaluationConfig":
        return cls(
            rf_estimators=80,
            xgb_reg_estimators=80,
            xgb_clf_estimators=80,
            weather_permutations=10,
            importance_repeats=5,
            quick_mode=True,
        )


def load_dataset(path: str | Path) -> pd.DataFrame:
    frame = read_csv(path)
    id_column = "student_id" if "student_id" in frame.columns else None
    required = {"year", "kolokvij1"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Feature dataset is missing required columns: {sorted(missing)}")

    frame["kolokvij1"] = pd.to_numeric(frame["kolokvij1"], errors="coerce")
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
    frame = frame.dropna(subset=["kolokvij1", "year"]).copy()
    frame["pass_fail"] = (frame["kolokvij1"] >= PASS_THRESHOLD).astype(int)
    if frame["pass_fail"].nunique() != 2:
        raise ValueError("Both pass and fail classes are required")
    return frame


def prepare_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    regression_target = frame["kolokvij1"].astype(float)
    classification_target = frame["pass_fail"].astype(int)
    features = frame.drop(
        columns=["student_id", "name", "year", "kolokvij1", "pass_fail"],
        errors="ignore",
    ).copy()

    for column in features.columns:
        if features[column].dtype == "object":
            features[column] = pd.to_numeric(
                features[column]
                .astype(str)
                .str.replace(",", ".", regex=False)
                .str.replace(" ", "", regex=False),
                errors="coerce",
            )

    features = (
        features.select_dtypes(include=[np.number])
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    return features, regression_target, classification_target


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def class_weight_ratio(target: pd.Series) -> float:
    positive = int((target == 1).sum())
    negative = int((target == 0).sum())
    return negative / positive if positive else 1.0


def regression_models(config: EvaluationConfig):
    return {
        "Baseline mean": DummyRegressor(strategy="mean"),
        "Random Forest": RandomForestRegressor(
            n_estimators=config.rf_estimators,
            max_depth=10,
            min_samples_leaf=4,
            random_state=config.random_state,
            n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=config.xgb_reg_estimators,
            learning_rate=0.02,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=config.random_state,
            n_jobs=-1,
            verbosity=0,
        ),
    }


def classification_models(config: EvaluationConfig, training_target: pd.Series):
    """Create classifiers using training-label information only."""
    return {
        "Baseline majority": DummyClassifier(strategy="most_frequent"),
        "Random Forest": RandomForestClassifier(
            n_estimators=config.rf_estimators,
            max_depth=10,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=config.random_state,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=config.xgb_clf_estimators,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=class_weight_ratio(training_target),
            random_state=config.random_state,
            n_jobs=-1,
            verbosity=0,
        ),
    }


def _mean_std_ci(values: pd.Series | np.ndarray) -> tuple[float, float, float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=0))
    if len(array) <= 1 or np.isclose(std, 0.0):
        return mean, std, mean, mean
    low, high = stats.t.interval(
        0.95,
        df=len(array) - 1,
        loc=mean,
        scale=stats.sem(array),
    )
    return mean, std, float(low), float(high)


def _summarize_folds(
    frame: pd.DataFrame,
    metrics: list[str],
) -> pd.DataFrame:
    rows = []
    for (feature_set, model), group in frame.groupby(["Feature set", "Model"], sort=False):
        row = {
            "Feature set": feature_set,
            "Model": model,
            "N_samples": int(group["N_samples"].iloc[0]),
            "N_features_min": int(group["N_features"].min()),
            "N_features_max": int(group["N_features"].max()),
        }
        if row["N_features_min"] == row["N_features_max"]:
            row["N_features"] = row["N_features_min"]
        else:
            row["N_features"] = np.nan

        for metric in metrics:
            mean, std, low, high = _mean_std_ci(group[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_CI95_low"] = low
            row[f"{metric}_CI95_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_cv(
    frame: pd.DataFrame,
    config: EvaluationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Leakage-safe five-fold evaluation.

    Constant removal and correlation pruning are fitted independently inside
    each training fold. The resulting column selection is then applied to the
    corresponding held-out fold. XGBoost's ``scale_pos_weight`` is also
    calculated from training labels only.
    """
    features, y_reg, y_clf = prepare_features(frame)
    sets = feature_sets(features)

    reg_splits = list(
        KFold(
            n_splits=config.folds,
            shuffle=True,
            random_state=config.random_state,
        ).split(features, y_reg)
    )
    clf_splits = list(
        StratifiedKFold(
            n_splits=config.folds,
            shuffle=True,
            random_state=config.random_state,
        ).split(features, y_clf)
    )

    regression_rows: list[dict] = []
    classification_rows: list[dict] = []
    pruning_rows: list[dict] = []

    for set_name, set_frame in sets.items():
        for fold, (train_idx, test_idx) in enumerate(reg_splits, 1):
            train_raw = set_frame.iloc[train_idx]
            test_raw = set_frame.iloc[test_idx]
            pruner = fit_feature_pruner(
                train_raw,
                correlation_threshold=config.correlation_threshold,
            )
            x_train = transform_feature_matrix(train_raw, pruner)
            x_test = transform_feature_matrix(test_raw, pruner)
            y_train = y_reg.iloc[train_idx]
            y_test = y_reg.iloc[test_idx]

            pruning_rows.append(
                {
                    "Task": "regression",
                    "Feature set": set_name,
                    "Fold": fold,
                    "N_features_before": train_raw.shape[1],
                    "N_features_after": len(pruner.selected_columns),
                    "Removed_constant": ";".join(pruner.removed_constant),
                    "Removed_correlated": ";".join(pruner.removed_correlated),
                }
            )

            for model_name, model in regression_models(config).items():
                model.fit(x_train, y_train)
                prediction = model.predict(x_test)
                regression_rows.append(
                    {
                        "Feature set": set_name,
                        "Model": model_name,
                        "Fold": fold,
                        "N_samples": len(y_reg),
                        "N_features": len(pruner.selected_columns),
                        "R2": float(r2_score(y_test, prediction)),
                        "RMSE": rmse(y_test, prediction),
                        "MAE": float(mean_absolute_error(y_test, prediction)),
                    }
                )

        for fold, (train_idx, test_idx) in enumerate(clf_splits, 1):
            train_raw = set_frame.iloc[train_idx]
            test_raw = set_frame.iloc[test_idx]
            pruner = fit_feature_pruner(
                train_raw,
                correlation_threshold=config.correlation_threshold,
            )
            x_train = transform_feature_matrix(train_raw, pruner)
            x_test = transform_feature_matrix(test_raw, pruner)
            y_train = y_clf.iloc[train_idx]
            y_test = y_clf.iloc[test_idx]

            pruning_rows.append(
                {
                    "Task": "classification",
                    "Feature set": set_name,
                    "Fold": fold,
                    "N_features_before": train_raw.shape[1],
                    "N_features_after": len(pruner.selected_columns),
                    "Removed_constant": ";".join(pruner.removed_constant),
                    "Removed_correlated": ";".join(pruner.removed_correlated),
                }
            )

            for model_name, model in classification_models(config, y_train).items():
                model.fit(x_train, y_train)
                prediction = model.predict(x_test)
                probability = (
                    model.predict_proba(x_test)[:, 1]
                    if hasattr(model, "predict_proba")
                    else None
                )
                classification_rows.append(
                    {
                        "Feature set": set_name,
                        "Model": model_name,
                        "Fold": fold,
                        "N_samples": len(y_clf),
                        "N_features": len(pruner.selected_columns),
                        "N_fail": int((y_clf == 0).sum()),
                        "N_pass": int((y_clf == 1).sum()),
                        "Accuracy": float(accuracy_score(y_test, prediction)),
                        "Balanced_Accuracy": float(
                            balanced_accuracy_score(y_test, prediction)
                        ),
                        "F1": float(f1_score(y_test, prediction, zero_division=0)),
                        "Precision": float(
                            precision_score(y_test, prediction, zero_division=0)
                        ),
                        "Recall": float(
                            recall_score(y_test, prediction, zero_division=0)
                        ),
                        "ROC_AUC": (
                            float(roc_auc_score(y_test, probability))
                            if probability is not None
                            else 0.5
                        ),
                    }
                )

    reg_folds = pd.DataFrame(regression_rows)
    clf_folds = pd.DataFrame(classification_rows)
    pruning = pd.DataFrame(pruning_rows)
    reg_summary = _summarize_folds(reg_folds, ["R2", "RMSE", "MAE"])
    clf_summary = _summarize_folds(
        clf_folds,
        ["Accuracy", "Balanced_Accuracy", "F1", "Precision", "Recall", "ROC_AUC"],
    )
    return reg_summary, clf_summary, reg_folds, clf_folds, pruning


def paired_feature_set_tests(
    reg_folds: pd.DataFrame,
    clf_folds: pd.DataFrame,
) -> pd.DataFrame:
    comparisons = [
        ("Behavior only", "Behavior + weather"),
        ("Behavior only", "Behavior + weather + interactions"),
        ("Behavior + weather", "Behavior + weather + interactions"),
    ]
    rows: list[dict] = []

    for task, frame, metrics in (
        ("regression", reg_folds, ["R2", "RMSE", "MAE"]),
        ("classification", clf_folds, ["ROC_AUC", "Balanced_Accuracy", "F1"]),
    ):
        for model_name in ("Random Forest", "XGBoost"):
            model_frame = frame[frame["Model"] == model_name]
            for set_a, set_b in comparisons:
                a = model_frame[model_frame["Feature set"] == set_a].sort_values("Fold")
                b = model_frame[model_frame["Feature set"] == set_b].sort_values("Fold")
                merged = a.merge(b, on="Fold", suffixes=("_A", "_B"))

                for metric in metrics:
                    x = merged[f"{metric}_A"].to_numpy(float)
                    y = merged[f"{metric}_B"].to_numpy(float)
                    delta = y - x
                    delta_mean, _, low, high = _mean_std_ci(delta)
                    t_result = stats.ttest_rel(y, x, nan_policy="omit")
                    if np.allclose(delta, 0):
                        w_stat, w_p = 0.0, 1.0
                    else:
                        w_result = stats.wilcoxon(
                            y,
                            x,
                            zero_method="wilcox",
                            alternative="two-sided",
                        )
                        w_stat, w_p = float(w_result.statistic), float(w_result.pvalue)

                    rows.append(
                        {
                            "Task": task,
                            "Model": model_name,
                            "Metric": metric,
                            "Feature_set_A": set_a,
                            "Feature_set_B": set_b,
                            "N_paired_folds": len(delta),
                            "Mean_A": float(np.mean(x)),
                            "Mean_B": float(np.mean(y)),
                            "Mean_delta_B_minus_A": delta_mean,
                            "Delta_CI95_low": low,
                            "Delta_CI95_high": high,
                            "Paired_t_statistic": float(t_result.statistic),
                            "Paired_t_p": float(t_result.pvalue),
                            "Wilcoxon_statistic": w_stat,
                            "Wilcoxon_p": w_p,
                        }
                    )
    return pd.DataFrame(rows)


def evaluate_cross_year(
    frame: pd.DataFrame,
    config: EvaluationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    regression_rows: list[dict] = []
    classification_rows: list[dict] = []
    pruning_rows: list[dict] = []
    years = sorted(frame["year"].unique())

    for train_year in years:
        for test_year in years:
            if train_year == test_year:
                continue

            train = frame[frame["year"] == train_year].copy()
            test = frame[frame["year"] == test_year].copy()
            x_train_all, y_train_reg, y_train_clf = prepare_features(train)
            x_test_all, y_test_reg, y_test_clf = prepare_features(test)
            common = [
                column for column in x_train_all.columns
                if column in x_test_all.columns
            ]
            train_sets = feature_sets(x_train_all[common])
            test_sets = feature_sets(x_test_all[common])

            for set_name in train_sets:
                pruner = fit_feature_pruner(
                    train_sets[set_name],
                    correlation_threshold=config.correlation_threshold,
                )
                x_train = transform_feature_matrix(train_sets[set_name], pruner)
                x_test = transform_feature_matrix(test_sets[set_name], pruner)

                pruning_rows.append(
                    {
                        "Train_year": int(train_year),
                        "Test_year": int(test_year),
                        "Feature set": set_name,
                        "N_features_before": train_sets[set_name].shape[1],
                        "N_features_after": len(pruner.selected_columns),
                        "Removed_constant": ";".join(pruner.removed_constant),
                        "Removed_correlated": ";".join(pruner.removed_correlated),
                    }
                )

                for model_name, model in regression_models(config).items():
                    model.fit(x_train, y_train_reg)
                    prediction = model.predict(x_test)
                    regression_rows.append(
                        {
                            "Train_year": int(train_year),
                            "Test_year": int(test_year),
                            "Feature set": set_name,
                            "Model": model_name,
                            "N_train": len(train),
                            "N_test": len(test),
                            "N_features": len(pruner.selected_columns),
                            "R2": float(r2_score(y_test_reg, prediction)),
                            "RMSE": rmse(y_test_reg, prediction),
                            "MAE": float(mean_absolute_error(y_test_reg, prediction)),
                        }
                    )

                for model_name, model in classification_models(
                    config, y_train_clf
                ).items():
                    model.fit(x_train, y_train_clf)
                    prediction = model.predict(x_test)
                    probability = (
                        model.predict_proba(x_test)[:, 1]
                        if hasattr(model, "predict_proba")
                        else None
                    )
                    classification_rows.append(
                        {
                            "Train_year": int(train_year),
                            "Test_year": int(test_year),
                            "Feature set": set_name,
                            "Model": model_name,
                            "N_train": len(train),
                            "N_test": len(test),
                            "N_features": len(pruner.selected_columns),
                            "Accuracy": float(
                                accuracy_score(y_test_clf, prediction)
                            ),
                            "Balanced_Accuracy": float(
                                balanced_accuracy_score(y_test_clf, prediction)
                            ),
                            "F1": float(
                                f1_score(y_test_clf, prediction, zero_division=0)
                            ),
                            "Precision": float(
                                precision_score(
                                    y_test_clf, prediction, zero_division=0
                                )
                            ),
                            "Recall": float(
                                recall_score(y_test_clf, prediction, zero_division=0)
                            ),
                            "ROC_AUC": (
                                float(roc_auc_score(y_test_clf, probability))
                                if probability is not None
                                else 0.5
                            ),
                        }
                    )

    return (
        pd.DataFrame(regression_rows),
        pd.DataFrame(classification_rows),
        pd.DataFrame(pruning_rows),
    )


def _cross_validated_rf_auc(
    features: pd.DataFrame,
    target: pd.Series,
    config: EvaluationConfig,
) -> float:
    cv = StratifiedKFold(
        n_splits=config.folds,
        shuffle=True,
        random_state=config.random_state,
    )
    values = []
    for train_idx, test_idx in cv.split(features, target):
        train_raw = features.iloc[train_idx]
        test_raw = features.iloc[test_idx]
        pruner = fit_feature_pruner(
            train_raw,
            correlation_threshold=config.correlation_threshold,
        )
        x_train = transform_feature_matrix(train_raw, pruner)
        x_test = transform_feature_matrix(test_raw, pruner)
        y_train = target.iloc[train_idx]
        y_test = target.iloc[test_idx]

        model = classification_models(config, y_train)["Random Forest"]
        model.fit(x_train, y_train)
        probability = model.predict_proba(x_test)[:, 1]
        values.append(roc_auc_score(y_test, probability))
    return float(np.mean(values))


def environmental_permutation_test(
    frame: pd.DataFrame,
    config: EvaluationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features, _, target = prepare_features(frame)
    full = feature_sets(features)["Behavior + weather + interactions"]
    environment = list(
        dict.fromkeys(
            weather_columns(full.columns) + interaction_columns(full.columns)
        )
    )

    observed = _cross_validated_rf_auc(full, target, config)
    generator = np.random.default_rng(config.random_state)
    permuted: list[float] = []

    for _ in range(config.weather_permutations):
        shuffled = full.copy()
        order = generator.permutation(len(shuffled))
        shuffled.loc[:, environment] = (
            shuffled.loc[:, environment].iloc[order].to_numpy()
        )
        permuted.append(_cross_validated_rf_auc(shuffled, target, config))

    permuted_array = np.asarray(permuted, dtype=float)
    p_value = (
        int((permuted_array >= observed).sum()) + 1
    ) / (len(permuted_array) + 1)

    summary = pd.DataFrame(
        [
            {
                "Observed_AUC": observed,
                "Permuted_AUC_mean": float(permuted_array.mean()),
                "Permuted_AUC_std": float(permuted_array.std(ddof=0)),
                "Permuted_AUC_min": float(permuted_array.min()),
                "Permuted_AUC_max": float(permuted_array.max()),
                "N_permutations": len(permuted_array),
                "p_value_right_tail": float(p_value),
            }
        ]
    )
    distribution = pd.DataFrame(
        {
            "Permutation": np.arange(1, len(permuted_array) + 1),
            "Permuted_AUC": permuted_array,
        }
    )
    return summary, distribution


def importance_outputs(
    frame: pd.DataFrame,
    config: EvaluationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Held-out permutation importance and fold-averaged tree importances."""
    features, y_reg, y_clf = prepare_features(frame)
    full = feature_sets(features)["Behavior + weather + interactions"]
    reg_splits = list(
        KFold(
            n_splits=config.folds,
            shuffle=True,
            random_state=config.random_state,
        ).split(full, y_reg)
    )
    clf_splits = list(
        StratifiedKFold(
            n_splits=config.folds,
            shuffle=True,
            random_state=config.random_state,
        ).split(full, y_clf)
    )

    permutation_rows: list[dict] = []
    model_rows: list[dict] = []

    for task, target, splits in (
        ("regression", y_reg, reg_splits),
        ("classification", y_clf, clf_splits),
    ):
        for fold, (train_idx, test_idx) in enumerate(splits, 1):
            train_raw = full.iloc[train_idx]
            test_raw = full.iloc[test_idx]
            pruner = fit_feature_pruner(
                train_raw,
                correlation_threshold=config.correlation_threshold,
            )
            x_train = transform_feature_matrix(train_raw, pruner)
            x_test = transform_feature_matrix(test_raw, pruner)
            y_train = target.iloc[train_idx]
            y_test = target.iloc[test_idx]

            if task == "regression":
                rf_model = regression_models(config)["Random Forest"]
                scoring = "r2"
            else:
                rf_model = classification_models(config, y_train)["Random Forest"]
                scoring = "roc_auc"

            rf_model.fit(x_train, y_train)
            permutation = permutation_importance(
                rf_model,
                x_test,
                y_test,
                scoring=scoring,
                n_repeats=config.importance_repeats,
                random_state=config.random_state + fold + (100 if task == "classification" else 0),
                n_jobs=-1,
            )
            for feature, mean, std in zip(
                pruner.selected_columns,
                permutation.importances_mean,
                permutation.importances_std,
            ):
                permutation_rows.append(
                    {
                        "Task": task,
                        "Fold": fold,
                        "Feature": feature,
                        "Importance_mean": float(mean),
                        "Importance_std_within_fold": float(std),
                    }
                )

            # Model-based rankings are averaged across training folds rather than
            # estimated from a single model trained/evaluated on the full data.
            for feature, importance in zip(
                pruner.selected_columns, rf_model.feature_importances_
            ):
                model_rows.append(
                    {
                        "Task": task,
                        "Model": "Random Forest",
                        "Fold": fold,
                        "Feature": feature,
                        "Importance": float(importance),
                    }
                )

            if task == "regression":
                xgb = regression_models(config)["XGBoost"]
                xgb.fit(x_train, y_train)
                for feature, importance in zip(
                    pruner.selected_columns, xgb.feature_importances_
                ):
                    model_rows.append(
                        {
                            "Task": task,
                            "Model": "XGBoost",
                            "Fold": fold,
                            "Feature": feature,
                            "Importance": float(importance),
                        }
                    )

    permutation_by_fold = pd.DataFrame(permutation_rows)
    summary_rows = []
    for task, group in permutation_by_fold.groupby("Task"):
        features_union = sorted(group["Feature"].unique())
        matrix = pd.DataFrame(
            0.0,
            index=range(1, config.folds + 1),
            columns=features_union,
        )
        for _, row in group.iterrows():
            matrix.loc[int(row["Fold"]), row["Feature"]] = row["Importance_mean"]
        for feature in features_union:
            values = matrix[feature].to_numpy(float)
            summary_rows.append(
                {
                    "Task": task,
                    "Feature": feature,
                    "Importance_mean": float(values.mean()),
                    "Importance_std": float(values.std(ddof=0)),
                }
            )
    permutation_summary = pd.DataFrame(summary_rows).sort_values(
        ["Task", "Importance_mean"], ascending=[True, False]
    )

    model_frame = pd.DataFrame(model_rows)
    model_summary = (
        model_frame.groupby(["Task", "Model", "Feature"], as_index=False)["Importance"]
        .mean()
        .sort_values(["Task", "Model", "Importance"], ascending=[True, True, False])
    )
    rf_importance = model_summary[
        (model_summary["Task"] == "regression")
        & (model_summary["Model"] == "Random Forest")
    ][["Feature", "Importance"]].reset_index(drop=True)
    xgb_importance = model_summary[
        (model_summary["Task"] == "regression")
        & (model_summary["Model"] == "XGBoost")
    ][["Feature", "Importance"]].reset_index(drop=True)

    return permutation_summary, permutation_by_fold, rf_importance, xgb_importance


def run_evaluation(
    dataset_path: str | Path,
    output_dir: str | Path,
    config: EvaluationConfig,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_dataset(dataset_path)

    cv_reg, cv_clf, reg_folds, clf_folds, pruning = evaluate_cv(frame, config)
    paired = paired_feature_set_tests(reg_folds, clf_folds)
    cross_reg, cross_clf, cross_pruning = evaluate_cross_year(frame, config)
    weather_test, weather_distribution = environmental_permutation_test(frame, config)
    permutation_summary, permutation_by_fold, rf_importance, xgb_importance = importance_outputs(frame, config)

    outputs = {
        "cv_regression_results.csv": cv_reg,
        "cv_classification_results.csv": cv_clf,
        "cv_regression_foldwise.csv": reg_folds,
        "cv_classification_foldwise.csv": clf_folds,
        "cv_pruning_audit.csv": pruning,
        "paired_feature_set_tests.csv": paired,
        "cross_year_regression_results.csv": cross_reg,
        "cross_year_classification_results.csv": cross_clf,
        "cross_year_pruning_audit.csv": cross_pruning,
        "weather_permutation_test.csv": weather_test,
        "weather_permutation_distribution.csv": weather_distribution,
        "permutation_importance.csv": permutation_summary,
        "permutation_importance_by_fold.csv": permutation_by_fold,
        "model_importance_random_forest.csv": rf_importance,
        "model_importance_xgboost.csv": xgb_importance,
    }
    for name, output in outputs.items():
        output.to_csv(output_dir / name, index=False, encoding="utf-8-sig")

    metadata = asdict(config)
    metadata.update(
        {
            "dataset": str(Path(dataset_path)),
            "n_samples": int(len(frame)),
            "n_fail": int((frame["pass_fail"] == 0).sum()),
            "n_pass": int((frame["pass_fail"] == 1).sum()),
            "cv_pruning_fitted_on_training_fold_only": True,
            "cross_year_pruning_fitted_on_training_cohort_only": True,
            "xgboost_class_weight_uses_training_labels_only": True,
            "permutation_importance_evaluated_on_held_out_folds": True,
        }
    )
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
