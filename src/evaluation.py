from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from xgboost import XGBClassifier, XGBRegressor

from .common import PASS_THRESHOLD, RANDOM_STATE, clean_feature_matrix, feature_sets, interaction_columns, read_csv, weather_columns


@dataclass(frozen=True)
class EvaluationConfig:
    random_state: int = RANDOM_STATE
    folds: int = 5
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
    required = {"student_id", "year", "kolokvij1"}
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
    features = frame.drop(columns=["student_id", "year", "kolokvij1", "pass_fail"], errors="ignore")
    for column in features.columns:
        features[column] = pd.to_numeric(features[column], errors="coerce")
    features = features.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
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


def classification_models(config: EvaluationConfig, target: pd.Series):
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
            scale_pos_weight=class_weight_ratio(target),
            random_state=config.random_state,
            n_jobs=-1,
            verbosity=0,
        ),
    }


def evaluate_cv(frame: pd.DataFrame, config: EvaluationConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    features, y_reg, y_clf = prepare_features(frame)
    cleaned_sets: dict[str, pd.DataFrame] = {}
    regression_rows = []
    classification_rows = []
    reg_cv = KFold(n_splits=config.folds, shuffle=True, random_state=config.random_state)
    clf_cv = StratifiedKFold(n_splits=config.folds, shuffle=True, random_state=config.random_state)
    regression_scoring = {
        "r2": "r2",
        "rmse": make_scorer(rmse, greater_is_better=False),
        "mae": "neg_mean_absolute_error",
    }
    classification_scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
        "roc_auc": "roc_auc",
    }

    for set_name, set_frame in feature_sets(features).items():
        clean, removed = clean_feature_matrix(set_frame)
        cleaned_sets[set_name] = clean
        for model_name, model in regression_models(config).items():
            scores = cross_validate(model, clean, y_reg, cv=reg_cv, scoring=regression_scoring, n_jobs=-1, error_score="raise")
            regression_rows.append(
                {
                    "Feature set": set_name,
                    "Model": model_name,
                    "N_samples": len(y_reg),
                    "N_features": clean.shape[1],
                    "R2_mean": float(np.mean(scores["test_r2"])),
                    "R2_std": float(np.std(scores["test_r2"])),
                    "RMSE_mean": float(-np.mean(scores["test_rmse"])),
                    "RMSE_std": float(np.std(-scores["test_rmse"])),
                    "MAE_mean": float(-np.mean(scores["test_mae"])),
                    "MAE_std": float(np.std(-scores["test_mae"])),
                    "Removed_correlated": ";".join(removed),
                }
            )
        for model_name, model in classification_models(config, y_clf).items():
            scores = cross_validate(model, clean, y_clf, cv=clf_cv, scoring=classification_scoring, n_jobs=-1, error_score="raise")
            classification_rows.append(
                {
                    "Feature set": set_name,
                    "Model": model_name,
                    "N_samples": len(y_clf),
                    "N_features": clean.shape[1],
                    "N_fail": int((y_clf == 0).sum()),
                    "N_pass": int((y_clf == 1).sum()),
                    "Accuracy_mean": float(np.mean(scores["test_accuracy"])),
                    "Accuracy_std": float(np.std(scores["test_accuracy"])),
                    "Balanced_Accuracy_mean": float(np.mean(scores["test_balanced_accuracy"])),
                    "Balanced_Accuracy_std": float(np.std(scores["test_balanced_accuracy"])),
                    "F1_mean": float(np.mean(scores["test_f1"])),
                    "F1_std": float(np.std(scores["test_f1"])),
                    "Precision_mean": float(np.mean(scores["test_precision"])),
                    "Recall_mean": float(np.mean(scores["test_recall"])),
                    "ROC_AUC_mean": float(np.mean(scores["test_roc_auc"])),
                    "ROC_AUC_std": float(np.std(scores["test_roc_auc"])),
                    "Removed_correlated": ";".join(removed),
                }
            )
    return pd.DataFrame(regression_rows), pd.DataFrame(classification_rows), cleaned_sets


def evaluate_cross_year(frame: pd.DataFrame, config: EvaluationConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    regression_rows = []
    classification_rows = []
    years = sorted(frame["year"].unique())
    for train_year in years:
        for test_year in years:
            if train_year == test_year:
                continue
            train = frame[frame["year"] == train_year].copy()
            test = frame[frame["year"] == test_year].copy()
            x_train_all, y_train_reg, y_train_clf = prepare_features(train)
            x_test_all, y_test_reg, y_test_clf = prepare_features(test)
            common = sorted(set(x_train_all.columns) & set(x_test_all.columns))
            train_sets = feature_sets(x_train_all[common])
            test_sets = feature_sets(x_test_all[common])
            for set_name in train_sets:
                x_train, _ = clean_feature_matrix(train_sets[set_name])
                x_test = test_sets[set_name].reindex(columns=x_train.columns, fill_value=0).replace([np.inf, -np.inf], np.nan).fillna(0)
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
                            "N_features": x_train.shape[1],
                            "R2": float(r2_score(y_test_reg, prediction)),
                            "RMSE": rmse(y_test_reg, prediction),
                            "MAE": float(mean_absolute_error(y_test_reg, prediction)),
                        }
                    )
                for model_name, model in classification_models(config, y_train_clf).items():
                    model.fit(x_train, y_train_clf)
                    prediction = model.predict(x_test)
                    probability = model.predict_proba(x_test)[:, 1] if hasattr(model, "predict_proba") else None
                    classification_rows.append(
                        {
                            "Train_year": int(train_year),
                            "Test_year": int(test_year),
                            "Feature set": set_name,
                            "Model": model_name,
                            "N_train": len(train),
                            "N_test": len(test),
                            "N_features": x_train.shape[1],
                            "Accuracy": float(accuracy_score(y_test_clf, prediction)),
                            "Balanced_Accuracy": float(balanced_accuracy_score(y_test_clf, prediction)),
                            "F1": float(f1_score(y_test_clf, prediction, zero_division=0)),
                            "Precision": float(precision_score(y_test_clf, prediction, zero_division=0)),
                            "Recall": float(recall_score(y_test_clf, prediction, zero_division=0)),
                            "ROC_AUC": float(roc_auc_score(y_test_clf, probability)) if probability is not None else np.nan,
                        }
                    )
    return pd.DataFrame(regression_rows), pd.DataFrame(classification_rows)


def cross_validated_rf_auc(features: pd.DataFrame, target: pd.Series, config: EvaluationConfig) -> float:
    clean, _ = clean_feature_matrix(features)
    model = classification_models(config, target)["Random Forest"]
    cv = StratifiedKFold(n_splits=config.folds, shuffle=True, random_state=config.random_state)
    scores = cross_validate(model, clean, target, cv=cv, scoring="roc_auc", n_jobs=-1, error_score="raise")
    return float(np.mean(scores["test_score"]))


def environmental_permutation_test(frame: pd.DataFrame, config: EvaluationConfig) -> pd.DataFrame:
    features, _, target = prepare_features(frame)
    environment = weather_columns(features.columns) + interaction_columns(features.columns)
    full = feature_sets(features)["Behavior + weather + interactions"]
    observed = cross_validated_rf_auc(full, target, config)
    generator = np.random.default_rng(config.random_state)
    permuted = []
    for _ in range(config.weather_permutations):
        shuffled = features.copy()
        order = generator.permutation(len(shuffled))
        shuffled.loc[:, environment] = shuffled.loc[:, environment].iloc[order].to_numpy()
        permuted.append(cross_validated_rf_auc(feature_sets(shuffled)["Behavior + weather + interactions"], target, config))
    permuted_array = np.asarray(permuted)
    p_value = (int((permuted_array >= observed).sum()) + 1) / (len(permuted_array) + 1)
    return pd.DataFrame(
        [
            {
                "Observed_AUC": observed,
                "Permuted_AUC_mean": float(permuted_array.mean()),
                "Permuted_AUC_std": float(permuted_array.std()),
                "Permuted_AUC_min": float(permuted_array.min()),
                "Permuted_AUC_max": float(permuted_array.max()),
                "N_permutations": len(permuted_array),
                "p_value_right_tail": float(p_value),
            }
        ]
    )


def importance_outputs(frame: pd.DataFrame, config: EvaluationConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features, y_reg, y_clf = prepare_features(frame)
    full, _ = clean_feature_matrix(feature_sets(features)["Behavior + weather + interactions"])
    classifier = classification_models(config, y_clf)["Random Forest"]
    regressor = regression_models(config)["Random Forest"]
    classifier.fit(full, y_clf)
    regressor.fit(full, y_reg)
    clf_perm = permutation_importance(
        classifier,
        full,
        y_clf,
        scoring="roc_auc",
        n_repeats=config.importance_repeats,
        random_state=config.random_state,
        n_jobs=-1,
    )
    reg_perm = permutation_importance(
        regressor,
        full,
        y_reg,
        scoring="r2",
        n_repeats=config.importance_repeats,
        random_state=config.random_state,
        n_jobs=-1,
    )
    permutation_frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "Task": "classification",
                    "Feature": full.columns,
                    "Importance_mean": clf_perm.importances_mean,
                    "Importance_std": clf_perm.importances_std,
                }
            ),
            pd.DataFrame(
                {
                    "Task": "regression",
                    "Feature": full.columns,
                    "Importance_mean": reg_perm.importances_mean,
                    "Importance_std": reg_perm.importances_std,
                }
            ),
        ],
        ignore_index=True,
    ).sort_values(["Task", "Importance_mean"], ascending=[True, False])

    rf_importance = pd.DataFrame({"Feature": full.columns, "Importance": regressor.feature_importances_}).sort_values("Importance", ascending=False)
    xgb = regression_models(config)["XGBoost"]
    xgb.fit(full, y_reg)
    xgb_importance = pd.DataFrame({"Feature": full.columns, "Importance": xgb.feature_importances_}).sort_values("Importance", ascending=False)
    return permutation_frame, rf_importance, xgb_importance


def run_evaluation(dataset_path: str | Path, output_dir: str | Path, config: EvaluationConfig) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_dataset(dataset_path)
    cv_reg, cv_clf, _ = evaluate_cv(frame, config)
    cross_reg, cross_clf = evaluate_cross_year(frame, config)
    weather_test = environmental_permutation_test(frame, config)
    permutation_frame, rf_importance, xgb_importance = importance_outputs(frame, config)

    outputs = {
        "cv_regression_results.csv": cv_reg,
        "cv_classification_results.csv": cv_clf,
        "cross_year_regression_results.csv": cross_reg,
        "cross_year_classification_results.csv": cross_clf,
        "weather_permutation_test.csv": weather_test,
        "permutation_importance.csv": permutation_frame,
        "model_importance_random_forest.csv": rf_importance,
        "model_importance_xgboost.csv": xgb_importance,
    }
    for filename, result in outputs.items():
        result.to_csv(output_dir / filename, index=False, encoding="utf-8-sig")

    run_metadata = {
        "synthetic_results_notice": "When run on data_dummy, these outputs are smoke-test results and are not findings reported in the manuscript.",
        "dataset": str(Path(dataset_path)),
        "rows": len(frame),
        "pass_fail_distribution": frame["pass_fail"].value_counts().sort_index().to_dict(),
        "config": asdict(config),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
