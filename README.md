# Supplementary implementation package

This package accompanies the manuscript **“Interpretable Feature Engineering from Moodle Logs for Student Performance Prediction Using Focused Learning Windows and Environmental Context.”**

It provides a privacy-safe, executable implementation of:

- Moodle-log preprocessing;
- focused-learning-window estimation with a default five-day window;
- session construction using a 50-minute inactivity threshold;
- behavioral, temporal, assignment, quiz, environmental, and interaction features;
- three feature-set ablations;
- Random Forest and XGBoost regression and pass/fail classification;
- leakage-safe five-fold cross-validation;
- train-cohort-only cross-year validation;
- 95% confidence intervals and paired feature-set comparisons;
- environmental-alignment permutation testing;
- held-out permutation importance;
- model-based feature rankings averaged across training folds;
- focused-window sensitivity analysis for `k = 3, 5, 7, 10`;
- generation of summary figures.

## Privacy statement

No real student data are included. Every file under `data_dummy/` is generated artificially and uses synthetic identifiers. The synthetic scores, Moodle events, course-record variables, and environmental values are not derived from the study data.

Results produced from the dummy data are **smoke-test outputs only**. They do not reproduce or replace the numerical findings reported in the manuscript.

## Leakage-safe preprocessing

All data-dependent predictor pruning is fitted **inside the training partition**. In each cross-validation fold, zero-variance removal and pairwise-correlation pruning (`|r| > 0.95`) are estimated using the training fold only. The resulting selected columns are then applied unchanged to the held-out fold. For XGBoost classification, `scale_pos_weight` is calculated from the training labels only.

The same rule is used for cross-year validation: pruning is fitted on the training cohort and then applied to the held-out cohort. Consequently, retained feature counts can vary slightly across folds or train/test directions.

## Repository structure

```text
.
├── data_dummy/                    # privacy-safe synthetic inputs
├── results_dummy/                 # generated smoke-test outputs (created by run_all.py)
├── src/                           # feature engineering, evaluation, and figures
├── .github/workflows/             # automatic synthetic-data smoke test
├── generate_dummy_data.py
├── 01_feature_engineering.py
├── 02_model_evaluation.py
├── 03_make_figures.py
├── 04_window_sensitivity.py
├── run_all.py
├── input_schema.md
└── requirements.txt
```

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

On Windows PowerShell, the environment can be used directly as:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## One-command synthetic-data test

Run the complete workflow using smaller models and fewer permutations:

```bash
python run_all.py --quick
```

This regenerates the synthetic inputs, creates `data_dummy/dummy_feature_dataset.csv`, runs the evaluation stages, and writes outputs to `results_dummy/`.

Quick mode uses 80 trees, 10 environmental permutations, and 5 permutationtance repeats. It is intended only to confirm that the pipeline executes.

## Paper-aligned configuration

The full evaluation configuration uses:

- 5 validation folds;
- 800 Random Forest trees;
- 800 XGBoost trees for regression;
- 500 XGBoost trees for classification;
- 300 environmental permutations;
- 30 held-out permutation-importance repeats;
- random state 42;
- correlation threshold `|r| > 0.95`.

To run the full configuration on the synthetic data:

```bash
python run_all.py
```

Runtime depends on the available CPU.

## Running individual stages

Generate synthetic data:

```bash
python generate_dummy_data.py --output-dir data_dummy --students-per-year 60 --seed 2026
```

Construct the primary five-day feature dataset:

```bash
python 01_feature_engineering.py \
  --logs-2023 data_dummy/moodle_logs_2023.csv \
  --grades-2023 data_dummy/grades_2023.csv \
  --exam-2023 "2023-12-13 18:00:00" \
  --logs-2024 data_dummy/moodle_logs_2024.csv \
  --grades-2024 data_dummy/grades_2024.csv \
  --exam-2024 "2024-12-03 16:00:00" \
  --weather data_dummy/weather.csv \
  --window-days 5 \
  --output data_dummy/dummy_feature_dataset.csv
```

Run the evaluation:

```bash
python 02_model_evaluation.py \
  --dataset data_dummy/dummy_feature_dataset.csv \
  --output-dir results_dummy \
  --quick
```

Regenerate figures:

```bash
python 03_make_figures.py --results-dir results_dummy
```

## Focused-window sensitivity

The manuscript uses `k=5` as the primary focused-learning window because it approximates a continuous working week of preparation. Robustness can be checked at alternative temporal scales with:

```bash
python 04_window_sensitivity.py \
  --logs-2023 data_dummy/moodle_logs_2023.csv \
  --grades-2023 data_dummy/grades_2023.csv \
  --exam-2023 "2023-12-13 18:00:00" \
  --logs-2024 data_dummy/moodle_logs_2024.csv \
  --grades-2024 data_dummy/grades_2024.csv \
  --exam-2024 "2024-12-03 16:00:00" \
  --weather data_dummy/weather.csv \
  --windows 3 5 7 10 \
  --output-dir window_sensitivity_dummy \
  --quick
```

The sensitivity script reports Random Forest performance and feature-ranking stability across the requested window lengths. The public synthetic results are illustrative only.

## Evaluation outputs

`02_model_evaluation.py` creates, among others:

- `cv_regression_results.csv` and `cv_classification_results.csv`;
- `cv_regression_foldwise.csv` and `cv_classification_foldwise.csv`;
- `cv_pruning_audit.csv`;
- `paired_feature_set_tests.csv`;
- `cross_year_regression_results.csv` and `cross_year_classification_results.csv`;
- `cross_year_pruning_audit.csv`;
- `weather_permutation_test.csv` and `weather_permutation_distribution.csv`;
- `permutation_importance.csv` and `permutation_importance_by_fold.csv`;
- `model_importance_random_forest.csv` and `model_importance_xgboost.csv`;
- `run_metadata.json`.

## Relation to the manuscript

The code implements the methodological operations described in the manuscript. Exact numerical reproduction of the protected study findings requires the original Moodle, assessment/course-record, and environmental data, which are not distributed because of student privacy and institutional restrictions.

Feature counts after zero-variance and correlation pruning are data-dependent and can vary by fold because preprocessing is fitted on training data only. Therefore, counts obtained from the synthetic data are not expected to match the protected study analysis.

See `input_schema.md` for the required columns.

## Citation

Use the repository's `CITATION.cff` file and cite the accompanying manuscript. After the article receives its final DOI, add the DOI and complete bibliographic information to `CITATION.cff`.

## License

The source code is distributed under the MIT License. The privacy restrictions applying to the original educational records remain unchanged; those records are not included in this repository.
