# Supplementary implementation package

This package accompanies the manuscript **“Interpretable Feature Engineering from Moodle Logs for Student Performance Prediction Using Focused Learning Windows and Environmental Context.”**

Recommended GitHub repository name: **`moodle-focused-learning-feature-engineering`**.

It provides a privacy-safe, executable implementation of:

- Moodle-log preprocessing;
- a five-day rolling focused-learning window;
- session construction using a 50-minute inactivity threshold;
- behavioral, temporal, assignment, quiz, environmental, and interaction features;
- three feature-set ablations;
- Random Forest and XGBoost regression and pass/fail classification;
- five-fold cross-validation and cross-year validation;
- environmental-alignment permutation testing;
- permutation-based and model-based feature importance;
- generation of summary figures.

## Privacy statement

No real student data are included. Every file under `data_dummy/` is generated artificially and uses identifiers such as `SYN2023_001`. The synthetic scores, Moodle events, and environmental values are not derived from the study data.

Results produced from the dummy data are **smoke-test outputs only**. They do not reproduce or replace the numerical findings reported in the manuscript.

> [!IMPORTANT]
> Do not commit protected Moodle logs, student identifiers, assessment records, or outputs derived from the original study data. Only the files under `data_dummy/` are intended for public distribution.

## Repository structure

```text
.
├── data_dummy/                    # privacy-safe synthetic inputs
├── results_dummy/                 # example outputs from quick mode
├── src/                           # feature engineering, evaluation, and figures
├── .github/workflows/             # automatic synthetic-data smoke test
├── generate_dummy_data.py
├── 01_feature_engineering.py
├── 02_model_evaluation.py
├── 03_make_figures.py
├── run_all.py
├── input_schema.md
└── requirements.txt
```

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

## One-command synthetic-data test

Run the complete workflow using smaller models and fewer permutations:

```bash
python run_all.py --quick
```

This command regenerates the synthetic inputs, creates `data_dummy/dummy_feature_dataset.csv`, runs all evaluation stages, and writes outputs to `results_dummy/`.

The quick mode preserves the same analysis structure but uses 80 trees, 10 weather permutations, and 5 permutation-importance repeats. It is intended only to confirm that the pipeline executes.

## Paper-aligned configuration

To use the full configuration on the synthetic data, run:

```bash
python run_all.py
```

The full configuration uses five folds, 800 Random Forest trees, 800 XGBoost trees for regression, 500 XGBoost trees for classification, 300 weather permutations, and 30 permutation-importance repeats. Runtime depends on the available CPU.

## Running individual stages

Generate synthetic data:

```bash
python generate_dummy_data.py --output-dir data_dummy --students-per-year 60 --seed 2026
```

Construct the feature dataset:

```bash
python 01_feature_engineering.py \
  --logs-2023 data_dummy/moodle_logs_2023.csv \
  --grades-2023 data_dummy/grades_2023.csv \
  --exam-2023 "2023-12-13 18:00:00" \
  --logs-2024 data_dummy/moodle_logs_2024.csv \
  --grades-2024 data_dummy/grades_2024.csv \
  --exam-2024 "2024-12-03 16:00:00" \
  --weather data_dummy/weather.csv \
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

## Output files

The evaluation creates:

- `cv_regression_results.csv`;
- `cv_classification_results.csv`;
- `cross_year_regression_results.csv`;
- `cross_year_classification_results.csv`;
- `weather_permutation_test.csv`;
- `permutation_importance.csv`;
- `model_importance_random_forest.csv`;
- `model_importance_xgboost.csv`;
- three PNG figures;
- `run_metadata.json`, which records the executed configuration.

## Relation to the manuscript

The code implements the methodological operations described in the manuscript. Exact numerical reproduction of the reported study results requires the protected original Moodle, grade, and environmental records, which are not distributed because of student privacy and institutional restrictions. The included synthetic data demonstrate input structure, execution order, and output generation without exposing any participant information.

Feature counts after removal of constant and highly correlated variables are data-dependent. Consequently, counts obtained from the synthetic data need not equal the 26, 33, and 46 features reported for the protected study dataset.

See `input_schema.md` for the required columns.

For manuscript wording, see `DATA_AND_CODE_AVAILABILITY.md`.

## Citation

Use the repository's `CITATION.cff` file and cite the accompanying manuscript. After the article receives its final DOI, add the DOI and complete bibliographic information to `CITATION.cff`.

## License

The source code is distributed under the MIT License. The privacy restrictions applying to the original educational records remain unchanged; those records are not included in this repository.

