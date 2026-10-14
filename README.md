# Bobathon ESILV 2026 — Parkinson’s UPDRS Regression

A student machine-learning project developed at **ESILV (École Supérieure d’Ingénieurs Léonard de Vinci)** during the **2026 Bobathon**, using **IBM Bob** and **Probabl’s Skore** to explore, evaluate and document regression models for a Kaggle challenge.

This repository contains our team’s experiments, feature engineering, results and workbook answers. It builds on the Bobathon lab provided by **Probabl and IBM**:

**Original repository: [probabl-ai/bobathon-esilv](https://github.com/probabl-ai/bobathon-esilv)**

The original teaching guides and scientific context are retained in [`docs/`](docs/). Our experiment code, journal and audit document the team’s work and subsequent verification.

## The challenge

Predict the **unbiased OFF MDS-UPDRS motor score** (`target`) for each patient visit. The synthetic dataset contains repeated visits, demographics, genetics, medication information, observed ON/OFF scores and time since medication intake.

The main challenges are missing clinical measurements, medication-related measurement effects and disease progression across visits. Kaggle holds out **entire patients**, so model selection must also separate patients between training and validation.

- **Training:** 44,590 visits from 5,576 patients.
- **Test:** 11,013 visits from 1,395 different patients.
- **Metric:** root mean squared error (**RMSE**); lower is better.
- **Competition:** [Bobathon ESILV on Kaggle](https://www.kaggle.com/competitions/bobathon-esilv/).
- **Kaggle public score:** **3.966**, reported by the team on 2026-10-14. The exact submitted CSV was not identified in the handoff.
- **Team report:** [Skore EstimatorReport 83726](https://skore.probabl.ai/parkinpredict/bobathon-esilv/estimators/83726) — supplied by the team; sign-in may be required. A Skore validation report and a Kaggle public leaderboard score are distinct results.

## Our approach and results

We started with a mean-predicting baseline, added a linear model, moved to patient-grouped validation, and then explored gradient boosting, categorical features and patient trajectories.

| Experiment | Approach | Local validation RMSE |
|---|---|---:|
| [01 — Dummy](experiments/01_dummy.py) | Training-target mean | 16.4779¹ |
| [02 — Ridge](experiments/02_ridge.py) | Eight numeric features, median imputation | 10.4877¹ |
| [04 — HGBR](experiments/04_hgbr.py) | Numeric gradient boosting with native missing-value handling | 7.7556 ± 0.1528 |
| [05 — Tabular pipeline](experiments/05_tabular_pipeline.py) | Numeric and categorical features with skrub | 7.5336 ± 0.1372 |
| [06 — DataOps](experiments/06_dataops.py) | Feature preparation and grouped splitting in a skrub graph | 7.5368 ± 0.1397 |
| [07 — Feature tuning](experiments/07_features.py) | Patient trajectories and tuned HGBR, 29 features | 4.5329 ± 0.0674 |
| [07 — Submission variant](experiments/07_features_submit.py) | Tuned HGBR with 32 features | **4.4814 ± 0.0601** |

¹ Day 1/early Ridge use a random 80/20 visit split and can share patients across both sides. They are educational baselines, not directly comparable unseen-patient estimates. Ridge evaluated with grouped CV scores **10.5965 ± 0.2670**.

Other rows use five-fold `GroupKFold` on `patient_id`; ± denotes sample standard deviation across fold RMSEs. These are **local validation results**, not Kaggle leaderboard scores. The unseeded string encoder causes small differences between runs. See the [audit](docs/DAY2_AUDIT.md) for historical results, exact configurations and verification details.

### Best current model

The submission variant combines `TableVectorizer` with `HistGradientBoostingRegressor`, using `learning_rate=0.03`, `max_leaf_nodes=63`, `max_iter=1000`, `l2_regularization=1` and `random_state=0`.

Its 32 inputs include the 11 raw clinical/categorical features, disease duration, visit timing, patient-level ON/OFF summaries, neighboring OFF measurements, a trajectory slope and missingness indicators. `patient_id` is used for grouping and then removed from model inputs. The additional ON median/minimum/maximum distinguish this model from the 29-feature tuning script.

The audit found no target-derived features or patient overlap in the outer validation folds. Limitations remain: tuning reuses those folds, internal early stopping is row-based, trajectory features assume batch access to the supplied visits, and the existing interpolation feature uses CSV index spacing rather than age. These are documented rather than hidden behind the score.

## Run the project

### 1. Set up the environment

Use **Python 3.12+** and follow the official [setup guide](docs/SETUP.md) for Bob, lab skills and Skore Hub authentication. The project uses a local virtual environment:

```bash
git clone https://github.com/JRF-cell/bobathon-parkinsons.git
cd bobathon-parkinsons
python3 -m venv --prompt bobathon .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

On Windows, activate with `.venv\Scripts\Activate.ps1`. The editable installation makes the `parkinsons` package in `src/` importable by the experiment scripts. The tested library versions are recorded in the [audit](docs/DAY2_AUDIT.md); `requirements.txt` is the upstream setup entry point, not a complete environment lockfile.

### 2. Download the competition data

Download the four files from the [Kaggle Data tab](https://www.kaggle.com/competitions/bobathon-esilv/data) into `data/`:

```text
data/
├── X_train.csv
├── y_train.csv
├── X_test.csv
└── sample_submission.csv
```

Raw CSVs are excluded from the current published tree. They were tracked in earlier commits; those historical commits have not been rewritten. Credentials (`.skore`, `.env`), the virtual environment and generated submissions are also excluded from new commits.

### 3. Evaluate and generate predictions

With the environment active, run experiments from the repository root in **Bob**:

```bash
python experiments/04_hgbr.py
python experiments/07_features_submit.py
```

The second command uses the selected configuration without repeating the tuning grid. It generates:

```text
submissions/07_features_32.csv
```

The script checks the template’s exact columns (`Index,target`), row count, ID order and finite predictions before saving with `index=False`. It also recomputes grouped out-of-fold errors. Experiment scripts can publish reports to the configured Skore Hub; generating an offline CSV alone does not establish competition eligibility.

For a valid Bobathon submission, follow the [official instructions](docs/DAY2.md): generate/review the work in Bob, obtain the **matching EstimatorReport URL**, include it in Kaggle’s Submission Description, and use a fresh report key for each new submission. The report and CSV names must match. Confirm the linked team report’s model before associating it with a regenerated CSV.

### 4. Check the setup

```bash
python -m pytest tests/test_setup.py
python -m pytest tests/smoke/test_01_dummy.py
```

The setup suite requires local Bob configuration and authenticated Hub access. The smoke test requires downloaded data. These are not credential-free CI checks.

## Documentation

| Document | Contents |
|---|---|
| [Workbook answers](docs/WORKBOOK_ANSWERS.md) | Answers to every Day 1 and Day 2 workbook question, with evidence and explicit limits |
| [Day 2 audit](docs/DAY2_AUDIT.md) | Experiment walkthrough, leakage checks, reproducibility and submission validation |
| [Experiment journal](journal/JOURNAL.md) | Historical experiments, decisions and next ideas |
| [Exploratory analysis](data/eda.md) | Initial EDA and pointers to corrected current measurements |
| [Challenge context](docs/CONTEXT.md) | Original scientific background and synthetic-data assumptions |
| [Setup](docs/SETUP.md), [Day 1](docs/DAY1.md), [Day 2](docs/DAY2.md) | Original Bobathon teaching guides |

## Repository structure

```text
experiments/       Baselines, model comparisons, feature tuning and submission
src/parkinsons/    Data loading, feature engineering and Skore integration
journal/           Experiment methods and historical results
docs/              Original guides, workbook answers and team audit
data/              EDA code/notes; downloaded CSVs stay local
submissions/       Generated Kaggle CSVs (local, Git-ignored)
tests/             Setup and baseline smoke checks
```

## Acknowledgements

This is an **ESILV student hackathon project**. Credit for the original challenge materials and lab goes to **Probabl and IBM**, through the [official Bobathon repository](https://github.com/probabl-ai/bobathon-esilv). We used **IBM Bob** for the hackathon workflow, **Probabl Skore** for model evaluation/reporting, **scikit-learn** and **skrub** for modeling, and **Kaggle** for the competition. The original guides and illustrations retain their source attributions.
