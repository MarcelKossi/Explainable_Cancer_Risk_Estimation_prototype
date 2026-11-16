# Audit of the AI-Cancer prototype: manuscript, code and results

Repository: https://github.com/MarcelKossi/Explainable_Cancer_Risk_Estimation_prototype

Material for the manuscript *Explainable Multi-Modal Cancer Risk Estimation from Public Data: A Critical Re-Evaluation of a Prototype* (Tamplare, Lushikwa Ellie, Upadhyay, Zoferana; GD Goenka University).

## Contents

| File | What it is |
|---|---|
| `manuscript.pdf` | The paper |
| `fig_splits.pdf`, `fig_calib.pdf`, `fig_importance.pdf` | Figures 1, 2 and 4, written by `audit_experiments.py` |
| `fig_blocks.pdf` | Figure 3, written by `fig_blocks.py` |
| `audit_experiments.py` | Main experiments on WDBC: cross-validation, single splits, tuning on the test set, permutation importance, ablation, noise control of the fused table. Writes `results.json` and three figures |
| `audit_experiments_supplement.py` | Per-repeat estimates, unpenalised logistic regression, paired LR minus RF difference, fused-table control with the corrected block structure. Adds to `results.json` |
| `audit_revision.py` | Further checks: calibration slope under three clipping values, leave-one-feature-out retraining, other split ratios and seed blocks, fused-table control without missingness indicators. Writes `results_revision.json` |
| `audit_artefacts.py` | Inspection of the two saved artefacts: scaler columns and counts, model weights, optimiser state, model outputs on synthetic inputs. Writes `results_artefacts.json` and `scaler_columns.csv` |
| `fig_blocks.py` | Draws Figure 3 from `scaler_columns.csv` |
| `results.json`, `results_revision.json`, `results_artefacts.json` | Numbers quoted in the paper |
| `scaler_columns.csv` | The 39 columns of the scaler with count, mean and standard deviation |
| `artefacts/solid_symptom_scaler.pkl` | Fitted `StandardScaler` saved by the prototype |
| `artefacts/multimodal_cancer_predictor.h5` | Keras model saved by the prototype |
| `requirements.txt` | Python packages |

## Which script produces which section

| Section of the paper | Script | Result file |
|---|---|---|
| 5.1 What the artefacts establish | `audit_artefacts.py` | `results_artefacts.json` |
| 5.2 Cross-validated performance, single splits, tuning | `audit_experiments.py`, `audit_experiments_supplement.py` | `results.json` |
| 5.3 Calibration | `audit_experiments.py`, `audit_experiments_supplement.py`, `audit_revision.py` | `results.json`, `results_revision.json` |
| 5.4 Fused table | `audit_artefacts.py` (structure), `audit_experiments_supplement.py` (noise control) | `results_artefacts.json`, `results.json` |
| 5.5 Image split | Counts taken from the image folders; the images are not in this repository | `results.json` (`breakhis`) |
| 5.6 Attributions | `audit_experiments.py` | `results.json` |

## How to run

```
pip install -r requirements.txt
python audit_experiments.py
python audit_experiments_supplement.py
python audit_revision.py
python audit_artefacts.py
python fig_blocks.py
```

`audit_experiments_supplement.py` reads the `results.json` written by `audit_experiments.py`. `fig_blocks.py` reads the `scaler_columns.csv` written by `audit_artefacts.py`.

On a machine with few cores, set `OMP_NUM_THREADS=1` before running the experiment scripts.

Run times measured on a two-core CPU with `OMP_NUM_THREADS=1`: `audit_experiments.py` 8.5 minutes, `audit_experiments_supplement.py` 1 minute, `audit_revision.py` 3 minutes. In the environment listed below, the first two scripts reproduce `results.json` with no numerical difference.

## Data

- The experiments use the copy of the Wisconsin Diagnostic Breast Cancer data bundled with scikit-learn (`load_breast_cancer`). No download is needed.
- The tabular sources of the prototype are public Kaggle datasets: `merishnasuwal/breast-cancer-prediction-dataset`, `sajidsaifi/prostate-cancer`, `rashadrmammadov/lung-cancer-prediction`, `ankushpanday2/colorectal-cancer-global-dataset-and-predictions`. They are not redistributed here.
- The image folders come from the Kaggle dataset `forderation/breakhis-400x`. They are not redistributed here.

## Seeds

- Cross-validation repeats: `StratifiedKFold(10, shuffle=True, random_state=r)` for `r` in 0 to 9.
- Single 80/20 splits: `random_state` 0 to 999.
- Tuning experiment: `random_state` 1000 to 1099.
- Permutation importance: `random_state` 100 to 104. Ablation: 0 to 4.
- Model seeds: `random_state=0`.

## Environment used for the files in this repository

Python 3.13, scikit-learn 1.9.1, NumPy 2.5.3, SciPy 1.18.1, Keras 3.15.1, TensorFlow 2.21.0. The scaler was saved with scikit-learn 1.6.1 and the model with Keras 3.11.3.

## Use of AI tools

The scripts were written with Claude (Anthropic) and checked by the authors. The paper states the full extent of AI use in its section *Use of AI tools*.
