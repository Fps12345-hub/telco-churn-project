# Telco customer churn project

This is the DSAI501 Option A Telco churn project. It uses the supplied Telco Customer Churn CSV to estimate whether a customer is labelled as churned. The workflow cleans the raw fields, creates two features, compares a tuned logistic regression with a tuned random forest, evaluates both on a held-out test split, and saves the selected random forest as a reusable pipeline.

The project is a reproducible educational analysis of a static IBM sample. It does not claim to be a live telecom deployment or a causal study.

## Results

The full file has 7,043 rows, 21 columns and a 26.54% churn rate. The data is split 80/20 with stratification and `random_state=42`. Model selection uses mean five-fold training cross-validation F1. Random forest is selected with `max_depth=8`, 200 trees, `min_samples_leaf=5` and balanced class weights. The final held-out test metrics at threshold 0.50 are:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Dummy | 0.735 | 0.000 | 0.000 | 0.000 | 0.500 |
| Logistic regression | 0.734 | 0.499 | 0.789 | 0.611 | 0.841 |
| Random forest | 0.763 | 0.537 | 0.783 | 0.637 | 0.843 |

The selected forest confusion matrix is TN 782, FP 253, FN 81, TP 293.

## Files

- `churn_analysis.ipynb`: executed notebook with visible outputs.
- `churn_analysis.py`: reproducible command-line analysis.
- `churn_features.py`: raw-input feature transformer used by the saved pipeline.
- `models/best_model.joblib`: selected fitted pipeline.
- `demo_predict.py`: reloads the model and scores raw customer rows.
- `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`: supplied full source copy used for training.
- `data/demo_customers.csv`: three held-out raw rows for the demo.
- `figures/`: training-only EDA and held-out evaluation figures.
- `results/`: source checks, cross-validation, test metrics, predictions, fairness audit and feature ablation.
- `report/Final_report.pdf`: assignment report.
- `docs/LIVE_DEMO.md`: 5–7 minute live demo script.
- `DATA_SOURCE.md`: provenance, hashes, data dictionary and preprocessing record.

## Run it

Use Python 3.12 if possible. From this project directory:

```bash
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
python -m pip install -r requirements.txt
python churn_analysis.py
python demo_predict.py --input data/demo_customers.csv
```

The analysis writes figures, results and the model under this directory. To view the submitted notebook:

```bash
python -m pip install jupyterlab  # if a notebook interface is not already installed
jupyter lab churn_analysis.ipynb
```

The saved model expects the raw predictor columns from the full schema. Do not use the supplied ten-column selected file as the training input because it has no `Churn` target and lacks contract and charge fields.

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`; in Command Prompt use `.venv\Scripts\activate.bat`. Keep `churn_features.py` beside the loading script because the saved model imports that transformer class. The submitted notebook already has outputs and can be read without rerunning it.

## Sources and limits

The dataset is attributed to BlastChar's Kaggle page and IBM's original-format CSV. Kaggle metadata labels the data licence “Data files © Original Authors”; this repository does not add a new licence claim for the data. The CSV has no reliable time sequence, intervention result or customer profit field. Use the scores to support a reviewed pilot only after newer data, calibration, subgroup impact and a controlled retention test are available.

## Verification and submission

The full analysis was rerun successfully before publication. Cross-validation results, held-out predictions, model metrics, feature checks and the demo match the supplied project. The notebook contains 12 executed code cells, seven embedded figures and no error outputs. See `docs/TECHNICAL_VALIDATION.md`.

The report cover contains the supplied Group 12 names, IDs and submission date of 4 October 2026. The group must review and sign the separately supplied ethics form, submit it through the course portal, and record or present the prepared 5–7 minute demo. Ethics documents are kept outside this repository. This private repository must be shared with the instructor to satisfy the assignment's public-or-shared requirement.

Code revision, analysis execution, drafting and checks used OpenAI Codex assistance, starting from the user-supplied `churn_analysis.py`. Members should understand the submission and describe their own contributions accurately.
