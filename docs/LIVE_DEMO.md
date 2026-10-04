# Five to seven minute live demo

This is a prepared live-demo route. It is not a claim that a recording has been made or that a presentation has already been delivered.

## 0:00–0:40  State the question

Say: “We selected Option A, telecom customer churn. The goal is to rank customers who look more likely to churn so a retention team can decide who to review. This is a prediction exercise on a static IBM educational sample, so we will also show what the data cannot prove.”

Open `churn_analysis.ipynb` and show the title and the printed shape: 7,043 rows and 21 columns.

## 0:40–1:25  Show the data checks

Scroll to the data-quality output. Point out 11 blank `TotalCharges` entries, all at zero tenure, no duplicate customer IDs and no negative tenure or charge values. Say that blanks are converted to missing numeric values and imputed inside the training pipeline. Explain that `customerID` is excluded from predictors and that `Churn` is the target.

## 1:25–2:10  Explain the features and EDA

Show the contract and internet-service charts. Say that month-to-month churn is 42.7% in the training data, compared with 11.1% for one-year and 2.9% for two-year contracts. Fiber optic is 42.1%, DSL 18.7% and no internet service 7.2%. Add: “These are associations, not proof that changing a contract or service causes churn.”

Point to `ServiceCount` and `TenureBand`. Explain that they count six optional services and group tenure into four relationship stages. The original fields remain available.

## 2:10–3:15  Explain modelling

Show the model cell and say: “We use an 80/20 stratified split: 5,634 training rows and 1,409 held-out test rows. Preprocessing is inside the pipeline, and five-fold training cross-validation compares a majority baseline, logistic regression and random forest. We select on mean training F1 before looking at the test results.”

Read the selection output: random forest CV F1 0.6316, logistic regression 0.6280 and dummy 0.0000. Mention that the forest uses balanced class weights, depth 8, 200 trees and minimum leaf size 5.

## 3:15–4:15  Show the test metrics

Show `results/test_metrics.csv` or the notebook table. Read the selected forest metrics: accuracy 0.763, precision 0.537, recall 0.783, F1 0.637 and ROC-AUC 0.843. Compare logistic regression briefly: its recall is slightly higher at 0.789, but its F1 and accuracy are lower. The dummy's 0.735 accuracy and zero recall explain why accuracy alone is misleading.

## 4:15–5:00  Explain the confusion matrix

Show `figures/07_model_evaluation.png`. Say: “At threshold 0.50, the forest has 782 true negatives, 253 false positives, 81 false negatives and 293 true positives.” A false negative is a churner we missed. A false positive is a customer we contacted or reviewed unnecessarily. The company must decide what those errors cost.

## 5:00–5:45  Run the saved model

In a terminal in the project folder, run:

```bash
python demo_predict.py --input data/demo_customers.csv
```

The expected probabilities are about 0.038, 0.859 and 0.166, with predictions No, Yes and No. Say that the second demo prediction is a false positive because its recorded test label is No. This makes the error concept concrete. Explain that the model is loaded from `models/best_model.joblib`; it is not retrained by the demo.

## 5:45–6:30  Business recommendation and limits

Say: “I would use the score to rank a limited reviewed outreach list, rather than automatically giving everyone a discount. The threshold or list size should be chosen with campaign cost and capacity, then tested in a controlled pilot.”

Finish with: “Contract and tenure are strongly associated with churn in this sample, but the model does not prove causes. The file has no reliable prediction date, customer profit or intervention outcome. Before deployment we need newer time-based validation, calibration, group impact checks and a retention experiment.”

## If asked about fairness or ethics

The held-out audit reports recall of 75.6% for females and 81.2% for males. For `SeniorCitizen` 0 and 1 it is 74.6% and 88.8%, with a smaller senior group of 222 test rows. Say these are descriptive checks, not a fairness certificate. Explain that the official ethics form still needs the actual members to complete and sign it.
