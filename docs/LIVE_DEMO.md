# Telco churn demo script

Group 12 — Andrea, Vanesa, Mia and Hasnain. Prepared 4 October 2026.

Allow about 6–7 minutes including scrolling, showing charts and running the command. These speaking parts are suggestions and can be swapped. Read the quoted text; screen actions are directions. Rehearse once to check your pace.

Before starting, open the executed notebook and a terminal in the project folder with the project's Python environment activated. Keep the model and supporting files in their existing folders. Do not rerun training during the presentation.

## 0:00–1:35 Andrea introduces the problem and data

**Show:** notebook title, dataset shape and data-quality output.

“Hi, we are Group 12: Andrea, Vanesa, Mia and Hasnain. Our project is Option A, predicting customer churn for a telecom provider.

Churn means a customer leaves a service. Our aim is to identify customers who look more likely to leave, so a retention team can decide who to review and contact.

We used the IBM Telco Customer Churn sample, available through Kaggle. It contains 7,043 customers and 21 columns. The target is Churn, recorded as Yes or No. About 26.5 percent of customers are labelled as churners.

We first checked missing values, duplicates and data types. We found 11 blank TotalCharges values, all belonging to customers with zero tenure. We converted this column to numbers and filled missing values using medians learned only from training data. We found no duplicate customer IDs.

We excluded customerID from the predictors because it identifies the record. We also kept the Churn target separate from the inputs.”

## 1:35–3:05 Vanesa explains patterns and training

**Show:** contract chart, engineered-feature example and training comparison.

“The contract chart shows one of the clearest patterns. In our training data, month-to-month customers had a churn rate of 42.7 percent. This was 11.1 percent for one-year contracts and 2.9 percent for two-year contracts. This is an association; it does not prove that changing someone's contract would stop them leaving.

We created two features. ServiceCount counts six optional services that a customer uses. TenureBand groups customers into four stages based on how long they have stayed. We also retained the original information.

We split the data into 80 percent training and 20 percent testing. That gives 5,634 training customers and 1,409 test customers, with similar churn proportions in both groups.

We compared logistic regression and random forest, plus a dummy model that always predicts the majority class. Missing-value handling, encoding and scaling are inside the training pipeline.

We used five-fold cross-validation on the training data to compare settings. We selected the model using F1 before evaluating it on the test set. Random forest was selected, with a validation F1 of about 0.632, slightly above logistic regression at 0.628.”

## 3:05–4:40 Mia explains the results

**Show:** test metrics and confusion matrix.

“On the held-out test set, random forest achieved 76.3 percent accuracy, 53.7 percent precision, 78.3 percent recall and an F1 score of 0.637.

Accuracy tells us how many predictions were correct overall. Precision means that about 54 percent of customers flagged as churners actually churned. Recall means we identified about 78 percent of all actual churners. F1 balances precision and recall.

Accuracy alone would be misleading here. The dummy model achieved about 73.5 percent accuracy by predicting that nobody would churn, but it identified zero churners. Logistic regression had slightly higher recall than our forest, but lower F1 and accuracy.

The confusion matrix shows what the errors mean. Our forest correctly identified 293 churners and correctly classified 782 customers who stayed. It missed 81 churners and incorrectly flagged 253 customers who stayed.

Those errors have different business costs. Missing a churner could mean losing a chance to help them. Flagging someone who stays could mean an unnecessary call or offer. The business needs to consider both before using the model.”

## 4:40–6:30 Hasnain runs the model and concludes

**Run in the project terminal:**

```bash
python demo_predict.py --input data/demo_customers.csv
```

**Show:** three output rows. Expected scores are approximately 0.0383, 0.8588 and 0.1660; predictions are No, Yes and No.

“This command loads our saved model and predicts for three held-out customers. It does not retrain the model. The saved pipeline handles the original input columns and creates the required features automatically.

Using a threshold of 0.5, the first and third customers are predicted not to churn, while the second is predicted to churn. The second customer's actual recorded outcome is No, so this example is a false positive. It shows why a high model score is not a certainty. These scores also need calibration before being treated as reliable individual probabilities.

Our recommendation is to use the model to rank a manageable list for a retention team to review. The company could investigate service problems or contact customers for support. Discounts should depend on costs, customer value and evidence that the offer helps.

There are limits. This is a static educational sample, and it does not contain reliable prediction dates, customer profit or responses to offers. Our results do not prove causes or show that the model will perform equally well for future customers.

Before deployment, we would test on newer, time-based data, check calibration and errors across customer groups, and run a controlled retention trial. Our project demonstrates a working prediction workflow and explains the further evidence a business would need.”

## Short answers for questions

- **Why random forest?** It had the highest mean training cross-validation F1. The choice was made before inspecting test results.
- **What about fairness?** We compared held-out errors across gender and senior-citizen groups. Differences require investigation; these checks do not certify fairness.
- **Did the new features help?** Their measured improvement was small: logistic regression CV F1 changed from about 0.6275 to 0.6280. We do not claim a major gain.
- **Did you use AI?** Yes. OpenAI Codex helped organise files, create and debug code, run and check analyses, and draft and edit documents. This assistance is disclosed in the declaration and project documentation.

This is a prepared script. The group still needs to record or deliver the demonstration.
