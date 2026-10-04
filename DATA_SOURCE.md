# Dataset source and preparation

## Source and access

**Dataset:** Telco Customer Churn, posted by BlastChar on [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), version 1. Kaggle metadata gives a last-updated date of 23 February 2018.

**Source access and local verification date:** 3 October 2026. The user supplied the CSV files; the user's original download date is unknown. This date records the present source check and inspection, not a claimed original download.

The complete supplied file has the same parsed cells, row order and column order as the [21-column CSV in IBM's Telco churn repository](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv). The only byte-level difference is CRLF versus LF line endings. The [IBM repository](https://github.com/IBM/telco-customer-churn-on-icp4d) provides the original-format file for an educational churn modelling example.

IBM's [Cognos sample description](https://community.ibm.com/community/user/blogs/steven-macko/2019/07/11/telco-customer-churn-1113) explicitly describes a fictional telco, but it documents an expanded five-table version. This project treats the original CSV as an educational sample, with no claim that it represents verified live subscribers. The expanded version's geography, quarter, satisfaction scores and lifetime-value fields must not be assumed to exist in this file.

## Usage and attribution

Kaggle's public dataset metadata reports the licence label **“Data files © Original Authors”**. This is not a verified CC0 or Creative Commons licence. The IBM repository identifies its code pattern as Apache 2.0; that does not establish a separate permissive licence for every underlying dataset. This project attributes the dataset to IBM via BlastChar/Kaggle and uses the supplied copy for the coursework. Source files retain their original attribution; no new licence is asserted for the data.

Metadata checked through [Kaggle's dataset-list endpoint](https://www.kaggle.com/api/v1/datasets/list?search=telco-customer-churn), selecting the record `blastchar/telco-customer-churn` (dataset ID 13996).

## Files and integrity

| File | Rows | Columns | Bytes | Use |
|---|---:|---:|---:|---|
| `WA_Fn-UseC_-Telco-Customer-Churn.csv` | 7,043 | 21 | 977,501 | Source used for this analysis |
| `WA_Fn-UseC_-Telco-Customer-Churn-selected-columns.csv` | 7,043 | 10 | 362,553 | Supplied subset, inspected but not used for training |

SHA-256 of supplied full file:

```text
88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a
```

SHA-256 of supplied selected-columns file:

```text
d21d4f617d91a30166d85b8349780edb79a1bc3571be94538aee59e6ac5f7303
```

SHA-256 of IBM's LF-line-ending source file:

```text
16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91
```

The selected file is exactly the first ten columns of the full CSV in the same row order. Its last column is `OnlineSecurity`; it has no `Churn`, contract or charge columns. It is therefore unsuitable as the sole input to this supervised assignment. The full CSV contains 19 candidate predictors, one identifier and one target, rather than 21 predictors.

## Data dictionary

Values and ranges below were verified in the supplied full CSV. Numeric charge units have no explicit currency marker in the file and are reported as charge units.

| Column | Meaning / observed values | Role |
|---|---|---|
| `customerID` | Unique customer record code; 7,043 distinct values | Identifier; excluded from predictors |
| `gender` | Recorded category: Female, Male | Categorical predictor; subgroup audit |
| `SeniorCitizen` | Senior-citizen indicator: 0, 1; exact ages are absent | Binary predictor; subgroup audit |
| `Partner` | Partner indicator: No, Yes | Categorical predictor |
| `Dependents` | Dependents indicator: No, Yes | Categorical predictor |
| `tenure` | Customer tenure in months, 0–72 | Numeric predictor |
| `PhoneService` | Phone subscription: No, Yes | Categorical predictor |
| `MultipleLines` | No, Yes, No phone service | Categorical predictor |
| `InternetService` | DSL, Fiber optic, No | Categorical predictor |
| `OnlineSecurity` | No, Yes, No internet service | Categorical predictor |
| `OnlineBackup` | No, Yes, No internet service | Categorical predictor |
| `DeviceProtection` | No, Yes, No internet service | Categorical predictor |
| `TechSupport` | No, Yes, No internet service | Categorical predictor |
| `StreamingTV` | No, Yes, No internet service | Categorical predictor |
| `StreamingMovies` | No, Yes, No internet service | Categorical predictor |
| `Contract` | Month-to-month, One year, Two year | Categorical predictor |
| `PaperlessBilling` | Paperless billing: No, Yes | Categorical predictor |
| `PaymentMethod` | Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic) | Categorical predictor |
| `MonthlyCharges` | Monthly billed amount, 18.25–118.75 | Numeric predictor |
| `TotalCharges` | Cumulative billed amount; observed numeric range 18.80–8,684.80; 11 blanks | Numeric predictor after conversion |
| `Churn` | Observed label: No (5,174), Yes (1,869) | Target; Yes = 1, No = 0 |

The positive-class proportion is 1,869 / 7,043 = 26.53699%. `No internet service` and `No phone service` describe subscription states; they are not missing values.

## Quality checks and preprocessing

The raw file is retained unchanged. It has no duplicate rows, no duplicated customer IDs, and no conventional missing cells detected on initial CSV loading. However, `TotalCharges` contains 11 whitespace entries; all 11 occur at zero tenure. Numeric conversion exposes these as missing values. No negative tenure or charge values occur. The observed numeric ranges are retained; values are not removed simply because they are large.

Preparation converts `TotalCharges` to numeric, codes `Churn` as 0/1, removes `customerID` from model inputs, and derives `ServiceCount` and `TenureBand`. `ServiceCount` counts Yes across six optional services: online security, online backup, device protection, technical support, streaming TV and streaming movies. `TenureBand` groups months into 0–12, 13–24, 25–48 and 49-plus (49–72 in this sample). Both features use existing input values, with no target lookup.

An 80/20 stratified split with random state 42 separates 5,634 training records and 1,409 test records. Numeric medians, numerical scaling and categorical encoding are learned inside the training pipeline, including each cross-validation fold. Missing numeric values use training medians. Categories are one-hot encoded with unknown categories ignored during transformation. The test data does not set preprocessing parameters or model-selection criteria. See the executed notebook for the actual final pipeline, cross-validation and evaluation outputs.

## Limits of the data

The file is one customer snapshot. It has no observation dates, churn-event timestamps or verified prediction cutoff. It cannot establish a specific future prediction horizon or prove every predictor was available before churn. Random-split test results describe classification within this sample, not a prospective trial at a telecom provider.

No intervention costs, customer margins, treatment assignments, campaign outcomes, complaint records or reliable lifetime values are provided. Discounts, callbacks and capacity examples are proposals; the data cannot estimate causal retention uplift or actual profit. Contract and service associations do not prove causes. Demographic fields are limited, and subgroup differences require investigation rather than automatic claims of fairness. Any operational use needs new dated data, documented feature availability, prospective evaluation and a controlled retention experiment.

## References

BlastChar. (2018). *Telco customer churn* (Version 1) [Data set]. Kaggle. https://www.kaggle.com/datasets/blastchar/telco-customer-churn

IBM. (n.d.). *Telco-Customer-Churn.csv* [Data set]. GitHub. Retrieved October 3, 2026, from https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

IBM. (n.d.). *Predict customer churn using Watson Machine Learning and Jupyter Notebooks on Cloud Pak for Data* [Code repository]. GitHub. Retrieved October 3, 2026, from https://github.com/IBM/telco-customer-churn-on-icp4d

Samples Team. (2019, July 11). *Telco customer churn (11.1.3+).* IBM Community. https://community.ibm.com/community/user/blogs/steven-macko/2019/07/11/telco-customer-churn-1113
