# Technical verification

Verified on 4 October 2026 using the pinned Python environment.

- Full training and evaluation script completed successfully.
- Rerun CV metrics, test metrics, all 1,409 test predictions, split membership, feature ablation, subgroup audit and demo cases match the supplied outputs (numerical tolerance 1e-12).
- Reloaded model reproduces all saved test probabilities.
- Training and test customers are disjoint; the imputer uses training medians.
- Changing customerID and Churn input columns does not change predictions.
- Missing numeric values and unseen categories are handled; missing required columns are rejected clearly.
- Notebook has 12 executed code cells, seven embedded chart images and zero error outputs.

Selected random forest test scores: accuracy 0.762952, precision 0.536630, recall 0.783422, F1 0.636957 and ROC-AUC 0.842699. Confusion matrix: TN 782, FP 253, FN 81, TP 293.

These checks establish that the supplied workflow runs and its reported technical results are reproducible. They do not guarantee a grade or predictive performance for a future telecom population. Report-cover details, signatures, instructor repository access and the actual demo remain group submission actions.
