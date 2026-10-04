"""Reproduce the student Telco churn analysis. Run: python churn_analysis.py"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import platform

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, make_scorer,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_features import TelcoFeatures, RAW_FEATURES, BASE_NUMERIC, OPTIONAL_SERVICES

ROOT = Path(__file__).resolve().parent
SEED = 42
SCORING = {'accuracy': 'accuracy', 'precision': make_scorer(precision_score, zero_division=0), 'recall': 'recall',
           'f1': 'f1', 'roc_auc': 'roc_auc'}


def prepare_dirs(out):
    for name in ['data', 'results', 'figures', 'models']:
        (out / name).mkdir(parents=True, exist_ok=True)


def inspect_data(df, source):
    """Administrative quality checks use all rows; relationship EDA uses train only."""
    total = pd.to_numeric(df.TotalCharges, errors='coerce')
    blank = df.TotalCharges.astype(str).str.strip().eq('')
    return {
        'source_file': source.name,
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'local_inspection_date': '2026-10-03',
        'rows': len(df), 'columns': len(df.columns),
        'unique_customer_ids': int(df.customerID.nunique()),
        'duplicate_rows': int(df.duplicated().sum()),
        'duplicate_ids': int(df.customerID.duplicated().sum()),
        'blank_TotalCharges': int(blank.sum()),
        'missing_TotalCharges_after_numeric_conversion': int(total.isna().sum()),
        'blank_TotalCharges_with_zero_tenure': int((blank & df.tenure.eq(0)).sum()),
        'other_missing_cells': int(df.drop(columns='TotalCharges').isna().sum().sum()),
        'negative_tenure_or_charges': int(((df.tenure < 0) | (df.MonthlyCharges < 0) | (total < 0)).sum()),
        'churn_count': int(df.Churn.eq('Yes').sum()),
        'nonchurn_count': int(df.Churn.eq('No').sum()),
        'churn_rate': float(df.Churn.eq('Yes').mean()),
    }


def make_pipeline(estimator, engineer=True):
    numeric = BASE_NUMERIC + (['ServiceCount'] if engineer else [])
    categorical = [c for c in RAW_FEATURES if c not in BASE_NUMERIC]
    categorical += ['TenureBand'] if engineer else []
    preprocess = ColumnTransformer([
        ('numeric', Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
        ]), numeric),
        ('categorical', Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('encode', OneHotEncoder(handle_unknown='ignore')),
        ]), categorical),
    ])
    return Pipeline([
        ('features', TelcoFeatures(engineer=engineer)),
        ('preprocess', preprocess),
        ('model', estimator),
    ])


def explore_training(X_train, y_train, out):
    """Six simple figures; no test-set relationships influence modelling choices."""
    data = TelcoFeatures().fit_transform(X_train)
    data['Churn'] = y_train.to_numpy()
    data['Churn label'] = data.Churn.map({0: 'No', 1: 'Yes'})
    sns.set_theme(style='whitegrid', context='notebook', font_scale=0.95)
    plt.rcParams.update({'figure.dpi': 100, 'savefig.dpi': 140})
    colors = ['#527B9A', '#D88B5C']
    stats = {}

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out / 'figures' / name, bbox_inches='tight')
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    counts = data['Churn label'].value_counts().reindex(['No', 'Yes'])
    ax.bar(counts.index, counts.values, color=colors)
    ax.bar_label(ax.containers[0], padding=3)
    ax.set(title='Training customers by churn status', xlabel='Churn', ylabel='Customers')
    save(fig, '01_churn_balance.png')

    for column, name, title in [
        ('Contract', '02_contract_churn.png', 'Training churn rate by contract'),
        ('InternetService', '05_internet_churn.png', 'Training churn rate by internet service'),
        ('PaymentMethod', '06_payment_churn.png', 'Training churn rate by payment method'),
    ]:
        table = data.groupby(column).Churn.agg(['size', 'sum', 'mean']).sort_values('mean', ascending=False)
        table.columns = ['customers', 'churners', 'churn_rate']
        table.to_csv(out / 'results' / ('eda_' + column + '.csv'))
        stats[column] = table.reset_index().to_dict(orient='records')
        fig, ax = plt.subplots(figsize=(7.2, 3.7))
        ax.barh(table.index, table.churn_rate * 100, color='#527B9A')
        labels = [f'{rate:.1%} (n={n:,})' for rate, n in zip(table.churn_rate, table.customers)]
        ax.bar_label(ax.containers[0], labels=labels, padding=4, fontsize=9)
        ax.set(xlabel='Churn rate (%)', title=title, xlim=(0, max(table.churn_rate) * 100 + 22))
        ax.invert_yaxis()
        save(fig, name)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    sns.histplot(data=data, x='tenure', hue='Churn label', hue_order=['No', 'Yes'],
                 bins=12, stat='density', common_norm=False, element='step',
                 palette=colors, ax=ax)
    ax.set(title='Training tenure distribution by churn', xlabel='Tenure (months)', ylabel='Density within churn group')
    save(fig, '03_tenure_distribution.png')
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    sns.boxplot(data=data, x='Churn label', y='MonthlyCharges', order=['No', 'Yes'],
                color='#94B7CD', ax=ax)
    ax.set(title='Training monthly charges by churn', xlabel='Churn', ylabel='Monthly charges (dataset units)')
    save(fig, '04_monthly_charges.png')
    stats['by_churn'] = data.groupby('Churn label').agg(
        customers=('Churn', 'size'), median_tenure=('tenure', 'median'),
        median_monthly_charges=('MonthlyCharges', 'median'),
    ).reset_index().to_dict(orient='records')
    stats['numeric_summary'] = data[BASE_NUMERIC].describe().to_dict()
    stats['training_missing_TotalCharges'] = int(data.TotalCharges.isna().sum())
    # IQR flags describe unusual values. Plausible values are retained.
    stats['iqr_flags'] = {}
    for col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
        q1, q3 = data[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        stats['iqr_flags'][col] = int(((data[col] < q1 - 1.5 * iqr) | (data[col] > q3 + 1.5 * iqr)).sum())
    return stats


def tune_models(X_train, y_train, out):
    """Predefined selection rule: highest mean training CV F1; threshold stays 0.5.

    CV scores used for tuning are not an unbiased performance estimate. The
    untouched test set below supplies the final evaluation.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    candidates = {
        'Dummy': (DummyClassifier(strategy='most_frequent'), {}),
        'Logistic regression': (
            LogisticRegression(max_iter=2000, random_state=SEED),
            {'model__C': [0.1, 1.0, 10.0], 'model__class_weight': [None, 'balanced']},
        ),
        'Random forest': (
            RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                   random_state=SEED, n_jobs=1),
            {'model__max_depth': [8, None], 'model__class_weight': [None, 'balanced']},
        ),
    }
    fitted, rows, params = {}, [], {}
    for name, (model, grid) in candidates.items():
        print('Training:', name, flush=True)
        search = GridSearchCV(make_pipeline(model), grid, scoring=SCORING, refit='f1',
                              cv=cv, n_jobs=2, return_train_score=False, error_score='raise')
        search.fit(X_train, y_train)
        fitted[name] = search.best_estimator_
        params[name] = search.best_params_
        row = {'model': name}
        for metric in SCORING:
            row['cv_' + metric] = float(search.cv_results_['mean_test_' + metric][search.best_index_])
            row['cv_' + metric + '_sd'] = float(search.cv_results_['std_test_' + metric][search.best_index_])
        rows.append(row)
        pd.DataFrame(search.cv_results_).to_csv(out / 'results' / ('tuning_' + name.lower().replace(' ', '_') + '.csv'), index=False)
    cv_table = pd.DataFrame(rows)
    # idxmax uses candidate insertion order for an exact tie, a rule fixed here.
    chosen = cv_table.loc[cv_table.cv_f1.idxmax(), 'model']
    cv_table.to_csv(out / 'results' / 'cv_metrics.csv', index=False)
    return fitted, cv_table, chosen, params


def feature_ablation(X_train, y_train, fitted, out):
    """Same chosen LR hyperparameters, same training folds, with/without features.

    This is a descriptive check, not another model selection step. It does not
    independently retune the base model, so it cannot prove optimal superiority.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for enabled in [False, True]:
        pipe = make_pipeline(clone(fitted['Logistic regression'].named_steps['model']), engineer=enabled)
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=2)
        rows.append({'features': 'Base + tenure band + service count' if enabled else 'Base features',
                     **{'cv_' + k: float(scores['test_' + k].mean()) for k in SCORING}})
    table = pd.DataFrame(rows)
    table.to_csv(out / 'results' / 'feature_ablation.csv', index=False)
    return table


def evaluate_models(fitted, chosen, X_test, y_test, out):
    """Final comparison only: neither model nor threshold changes after this."""
    rows, matrices = [], {}
    predictions = pd.DataFrame({'customerID': X_test.customerID, 'actual_churn': y_test})
    for name, pipe in fitted.items():
        probability = pipe.predict_proba(X_test)[:, 1]
        predicted = (probability >= 0.5).astype(int)
        rows.append({'model': name, 'accuracy': accuracy_score(y_test, predicted),
                     'precision': precision_score(y_test, predicted, zero_division=0),
                     'recall': recall_score(y_test, predicted, zero_division=0),
                     'f1': f1_score(y_test, predicted, zero_division=0),
                     'roc_auc': roc_auc_score(y_test, probability)})
        matrices[name] = confusion_matrix(y_test, predicted, labels=[0, 1]).tolist()
        if name == chosen:
            predictions['predicted_churn'] = predicted
            predictions['churn_probability'] = probability
    table = pd.DataFrame(rows)
    table.to_csv(out / 'results' / 'test_metrics.csv', index=False)
    predictions.to_csv(out / 'results' / 'test_predictions.csv', index=False)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for name, pipe in fitted.items():
        RocCurveDisplay.from_estimator(pipe, X_test, y_test, ax=ax[0], name=name)
    ax[0].plot([0, 1], [0, 1], '--', color='grey', linewidth=1)
    ax[0].set_title('Held-out ROC curves')
    ConfusionMatrixDisplay(np.array(matrices[chosen]), display_labels=['No churn', 'Churn']).plot(ax=ax[1], colorbar=False, cmap='Blues')
    ax[1].set_title(chosen + ' (threshold 0.5)')
    fig.tight_layout()
    fig.savefig(out / 'figures' / '07_model_evaluation.png', dpi=140, bbox_inches='tight')
    plt.close(fig)
    return table, matrices, predictions


def fairness_audit(X_test, predictions, out):
    """Descriptive held-out subgroup audit; no tuning and no fairness certification."""
    rows = []
    for attribute in ['gender', 'SeniorCitizen']:
        for group in sorted(X_test[attribute].unique()):
            selected = X_test[attribute].eq(group)
            y = predictions.loc[selected, 'actual_churn']
            pred = predictions.loc[selected, 'predicted_churn']
            tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
            rows.append({'attribute': attribute, 'group': str(group), 'customers': len(y),
                         'actual_churners': int(y.sum()), 'actual_nonchurners': int((y == 0).sum()),
                         'base_churn_rate': float(y.mean()), 'flag_rate': float(pred.mean()),
                         'recall': float(tp / (tp + fn)) if tp + fn else None,
                         'false_positive_rate': float(fp / (fp + tn)) if fp + tn else None,
                         'precision': float(tp / (tp + fp)) if tp + fp else None,
                         'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)})
    table = pd.DataFrame(rows)
    table.to_csv(out / 'results' / 'fairness_audit.csv', index=False)
    return table


def export_model(best, X_test, y_test, out):
    model_path = out / 'models' / 'best_model.joblib'
    joblib.dump(best, model_path, compress=3)
    reloaded = joblib.load(model_path)
    np.testing.assert_allclose(best.predict_proba(X_test), reloaded.predict_proba(X_test), rtol=0, atol=0)
    # Fixed test cases by split order, not selected because they look persuasive.
    demo = X_test.iloc[:3].copy()
    demo.to_csv(out / 'data' / 'demo_customers.csv', index=False)
    demo_probability = reloaded.predict_proba(demo)[:, 1]
    expected = pd.DataFrame({'customerID': demo.customerID, 'actual_churn': y_test.iloc[:3],
                            'churn_probability': demo_probability,
                            'predicted_churn': (demo_probability >= 0.5).astype(int)})
    expected.to_csv(out / 'results' / 'demo_expected.csv', index=False)
    return expected


def run(data_path, out):
    prepare_dirs(out)
    df = pd.read_csv(data_path)
    if set(df.Churn.unique()) != {'No', 'Yes'}:
        raise ValueError('Expected a complete binary Churn target: Yes/No.')
    quality = inspect_data(df, data_path)
    if quality['duplicate_ids'] or quality['duplicate_rows']:
        raise ValueError('Duplicate customers require a reviewed group-based split before training.')
    X, y = df.drop(columns='Churn'), df.Churn.eq('Yes').astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    assert set(X_train.customerID).isdisjoint(X_test.customerID)
    print('Quality:', json.dumps(quality, indent=2))
    print('Train / test:', len(X_train), len(X_test))
    pd.DataFrame({'customerID': df.customerID,
                  'split': np.where(df.index.isin(X_train.index), 'train', 'test')}).to_csv(out / 'results' / 'split_membership.csv', index=False)
    eda = explore_training(X_train, y_train, out)
    fitted, cv_table, chosen, params = tune_models(X_train, y_train, out)
    print('CV comparison:\n', cv_table.round(4).to_string(index=False))
    print('Selected BEFORE test evaluation:', chosen, params[chosen])
    ablation = feature_ablation(X_train, y_train, fitted, out)
    test_table, matrices, predictions = evaluate_models(fitted, chosen, X_test, y_test, out)
    fairness = fairness_audit(X_test, predictions, out)
    demo = export_model(fitted[chosen], X_test, y_test, out)
    packages = ['numpy', 'pandas', 'scikit-learn', 'scipy', 'joblib', 'matplotlib', 'seaborn', 'nbformat', 'nbclient', 'ipykernel']
    summary = {
        'quality': quality,
        'split': {'random_state': SEED, 'train_rows': len(X_train), 'test_rows': len(X_test),
                  'train_churners': int(y_train.sum()), 'test_churners': int(y_test.sum()),
                  'train_churn_rate': float(y_train.mean()), 'test_churn_rate': float(y_test.mean())},
        'primary_selection_metric': 'mean training 5-fold stratified CV F1',
        'classification_threshold': 0.5, 'chosen_model': chosen,
        'best_parameters': params, 'cv_metrics': cv_table.to_dict(orient='records'),
        'test_metrics': test_table.to_dict(orient='records'), 'confusion_matrices': matrices,
        'eda_train_only': eda, 'feature_ablation': ablation.to_dict(orient='records'),
        'fairness_test_posthoc': fairness.to_dict(orient='records'),
        'demo_cases': demo.to_dict(orient='records'), 'reload_probabilities_identical': True,
        'python': platform.python_version(),
        'package_versions': {p: importlib.metadata.version(p) for p in packages},
    }
    (out / 'results' / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('Test comparison:\n', test_table.round(4).to_string(index=False))
    print('Confusion matrices [TN, FP; FN, TP]:', matrices)
    print('Feature check:\n', ablation.round(4).to_string(index=False))
    print('Demo:\n', demo.round(4).to_string(index=False))
    print('Saved:', out / 'models' / 'best_model.joblib')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data' / 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
    parser.add_argument('--output-dir', type=Path, default=ROOT)
    args = parser.parse_args()
    run(args.data.resolve(), args.output_dir.resolve())
