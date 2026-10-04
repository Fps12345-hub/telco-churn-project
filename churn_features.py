"""Raw CSV cleaning and deterministic features used during training and prediction.

Keep this module beside the notebook and scripts so joblib can import the class.
No medians, scaling values, categories, or target information are learned here.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


RAW_FEATURES = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges',
]
BASE_NUMERIC = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
OPTIONAL_SERVICES = [
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies',
]


class TelcoFeatures(TransformerMixin, BaseEstimator):
    """Accept the original CSV columns, exclude the ID/target, and add features."""

    def __init__(self, engineer=True):
        self.engineer = engineer

    def fit(self, X, y=None):
        # There is deliberately nothing to fit: the rules are set in advance.
        self.transform(X)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError('Prediction input must be a pandas DataFrame with CSV column names.')
        missing = sorted(set(RAW_FEATURES) - set(X.columns))
        if missing:
            raise ValueError('Missing required raw columns: ' + ', '.join(missing))
        data = X.loc[:, RAW_FEATURES].copy()
        for column in data.select_dtypes(include=['object', 'string']):
            data[column] = data[column].str.strip().replace('', np.nan)
        for column in BASE_NUMERIC:
            data[column] = pd.to_numeric(data[column], errors='coerce')
        if self.engineer:
            # 'No internet service' is a real state, not a missing value.
            # Unknown/missing service answers leave the count missing to impute.
            known = data[OPTIONAL_SERVICES].isin(['Yes', 'No', 'No internet service']).all(axis=1)
            data['ServiceCount'] = data[OPTIONAL_SERVICES].eq('Yes').sum(axis=1).where(known)
            data['TenureBand'] = pd.cut(
                data['tenure'], bins=[-np.inf, 12, 24, 48, np.inf],
                labels=['0-12', '13-24', '25-48', '49+'],
            ).astype(object)
        return data
