"""Load the complete pipeline and predict from original customer CSV columns."""
from pathlib import Path
import argparse
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'data' / 'demo_customers.csv')
    parser.add_argument('--model', type=Path, default=ROOT / 'models' / 'best_model.joblib')
    parser.add_argument('--output', type=Path, default=ROOT / 'results' / 'demo_predictions.csv')
    args = parser.parse_args()
    # Only load the model distributed with this project, or another trusted file.
    pipeline = joblib.load(args.model)
    raw = pd.read_csv(args.input)
    probability = pipeline.predict_proba(raw)[:, 1]
    result = pd.DataFrame({'customerID': raw.customerID if 'customerID' in raw else np.arange(len(raw)),
                           'churn_probability': probability,
                           'predicted_churn': np.where(probability >= 0.5, 'Yes', 'No')})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print('Saved:', args.output)
    print('Threshold: 0.5. A prediction is not a certainty or evidence of treatment benefit.')


if __name__ == '__main__':
    main()
