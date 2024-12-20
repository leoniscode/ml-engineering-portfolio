"""Run predictions on new test data."""

import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from feature_engineering import engineer_features


def predict_churn(data_path=None):
    """Load model and predict on new data. Uses synthetic data if no path given."""
    if data_path:
        test_data = pd.read_csv(data_path)
    else:
        from feature_engineering import create_synthetic_data
        test_data = create_synthetic_data(n=500)
        print('Using synthetic test data (500 samples)\n')

    engineered = engineer_features(test_data)

    drop_cols = ['customer_id', 'churned', 'tenure_bucket']
    cat_cols = ['contract_type', 'internet_service', 'payment_method']
    for col in cat_cols:
        if col in engineered.columns:
            engineered[col] = LabelEncoder().fit_transform(engineered[col])

    X = engineered.drop(columns=[c for c in drop_cols if c in engineered.columns])

    # In production, load trained model from disk
    # For demo, train a quick model
    from train import prepare_data
    X_train, _, y_train, _ = prepare_data()
    model = xgb.XGBClassifier(n_estimators=200, max_depth=6,
                               use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict_proba(X)[:, 1]
    test_data['churn_probability'] = predictions
    test_data['predicted_churn'] = (predictions > 0.5).astype(int)

    print(f'Predictions on {len(test_data)} customers:')
    print(f'  Predicted churn: {test_data["predicted_churn"].sum()} ({test_data["predicted_churn"].mean():.1%})')
    print(f'  Avg churn probability: {predictions.mean():.3f}')

    if 'churned' in test_data.columns:
        from sklearn.metrics import roc_auc_score, accuracy_score
        auc = roc_auc_score(test_data['churned'], predictions)
        acc = accuracy_score(test_data['churned'], test_data['predicted_churn'])
        print(f'\n  Actual churn rate: {test_data["churned"].mean():.1%}')
        print(f'  ROC-AUC on test: {auc:.4f}')
        print(f'  Accuracy on test: {acc:.4f}')

    return test_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=None)
    args = parser.parse_args()
    predict_churn(args.data)
