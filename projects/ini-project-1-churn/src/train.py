"""Train and evaluate churn prediction models."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, classification_report,
)
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler

from feature_engineering import create_synthetic_data, engineer_features

optuna.logging.set_verbosity(optuna.logging.WARNING)


def prepare_data():
    raw = create_synthetic_data(n=5000)
    data = engineer_features(raw)

    drop_cols = ['customer_id', 'churned', 'tenure_bucket']
    cat_cols = ['contract_type', 'internet_service', 'payment_method']
    for col in cat_cols:
        data[col] = LabelEncoder().fit_transform(data[col])

    X = data.drop(columns=drop_cols, errors='ignore')
    y = data['churned']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42,
    )

    scaler = StandardScaler()
    num_cols = X_train.select_dtypes(include=[np.number]).columns
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    return X_train, X_test, y_train, y_test


def evaluate(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f'{name:<35} AUC={auc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}')
    return {'AUC': auc, 'Precision': prec, 'Recall': rec, 'F1': f1}


def tune_xgboost(X_train, y_train):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 5),
        }
        model = xgb.XGBClassifier(**params, use_label_encoder=False,
                                   eval_metric='logloss', random_state=42)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in skf.split(X_train, y_train):
            model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            y_prob = model.predict_proba(X_train.iloc[val_idx])[:, 1]
            scores.append(roc_auc_score(y_train.iloc[val_idx], y_prob))
        return np.mean(scores)

    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=50)
    return study.best_params


def main():
    X_train, X_test, y_train, y_test = prepare_data()

    print('=' * 80)
    print('CHURN PREDICTION — MODEL COMPARISON')
    print('=' * 80)
    print(f'Train: {len(X_train)}, Test: {len(X_test)}, '
          f'Churn rate: {y_test.mean():.2%}\n')

    # Baseline
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    evaluate(lr, X_test, y_test, 'Logistic Regression (baseline)')

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    evaluate(rf, X_test, y_test, 'Random Forest')

    # XGBoost default
    xgb_default = xgb.XGBClassifier(n_estimators=200, max_depth=6,
                                     use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_default.fit(X_train, y_train)
    evaluate(xgb_default, X_test, y_test, 'XGBoost (default)')

    # XGBoost tuned with Optuna
    print('\nTuning XGBoost with Optuna (50 trials)...')
    best_params = tune_xgboost(X_train, y_train)
    xgb_tuned = xgb.XGBClassifier(**best_params, use_label_encoder=False,
                                    eval_metric='logloss', random_state=42)
    xgb_tuned.fit(X_train, y_train)
    results = evaluate(xgb_tuned, X_test, y_test, 'XGBoost (Optuna tuned)')

    print(f'\nBest Optuna params: {best_params}')
    print(f'\nFinal classification report (XGBoost tuned):')
    print(classification_report(y_test, xgb_tuned.predict(X_test)))


if __name__ == '__main__':
    main()
