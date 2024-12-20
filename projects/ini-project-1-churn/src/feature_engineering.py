"""Feature engineering for customer churn prediction."""

import numpy as np
import pandas as pd


def create_synthetic_data(n=5000):
    np.random.seed(42)
    data = pd.DataFrame({
        'customer_id': range(n),
        'tenure_months': np.random.exponential(24, n).clip(1, 72).astype(int),
        'monthly_charges': np.random.normal(65, 25, n).clip(20, 150),
        'total_charges': np.zeros(n),
        'contract_type': np.random.choice(['month-to-month', 'one_year', 'two_year'], n, p=[0.5, 0.3, 0.2]),
        'internet_service': np.random.choice(['fiber', 'dsl', 'none'], n, p=[0.45, 0.35, 0.2]),
        'payment_method': np.random.choice(['electronic', 'mailed_check', 'bank_transfer', 'credit_card'], n),
        'num_support_tickets': np.random.poisson(2, n),
        'num_referrals': np.random.poisson(1, n),
        'online_security': np.random.choice([0, 1], n, p=[0.5, 0.5]),
        'tech_support': np.random.choice([0, 1], n, p=[0.55, 0.45]),
        'streaming_tv': np.random.choice([0, 1], n, p=[0.4, 0.6]),
        'paperless_billing': np.random.choice([0, 1], n, p=[0.4, 0.6]),
    })
    data['total_charges'] = data['monthly_charges'] * data['tenure_months']

    churn_prob = (
        0.15
        + 0.2 * (data['contract_type'] == 'month-to-month')
        + 0.1 * (data['internet_service'] == 'fiber')
        + 0.05 * (data['num_support_tickets'] > 3)
        - 0.1 * (data['tenure_months'] > 24)
        - 0.05 * data['online_security']
    ).clip(0.05, 0.85)

    data['churned'] = (np.random.random(n) < churn_prob).astype(int)
    return data


def engineer_features(df):
    """Create engineered features from raw customer data."""
    result = df.copy()

    # Temporal features
    result['charge_per_month_avg'] = result['total_charges'] / result['tenure_months'].clip(1)
    result['tenure_bucket'] = pd.cut(result['tenure_months'], bins=[0, 6, 12, 24, 48, 100],
                                     labels=['0-6m', '6-12m', '1-2y', '2-4y', '4y+'])
    result['is_new_customer'] = (result['tenure_months'] <= 6).astype(int)

    # Interaction features
    result['support_per_month'] = result['num_support_tickets'] / result['tenure_months'].clip(1)
    result['high_charge_short_tenure'] = (
        (result['monthly_charges'] > result['monthly_charges'].median()) &
        (result['tenure_months'] < 12)
    ).astype(int)

    # Service aggregation
    service_cols = ['online_security', 'tech_support', 'streaming_tv']
    result['num_services'] = result[service_cols].sum(axis=1)
    result['has_all_services'] = (result['num_services'] == len(service_cols)).astype(int)

    # RFM-inspired scores
    result['engagement_score'] = (
        result['num_referrals'] * 2
        + result['num_services']
        - result['num_support_tickets'] * 0.5
    ).clip(0)

    # Contract risk
    result['is_month_to_month'] = (result['contract_type'] == 'month-to-month').astype(int)
    result['is_fiber'] = (result['internet_service'] == 'fiber').astype(int)
    result['is_paperless_electronic'] = (
        (result['paperless_billing'] == 1) &
        (result['payment_method'] == 'electronic')
    ).astype(int)

    return result


if __name__ == '__main__':
    data = create_synthetic_data()
    engineered = engineer_features(data)
    print(f'Original features: {data.shape[1]}')
    print(f'After engineering: {engineered.shape[1]}')
    print(f'Churn rate: {data["churned"].mean():.2%}')
    print(f'\nNew features: {[c for c in engineered.columns if c not in data.columns]}')
