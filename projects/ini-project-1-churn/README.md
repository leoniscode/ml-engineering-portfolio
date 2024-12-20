# Customer Churn Prediction

**Score: 92/100** — Evaluated by industry expert

## Executive Summary

Telecom companies lose an estimated 15-25% of their subscribers annually, with acquisition costs running 5-10x higher than retention costs. This project builds a predictive model that identifies at-risk customers before they churn, enabling targeted retention campaigns that reduce churn rate and protect revenue.

## Business Problem

A telecom provider with 5,000+ subscribers faces rising churn. The current approach — blanket retention offers to all customers — is expensive and inefficient.

**Current state:**
- ~25% annual churn rate
- Retention campaigns sent to the entire customer base
- No prioritization of high-risk accounts
- Estimated $2.4M annual revenue loss from preventable churn

**Goal:** Build a model that identifies 80%+ of churning customers while targeting less than 30% of the base — a 2.7x improvement over random targeting.

## Approach

| Step | Method | Rationale |
|------|--------|-----------|
| Feature Engineering | Temporal, interaction, RFM-inspired features | Raw features tell WHAT happened; engineered features capture WHY it matters |
| Baseline | Logistic Regression | Interpretable, fast, sets the performance floor |
| Improvement 1 | Random Forest | Captures non-linear relationships the linear model misses |
| Improvement 2 | XGBoost (default) | State-of-the-art for tabular data |
| Improvement 3 | XGBoost + Optuna (50 trials) | Bayesian hyperparameter optimization for the final push |

## Results

| Model | ROC-AUC | Precision | Recall | F1 |
|-------|---------|-----------|--------|-----|
| Logistic Regression (baseline) | 0.823 | 0.71 | 0.68 | 0.69 |
| Random Forest | 0.876 | 0.78 | 0.74 | 0.76 |
| XGBoost (default) | 0.891 | 0.81 | 0.77 | 0.79 |
| XGBoost (Optuna tuned) | **0.905** | **0.83** | **0.79** | **0.81** |

**Improvement over baseline:** +0.082 AUC (+10% relative)

### Top Predictive Features
1. `is_month_to_month` — contract type is the single strongest churn signal
2. `engagement_score` — composite metric combining referrals, services, and support tickets
3. `support_per_month` — frequent support calls signal dissatisfaction
4. `tenure_months` — new customers churn at 2x the rate of long-tenured ones

## Evaluation Criteria (Graded)

| Criteria | Score | Notes |
|----------|-------|-------|
| Feature Engineering | 95% | 8 engineered features with temporal, interaction, and RFM patterns |
| Baseline Model Training | 90% | Logistic regression with proper scaling and stratified split |
| Model Evaluation | 93% | ROC-AUC, precision/recall, confusion matrix, feature importance |
| Improvements over Baseline | 92% | Progressive improvement with Optuna Bayesian optimization |
| Final Running Code | 90% | Clean pipeline with prediction support for new data |

## Limitations and Next Steps

- **Synthetic data:** Real telecom data would include richer behavioral signals (call logs, app usage, billing disputes)
- **Threshold optimization:** Current default 0.5 threshold; a cost-sensitive threshold could improve business ROI
- **Temporal validation:** Time-based train/test split would better simulate production deployment
- **Model monitoring:** No drift detection — production deployment would need Evidently or similar

## How to Run

```bash
pip install -r requirements.txt

# Full analysis (recommended)
jupyter notebook customer_churn_prediction.ipynb

# Or run the pipeline scripts
python src/train.py                    # Train all models
python src/predict.py --data test.csv  # Predict on new data
```

## Tech Stack

Python, pandas, scikit-learn, XGBoost, Optuna, SHAP, Matplotlib
