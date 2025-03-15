"""Train medical text classifiers: baseline through BERT."""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, f1_score

from preprocess import create_synthetic_notes, clean_text, build_tfidf_features, MEDICAL_CATEGORIES


def train_baselines():
    data = create_synthetic_notes(n=2000)
    data['clean_text'] = data['text'].apply(clean_text)

    train_df, test_df = train_test_split(data, test_size=0.2, stratify=data['label'], random_state=42)
    X_train, X_test, vectorizer = build_tfidf_features(
        train_df['clean_text'].tolist(), test_df['clean_text'].tolist(),
    )
    y_train, y_test = train_df['label'].values, test_df['label'].values

    print('=' * 70)
    print('MEDICAL TEXT CLASSIFICATION — MODEL COMPARISON')
    print('=' * 70)

    models = {
        'Logistic Regression (baseline)': LogisticRegression(max_iter=1000, random_state=42),
        'Linear SVM': LinearSVC(max_iter=2000, random_state=42),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        print(f'\n{name}:')
        print(f'  Accuracy={acc:.4f}  Macro-F1={f1_macro:.4f}  Weighted-F1={f1_weighted:.4f}')
        print(classification_report(y_test, y_pred, target_names=MEDICAL_CATEGORIES))


if __name__ == '__main__':
    train_baselines()
