"""Text preprocessing for medical classification."""

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


MEDICAL_CATEGORIES = [
    'cardiology', 'neurology', 'orthopedics', 'gastroenterology', 'pulmonology',
]

CATEGORY_TERMS = {
    'cardiology': ['heart', 'cardiac', 'chest pain', 'ecg', 'arrhythmia', 'hypertension', 'bp'],
    'neurology': ['headache', 'seizure', 'brain', 'mri', 'numbness', 'stroke', 'cognitive'],
    'orthopedics': ['fracture', 'joint', 'bone', 'knee', 'arthritis', 'spine', 'mobility'],
    'gastroenterology': ['abdomen', 'liver', 'nausea', 'digestive', 'endoscopy', 'gastric'],
    'pulmonology': ['lung', 'breathing', 'cough', 'asthma', 'pulmonary', 'oxygen', 'respiratory'],
}


def create_synthetic_notes(n=2000):
    np.random.seed(42)
    templates = {
        'cardiology': [
            'Patient presents with chest pain and elevated blood pressure. ECG shows {finding}. '
            'History of hypertension. Recommend cardiac monitoring and {treatment}.',
        ],
        'neurology': [
            'Patient reports persistent headache and numbness in extremities. MRI of brain shows {finding}. '
            'Neurological exam reveals {symptom}. Recommend {treatment}.',
        ],
        'orthopedics': [
            'Patient presents with joint pain and limited mobility in {location}. '
            'X-ray reveals {finding}. Physical therapy recommended along with {treatment}.',
        ],
        'gastroenterology': [
            'Patient complains of abdominal pain and nausea for {duration}. '
            'Endoscopy findings: {finding}. Liver function tests {result}. Prescribe {treatment}.',
        ],
        'pulmonology': [
            'Patient presents with chronic cough and difficulty breathing. '
            'Pulmonary function test shows {finding}. Chest X-ray {result}. Start {treatment}.',
        ],
    }

    notes, labels = [], []
    for _ in range(n):
        category = np.random.choice(MEDICAL_CATEGORIES)
        template = np.random.choice(templates[category])
        note = template.format(
            finding='mild abnormality', treatment='standard care',
            symptom='mild deficit', location='lower extremity',
            duration='2 weeks', result='within normal limits',
        )
        extra = np.random.choice(CATEGORY_TERMS[category], size=3, replace=True)
        note += f' Additional findings related to {", ".join(extra)}.'
        notes.append(note)
        labels.append(MEDICAL_CATEGORIES.index(category))

    return pd.DataFrame({'text': notes, 'label': labels, 'category': [MEDICAL_CATEGORIES[l] for l in labels]})


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def build_tfidf_features(train_texts, test_texts, max_features=5000):
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=2)
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    return X_train, X_test, vectorizer


if __name__ == '__main__':
    data = create_synthetic_notes()
    print(f'Dataset: {len(data)} notes, {data["category"].nunique()} categories')
    print(f'\nCategory distribution:\n{data["category"].value_counts()}')
