# Medical Text Classification

**Score: 88/100** — Evaluated by industry expert

## Executive Summary

Clinical documentation — discharge summaries, consultation notes, diagnostic reports — accounts for millions of documents processed daily across healthcare systems. Manual routing of these notes to the correct medical department is slow, error-prone, and contributes to delayed patient care. This project builds an NLP classifier that automatically categorizes clinical notes into medical specialties, demonstrating a progression from classical baselines to transformer-based models.

## Business Problem

A healthcare system processes 2,000+ clinical notes daily across five departments:

**Current state:**
- Manual triage by administrative staff with limited medical vocabulary
- Average routing time: 15-30 minutes per note
- 8-12% misrouting rate, leading to delayed consultations
- Annual cost of misrouting estimated at $500K+ (delayed care, rework)

**Goal:** Automate clinical note classification with >85% accuracy, reducing routing errors and enabling faster specialist referrals.

### Target Categories
| Department | Key Signals | Note Volume |
|-----------|-------------|-------------|
| Cardiology | heart, cardiac, ECG, arrhythmia, hypertension | ~22% |
| Neurology | headache, seizure, brain, MRI, stroke | ~20% |
| Orthopedics | fracture, joint, bone, knee, arthritis | ~18% |
| Gastroenterology | abdomen, liver, nausea, endoscopy | ~19% |
| Pulmonology | lung, breathing, cough, asthma, respiratory | ~21% |

## Approach

| Step | Method | Rationale |
|------|--------|-----------|
| Text Preprocessing | Lowercasing, regex cleaning, whitespace normalization | Reduces vocabulary noise without losing clinical signal |
| Feature Extraction | TF-IDF with bigrams (max 5,000 features) | Captures multi-word medical terms ("chest pain", "heart failure") |
| Baseline | TF-IDF + Logistic Regression | Standard NLP baseline — fast, interpretable, surprisingly strong |
| Improvement 1 | TF-IDF + Linear SVM | Max-margin objective works well in high-dimensional sparse spaces |
| Improvement 2 | BiLSTM + GloVe | Sequence modeling captures word order and context |
| Improvement 3 | BERT fine-tuned | Pre-trained on biomedical text; understands medical language |

## Results

| Model | Accuracy | Macro F1 | Weighted F1 |
|-------|----------|----------|-------------|
| TF-IDF + Logistic Regression (baseline) | 0.78 | 0.76 | 0.78 |
| TF-IDF + Linear SVM | 0.81 | 0.79 | 0.81 |
| BiLSTM + GloVe | 0.84 | 0.82 | 0.84 |
| BERT fine-tuned | **0.89** | **0.88** | **0.89** |

**Improvement over baseline:** +11% accuracy, +12% F1

### Key Findings
- **Domain vocabulary dominates:** The top 5 TF-IDF features per category are almost exclusively domain-specific terms
- **Bigrams matter:** "chest pain" is highly discriminative; "chest" and "pain" individually are not
- **BERT's contextual understanding:** Distinguishes "patient failed to respond" (neurology) from "heart failure" (cardiology) — TF-IDF cannot

## Evaluation Criteria (Graded)

| Criteria | Score | Notes |
|----------|-------|-------|
| Feature Engineering | 87% | TF-IDF with bigrams, clinical text preprocessing |
| Baseline Model Training | 90% | Logistic regression with proper stratified split |
| Model Evaluation | 88% | Per-class F1, confusion matrix, error analysis |
| Improvements over Baseline | 88% | Clear progression from TF-IDF → SVM → BiLSTM → BERT |
| Final Running Code | 87% | Clean pipeline, reproducible |

## Limitations and Next Steps

- **Synthetic data:** Real clinical notes have far more variability (abbreviations, typos, mixed languages)
- **PHI handling:** Production deployment requires PII/PHI detection and redaction (Presidio)
- **More categories:** Real systems route to 20+ specialties, not 5
- **Confidence thresholds:** Low-confidence predictions should be routed to human review
- **Active learning:** Use misclassified notes to improve the model iteratively

## How to Run

```bash
pip install -r requirements.txt

# Full analysis (recommended)
jupyter notebook medical_text_classification.ipynb

# Or run the pipeline scripts
python src/preprocess.py    # Preprocess clinical notes
python src/train.py         # Train all models
```

## Tech Stack

Python, scikit-learn, TF-IDF, PyTorch, HuggingFace Transformers, BERT, Matplotlib
