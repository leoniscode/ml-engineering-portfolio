# Interview Kickstart — ML Engineering Curriculum

Complete coursework and projects from the Interview Kickstart ML Engineering program, covering classical machine learning through production LLM systems. Each notebook includes detailed reasoning, analysis, and key takeaways — not just code.

## Curriculum Overview

### Modules

| # | Module | Key Topics | Notebook |
|---|--------|-----------|----------|
| 1 | Build Your First ML Model | Linear/Ridge/Lasso regression, feature scaling, cross-validation | [Open](modules/01-first-ml-model/first_ml_model.ipynb) |
| 2 | Model Evaluation & Interpretation | Precision/Recall/F1, ROC-AUC, confusion matrix, SHAP | [Open](modules/02-model-evaluation/model_evaluation.ipynb) |
| 3 | Model Optimization | Grid Search, Random Search, Optuna (Bayesian), ensembles | [Open](modules/03-model-optimization/model_optimization.ipynb) |
| 4 | Neural Networks Basics | PyTorch MLP, BatchNorm, Dropout, activation functions | [Open](modules/04-neural-networks/neural_networks.ipynb) |
| 5 | ML Architectures | CNN for images, BiLSTM for text, architecture selection | [Open](modules/05-ml-architectures/ml_architectures.ipynb) |
| 6 | Transformer Based Models | Self-attention, multi-head attention, transformer blocks | [Open](modules/06-transformers/transformers.ipynb) |
| 7 | Deep Dive into LLMs | Prompt engineering, LoRA/PEFT, RAG, fine-tuning (SFT/DPO) | [Open](modules/07-deep-dive-llms/deep_dive_llms.ipynb) |
| 8 | Behavioral Interview | STAR method, impact quantification, ML-specific frameworks | [Open](modules/08-behavioral-interview/behavioral_interview.ipynb) |

### Projects

| Project | Score | Description |
|---------|-------|-------------|
| [Customer Churn Prediction](projects/ini-project-1-churn/) | **92/100** | End-to-end churn model: feature engineering → XGBoost → Optuna tuning |
| [Medical Text Classification](projects/ini-project-2-medical-text/) | **88/100** | Clinical NLP pipeline: TF-IDF → SVM → BERT fine-tuning |
| [Capstone: RAG Pipeline](projects/capstone-rag-pipeline/) | **95/100** | Production RAG: hybrid retrieval, re-ranking, RAGAS evaluation |

## Completion Summary

- **8/8 modules** completed with detailed analysis and reasoning
- **2 INI projects** scored by industry experts (92%, 88%)
- **1 Capstone project** with production-grade RAG pipeline (95%)

## How to Run

```bash
# Clone the repository
git clone https://github.com/<your-username>/interview-kickstart.git
cd interview-kickstart

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter and open any notebook
jupyter notebook
```

Individual projects have their own `requirements.txt` for project-specific dependencies:

```bash
cd projects/capstone-rag-pipeline
pip install -r requirements.txt
```

## Tech Stack

**Core ML:** Python, NumPy, pandas, scikit-learn, XGBoost, Optuna

**Deep Learning:** PyTorch, HuggingFace Transformers, PEFT/LoRA

**NLP & LLMs:** TF-IDF, BERT, sentence-transformers, LangChain

**RAG & Retrieval:** FAISS, BM25, cross-encoder re-ranking

**Evaluation:** SHAP, RAGAS, DeepEval, Evidently AI

**Visualization:** Matplotlib, Seaborn
