# Credit Card Fraud Detection using Imbalanced Machine Learning

## Applied Research Overview
This project presents an applied machine learning study focused on **credit card fraud detection under extreme class imbalance**. The objective is to design, train, and evaluate a supervised classification model that prioritizes **fraud recall and probabilistic risk ranking**, rather than misleading aggregate accuracy.

The work follows industry-aligned practices used in financial risk analytics, fraud monitoring systems, and applied data science research.

---

## Research Problem
Credit card fraud detection presents a fundamental challenge due to **rare-event classification**, where fraudulent transactions represent less than 0.2% of all observations. Traditional accuracy-based evaluation metrics fail in such contexts, often masking poor fraud detection performance.

**Research Question:**  
How effectively can a cost-sensitive logistic regression model identify fraudulent transactions while maintaining interpretable probabilistic risk scores in an imbalanced dataset?

---

## Dataset Description
- Source: Kaggle – Credit Card Fraud Detection Dataset  
- Observations: 284,807 transactions  
- Fraud cases: 492 (~0.17%)  
- Features:
  - `V1–V28`: PCA-transformed features for privacy
  - `Time`: seconds elapsed since first transaction
  - `Amount`: transaction value
- Target variable:
  - `Class` (1 = fraud, 0 = legitimate)

Due to confidentiality constraints, original feature semantics are unavailable, making this dataset well-suited for methodological evaluation rather than feature interpretability studies.

---

## Methodology

### Data Partitioning
- Stratified train/test split (80/20) to preserve fraud prevalence across sets.

### Model Selection
- **Logistic Regression** with:
  - `class_weight="balanced"` to counter class imbalance
  - L2 regularization
  - `lbfgs` optimizer

Logistic Regression was selected due to:
- Interpretability
- Probabilistic output suitability
- Alignment with financial risk scoring systems

### Feature Scaling
- `StandardScaler` applied within a Scikit-Learn pipeline to:
  - Improve convergence
  - Normalize feature magnitudes
  - Ensure reproducibility

---

## Evaluation Framework

Given the severe class imbalance, evaluation emphasizes **precision-recall tradeoffs** rather than accuracy.

### Metrics Used
- **Recall (Fraud Class):** Measures fraud capture rate
- **Precision (Fraud Class):** Measures false alarm burden
- **F1-Score:** Harmonic balance
- **PR-AUC (Average Precision):** Primary performance metric

PR-AUC was selected because it provides a meaningful assessment of model performance when positive cases are rare.

---

## Results Summary

- **Fraud Recall:** ~91.8%
- **PR-AUC:** ~0.72
- **False Positives:** Expected and acceptable at default threshold
- **Missed Frauds:** Very low (single-digit count in test set)

These results indicate the model is effective as a **fraud screening and alerting system**, suitable for downstream analyst review rather than autonomous decision-making.

---

## Risk Scoring and Business Application
Instead of binary-only predictions, the model outputs **continuous fraud probabilities**, enabling:

- Transaction risk ranking
- Analyst triage prioritization
- Adjustable decision thresholds based on business cost tolerance
- Integration into fraud monitoring pipelines

The system produces a ranked list of highest-risk transactions, reflecting real-world operational workflows in financial institutions.

---

## Limitations
- PCA-transformed features limit interpretability
- No temporal sequence modeling
- Threshold selection requires cost-sensitive calibration
- Logistic model assumes linear decision boundary

These limitations are consistent with baseline fraud detection systems and motivate future extensions.

---

## Future Work
- Threshold optimization using cost matrices
- Comparison with tree-based models (Random Forest, XGBoost)
- Precision-Recall curve visualization
- Temporal fraud drift analysis
- Integration with real-time scoring pipelines

---

## Reproducibility
This project emphasizes:
- Deterministic random seeds
- Pipeline-based preprocessing
- Portable relative paths
- Explicit metric reporting

Dataset is intentionally excluded from the repository due to size constraints and must be downloaded separately from Kaggle.

---

## How to Run

1. Download `creditcard.csv` from Kaggle
2. Place it in the project directory
3. Install dependencies:
   ```bash
   pip install pandas scikit-learn
