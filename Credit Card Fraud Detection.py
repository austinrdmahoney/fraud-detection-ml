import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    average_precision_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import os

# -----------------------------
# 1) Verify working directory
# -----------------------------
print("Current working directory:", os.getcwd())
print("Files in directory:", os.listdir())

# -----------------------------
# 2) Load dataset (relative path)
# -----------------------------
data = pd.read_csv(r"C:\Users\austi\Downloads\creditcard.csv")

print("\nDataset shape:", data.shape)
print("\nFraud counts:\n", data["Class"].value_counts())
print("\nFraud rate:", data["Class"].mean())

# -----------------------------
# 3) Features / target
# -----------------------------
X = data.drop("Class", axis=1)
y = data["Class"]

# -----------------------------
# 4) Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# 5) Scaled Logistic Regression pipeline
# -----------------------------
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        solver="lbfgs"
    ))
])

model.fit(X_train, y_train)

# -----------------------------
# 6) Predictions
# -----------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# -----------------------------
# 7) Evaluation (imbalanced-safe)
# -----------------------------
print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, digits=4))

print("\n=== PR-AUC (Average Precision) ===")
print(average_precision_score(y_test, y_prob))

# -----------------------------
# 8) Business output: ranked fraud risk
# -----------------------------
results = X_test.copy()
results["actual_class"] = y_test.values
results["fraud_probability"] = y_prob

top_risk = results.sort_values(
    "fraud_probability",
    ascending=False
).head(10)

print("\n=== Top 10 Highest-Risk Transactions ===")
print(top_risk[["Time", "Amount", "actual_class", "fraud_probability"]])
