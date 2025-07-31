import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Step 1: Load Data
df = pd.read_csv("Task 3 and 4_Loan_Data.csv")

# Step 2: Prepare Features and Target
X = df.drop(columns=["customer_id", "default"])
y = df["default"]

# Step 3: Standardize Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 5: Train Logistic Regression Model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Optional: Evaluate the model
y_probs = log_reg.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_probs)
print("ROC AUC Score:", roc_auc)
print(classification_report(y_test, log_reg.predict(X_test)))

# Step 6: Define Expected Loss Function
def predict_expected_loss(credit_lines_outstanding, loan_amt_outstanding, total_debt_outstanding,
                          income, years_employed, fico_score, recovery_rate=0.1):
    features = np.array([[credit_lines_outstanding, loan_amt_outstanding, total_debt_outstanding,
                          income, years_employed, fico_score]])
    features_scaled = scaler.transform(features)
    pd = log_reg.predict_proba(features_scaled)[0, 1]
    expected_loss = pd * (1 - recovery_rate) * loan_amt_outstanding
    return {
        "Predicted_PD": pd,
        "Expected_Loss": expected_loss
    }

# Step 7: Example Prediction
example = predict_expected_loss(
    credit_lines_outstanding=2,
    loan_amt_outstanding=5000,
    total_debt_outstanding=8000,
    income=40000,
    years_employed=4,
    fico_score=620
)
print("Example Prediction:", example)
