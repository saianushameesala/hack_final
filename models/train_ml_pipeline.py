"""
Train and save ML models for the explainability app
"""
import os
import pickle as pkl
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb

# Create directories if they don't exist
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

def save_model_and_data(model, X, y, model_path, data_path, y_col="y"):
    """Save model and dataset"""
    # Save model
    with open(model_path, 'wb') as f:
        pkl.dump(model, f)
    
    # Create and save dataset
    data = X.copy()
    data[y_col] = y
    data.to_csv(data_path, index=False)
    
    print(f"Saved model to {model_path}")
    print(f"Saved data to {data_path}")

# 1. Give Me Some Credit Dataset (Classification)
print("Training Give Me Some Credit models...")

# Load data
data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
cs_data_path = os.path.join(data_dir, "cs-training.csv")
cs_data = pd.read_csv(cs_data_path)

# The dataset may have an unnamed index column, drop it if present
if "Unnamed: 0" in cs_data.columns:
    cs_data = cs_data.drop(columns=["Unnamed: 0"])

# Target and features
y_cs = cs_data["SeriousDlqin2yrs"]
X_cs = cs_data.drop(columns=["SeriousDlqin2yrs"])

# Impute missing values for all features (use median for numeric columns)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
X_cs_imputed = pd.DataFrame(imputer.fit_transform(X_cs), columns=X_cs.columns)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_cs_imputed, y_cs, test_size=0.2, random_state=42)

# Logistic Regression
logistic_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=10, random_state=42))
])
logistic_pipeline.fit(X_train, y_train)
accuracy = logistic_pipeline.score(X_test, y_test)
print(f"Logistic Regression Accuracy (GMSC): {accuracy:.4f}")

save_model_and_data(
    logistic_pipeline,
    X_cs_imputed,
    y_cs,
    os.path.join("models", "logistic_model.pkl"),
    os.path.join("data", "cs_training_processed.csv"),
    y_col="SeriousDlqin2yrs"
)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=3)  # Reduced from 50 to 10 estimators, added max_depth
rf_model.fit(X_train, y_train)
accuracy = rf_model.score(X_test, y_test)
print(f"Random Forest Accuracy (GMSC): {accuracy:.4f}")

save_model_and_data(
    rf_model,
    X_cs_imputed,
    y_cs,
    os.path.join("models", "rf_model.pkl"),
    os.path.join("data", "cs_training_rf.csv")
)

# 2. Loan Sanction Dataset (Classification, replaces Diabetes)
print("\nTraining loan sanction models...")

# Load data
loan_data_path = os.path.join(data_dir, "loan_sanction_train.csv")
loan_data = pd.read_csv(loan_data_path)

# Drop ID column if present
if "Loan_ID" in loan_data.columns:
    loan_data = loan_data.drop(columns=["Loan_ID"])

# Target and features
y_loan = loan_data["Loan_Status"]
X_loan = loan_data.drop(columns=["Loan_Status"])

# Encode categorical variables
from sklearn.preprocessing import LabelEncoder

# Encode target variable if it's not numeric (Y/N to 1/0)
if y_loan.dtype == object or str(y_loan.dtype).startswith("category"):
    y_loan = y_loan.map({"Y": 1, "N": 0})

# Encode categorical features
for col in X_loan.select_dtypes(include=["object", "category"]).columns:
    X_loan[col] = X_loan[col].astype(str).fillna("missing")
    le = LabelEncoder()
    X_loan[col] = le.fit_transform(X_loan[col])

# Impute missing values for all features (use median for numeric columns)
imputer = SimpleImputer(strategy="median")
X_loan_imputed = pd.DataFrame(imputer.fit_transform(X_loan), columns=X_loan.columns)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_loan_imputed, y_loan, test_size=0.2, random_state=42)

# XGBoost Classifier
xgb_model = xgb.XGBClassifier(n_estimators=10, learning_rate=0.1, random_state=42, max_depth=3, use_label_encoder=False, eval_metric='logloss')  # Reduced from 50 to 10 estimators, added max_depth
xgb_model.fit(X_train, y_train)
accuracy = xgb_model.score(X_test, y_test)
print(f"XGBoost Accuracy (Loan Sanction): {accuracy:.4f}")

save_model_and_data(
    xgb_model,
    X_loan_imputed,
    y_loan,
    os.path.join("models", "xgb_model.pkl"),
    os.path.join("data", "loan_sanction_processed.csv"),
    y_col="Loan_Status"
)

print("\nAll models and datasets have been created!")
