# scripts/ml/model_trainer.py
"""
End-to-End Model Training Pipeline
Steps:
1. Load gold features and labels
2. Time-based train/validation split
3. Preprocessing and feature engineering
4. Train multiple models with hyperparameter tuning
5. Evaluate models and select best performer
6. Save best model to model_bank
"""

# --- 1. IMPORT LIBRARIES ---
# Main goal: Load required dependencies
import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

# --- 2. CONFIGURATION ---
# Main goal: Set up paths and parameters
# Add these paths to your directory structure
FEATURE_DIR = "datamart/gold/feature_store"
LABEL_DIR = "datamart/gold/label_store"
MODEL_BANK = "model_bank"
os.makedirs(MODEL_BANK, exist_ok=True)

# Time-based split parameters
TRAIN_START = "2023-01-01"
TRAIN_END = "2023-12-31"
VAL_START = "2024-01-01"
VAL_END = "2024-06-30"

# --- 3. DATA LOADING ---
# Main goal: Load and merge features/labels from gold layer
def load_training_data():
    """Load and merge engagement, financial risk, and label data"""
    # Load engagement features
    eng_files = [f"{FEATURE_DIR}/eng/{f}" for f in os.listdir(f"{FEATURE_DIR}/eng") 
                 if f.endswith(".parquet")]
    eng_df = pd.concat([pd.read_parquet(f) for f in eng_files])
    
    # Load financial risk features
    fin_files = [f"{FEATURE_DIR}/cust_fin_risk/{f}" for f in os.listdir(f"{FEATURE_DIR}/cust_fin_risk") 
                 if f.endswith(".parquet")]
    fin_df = pd.concat([pd.read_parquet(f) for f in fin_files])
    
    # Load labels
    label_files = [f"{LABEL_DIR}/{f}" for f in os.listdir(LABEL_DIR) 
                   if f.endswith(".parquet")]
    label_df = pd.concat([pd.read_parquet(f) for f in label_files])
    
    # Merge datasets
    features_df = pd.merge(eng_df, fin_df, on=["Customer_ID", "snapshot_date"])
    full_df = pd.merge(features_df, label_df, on=["Customer_ID", "snapshot_date"])
    
    return full_df

# --- 4. TIME-BASED SPLIT ---
# Main goal: Prevent temporal leakage using chronological split
def time_based_split(df):
    """Split data into training and validation sets by time"""
    train_df = df[(df["snapshot_date"] >= TRAIN_START) & 
                 (df["snapshot_date"] <= TRAIN_END)]
    val_df = df[(df["snapshot_date"] >= VAL_START) & 
               (df["snapshot_date"] <= VAL_END)]
    
    # Prepare features and labels
    X_train = train_df.drop(columns=["label", "label_def", "snapshot_date"])
    y_train = train_df["label"]
    X_val = val_df.drop(columns=["label", "label_def", "snapshot_date"])
    y_val = val_df["label"]
    
    return X_train, X_val, y_train, y_val

# --- 5. PREPROCESSING PIPELINE ---
# Main goal: Handle missing values and scale features
def build_preprocessor():
    """Create preprocessing pipeline for different feature types"""
    # Identify feature types
    engagement_features = [f"click_{i}m" for i in range(1, 7)]
    financial_features = [
        'Credit_History_Age', 'Num_Fin_Pdts', 'EMI_to_Salary', 
        'Debt_to_Salary', 'Repayment_Ability', 'Loans_per_Credit_Item',
        'Loan_Extent', 'Outstanding_Debt', 'Interest_Rate', 
        'Delay_from_due_date', 'Changed_Credit_Limit'
    ]
    
    # Numerical preprocessing pipeline
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Full preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, engagement_features + financial_features)
        ])
    
    return preprocessor

# --- 6. MODEL TRAINING ---
# Main goal: Train multiple models with default hyperparameters
def train_models(X_train, y_train, preprocessor):
    """Train and return multiple classifier models"""
    models = {
        "RandomForest": Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=42))
        ]),
        "XGBoost": Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', XGBClassifier(random_state=42, eval_metric='logloss'))
        ]),
        "LogisticRegression": Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])
    }
    
    # Train all models
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"Trained {name}")
    
    return models

# --- 7. MODEL EVALUATION ---
# Main goal: Select best model based on validation AUC
def evaluate_models(models, X_val, y_val):
    """Evaluate models and return best performer"""
    best_model = None
    best_score = 0
    
    for name, model in models.items():
        y_pred = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)
        print(f"{name} AUC: {auc:.4f}")
        
        if auc > best_score:
            best_score = auc
            best_model = model
            best_model_name = name
    
    print(f"Selected best model: {best_model_name} with AUC: {best_score:.4f}")
    return best_model, best_model_name

# --- 8. SAVE MODEL ---
# Main goal: Persist best model to model_bank
def save_model(model, model_name):
    """Save model as .pkl file with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{MODEL_BANK}/{model_name}_{timestamp}.pkl"
    joblib.dump(model, filename)
    print(f"Saved model to {filename}")
    return filename

# --- 9. MAIN EXECUTION ---
if __name__ == "__main__":
    print("Starting model training pipeline...")
    
    # Step 1: Load data
    df = load_training_data()
    print(f"Loaded training data with shape: {df.shape}")
    
    # Step 2: Time-based split
    X_train, X_val, y_train, y_val = time_based_split(df)
    print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}")
    
    # Step 3: Build preprocessor
    preprocessor = build_preprocessor()
    
    # Step 4: Train models
    models = train_models(X_train, y_train, preprocessor)
    
    # Step 5: Evaluate and select best model
    best_model, best_model_name = evaluate_models(models, X_val, y_val)
    
    # Step 6: Save best model
    model_path = save_model(best_model, best_model_name)
    
    print("Training pipeline completed successfully!")
