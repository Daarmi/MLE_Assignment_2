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
import os
import sys
import joblib
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

# --- 2. CONFIGURATION ---
def parse_args():

    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-date", required=True, help="Airflow execution date (YYYY-MM-DD)")
    parser.add_argument("--feature-dir", default="datamart/gold/feature_store", help="Base directory for features")
    parser.add_argument("--label-dir", default="datamart/gold/label_store", help="Base directory for labels")
    parser.add_argument("--model-bank", default="model_bank", help="Directory to save trained models")
    return parser.parse_args()

# --- 3. DATA LOADING ---
def load_training_data(feature_dir, label_dir, max_date):
    """Load and merge features/labels with date validation"""
    # Verify data exists for all required dates
    required_dates = [max_date - relativedelta(months=i) for i in range(1, 8)]
    date_strs = [d.strftime("%Y_%m_%d") for d in required_dates]
    
    # Check feature store files
    missing_files = []
    for date_str in date_strs:
        eng_path = f"{feature_dir}/eng/gold_ft_store_engagement_{date_str}.parquet"
        fin_path = f"{feature_dir}/cust_fin_risk/gold_ft_store_cust_fin_risk_{date_str}.parquet"
        label_path = f"{label_dir}/gold_label_store_{date_str}.parquet"
        
        if not os.path.exists(eng_path): missing_files.append(eng_path)
        if not os.path.exists(fin_path): missing_files.append(fin_path)
        if not os.path.exists(label_path): missing_files.append(label_path)
    
    if missing_files:
        raise FileNotFoundError(f"Missing {len(missing_files)} required data files. "
                                f"First 3 missing: {missing_files[:3]}")
                                
    """Load and merge features/labels with date validation"""
    # Verify data exists for all required dates
    required_dates = [max_date - relativedelta(months=i) for i in range(1, 8)]
    date_strs = [d.strftime("%Y_%m_%d") for d in required_dates]
    
    # Check feature store files
    for date_str in date_strs:
        eng_path = f"{feature_dir}/eng/gold_ft_store_engagement_{date_str}.parquet"
        fin_path = f"{feature_dir}/cust_fin_risk/gold_ft_store_cust_fin_risk_{date_str}.parquet"
        label_path = f"{label_dir}/gold_label_store_{date_str}.parquet"
        
        if not all(os.path.exists(p) for p in [eng_path, fin_path, label_path]):
            missing = [p for p in [eng_path, fin_path, label_path] if not os.path.exists(p)]
            raise FileNotFoundError(f"Missing data files: {missing}")

    # Load data using Spark would be better, but we'll use pandas for simplicity
    # For large datasets, consider PySpark implementation
    eng_files = [f"{feature_dir}/eng/{f}" for f in os.listdir(f"{feature_dir}/eng") 
                 if f.startswith("gold_ft_store_engagement") and f.endswith(".parquet")]
    fin_files = [f"{feature_dir}/cust_fin_risk/{f}" for f in os.listdir(f"{feature_dir}/cust_fin_risk") 
                 if f.startswith("gold_ft_store_cust_fin_risk") and f.endswith(".parquet")]
    label_files = [f"{label_dir}/{f}" for f in os.listdir(label_dir) 
                   if f.startswith("gold_label_store") and f.endswith(".parquet")]
    
    # Load and merge datasets
    eng_df = pd.concat([pd.read_parquet(f) for f in eng_files])
    fin_df = pd.concat([pd.read_parquet(f) for f in fin_files])
    label_df = pd.concat([pd.read_parquet(f) for f in label_files])
    
    features_df = pd.merge(eng_df, fin_df, on=["Customer_ID", "snapshot_date"])
    full_df = pd.merge(features_df, label_df, on=["Customer_ID", "snapshot_date"])
    
    return full_df

# --- 4. TIME-BASED SPLIT ---
def time_based_split(df, execution_date):
    """Split data into training and validation sets by time"""
    # Convert to datetime if needed
    if not isinstance(df['snapshot_date'].dtype, pd.DatetimeTZDtype):
        df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])
    
    # Calculate split points (using 70/30 time-based split)
    max_date = execution_date
    train_end = max_date - relativedelta(months=3)
    val_start = train_end + relativedelta(days=1)
    
    train_df = df[df["snapshot_date"] <= train_end]
    val_df = df[df["snapshot_date"] >= val_start]
    
    # Prepare features and labels
    X_train = train_df.drop(columns=["label", "label_def", "snapshot_date"])
    y_train = train_df["label"]
    X_val = val_df.drop(columns=["label", "label_def", "snapshot_date"])
    y_val = val_df["label"]
    
    return X_train, X_val, y_train, y_val

# --- 5. PREPROCESSING PIPELINE ---
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
def train_models(X_train, y_train, preprocessor):
    """Train and return multiple classifier models"""
    models = {
        "RandomForest": Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=100, 
                random_state=42,
                class_weight='balanced'
            ))
        ]),
        "XGBoost": Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', XGBClassifier(
                random_state=42, 
                eval_metric='auc',
                scale_pos_weight=np.sum(y_train == 0) / np.sum(y_train == 1)
            )
        ]),
        "LogisticRegression": Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(
                random_state=42, 
                max_iter=1000,
                class_weight='balanced'
            ))
        ])
    }
    
    # Train all models
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"Trained {name}")
    
    return models

# --- 7. MODEL EVALUATION ---
def evaluate_models(models, X_val, y_val):
    """Evaluate models and return best performer"""
    best_model = None
    best_score = 0
    best_model_name = ""
    
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
def save_model(model, model_name, execution_date):
    """Save model with execution date in filename"""
    os.makedirs(MODEL_BANK, exist_ok=True)
    date_str = execution_date.strftime("%Y%m%d")
    filename = f"{MODEL_BANK}/{model_name}_{date_str}.pkl"
    joblib.dump(model, filename)
    print(f"Saved model to {filename}")
    return filename

# --- 9. MAIN EXECUTION ---
if __name__ == "__main__":
    print("Starting model training pipeline...")
    
    # Parse arguments
    args = parse_args()
    execution_date = datetime.strptime(args.execution_date, "%Y-%m-%d")
    FEATURE_DIR = args.feature_dir
    LABEL_DIR = args.label_dir
    MODEL_BANK = args.model_bank
    
    print(f"Execution date: {execution_date.strftime('%Y-%m-%d')}")
    print(f"Feature directory: {FEATURE_DIR}")
    print(f"Label directory: {LABEL_DIR}")
    print(f"Model bank: {MODEL_BANK}")
    
    # Step 1: Load data with date validation
    try:
        df = load_training_data(FEATURE_DIR, LABEL_DIR, execution_date)
        print(f"Loaded training data with shape: {df.shape}")
    except Exception as e:
        print(f"Data loading failed: {str(e)}")
        sys.exit(1)
    
    # Step 2: Time-based split
    X_train, X_val, y_train, y_val = time_based_split(df, execution_date)
    print(f"Train size: {len(X_train)} ({X_train['snapshot_date'].min()} to {X_train['snapshot_date'].max()})")
    print(f"Validation size: {len(X_val)} ({X_val['snapshot_date'].min()} to {X_val['snapshot_date'].max()})")
    
    # Step 3: Build preprocessor
    preprocessor = build_preprocessor()
    
    # Step 4: Train models
    models = train_models(X_train, y_train, preprocessor)
    
    # Step 5: Evaluate and select best model
    best_model, best_model_name = evaluate_models(models, X_val, y_val)
    
    # Step 6: Save best model
    model_path = save_model(best_model, best_model_name, execution_date)
    
    print("Training pipeline completed successfully!")