# scripts/ml/predictor.py
"""
End-to-End Batch Inference Pipeline
Steps:
1. Check if required data exists
2. Load the latest production model from model_bank
3. Load gold features for a specific snapshot date
4. Preprocess features (using the model's built-in pipeline)
5. Generate predictions and confidence scores
6. Store predictions in inference store for downstream monitoring
"""

# --- 1. IMPORT LIBRARIES ---
import os
import sys
import joblib
import pandas as pd
import argparse
from datetime import datetime

# --- 2. CONFIGURATION ---
# Defaults (overridable via command line)
DEFAULT_MODEL_BANK = "model_bank"
DEFAULT_FEATURE_DIR = "datamart/gold/feature_store"
DEFAULT_INFERENCE_STORE = "datamart/gold/inference_store"

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-date", required=True)
    parser.add_argument("--model-bank", default=DEFAULT_MODEL_BANK)
    parser.add_argument("--feature-dir", default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--inference-store", default=DEFAULT_INFERENCE_STORE)
    return parser.parse_args()

# --- 3. DATA READINESS CHECK ---
def check_prediction_data(snapshot_date, feature_dir):
    """Check if required prediction data exists"""
    date_str = datetime.strptime(snapshot_date, "%Y-%m-%d").strftime("%Y_%m_%d")
    paths_to_check = [
        f"{feature_dir}/eng/gold_ft_store_engagement_{date_str}.parquet",
        f"{feature_dir}/cust_fin_risk/gold_ft_store_cust_fin_risk_{date_str}.parquet"
    ]
    missing = [p for p in paths_to_check if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing prediction data: {missing}")
    return True

# --- 4. MODEL LOADING ---
def load_latest_model(model_bank):
    """Load most recent model from model_bank"""
    if not os.path.exists(model_bank):
        raise FileNotFoundError(f"Model bank directory not found: {model_bank}")
    
    model_files = [f for f in os.listdir(model_bank) if f.endswith(".pkl")]
    if not model_files:
        raise ValueError("No models found in model bank")
    
    # Get latest by timestamp in filename
    latest_model = sorted(model_files, reverse=True)[0]
    model_path = os.path.join(model_bank, latest_model)
    return joblib.load(model_path), latest_model

# --- 5. FEATURE LOADING ---
def load_features(snapshot_date, feature_dir):
    """Load engagement and financial features for a specific date"""
    # Convert to file naming convention
    date_str = snapshot_date.replace("-", "_")
    
    # Engagement features
    eng_path = f"{feature_dir}/eng/gold_ft_store_engagement_{date_str}.parquet"
    eng_df = pd.read_parquet(eng_path) if os.path.exists(eng_path) else None
    
    # Financial risk features
    fin_path = f"{feature_dir}/cust_fin_risk/gold_ft_store_cust_fin_risk_{date_str}.parquet"
    fin_df = pd.read_parquet(fin_path) if os.path.exists(fin_path) else None
    
    # Merge features
    if eng_df is not None and fin_df is not None:
        return pd.merge(eng_df, fin_df, on=["Customer_ID", "snapshot_date"])
    raise FileNotFoundError(f"Feature files missing for {snapshot_date}")

# --- 6. PREDICTION GENERATION ---
def generate_predictions(model, features_df):
    """Generate predictions using model pipeline"""
    # Prepare features (exclude metadata)
    X = features_df.drop(columns=["Customer_ID", "snapshot_date"])
    
    # Generate predictions
    features_df["default_probability"] = model.predict_proba(X)[:, 1]
    features_df["prediction_class"] = model.predict(X)
    
    return features_df[["Customer_ID", "snapshot_date", 
                       "default_probability", "prediction_class"]]

# --- 7. STORE PREDICTIONS ---
def store_predictions(preds_df, snapshot_date, inference_store):
    """Save predictions to inference store"""
    os.makedirs(inference_store, exist_ok=True)
    date_str = snapshot_date.replace("-", "_")
    output_path = f"{inference_store}/predictions_{date_str}.parquet"
    preds_df.to_parquet(output_path)
    return output_path

# --- 8. MAIN EXECUTION ---
def run_inference(snapshot_date, model_bank, feature_dir, inference_store):
    """Orchestrate inference pipeline for a single date"""
    print(f"\nStarting inference for {snapshot_date}")
    
    # Load model
    model, model_name = load_latest_model(model_bank)
    print(f"Loaded model: {model_name}")
    
    # Load features
    features_df = load_features(snapshot_date, feature_dir)
    print(f"Loaded features with {len(features_df)} records")
    
    # Generate predictions
    preds_df = generate_predictions(model, features_df)
    print(f"Generated predictions for {len(preds_df)} customers")
    
    # Store results
    output_path = store_predictions(preds_df, snapshot_date, inference_store)
    print(f"Predictions saved to {output_path}")
    
    return preds_df

# For Airflow integration
if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    try:
        # Check data readiness first
        check_prediction_data(args.snapshot_date, args.feature_dir)
        
        # Run inference if data exists
        run_inference(
            snapshot_date=args.snapshot_date,
            model_bank=args.model_bank,
            feature_dir=args.feature_dir,
            inference_store=args.inference_store
        )
    
    except (FileNotFoundError, ValueError) as e:
        print(f"Skipping prediction: {str(e)}")
        sys.exit(0)  # Clean exit