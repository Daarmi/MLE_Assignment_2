# scripts/ml/predictor.py
"""
End-to-End Batch Inference Pipeline
Steps:
1. Load the latest production model from model_bank
2. Load gold features for a specific snapshot date
3. Preprocess features (using the model's built-in pipeline)
4. Generate predictions and confidence scores
5. Store predictions in inference store for downstream monitoring
"""

# --- 1. IMPORT LIBRARIES ---
# Main goal: Reuse existing dependencies (no new packages needed)
import os
import joblib
import pandas as pd
from datetime import datetime

# --- 2. CONFIGURATION ---
# Main goal: Define paths and parameters
MODEL_BANK = "model_bank"
FEATURE_DIR = "datamart/gold/feature_store"
INFERENCE_STORE = "datamart/gold/inference_store"
os.makedirs(INFERENCE_STORE, exist_ok=True)

# --- 3. MODEL LOADING ---
# Main goal: Retrieve the latest trained model
def load_latest_model():
    """Load most recent model from model_bank"""
    model_files = [f for f in os.listdir(MODEL_BANK) if f.endswith(".pkl")]
    if not model_files:
        raise ValueError("No models found in model bank")
    
    # Get latest by timestamp in filename
    latest_model = sorted(model_files, reverse=True)[0]
    model_path = os.path.join(MODEL_BANK, latest_model)
    return joblib.load(model_path), latest_model

# --- 4. FEATURE LOADING ---
# Main goal: Fetch point-in-time correct features
def load_features(snapshot_date):
    """Load engagement and financial features for a specific date"""
    # Convert to file naming convention
    date_str = snapshot_date.replace("-", "_")
    
    # Engagement features
    eng_path = f"{FEATURE_DIR}/eng/gold_ft_store_engagement_{date_str}.parquet"
    eng_df = pd.read_parquet(eng_path) if os.path.exists(eng_path) else None
    
    # Financial risk features
    fin_path = f"{FEATURE_DIR}/cust_fin_risk/gold_ft_store_cust_fin_risk_{date_str}.parquet"
    fin_df = pd.read_parquet(fin_path) if os.path.exists(fin_path) else None
    
    # Merge features
    if eng_df is not None and fin_df is not None:
        return pd.merge(eng_df, fin_df, on=["Customer_ID", "snapshot_date"])
    raise FileNotFoundError(f"Feature files missing for {snapshot_date}")

# --- 5. PREDICTION GENERATION ---
# Main goal: Generate predictions with probabilities
def generate_predictions(model, features_df):
    """Generate predictions using model pipeline"""
    # Prepare features (exclude metadata)
    X = features_df.drop(columns=["Customer_ID", "snapshot_date"])
    
    # Generate predictions
    features_df["default_probability"] = model.predict_proba(X)[:, 1]
    features_df["prediction_class"] = model.predict(X)
    
    return features_df[["Customer_ID", "snapshot_date", 
                       "default_probability", "prediction_class"]]

# --- 6. STORE PREDICTIONS ---
# Main goal: Save predictions for monitoring
def store_predictions(preds_df, snapshot_date):
    """Save predictions to inference store"""
    date_str = snapshot_date.replace("-", "_")
    output_path = f"{INFERENCE_STORE}/predictions_{date_str}.parquet"
    preds_df.to_parquet(output_path)
    return output_path

# --- 7. MAIN EXECUTION ---
def run_inference(snapshot_date):
    """Orchestrate inference pipeline for a single date"""
    print(f"\nStarting inference for {snapshot_date}")
    
    # Load model
    model, model_name = load_latest_model()
    print(f"Loaded model: {model_name}")
    
    # Load features
    features_df = load_features(snapshot_date)
    print(f"Loaded features with {len(features_df)} records")
    
    # Generate predictions
    preds_df = generate_predictions(model, features_df)
    print(f"Generated predictions for {len(preds_df)} customers")
    
    # Store results
    output_path = store_predictions(preds_df, snapshot_date)
    print(f"Predictions saved to {output_path}")
    
    return preds_df

# For Airflow integration
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-date", required=True)
    args = parser.parse_args()
    
    run_inference(args.snapshot_date)