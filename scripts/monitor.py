# scripts/ml/monitor.py
"""
End-to-End Model Monitoring Pipeline
Steps:
1. Load predictions and actual labels
2. Compute performance metrics (AUC, precision, recall, F1)
3. Compute stability metrics (PSI for features and predictions)
4. Track latency/resource utilization
5. Store monitoring results
6. Generate visualizations
"""

# --- 1. IMPORT LIBRARIES ---
# Main goal: Load required dependencies
import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import psutil  # For resource monitoring
import time

# --- 2. CONFIGURATION ---
# Main goal: Set paths and parameters
MODEL_BANK = "model_bank"
INFERENCE_STORE = "datamart/gold/inference_store"
LABEL_STORE = "datamart/gold/label_store"
MONITORING_STORE = "datamart/gold/monitoring_store"
VIZ_STORE = "datamart/gold/monitoring_viz"
os.makedirs(MONITORING_STORE, exist_ok=True)
os.makedirs(VIZ_STORE, exist_ok=True)

# PSI thresholds (from lecture)
PSI_THRESHOLDS = {
    "no_drift": 0.1,
    "moderate_drift": 0.25,
    "significant_drift": 0.25
}

# --- 3. METRICS COMPUTATION ---
# Main goal: Implement monitoring metrics from lecture
def calculate_psi(baseline, current, bins=10):
    """Calculate Population Stability Index (PSI)"""
    # Create bins based on baseline distribution
    breakpoints = np.percentile(baseline, [100/bins*i for i in range(1, bins)])
    breakpoints = np.unique(breakpoints)
    
    # Bin both datasets
    baseline_counts = np.histogram(baseline, bins=np.concatenate([[-np.inf], breakpoints, [np.inf]]))[0]
    current_counts = np.histogram(current, bins=np.concatenate([[-np.inf], breakpoints, [np.inf]]))[0]
    
    # Convert to percentages
    baseline_pct = baseline_counts / len(baseline)
    current_pct = current_counts / len(current)
    
    # Avoid division by zero
    baseline_pct = np.where(baseline_pct == 0, 0.0001, baseline_pct)
    current_pct = np.where(current_pct == 0, 0.0001, current_pct)
    
    # Calculate PSI
    psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
    return psi

def compute_performance_metrics(y_true, y_pred, y_prob):
    """Compute performance metrics (AUC, precision, recall, F1)"""
    return {
        "auc": roc_auc_score(y_true, y_prob),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred)
    }

# --- 4. DATA LOADING ---
# Main goal: Retrieve data for monitoring
def load_monitoring_data(application_date):
    """Load predictions and actual labels"""
    # Convert date format
    date_str = application_date.replace("-", "_")
    
    # Load predictions
    pred_path = f"{INFERENCE_STORE}/predictions_{date_str}.parquet"
    pred_df = pd.read_parquet(pred_path)
    
    # Load actual labels (with 6-month delay)
    label_date = (datetime.strptime(application_date, "%Y-%m-%d") - pd.DateOffset(months=6)).strftime("%Y-%m-%d")
    label_str = label_date.replace("-", "_")
    label_path = f"{LABEL_STORE}/gold_label_store_{label_str}.parquet"
    label_df = pd.read_parquet(label_path)
    
    # Merge datasets
    return pd.merge(pred_df, label_df, on="Customer_ID")

def load_baseline_data(model_version):
    """Load baseline features from training period"""
    baseline_path = f"{MODEL_BANK}/baseline_{model_version}.parquet"
    return pd.read_parquet(baseline_path)

# --- 5. MONITORING EXECUTION ---
# Main goal: Compute all monitoring metrics
def run_monitoring(application_date):
    """Execute monitoring for a specific application date"""
    start_time = time.time()
    print(f"\nStarting monitoring for {application_date}")
    
    # Load data
    monitor_df = load_monitoring_data(application_date)
    model_version = monitor_df["model_version"].iloc[0]
    
    # Get baseline data
    baseline_df = load_baseline_data(model_version)
    
    # Track resource utilization
    cpu_percent = psutil.cpu_percent()
    mem_usage = psutil.virtual_memory().percent
    
    # Compute performance metrics
    metrics = compute_performance_metrics(
        monitor_df["label"],
        monitor_df["prediction_class"],
        monitor_df["default_probability"]
    )
    
    # Compute stability metrics (PSI)
    psi_results = {}
    for feature in baseline_df.columns:
        if feature not in ["Customer_ID", "snapshot_date"]:
            psi = calculate_psi(
                baseline_df[feature].dropna(),
                monitor_df[feature].dropna()
            )
            psi_results[feature] = psi
    
    # Prediction distribution PSI
    pred_psi = calculate_psi(
        baseline_df["default_probability"],
        monitor_df["default_probability"]
    )
    
    # Track latency
    latency = time.time() - start_time
    
    # Prepare monitoring record
    record = {
        "application_date": application_date,
        "monitoring_date": datetime.now().strftime("%Y-%m-%d"),
        "model_version": model_version,
        "performance_metrics": metrics,
        "feature_psi": psi_results,
        "prediction_psi": pred_psi,
        "resource_usage": {"cpu": cpu_percent, "memory": mem_usage},
        "latency_seconds": latency
    }
    
    # Store results
    save_monitoring_record(record)
    
    # Generate visualizations
    generate_visualizations(record, application_date)
    
    # Check alert conditions
    check_alerts(record)
    
    print("Monitoring completed successfully!")
    return record

# --- 6. STORAGE & VISUALIZATION ---
# Main goal: Save results and create visualizations
def save_monitoring_record(record):
    """Save monitoring record to datamart"""
    date_str = record["application_date"].replace("-", "_")
    output_path = f"{MONITORING_STORE}/monitoring_{date_str}.json"
    with open(output_path, "w") as f:
        json.dump(record, f)

def generate_visualizations(record, application_date):
    """Create monitoring visualizations"""
    date_str = application_date.replace("-", "_")
    
    # 1. Performance trend visualization
    plt.figure(figsize=(10, 6))
    metrics = record["performance_metrics"]
    plt.bar(metrics.keys(), metrics.values())
    plt.title(f"Model Performance ({application_date})")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.savefig(f"{VIZ_STORE}/performance_{date_str}.png")
    plt.close()
    
    # 2. PSI visualization
    plt.figure(figsize=(12, 8))
    psi_data = record["feature_psi"]
    psi_data["prediction"] = record["prediction_psi"]
    plt.bar(psi_data.keys(), psi_data.values())
    plt.axhline(y=PSI_THRESHOLDS["no_drift"], color="g", linestyle="--", label="No Drift")
    plt.axhline(y=PSI_THRESHOLDS["moderate_drift"], color="y", linestyle="--", label="Moderate Drift")
    plt.axhline(y=PSI_THRESHOLDS["significant_drift"], color="r", linestyle="--", label="Significant Drift")
    plt.title(f"Feature Stability (PSI) - {application_date}")
    plt.ylabel("PSI Score")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{VIZ_STORE}/stability_{date_str}.png")
    plt.close()

# --- 7. ALERTING SYSTEM ---
# Main goal: Implement alert thresholds from lecture
def check_alerts(record):
    """Check monitoring results against alert thresholds"""
    alerts = []
    
    # Performance alerts
    if record["performance_metrics"]["auc"] < 0.7:
        alerts.append("Critical: AUC dropped below 0.7")
    
    # Stability alerts
    if record["prediction_psi"] > PSI_THRESHOLDS["significant_drift"]:
        alerts.append(f"Critical: Prediction PSI ({record['prediction_psi']:.4f}) > 0.25")
    
    high_drift_features = [
        (f, psi) for f, psi in record["feature_psi"].items() 
        if psi > PSI_THRESHOLDS["significant_drift"]
    ]
    for feature, psi in high_drift_features:
        alerts.append(f"Warning: High PSI ({psi:.4f}) for feature '{feature}'")
    
    # Resource alerts
    if record["resource_usage"]["cpu"] > 90:
        alerts.append(f"Warning: High CPU usage ({record['resource_usage']['cpu']}%)")
    
    # Save alerts if any
    if alerts:
        alert_path = f"{MONITORING_STORE}/alerts_{record['application_date'].replace('-', '_')}.txt"
        with open(alert_path, "w") as f:
            f.write("\n".join(alerts))

# --- 8. BASELINE CREATION (To be added to training script) ---
# Main goal: Save training data snapshot for PSI calculations
# Add this to model_trainer.py after training completes
"""
# In model_trainer.py
def save_baseline_data(model_version, X_train, y_prob):
    baseline_df = X_train.copy()
    baseline_df["default_probability"] = y_prob
    baseline_path = f"{MODEL_BANK}/baseline_{model_version}.parquet"
    baseline_df.to_parquet(baseline_path)
"""

# --- 9. MAIN EXECUTION ---
def main(application_date):
    return run_monitoring(application_date)

# For Airflow integration
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--application-date", required=True)
    args = parser.parse_args()
    main(args.application_date)