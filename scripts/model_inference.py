import os
import argparse
import pickle
import pandas as pd
from sklearn.metrics import roc_auc_score

def _log(msg: str):
    print(f"[model_inference] {msg}", flush=True)

def run_inference(snapshot_date_str: str) -> None:
    # Define paths
    SCRIPTS_ROOT = os.path.dirname(os.path.abspath(__file__))
    # Get the parent directory of SCRIPTS_ROOT, which would be '/opt/airflow'
    AIRFLOW_ROOT = os.path.dirname(SCRIPTS_ROOT)
    DATAMART_ROOT = os.path.join(SCRIPTS_ROOT, "datamart")
    DC_ROOT = os.path.join(DATAMART_ROOT, "data_check")
    GOLD_ROOT = os.path.join(DATAMART_ROOT, "gold")
    MODEL_BANK = os.path.join(AIRFLOW_ROOT, "model_bank")

    # Input data paths
    paths = {
        "test": os.path.join(DC_ROOT, "test", f"test_{snapshot_date_str}.parquet"),
        "validation": os.path.join(DC_ROOT, "val", f"val_{snapshot_date_str}.parquet"),
    }
    eng_path = os.path.join(GOLD_ROOT, "feature_store", "eng")
    risk_path = os.path.join(GOLD_ROOT, "feature_store", "cust_fin_risk")

    # Load feature stores
    _log("Loading feature stores...")
    eng_df = pd.read_parquet(eng_path)
    risk_df = pd.read_parquet(risk_path)
    eng_df = eng_df.rename(columns={"Customer_ID": "loan_id"})
    risk_df = risk_df.rename(columns={"Customer_ID": "loan_id"})

    # Load the pickled scikit-learn model
    model_file = os.path.join(MODEL_BANK, "logreg_pipeline_6m.pkl")
    _log(f"Loading model from {model_file}")
    with open(model_file, "rb") as f:
        model = pickle.load(f)

    # Prepare outputs
    preds_dir = os.path.join(GOLD_ROOT, "predictions")
    os.makedirs(preds_dir, exist_ok=True)
    outputs = {}

    # Iterate over datasets
    for set_name, path in paths.items():
        if not os.path.exists(path):
            _log(f"{set_name.capitalize()} data not found at {path}, skipping.")
            continue

        _log(f"Loading {set_name} data from {path}...")
        df = pd.read_parquet(path)

        # Merge features
        full_df = df.merge(eng_df, on=["loan_id", "snapshot_date"], how="left") \
                    .merge(risk_df, on=["loan_id", "snapshot_date"], how="left")

        # Determine features
        feature_cols = [c for c in full_df.columns if c not in ("loan_id", "snapshot_date", "label")]
        X = full_df[feature_cols]

        # Predict
        probs = model.predict_proba(X)[:, 1]
        preds = model.predict(X)
        full_df["probability"] = probs
        full_df["prediction"] = preds

        # Save
        out_path = os.path.join(preds_dir, f"{set_name}_preds_{snapshot_date_str}.parquet")
        full_df[["loan_id", "snapshot_date", "label", "probability", "prediction"]] \
               .to_parquet(out_path, index=False)
        _log(f"{set_name.capitalize()} predictions saved to {out_path}")
        outputs[set_name] = full_df

    # If no outputs, exit
    if not outputs:
        _log("No datasets processed, exiting.")
        return

    # Evaluate AUC if both present or individually
    for set_name, full_df in outputs.items():
        auc = roc_auc_score(full_df["label"], full_df["probability"])
        _log(f"{set_name.capitalize()} AUC: {auc:.4f}")

    # Append to CSV manifest
    manifest_dir = os.path.join(GOLD_ROOT, "meta")
    os.makedirs(manifest_dir, exist_ok=True)
    manifest_path = os.path.join(manifest_dir, "model_manifest.csv")
    header = not os.path.exists(manifest_path)
    import csv
    with open(manifest_path, "a", newline="") as mf:
        writer = csv.writer(mf)
        if header:
            writer.writerow(["model_file", "loaded_on", "set", "auc", "preds_path"])
        for set_name, _ in outputs.items():
            out_path = os.path.join(preds_dir, f"{set_name}_preds_{snapshot_date_str}.parquet")
            auc = roc_auc_score(outputs[set_name]["label"], outputs[set_name]["probability"])
            writer.writerow([model_file, snapshot_date_str, set_name, auc, out_path])
    _log(f"Appended inference results to manifest: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run inference via pickle-loaded model and save outputs."
    )
    parser.add_argument(
        "--snapshot-date", required=True, help="YYYY-MM-DD"
    )
    args = parser.parse_args()
    run_inference(args.snapshot_date)

if __name__ == "__main__":
    main()
