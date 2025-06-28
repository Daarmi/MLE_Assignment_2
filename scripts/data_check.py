# scripts/data_check.py
import os, sys, glob, argparse
from pyspark.sql import SparkSession

# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────
def _log(msg: str):
    print(f"[data_check] {msg}", flush=True)

# Make sure sibling utils are importable (if you need them later)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# ────────────────────────────────────────────────────────────────
# Main routine
# ────────────────────────────────────────────────────────────────
def build_data_check(snapshot_date_str: str):
    # 1️⃣ Spark session
    spark = (
        SparkSession.builder
        .appName("data_check")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    # 2️⃣ Paths
    DATAMART_ROOT = "/opt/airflow/scripts/datamart"   # ← adjust once if needed
    MODEL_DIR     = os.path.join(DATAMART_ROOT, "gold", "model_store")
    OUT_ROOT      = os.path.join(DATAMART_ROOT, "data_check")
    os.makedirs(OUT_ROOT, exist_ok=True)

    # 3️⃣ Count model months
    model_files   = sorted(glob.glob(os.path.join(MODEL_DIR, "gold_model_table_*.parquet")))
    months_loaded = len(model_files)
    _log(f"Found {months_loaded} months of model tables")

    # 4️⃣ Business rule gate
    if months_loaded < 6:
        _log("Waiting for 12 months before we create splits.")
        spark.stop()
        return

    # 5️⃣ Load all model data
    df_all = spark.read.parquet(*model_files)

    # ------------------------------------------------------------------
    # 6️⃣ Train / Test / Val split  (first time only)
    # ------------------------------------------------------------------
    if months_loaded == 6:
        _log("Splitting first 12 months into train / test / val (70 / 20 / 10)")
        train, test, val = df_all.randomSplit([0.7, 0.2, 0.1], seed=42)

        for name, subset in [("train", train), ("test", test), ("val", val)]:
            out_dir  = os.path.join(OUT_ROOT, name)
            os.makedirs(out_dir, exist_ok=True)

            first_mon = os.path.basename(model_files[0]).split("gold_model_table_")[-1].replace(".parquet", "")
            last_mon  = os.path.basename(model_files[11]).split("gold_model_table_")[-1].replace(".parquet", "")
            out_path  = os.path.join(out_dir, f"{name}_{first_mon}_to_{last_mon}.parquet")

            # Optional: rename snapshot_date for downstream convenience
            if "snapshot_date" in subset.columns:
                subset = subset.withColumnRenamed("snapshot_date", "snapshot_ts")

            subset.write.mode("overwrite").parquet(out_path)
            _log(f"  • Saved {name} → {out_path}")

    # ------------------------------------------------------------------
    # 7️⃣  OOT tables  (months 13–15)
    # ------------------------------------------------------------------
    elif 7 <= months_loaded <= 9:
        oot_idx = months_loaded - 6
        oot_tag = f"oot{oot_idx}"

        latest_file = model_files[-1]
        ds_tag      = os.path.basename(latest_file).split("gold_model_table_")[-1].replace(".parquet", "")
        df_this     = spark.read.parquet(latest_file)

        out_dir  = os.path.join(OUT_ROOT, oot_tag)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{oot_tag}_{ds_tag}.parquet")

        if "snapshot_date" in df_this.columns:
            df_this = df_this.withColumnRenamed("snapshot_date", "snapshot_ts")

        df_this.write.mode("overwrite").parquet(out_path)
        _log(f"  • Tagged {oot_tag.upper()} → {out_path}")

    # ------------------------------------------------------------------
    # 8️⃣ Monitoring append  (16 + months)
    # ------------------------------------------------------------------
    else:
        latest_file = model_files[-1]
        ds_tag      = os.path.basename(latest_file).split("gold_model_table_")[-1].replace(".parquet", "")
        df_this     = spark.read.parquet(latest_file)

        mon_dir  = os.path.join(OUT_ROOT, "monitor")
        os.makedirs(mon_dir, exist_ok=True)
        out_path = os.path.join(mon_dir, f"monitor_{ds_tag}.parquet")

        if "snapshot_date" in df_this.columns:
            df_this = df_this.withColumnRenamed("snapshot_date", "snapshot_ts")

        df_this.write.mode("append").parquet(out_path)
        _log(f"  • Appended monitoring data → {out_path}")

    spark.stop()

# ------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check model-store readiness and create splits / OOT / monitor")
    parser.add_argument("--snapshot-date", required=True, help="YYYY-MM-DD (pass-through, unused here)")
    args = parser.parse_args()
    build_data_check(args.snapshot_date)
