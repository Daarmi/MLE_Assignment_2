# scripts/data_check.py
import os
import glob
import argparse
from datetime import datetime
from pyspark.sql import SparkSession

def _log(msg: str):
    print(f"[data_check] {msg}", flush=True)

def build_data_check(snapshot_date_str: str):
    # ─── Init Spark & paths ─────────────────────────────────────────────
    spark = (
        SparkSession.builder
        .appName("data_check")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    SCRIPTS_ROOT   = os.path.dirname(os.path.abspath(__file__))
    DATAMART_ROOT  = os.path.join(SCRIPTS_ROOT, "datamart")
    LABEL_DIR      = os.path.join(DATAMART_ROOT, "gold", "label_store")
    OUT_ROOT       = os.path.join(DATAMART_ROOT, "data_check")
    os.makedirs(OUT_ROOT, exist_ok=True)

    # ─── Discover how many months loaded ─────────────────────────────────
    files = sorted(glob.glob(os.path.join(LABEL_DIR, "gold_label_store_*.parquet")))
    months_loaded = len(files)
    _log(f"Found {months_loaded} months of gold labels")

    # ─── Apply business rules ───────────────────────────────────────────
    if months_loaded < 12:
        _log(f"Only {months_loaded} months available – waiting for 12 before splitting.")
        spark.stop()
        return

    # Read *all* label data so far
    df_all = spark.read.parquet(*files)

    if months_loaded == 12:
        _log("Splitting first 12 months into train/test/val (70/20/10)")
        train, test, val = df_all.randomSplit([0.7, 0.2, 0.1], seed=42)
        for name, subset in [("train", train), ("test", test), ("val", val)]:
            out_dir = os.path.join(OUT_ROOT, name)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{name}_{snapshot_date_str}.parquet")
            subset.write.mode("overwrite").parquet(out_path)
            _log(f"  • Saved {name} → {out_path}")

    elif 13 <= months_loaded <= 15:
        oot_idx = months_loaded - 12
        oot_tag = f"oot{oot_idx}"
        # Grab just this month’s file
        ds_tag = files[-1].split("_")[-1].replace(".parquet", "")
        df_this = spark.read.parquet(files[-1])
        out_dir = os.path.join(OUT_ROOT, oot_tag)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{oot_tag}_{ds_tag}.parquet")
        df_this.write.mode("overwrite").parquet(out_path)
        _log(f"  • Tagged OOT{oot_idx} → {out_path}")

    else:  # months_loaded >= 16
        ds_tag = files[-1].split("_")[-1].replace(".parquet", "")
        df_this = spark.read.parquet(files[-1])
        mon_dir = os.path.join(OUT_ROOT, "monitor")
        os.makedirs(mon_dir, exist_ok=True)
        out_path = os.path.join(mon_dir, f"monitor_{ds_tag}.parquet")
        # append each month’s data
        df_this.write.mode("append").parquet(out_path)
        _log(f"  • Appended monitoring data → {out_path}")

    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check gold data readiness and split/OOT/monitor")
    parser.add_argument("--snapshot-date", required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    build_data_check(args.snapshot_date)
