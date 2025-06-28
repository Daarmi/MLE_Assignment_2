# scripts/build_gold_model_table.py
import os, sys, argparse
from typing import Optional, List

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import to_date, col

from utils.data_processing_gold_table import (
    process_labels_gold_table,
    process_fts_gold_engag_table,
    process_fts_gold_cust_risk_table,
)

# ────────────────────────────────────────────────────────────────
# Helper: join only if RHS is real and non-empty
# ────────────────────────────────────────────────────────────────
def safe_join(
    left: DataFrame,
    right: Optional[DataFrame],
    on: List[str],
    how: str = "left"
) -> DataFrame:
    if right is None or len(right.columns) == 0:
        return left
    return left.join(right, on=on, how=how)

def cast_snapshot_to_date(df: Optional[DataFrame]) -> Optional[DataFrame]:
    """Ensure snapshot_date is DateType so join keys match."""
    if df is None or "snapshot_date" not in df.columns:
        return df
    return df.withColumn("snapshot_date", to_date(col("snapshot_date")))

# ────────────────────────────────────────────────────────────────
# Core builder
# ────────────────────────────────────────────────────────────────
def build_gold_model_table(
    snapshot_date_str: str,
    silver_lms_dir: str,
    silver_clks_dir: str,
    silver_fin_dir: str,
    gold_label_dir: str,
    gold_clks_dir: str,
    gold_fin_dir: str,
    gold_model_dir: str,
    dpd: int,
    mob: int,
    spark: SparkSession
):
    # 1️⃣ create each gold piece
    df_label = process_labels_gold_table(
        snapshot_date_str, silver_lms_dir, gold_label_dir, spark, dpd, mob
    )

    df_engag = process_fts_gold_engag_table(
        snapshot_date_str, silver_clks_dir, gold_clks_dir, spark
    )
    df_engag = cast_snapshot_to_date(df_engag)

    df_risk = process_fts_gold_cust_risk_table(
        snapshot_date_str, silver_fin_dir, gold_fin_dir, spark
    )
    df_risk = cast_snapshot_to_date(df_risk)

    # 2️⃣ safely join into one model-ready table
    JOIN_KEYS = ["Customer_ID"]
    df_full = safe_join(df_label, df_engag, ["Customer_ID", "snapshot_date"])
    df_full = safe_join(df_full, df_risk,  ["Customer_ID", "snapshot_date"])

    # 3️⃣ write out
    os.makedirs(gold_model_dir, exist_ok=True)
    out_path = os.path.join(
        gold_model_dir,
        f"gold_model_table_{snapshot_date_str.replace('-','_')}.parquet"
    )
    df_full.write.mode("overwrite").parquet(out_path)
    print(f"✅ Saved combined model table to: {out_path}")

# ────────────────────────────────────────────────────────────────
# CLI entry-point with defaults
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    DEFAULT_DM = "/opt/airflow/scripts/datamart"

    parser = argparse.ArgumentParser(
        description="Build combined gold-model Parquet for loan-default modeling"
    )
    parser.add_argument("--snapshot-date", required=True,
                        help="YYYY-MM-DD for the data slice")

    # optional directory flags
    parser.add_argument("--silver-lms-dir",
                        default=os.path.join(DEFAULT_DM, "silver", "lms"))
    parser.add_argument("--silver-clks-dir",
                        default=os.path.join(DEFAULT_DM, "silver", "clks"))
    parser.add_argument("--silver-fin-dir",
                        default=os.path.join(DEFAULT_DM, "silver", "fin"))
    parser.add_argument("--gold-label-dir",
                        default=os.path.join(DEFAULT_DM, "gold", "label_store"))
    parser.add_argument("--gold-clks-dir",
                        default=os.path.join(DEFAULT_DM, "gold", "engagement_store"))
    parser.add_argument("--gold-fin-dir",
                        default=os.path.join(DEFAULT_DM, "gold", "fin_risk_store"))
    parser.add_argument("--gold-model-dir",
                        default=os.path.join(DEFAULT_DM, "gold", "model_store"))

    parser.add_argument("--dpd", type=int, default=30,
                        help="Days-past-due threshold")
    parser.add_argument("--mob", type=int, default=6,
                        help="Months-on-book threshold")

    args = parser.parse_args()

    spark = (
        SparkSession.builder
        .appName("build_gold_model_table")
        .getOrCreate()
    )

    build_gold_model_table(
        snapshot_date_str = args.snapshot_date,
        silver_lms_dir    = args.silver_lms_dir,
        silver_clks_dir   = args.silver_clks_dir,
        silver_fin_dir    = args.silver_fin_dir,
        gold_label_dir    = args.gold_label_dir,
        gold_clks_dir     = args.gold_clks_dir,
        gold_fin_dir      = args.gold_fin_dir,
        gold_model_dir    = args.gold_model_dir,
        dpd               = args.dpd,
        mob               = args.mob,
        spark             = spark
    )
    spark.stop()
