# gold_store.py
# ------------------------------------------------------------------
import os
import sys
import argparse
import pyspark
import utils.data_processing_gold_table as gold_etl

# Determine the absolute path of this script folder
SCRIPTS_ROOT = os.path.dirname(os.path.abspath(__file__))

# Datamart will live inside …/scripts/datamart
DATAMART_ROOT = os.path.join(SCRIPTS_ROOT, "datamart")
os.makedirs(DATAMART_ROOT, exist_ok=True)          # ensure base folder exists

def _log(msg: str) -> None:
    print(f"[gold_store] {msg}", flush=True)

def build_gold(snapshot_date: str) -> None:
    spark = (
        pyspark.sql.SparkSession.builder
        .appName("gold_store")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    silver_root = os.path.join(DATAMART_ROOT, "silver")
    gold_root   = os.path.join(DATAMART_ROOT, "gold")
    silver = {
        "lms":  os.path.join(silver_root, "lms"),
        "clks": os.path.join(silver_root, "clks"),
        "fin":  os.path.join(silver_root, "fin"),
    }
    gold = {
        "eng":   os.path.join(gold_root, "feature_store", "eng"),
        "cust":  os.path.join(gold_root, "feature_store", "cust_fin_risk"),
        "label": os.path.join(gold_root, "label_store"),
    }
    for p in gold.values():
        os.makedirs(p, exist_ok=True)

    try:
        _log(f"START  eng_features  {snapshot_date}")
        gold_etl.process_fts_gold_engag_table(snapshot_date, silver["clks"], gold["eng"],   spark)
        _log(f"✅  SUCCESS eng_features  {snapshot_date}")

        _log(f"START  cust_fin_risk_features  {snapshot_date}")
        gold_etl.process_fts_gold_cust_risk_table(snapshot_date, silver["fin"], gold["cust"],  spark)
        _log(f"✅  SUCCESS cust_fin_risk_features  {snapshot_date}")

        _log(f"START  label_store  {snapshot_date}")
        gold_etl.process_labels_gold_table(snapshot_date, silver["lms"], gold["label"], spark, dpd=30, mob=6)
        _log(f"✅  SUCCESS label_store  {snapshot_date}")

        _log(f"ALL TABLES COMPLETE  {snapshot_date}")

    except Exception as exc:
        _log(f"❌  FAILURE: {exc}")
        raise
    finally:
        spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-date", required=True, help="YYYY-MM-DD")
    build_gold(parser.parse_args().snapshot_date)
