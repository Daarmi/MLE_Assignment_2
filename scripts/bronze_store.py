# bronze_store.py
# ------------------------------------------------------------------
import os
import sys
import argparse
import pyspark
import utils.data_processing_bronze_table as bronze_etl

# Determine the absolute path of this script folder
SCRIPTS_ROOT = os.path.dirname(os.path.abspath(__file__))

# Datamart will live inside …/scripts/datamart
DATAMART_ROOT = os.path.join(SCRIPTS_ROOT, "datamart")
os.makedirs(DATAMART_ROOT, exist_ok=True)          # ensure base folder exists

def _log(msg: str) -> None:
    print(f"[bronze_store] {msg}", flush=True)

def build_bronze(snapshot_date: str) -> None:
    spark = (
        pyspark.sql.SparkSession.builder
        .appName("bronze_store")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    bronze_root = os.path.join(DATAMART_ROOT, "bronze")
    paths = {
        "loan":        os.path.join(bronze_root, "lms"),
        "clickstream": os.path.join(bronze_root, "clks"),
        "attributes":  os.path.join(bronze_root, "attr"),
        "financials":  os.path.join(bronze_root, "fin"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)

    try:
        _log(f"START  loan  {snapshot_date}")
        bronze_etl.process_bronze_loan_table(snapshot_date, paths["loan"], spark)
        _log(f"✅  SUCCESS loan  {snapshot_date}")

        _log(f"START  clickstream  {snapshot_date}")
        bronze_etl.process_bronze_clickstream_table(snapshot_date, paths["clickstream"], spark)
        _log(f"✅  SUCCESS clickstream  {snapshot_date}")

        _log(f"START  attributes  {snapshot_date}")
        bronze_etl.process_bronze_attributes_table(snapshot_date, paths["attributes"], spark)
        _log(f"✅  SUCCESS attributes  {snapshot_date}")

        _log(f"START  financials  {snapshot_date}")
        bronze_etl.process_bronze_financials_table( snapshot_date, paths["financials"],  spark)
        _log(f"✅  SUCCESS financials  {snapshot_date}")

        _log(f"ALL TABLES COMPLETE  {snapshot_date}")

    except Exception as exc:
        _log(f"❌  FAILURE: {exc}")
        raise
    finally:
        spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-date", required=True, help="YYYY-MM-DD")
    build_bronze(parser.parse_args().snapshot_date)
