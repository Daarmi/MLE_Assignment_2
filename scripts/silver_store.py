# silver_store.py
# ------------------------------------------------------------------
import os
import sys
import argparse
import pyspark
import utils.data_processing_silver_table as silver_etl

# Determine the absolute path of this script folder
SCRIPTS_ROOT = os.path.dirname(os.path.abspath(__file__))

# Datamart will live inside …/scripts/datamart
DATAMART_ROOT = os.path.join(SCRIPTS_ROOT, "datamart")
os.makedirs(DATAMART_ROOT, exist_ok=True)          # ensure base folder exists


def _log(msg: str) -> None:
    print(f"[silver_store] {msg}", flush=True)


def build_silver(snapshot_date: str) -> None:
    spark = (
        pyspark.sql.SparkSession.builder
        .appName("silver_store")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    bronze_root = os.path.join(DATAMART_ROOT, "bronze")
    silver_root = os.path.join(DATAMART_ROOT, "silver")

    paths = {
        "loan":        {"bronze": os.path.join(bronze_root, "lms"),  "silver": os.path.join(silver_root, "lms")},
        "clickstream": {"bronze": os.path.join(bronze_root, "clks"), "silver": os.path.join(silver_root, "clks")},
        "attributes":  {"bronze": os.path.join(bronze_root, "attr"), "silver": os.path.join(silver_root, "attr")},
        "financials":  {"bronze": os.path.join(bronze_root, "fin"),  "silver": os.path.join(silver_root, "fin")},
    }
    for p in paths.values():
        os.makedirs(p["silver"], exist_ok=True)

    try:
        # ---------------- Loan -----------------
        _log(f"START  loan  {snapshot_date}")
        silver_etl.process_silver_loan_table(snapshot_date, paths["loan"]["bronze"], paths["loan"]["silver"], spark)
        _log(f"✅  SUCCESS loan  {snapshot_date}")

        # -------------- Clickstream ------------
        _log(f"START  clickstream  {snapshot_date}")
        silver_etl.process_silver_clickstream_table(snapshot_date, paths["clickstream"]["bronze"], paths["clickstream"]["silver"], spark)
        _log(f"✅  SUCCESS clickstream  {snapshot_date}")

        # -------------- Attributes -------------
        _log(f"START  attributes  {snapshot_date}")
        silver_etl.process_silver_attributes_table(snapshot_date, paths["attributes"]["bronze"], paths["attributes"]["silver"], spark)
        _log(f"✅  SUCCESS attributes  {snapshot_date}")

        # -------------- Financials -------------
        _log(f"START  financials  {snapshot_date}")
        silver_etl.process_silver_financials_table(snapshot_date, paths["financials"]["bronze"], paths["financials"]["silver"], spark)
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
    build_silver(parser.parse_args().snapshot_date)
