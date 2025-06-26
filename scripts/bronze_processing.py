import os
import argparse
import pyspark
import utils.data_processing_bronze_table

def main(snapshot_date):
    spark = pyspark.sql.SparkSession.builder \
        .appName("bronze_processing") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # Define a safe base directory
    AIRFLOW_HOME = os.getenv("AIRFLOW_HOME", "/opt/airflow")
    BRONZE_DIR = os.path.join(AIRFLOW_HOME, "datamart", "bronze")

    directories = {
        "loan": os.path.join(BRONZE_DIR, "lms"),
        "clickstream": os.path.join(BRONZE_DIR, "clks"),
        "attributes": os.path.join(BRONZE_DIR, "attr"),
        "financials": os.path.join(BRONZE_DIR, "fin")
    }

    # Process each bronze table
    for name, path in directories.items():
        os.makedirs(path, exist_ok=True)
        processor = getattr(utils.data_processing_bronze_table, f"process_bronze_{name}_table")
        processor(snapshot_date, path, spark)
        print(f"Processed bronze {name} table for {snapshot_date}")

    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-date", required=True)
    args = parser.parse_args()
    main(args.snapshot_date)
