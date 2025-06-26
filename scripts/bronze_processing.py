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

    # Define directories
    directories = {
        "loan": "datamart/bronze/lms/",
        "clickstream": "datamart/bronze/clks/",
        "attributes": "datamart/bronze/attr/",
        "financials": "datamart/bronze/fin/"
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