import os
import argparse
import pyspark
import utils.data_processing_silver_table

def main(snapshot_date):
    spark = pyspark.sql.SparkSession.builder \
        .appName("silver_processing") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # Define input/output directories
    bronze_dirs = {
        "loan": "datamart/bronze/lms/",
        "clickstream": "datamart/bronze/clks/",
        "attributes": "datamart/bronze/attr/",
        "financials": "datamart/bronze/fin/"
    }
    
    silver_dirs = {
        "loan": "datamart/silver/lms/",
        "clickstream": "datamart/silver/clks/",
        "attributes": "datamart/silver/attr/",
        "financials": "datamart/silver/fin/"
    }

    # Process each silver table
    for name in bronze_dirs.keys():
        os.makedirs(silver_dirs[name], exist_ok=True)
        processor = getattr(utils.data_processing_silver_table, f"process_silver_{name}_table")
        processor(snapshot_date, bronze_dirs[name], silver_dirs[name], spark)
        print(f"Processed silver {name} table for {snapshot_date}")

    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-date", required=True)
    args = parser.parse_args()
    main(args.snapshot_date)