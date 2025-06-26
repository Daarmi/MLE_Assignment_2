import os
import argparse
import pyspark
import utils.data_processing_gold_table

def main(snapshot_date):
    spark = pyspark.sql.SparkSession.builder \
        .appName("gold_processing") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # Feature Store - Engagement
    silver_clks_dir = "datamart/silver/clks/"
    gold_eng_dir = "datamart/gold/feature_store/eng/"
    os.makedirs(gold_eng_dir, exist_ok=True)
    utils.data_processing_gold_table.process_fts_gold_engag_table(
        snapshot_date, silver_clks_dir, gold_eng_dir, spark
    )

    # Feature Store - Financial Risk
    silver_fin_dir = "datamart/silver/fin/"
    gold_fin_risk_dir = "datamart/gold/feature_store/cust_fin_risk/"
    os.makedirs(gold_fin_risk_dir, exist_ok=True)
    utils.data_processing_gold_table.process_fts_gold_cust_risk_table(
        snapshot_date, silver_fin_dir, gold_fin_risk_dir, spark
    )

    # Label Store
    silver_lms_dir = "datamart/silver/lms/"
    gold_label_dir = "datamart/gold/label_store/"
    os.makedirs(gold_label_dir, exist_ok=True)
    utils.data_processing_gold_table.process_labels_gold_table(
        snapshot_date, silver_lms_dir, gold_label_dir, spark, dpd=30, mob=6
    )

    print(f"Completed gold processing for {snapshot_date}")
    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-date", required=True)
    args = parser.parse_args()
    main(args.snapshot_date)