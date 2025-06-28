#!/usr/bin/env python3
import os
import argparse
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import BinaryClassificationEvaluator

def _log(msg: str):
    print(f"[model_monitoring] {msg}", flush=True)

def run_monitoring(start_date: str, end_date: str) -> None:
    spark = (
        SparkSession.builder
        .appName("model_monitoring")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    GOLD_ROOT = os.path.join(os.path.dirname(__file__), "datamart", "gold")
    preds_dir = os.path.join(GOLD_ROOT, "predictions")
    monitor_dir = os.path.join(GOLD_ROOT, "monitoring")
    os.makedirs(monitor_dir, exist_ok=True)

    evaluator = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )

    # loop over snapshot dates in range (inclusive)
    from datetime import datetime, timedelta
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end   = datetime.strptime(end_date,   "%Y-%m-%d")
    delta = timedelta(days=1)

    current = start
    metrics = []
    while current <= end:
        ds = current.strftime("%Y-%m-%d")
        path = os.path.join(preds_dir, f"test_preds_{ds}.parquet")
        if os.path.exists(path):
            df = spark.read.parquet(path)
            auc = evaluator.evaluate(df)
            metrics.append((ds, float(auc)))
            _log(f"{ds} → AUC={auc:.4f}")
        else:
            _log(f"{ds} → no preds found, skipping")
        current += delta

    # build and write monitoring table
    monitor_df = spark.createDataFrame(
        [(d, auc) for d, auc in metrics],
        schema=["snapshot_date", "test_auc"]
    )
    out_path = os.path.join(monitor_dir, f"metrics_{start_date}_to_{end_date}.parquet")
    monitor_df.write.mode("overwrite").parquet(out_path)
    _log(f"Written monitoring metrics to {out_path}")

    spark.stop()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    p.add_argument("--end-date",   required=True, help="YYYY-MM-DD")
    args = p.parse_args()
    run_monitoring(args.start_date, args.end_date)
