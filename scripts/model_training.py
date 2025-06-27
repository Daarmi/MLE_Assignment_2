# model_training.py
# ------------------------------------------------------------------
# Train a logistic regression model on the loan-default data
# ------------------------------------------------------------------
import os
import argparse
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator


def _log(msg: str):
    print(f"[model_training] {msg}", flush=True)


def build_model(snapshot_date_str: str) -> None:
    # --- Initialize Spark --------------------------------------------------
    spark = (
        SparkSession.builder
        .appName("model_training")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    # --- Paths ---------------------------------------------------------------
    SCRIPTS_ROOT = os.path.dirname(os.path.abspath(__file__))
    DATAMART_ROOT = os.path.join(SCRIPTS_ROOT, "datamart")
    DC_ROOT       = os.path.join(DATAMART_ROOT, "data_check")
    GOLD_ROOT     = os.path.join(DATAMART_ROOT, "gold")

    # Split paths
    train_path = os.path.join(DC_ROOT, "train", f"train_{snapshot_date_str}.parquet")
    test_path  = os.path.join(DC_ROOT, "test",  f"test_{snapshot_date_str}.parquet")
    val_path   = os.path.join(DC_ROOT, "val",   f"val_{snapshot_date_str}.parquet")

    # Feature-store paths
    eng_path  = os.path.join(GOLD_ROOT, "feature_store", "eng")
    risk_path = os.path.join(GOLD_ROOT, "feature_store", "cust_fin_risk")

    # Model output
    model_bank = os.path.join(DATAMART_ROOT, "model_bank")
    os.makedirs(model_bank, exist_ok=True)

    # --- Load data -----------------------------------------------------------
    _log("Loading train/test/val and feature sets...")
    train_df = spark.read.parquet(train_path)
    test_df  = spark.read.parquet(test_path)
    val_df   = spark.read.parquet(val_path)
    # read nested parquet directories
    eng_df   = (
        spark.read
             .option("recursiveFileLookup", "true")
             .parquet(eng_path)
    )
    risk_df  = (
        spark.read
             .option("recursiveFileLookup", "true")
             .parquet(risk_path)
    )

    # --- Prepare feature tables for join ------------------------------------
    # Preserve original Customer_ID under cust_id, then rename for join on loan_id
    eng_df = (
        eng_df
        .withColumn("cust_id", col("Customer_ID"))
        .withColumnRenamed("Customer_ID", "loan_id")
    )
    risk_df = (
        risk_df
        .withColumn("cust_id", col("Customer_ID"))
        .withColumnRenamed("Customer_ID", "loan_id")
    )

    # --- Assemble features --------------------------------------------------
    def assemble(df_split):
        return (
            df_split
            .join(eng_df, ["loan_id", "snapshot_date"], "left")
            .join(risk_df, ["loan_id", "snapshot_date"], "left")
        )

    train_full = assemble(train_df)
    test_full  = assemble(test_df)
    val_full   = assemble(val_df)

    # Identify feature columns (exclude keys and label)
    feature_cols = [c for c in train_full.columns if c not in ("loan_id", "snapshot_date", "label")]

    # --- Build pipeline ------------------------------------------------------
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
    scaler    = StandardScaler(inputCol="raw_features", outputCol="features", withMean=True, withStd=True)
    lr        = LogisticRegression(labelCol="label", featuresCol="features", maxIter=50)
    pipeline  = Pipeline(stages=[assembler, scaler, lr])

    # --- Train model ---------------------------------------------------------
    _log("Training Logistic Regression model...")
    model = pipeline.fit(train_full)
    _log("Model training complete.")

    # --- Evaluate -----------------------------------------------------------
    evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    auc_test = evaluator.evaluate(model.transform(test_full))
    auc_val  = evaluator.evaluate(model.transform(val_full))
    _log(f"Test AUC: {auc_test:.4f}")
    _log(f"Validation AUC: {auc_val:.4f}")

    # --- Persist model ------------------------------------------------------
    ts = snapshot_date_str.replace("-", "_")
    model_dir = os.path.join(model_bank, f"model_{ts}")
    model.write().overwrite().save(model_dir)
    _log(f"Saved model artifact to {model_dir}")

    # --- Write manifest -----------------------------------------------------
    manifest_dir = os.path.join(GOLD_ROOT, "meta")
    os.makedirs(manifest_dir, exist_ok=True)
    manifest_path = os.path.join(manifest_dir, "model_manifest.parquet")

    manifest_df = spark.createDataFrame([
        {
            "model_dir": model_dir,
            "trained_on": snapshot_date_str,
            "auc_test": auc_test,
            "auc_val": auc_val
        }
    ])
    manifest_df.write.mode("append").parquet(manifest_path)
    _log(f"Appended training results to manifest: {manifest_path}")

    spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train logistic regression on loan default data.")
    parser.add_argument("--snapshot-date", required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    build_model(args.snapshot_date)
