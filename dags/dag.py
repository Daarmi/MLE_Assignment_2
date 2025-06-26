# dags/loan_ml_pipeline.py
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import os

# Configuration
DEFAULT_START_DATE = datetime(2023, 1, 1)
SCRIPTS_DIR = "/opt/airflow/scripts"

def run_bronze(snapshot_date):
    os.system(f"python {SCRIPTS_DIR}/bronze_processing.py --snapshot-date {snapshot_date}")

def run_silver(snapshot_date):
    os.system(f"python {SCRIPTS_DIR}/silver_processing.py --snapshot-date {snapshot_date}")

def run_gold(snapshot_date):
    os.system(f"python {SCRIPTS_DIR}/gold_processing.py --snapshot-date {snapshot_date}")

def run_model_training(execution_date):
    """Train model using data up to previous month"""
    exec_date_str = execution_date.strftime("%Y-%m-%d")
    
    os.system(
        f"python /opt/airflow/scripts/model_trainer.py "
        f"--execution-date {exec_date_str} "
        f"--feature-dir /opt/airflow/datamart/gold/feature_store "
        f"--label-dir /opt/airflow/datamart/gold/label_store "
        f"--model-bank /opt/airflow/model_bank"
    )

def run_prediction(snapshot_date):
    os.system(f"python {SCRIPTS_DIR}/predictor.py --snapshot-date {snapshot_date}")

def run_monitoring(application_date):
    os.system(f"python {SCRIPTS_DIR}/monitor.py --application-date {application_date}")

default_args = {
    'owner': 'airflow',
    'depends_on_past': True,  # Critical for data pipeline integrity
    'start_date': DEFAULT_START_DATE,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'max_active_runs': 1  # Ensure sequential processing
}

with DAG(
    'loan_ml_pipeline',
    default_args=default_args,
    description='End-to-End ML Pipeline for Loan Default Prediction',
    schedule_interval='@monthly',  # Runs daily at midnight
    catchup=True,  # Enable backfilling
    max_active_runs=1,
    tags=['loan_ml']
) as dag:
    
    # Bronze Layer Processing
    bronze = PythonOperator(
        task_id='bronze_processing',
        python_callable=run_bronze,
        op_kwargs={'snapshot_date': '{{ ds }}'}  # ds = execution date in YYYY-MM-DD
    )
    
    # Silver Layer Processing
    silver = PythonOperator(
        task_id='silver_processing',
        python_callable=run_silver,
        op_kwargs={'snapshot_date': '{{ ds }}'}
    )
    
    # Gold Layer Processing
    gold = PythonOperator(
        task_id='gold_processing',
        python_callable=run_gold,
        op_kwargs={'snapshot_date': '{{ ds }}'}
    )
    
    # Model Training (runs monthly on 1st)
    train_model = PythonOperator(
        task_id='train_model',
        python_callable=run_model_training,
        op_kwargs={'execution_date': '{{ ds }}'},
        execution_timeout=timedelta(hours=2)
    )
    
    # Daily Predictions
    predict = PythonOperator(
        task_id='make_predictions',
        python_callable=run_prediction,
        op_kwargs={'snapshot_date': '{{ ds }}'}
    )
    
    # Model Monitoring (6-month delayed performance)
    monitor = PythonOperator(
        task_id='monitor_performance',
        python_callable=run_monitoring,
        op_kwargs={'application_date': '{{ (execution_date - macros.dateutil.relativedelta.relativedelta(months=6)).strftime("%Y-%m-%d") }}'},
        execution_timeout=timedelta(minutes=30)
    )
    
    # Define workflow dependencies
    bronze >> silver >> gold
    gold >> [train_model, predict]  # Gold must complete before training/predictions
    train_model >> predict  # Training must complete before predictions
    predict >> monitor  # Predictions feed into monitoring

    # Only run training on the 1st of each month
    def _needs_training(execution_date, **kwargs):
        return execution_date.day == 1
    
    # Apply conditional execution
    train_model.trigger_rule = 'none_failed_or_skipped'
    train_model.condition = _needs_training