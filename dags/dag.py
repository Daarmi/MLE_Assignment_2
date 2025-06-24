# dags/loan_ml_pipeline.py
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from airflow.models import Variable
import os

# Configuration - adjust as needed
DEFAULT_START_DATE = datetime(2023, 1, 1)
DEFAULT_END_DATE = datetime(2024, 12, 1)
DATA_DIR = "/opt/airflow/scripts/data"
MODEL_BANK = "/opt/airflow/model_bank"
DATAMART_DIR = "/opt/airflow/scripts/datamart"

def run_data_processing(snapshot_date):
    """Execute data processing pipeline for a specific date"""
    os.system(f"python /opt/airflow/scripts/main.py --snapshot-date {snapshot_date}")

def run_model_training(execution_date):
    """Train model using data up to previous month"""
    # Calculate training end date (last day of previous month)
    train_end = (execution_date - timedelta(days=1)).replace(day=1) - timedelta(days=1)
    train_end_str = train_end.strftime("%Y-%m-%d")
    
    os.system(f"python /opt/airflow/scripts/model_trainer.py "
              f"--train-start {DEFAULT_START_DATE.strftime('%Y-%m-%d')} "
              f"--train-end {train_end_str}")

def run_prediction(snapshot_date):
    """Generate predictions for a specific date"""
    os.system(f"python /opt/airflow/scripts/predictor.py "
              f"--snapshot-date {snapshot_date}")

def run_monitoring(application_date):
    """Monitor model performance for applications from 6 months ago"""
    os.system(f"python /opt/airflow/scripts/monitor.py "
              f"--application-date {application_date}")

default_args = {
    'owner': 'airflow',
    'depends_on_past': True,  # Critical for temporal integrity
    'start_date': DEFAULT_START_DATE,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'max_active_runs': 1  # Ensure sequential processing for data dependencies
}

with DAG(
    'loan_ml_pipeline',
    default_args=default_args,
    description='End-to-End ML Pipeline for Loan Default Prediction',
    schedule_interval='@daily',
    catchup=True,  # Enable backfilling
    max_active_runs=1,
    tags=['cs611', 'ml-pipeline']
) as dag:
    
    # Task 1: Data Processing (Bronze → Silver → Gold)
    process_data = PythonOperator(
        task_id='process_data',
        python_callable=run_data_processing,
        op_kwargs={'snapshot_date': '{{ ds }}'}
    )
    
    # Task 2: Model Training (Runs monthly on 1st)
    train_model = PythonOperator(
        task_id='train_model',
        python_callable=run_model_training,
        op_kwargs={'execution_date': '{{ ds }}'},
        execution_timeout=timedelta(hours=2),
        trigger_rule='all_done'
    )
    
    # Task 3: Daily Predictions
    make_predictions = PythonOperator(
        task_id='make_predictions',
        python_callable=run_prediction,
        op_kwargs={'snapshot_date': '{{ ds }}'}
    )
    
    # Task 4: Model Monitoring (6-month delayed performance)
    monitor_performance = PythonOperator(
        task_id='monitor_performance',
        python_callable=run_monitoring,
        op_kwargs={'application_date': '{{ (execution_date - macros.dateutil.relativedelta.relativedelta(months=6)).strftime("%Y-%m-%d") }}'},
        execution_timeout=timedelta(minutes=30)
    )
    
    # Define workflow dependencies
    process_data >> train_model
    process_data >> make_predictions
    train_model >> make_predictions  # Ensure predictions use latest model
    make_predictions >> monitor_performance

# Only run training on the 1st of each month
def _training_condition(execution_date, **kwargs):
    return execution_date.day == 1

train_model.trigger_rule = 'none_failed'
train_model.condition = _training_condition