# dags/loan_ml_pipeline.py
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.dates import days_ago
from dateutil.relativedelta import relativedelta
import os
import glob

# Configuration
DEFAULT_START_DATE = datetime(2023, 1, 1)
SCRIPTS_DIR = "/opt/airflow/scripts"
DATAMART_DIR = "/opt/airflow/datamart"  # Base directory for all data

def check_data_ready(snapshot_date, months_needed=6):
    """Check if required historical data exists"""
    date_objs = [datetime.strptime(snapshot_date, "%Y-%m-%d") - relativedelta(months=i) 
                 for i in range(months_needed)]
    
    # Check if all required gold files exist
    all_exist = True
    for dt in date_objs:
        date_str = dt.strftime("%Y_%m_%d")
        paths_to_check = [
            f"{DATAMART_DIR}/gold/feature_store/eng/gold_ft_store_engagement_{date_str}.parquet",
            f"{DATAMART_DIR}/gold/feature_store/cust_fin_risk/gold_ft_store_cust_fin_risk_{date_str}.parquet",
            f"{DATAMART_DIR}/gold/label_store/gold_label_store_{date_str}.parquet"
        ]
        if not all(os.path.exists(p) for p in paths_to_check):
            print(f"Missing data for {date_str}")
            all_exist = False
    
    return "proceed" if all_exist else "skip"

def run_bronze(snapshot_date):
    os.system(f"python {SCRIPTS_DIR}/bronze_processing.py --snapshot-date {snapshot_date}")

def run_silver(snapshot_date):
    os.system(f"python {SCRIPTS_DIR}/silver_processing.py --snapshot-date {snapshot_date}")

def run_gold(snapshot_date):
    os.system(f"python {SCRIPTS_DIR}/gold_processing.py --snapshot-date {snapshot_date}")

def run_model_training(execution_date):
    os.system(f"python {SCRIPTS_DIR}/model_trainer.py --execution-date {execution_date}")

def run_prediction(snapshot_date):
    os.system(f"python {SCRIPTS_DIR}/predictor.py --snapshot-date {snapshot_date}")

def run_monitoring(application_date):
    os.system(f"python {SCRIPTS_DIR}/monitor.py --application-date {application_date}")

default_args = {
    'owner': 'airflow',
    'depends_on_past': True,
    'start_date': DEFAULT_START_DATE,
    'email_on_failure': False,
    'retries': 0,  # Disable retries for now
    'retry_delay': timedelta(minutes=5),
    'max_active_runs': 1
}

with DAG(
    'loan_ml_pipeline',
    default_args=default_args,
    schedule_interval='@monthly',
    catchup=True,
    max_active_runs=1,
    tags=['loan_ml']
) as dag:
    
    # Data processing tasks
    bronze = PythonOperator(
        task_id='bronze_processing',
        python_callable=run_bronze,
        op_kwargs={'snapshot_date': '{{ ds }}'}
    )
    
    silver = PythonOperator(
        task_id='silver_processing',
        python_callable=run_silver,
        op_kwargs={'snapshot_date': '{{ ds }}'}
    )
    
    gold = PythonOperator(
        task_id='gold_processing',
        python_callable=run_gold,
        op_kwargs={'snapshot_date': '{{ ds }}'}
    )
    
    # Data readiness check for training
    check_train_data = BranchPythonOperator(
        task_id='check_training_data_ready',
        python_callable=check_data_ready,
        op_kwargs={
            'snapshot_date': '{{ ds }}',
            'months_needed': 6
        }
    )
    
    # Training path
    proceed_train = DummyOperator(task_id='proceed_train')
    skip_train = DummyOperator(task_id='skip_train')
    
    train_model = PythonOperator(
        task_id='train_model',
        python_callable=run_model_training,
        op_kwargs={'execution_date': '{{ ds }}'},
        execution_timeout=timedelta(hours=2)
    )
    
    # Data readiness check for prediction
    check_prediction_data = BranchPythonOperator(
        task_id='check_prediction_data_ready',
        python_callable=check_data_ready,
        op_kwargs={
            'snapshot_date': '{{ ds }}',
            'months_needed': 1  # Only need current month
        }
    )
    
    # Prediction path
    proceed_pred = DummyOperator(task_id='proceed_prediction')
    skip_pred = DummyOperator(task_id='skip_prediction')
    
    predict = PythonOperator(
        task_id='make_predictions',
        python_callable=run_prediction,
        op_kwargs={'snapshot_date': '{{ ds }}'}
    )
    
    monitor = PythonOperator(
        task_id='monitor_performance',
        python_callable=run_monitoring,
        op_kwargs={'application_date': '{{ (execution_date - macros.dateutil.relativedelta.relativedelta(months=6)).strftime("%Y-%m-%d") }}'},
        execution_timeout=timedelta(minutes=30)
    )
    
    # Workflow dependencies
    bronze >> silver >> gold
    
    # Training branch
    gold >> check_train_data
    check_train_data >> [proceed_train, skip_train]
    proceed_train >> train_model
    
    # Prediction branch
    gold >> check_prediction_data
    check_prediction_data >> [proceed_pred, skip_pred]
    
    # Synchronization point
    [train_model, proceed_train, skip_train, proceed_pred, skip_pred] >> predict
    predict >> monitor