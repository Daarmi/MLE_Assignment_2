# dags/loan_ml_pipeline.py  (MINIMAL: bronze → silver → gold)
# ------------------------------------------------------------------
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import os
import subprocess
import subprocess, textwrap

# ------------------------------------------------------------------
# Global configuration
# ------------------------------------------------------------------
DEFAULT_START_DATE = datetime(2023, 1, 1)
DEFAULT_END_DATE = datetime(2024, 12, 31)
SCRIPTS_DIR = "/opt/airflow/scripts"          # where bronze_store.py etc. live

def run_script(script_name: str, snapshot_date: str) -> None:
    """
    Generic launcher so we don’t repeat os.system calls.
    Airflow substitutes {{ ds }} with the execution date (YYYY-MM-DD).
    """
    cmd = ["python", f"{SCRIPTS_DIR}/{script_name}", "--snapshot-date", snapshot_date]
    subprocess.run(cmd, check=True, cwd=SCRIPTS_DIR) 

# ------------------------------------------------------------------
# DAG definition
# ------------------------------------------------------------------
default_args = {
    "owner": "airflow",
    "depends_on_past": True,
    "start_date": DEFAULT_START_DATE,
    "end_date": DEFAULT_END_DATE,
    "email_on_failure": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
    "max_active_runs": 1,
}

with DAG(
    dag_id="loan_ml_pipeline_minimal",
    start_date=DEFAULT_START_DATE, 
    end_date=DEFAULT_END_DATE,     # ← earliest logical date
    schedule_interval="@monthly",
    catchup=True,
    max_active_runs=1,
    tags=["loan_ml"],
) as dag:

    # -------------------------------
    # 1️⃣  Bronze layer
    # -------------------------------
    bronze = PythonOperator(
        task_id="bronze_store",
        python_callable=run_script,
        op_kwargs={
            "script_name": "bronze_store.py",
            "snapshot_date": "{{ ds }}",
        },
    )

    # -------------------------------
    # 2️⃣  Silver layer
    # -------------------------------
    silver = PythonOperator(
        task_id="silver_store",
        python_callable=run_script,
        op_kwargs={
            "script_name": "silver_store.py",
            "snapshot_date": "{{ ds }}",
        },
    )

    # -------------------------------
    # 3️⃣  Gold layer
    # -------------------------------
    gold = PythonOperator(
        task_id="gold_store",
        python_callable=run_script,
        op_kwargs={
            "script_name": "gold_store.py",
            "snapshot_date": "{{ ds }}",
        },
    )

    # Workflow: bronze ➜ silver ➜ gold
    bronze >> silver >> gold

    # ------------------------------------------------------------------
    # ✂️  The following MLOps tasks are kept for later but disabled now
    # ------------------------------------------------------------------
    # from airflow.operators.branch import BranchPythonOperator
    # from airflow.operators.dummy import DummyOperator
    #
    # def check_data_ready(...):
    #     ...
    #
    # model_training = PythonOperator(...)
    # prediction      = PythonOperator(...)
    # monitoring      = PythonOperator(...)
    #
    # gold >> check_data_ready >> ...
