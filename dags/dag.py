# dags/loan_ml_pipeline.py  (with conditional model_training)
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.models import Variable
import os
import subprocess

# ------------------------------------------------------------------
# Global configuration
# ------------------------------------------------------------------
DEFAULT_START_DATE = datetime(2023, 1, 1)
DEFAULT_END_DATE   = datetime(2024, 12, 31)
SCRIPTS_DIR        = "/opt/airflow/scripts"

# -----------------------
# Helper to run scripts
# -----------------------
def run_script(script_name: str, snapshot_date: str) -> None:
    cmd = ["python", f"{SCRIPTS_DIR}/{script_name}", "--snapshot-date", snapshot_date]
    # capture both streams so we can print them on error
    result = subprocess.run(
        cmd,
        cwd=SCRIPTS_DIR,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print("---- script stdout ----")
        print(result.stdout)
        print("---- script stderr ----")
        print(result.stderr)
        # re-raise so Airflow still marks the task FAILED
        raise subprocess.CalledProcessError(result.returncode, cmd)

# -----------------------
# Decide whether to train
# -----------------------
def _should_train(snapshot_date: str, **kwargs) -> str:
    # Paths
    scripts_root  = SCRIPTS_DIR
    dm_root       = os.path.join(scripts_root, "datamart")
    label_dir     = os.path.join(dm_root, "gold", "label_store")
    model_bank    = os.path.join(dm_root, "model_bank")

    # Count months of labels
    files = [f for f in os.listdir(label_dir) if f.startswith("gold_label_store_")]
    months_loaded = len(files)

    # Path for this model version
    model_ts      = snapshot_date.replace('-', '_')
    model_path    = os.path.join(model_bank, f"model_{model_ts}")

    # Initial train on 12th month
    if months_loaded == 12 and not os.path.exists(model_path):
        return "model_training"
    # Skip otherwise
    return "skip_model_training"

# ------------------------------------------------------------------
# DAG definition
# ------------------------------------------------------------------
def create_dag():
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
        default_args=default_args,
        start_date=DEFAULT_START_DATE,
        end_date=DEFAULT_END_DATE,
        schedule_interval="@monthly",
        catchup=True,
        max_active_runs=1,
        tags=["loan_ml"],
    ) as dag:

        bronze = PythonOperator(
            task_id="bronze_store",
            python_callable=run_script,
            op_kwargs={"script_name": "bronze_store.py", "snapshot_date": "{{ ds }}"},
        )

        silver = PythonOperator(
            task_id="silver_store",
            python_callable=run_script,
            op_kwargs={"script_name": "silver_store.py", "snapshot_date": "{{ ds }}"},
        )

        gold = PythonOperator(
            task_id="gold_store",
            python_callable=run_script,
            op_kwargs={"script_name": "gold_store.py", "snapshot_date": "{{ ds }}"},
        )

        data_check = PythonOperator(
            task_id="data_check",
            python_callable=run_script,
            op_kwargs={"script_name": "data_check.py", "snapshot_date": "{{ ds }}"},
        )

        # Branch: decide to train or skip
        decide = BranchPythonOperator(
            task_id="branch_model_training",
            python_callable=_should_train,
            op_kwargs={"snapshot_date": "{{ ds }}"},
        )

        model_training = PythonOperator(
            task_id="model_training",
            python_callable=run_script,
            op_kwargs={"script_name": "model_training.py", "snapshot_date": "{{ ds }}"},
        )

        skip_training = DummyOperator(
            task_id="skip_model_training"
        )

        # Join after branch
        join = DummyOperator(
            task_id="after_model_training",
            trigger_rule="none_failed_min_one_success"
        )

        # Define dependencies
        bronze >> silver >> gold >> data_check >> decide
        decide >> model_training >> join
        decide >> skip_training  >> join

        return dag

    # instantiate the DAG
global dag
dag = create_dag()
