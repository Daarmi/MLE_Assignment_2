# dags/loan_ml_pipeline.py  (simplified: direct inference, monitoring, visualization)
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
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
def run_script(script_name: str, **kwargs) -> None:
    cmd = ["python", os.path.join(SCRIPTS_DIR, script_name)]
    if kwargs.get('snapshot_date'):
        cmd += ["--snapshot-date", kwargs['snapshot_date']]
    if kwargs.get('start_date'):
        cmd += ["--start-date", kwargs['start_date']]
    if kwargs.get('end_date'):
        cmd += ["--end-date", kwargs['end_date']]
    result = subprocess.run(
        cmd, cwd=SCRIPTS_DIR, capture_output=True, text=True
    )
    if result.returncode != 0:
        print("---- stdout ----", result.stdout)
        print("---- stderr ----", result.stderr)
        raise subprocess.CalledProcessError(result.returncode, cmd)

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
        dag_id="loan_ml_pipeline",
        default_args=default_args,
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

        build_model_gold = PythonOperator(
            task_id="build_gold_model_table",
            python_callable=run_script,
            op_kwargs={"script_name": "gold_model_table.py", "snapshot_date": "{{ ds }}"},
        )

        data_check = PythonOperator(
            task_id="data_check",
            python_callable=run_script,
            op_kwargs={"script_name": "data_check.py", "snapshot_date": "{{ ds }}"},
        )

        model_inference = PythonOperator(
            task_id="model_inference",
            python_callable=run_script,
            op_kwargs={"script_name": "model_inference.py", "snapshot_date": "{{ ds }}"},
        )

        model_monitoring = PythonOperator(
            task_id="model_monitoring",
            python_callable=run_script,
            op_kwargs={
                "script_name": "model_monitoring.py",
                "start_date": "{{ macros.ds_add(ds, -30) }}",
                "end_date": "{{ ds }}",
            },
        )

        performance_visualization = PythonOperator(
            task_id="performance_visualization",
            python_callable=run_script,
            op_kwargs={"script_name": "performance_visualization.py"},
        )

        # Define linear dependencies
        bronze >> silver >> gold >> build_model_gold >> data_check \
               >> model_inference >> model_monitoring >> performance_visualization

        return dag

# instantiate the DAG
global dag
dag = create_dag()
