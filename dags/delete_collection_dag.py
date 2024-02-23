
import textwrap
from datetime import datetime, timedelta

# The DAG object; we'll need this to instantiate a DAG
from airflow.models.dag import DAG
from airflow.models.param import Param

# Operators; we need this to operate!
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

from machine_learning.utils.web_scraper import ChromDBClient

import logging

with DAG(
    "ml_delete_collection",
    # These args will get passed on to each operator
    # You can override them on a per-task basis during operator initialization
    default_args={
        "depends_on_past": False,
        "email": ["sshindetech@gmail.com"],
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
        # 'queue': 'bash_queue',
        # 'pool': 'backfill',
        # 'priority_weight': 10,
        # 'end_date': datetime(2016, 1, 1),
        # 'wait_for_downstream': False,
        # 'sla': timedelta(hours=2),
        # 'execution_timeout': timedelta(seconds=300),
        # 'on_failure_callback': some_function, # or list of functions
        # 'on_success_callback': some_other_function, # or list of functions
        # 'on_retry_callback': another_function, # or list of functions
        # 'sla_miss_callback': yet_another_function, # or list of functions
        # 'trigger_rule': 'all_success'
    },
    description="A simple DAG to delete collection",
    schedule=None,
    catchup=False,
    params={
         "chromadb_host_url": Param('10.0.1.104', type=["null", "string"]),
         "chromadb_collection_name": Param('a-test-collection', type=["null", "string"])
     }, 
    tags=["machine_learning"],
) as dag:

    # t1, and t2 are examples of tasks created by instantiating operators
    t1 = BashOperator(
        task_id="print_date",
        bash_command="date",
        dag=dag,
    )
    
    def delete_collection(**context):
        chromadb_host = context["params"]["chromadb_host_url"]
        collection_name = context["params"]["chromadb_collection_name"]  
        logging.info(f"Sending request for deleting collection: {collection_name}")   
        chroma_client = ChromDBClient(chromadb_host=chromadb_host, collection_name=collection_name)   
        return chroma_client.delete_collection()
        
    # Task: Scrape URLs and write to a file
    t2 = PythonOperator(
        task_id='delete_collection_task',
        python_callable=delete_collection,
        dag=dag,
    )

    t1 >> [t2]