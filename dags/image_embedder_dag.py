from datetime import datetime, timedelta
import logging
import json

# The DAG object; we'll need this to instantiate a DAG
from airflow.models.dag import DAG
from airflow.models.param import Param

# Operators; we need this to operate!
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

from machine_learning.utils.image_embedder import ImageEmbedder
from machine_learning.utils.clip_embeddings import CLIPEmbeddings
from machine_learning.utils.pdf_to_image_converter import PDFToImageConverter

import machine_learning.utils.constants as CONST;

with DAG(
    "ml_image_embedder_dag",
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
    description="A simple DAG to process Images from PDF and create embeddings",
    schedule=None,
    catchup=False,
    tags=["machine_learning"],
    params={
         "docs_folder": Param(CONST.DOCUMENT_SOURCE, type=["null", "string"]),
         "chromadb_host_url": Param(CONST.CHROM_DB_HOST, type=["null", "string"]),
         "collection_name": Param(CONST.CHROME_IMAGE_COLLECTION, type=["null", "string"])
     }    
) as dag:

    def convert_pdf_to_images(**context):
        docs_folder = context["params"]["docs_folder"]
        
        converter = PDFToImageConverter(doc_path=docs_folder)
        image_list = converter.convert()
        context['ti'].xcom_push(key='image_list', value=json.dumps(image_list))

    def create_emebeddings(**context):
        docs_folder = context["params"]["docs_folder"]
        chromadb_host = context["params"]["chromadb_host_url"]
        collection_name = context["params"]["collection_name"] 
        print(f"Indexing PDF from {docs_folder}")
        
        xcom_image_list = context['ti'].xcom_pull(key='image_list')
        image_list = json.loads(xcom_image_list)
        
        if(docs_folder):
            image_embedder = ImageEmbedder(image_list=image_list,
                                           chromadb_host=chromadb_host,
                                           collection_name=collection_name, 
                                           embeddings=CLIPEmbeddings(model_name=CONST.IMAGE_MODEL_NAME))
            image_embedder.embedded()
    
    # t1, and t2 are examples of tasks created by instantiating operators
    t1 = BashOperator(
        task_id="print_date",
        bash_command="date",
        dag=dag,
    )
        
    # Task: Scrape URLs and write to a file
    t2 = PythonOperator(
        task_id='convert_pdf_to_images',
        python_callable=convert_pdf_to_images,
        dag=dag,
    )
    
    # Task: Scrape URLs and write to a file
    t3 = PythonOperator(
        task_id='create_emebeddings',
        python_callable=create_emebeddings,
        dag=dag,
    )

    # t1.set_downstream(t2)
    # t2.set_downstream(t3)
    
    t1 >> t2 >> t3