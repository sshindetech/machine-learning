from datetime import datetime, timedelta

# The DAG object; we'll need this to instantiate a DAG
from airflow.models.dag import DAG
from airflow.models.param import Param

# Operators; we need this to operate!
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

from machine_learning.utils.image_embedder import ImageEmbedder
from machine_learning.utils.clip_embeddings import CLIPEmbeddings
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
         "chromadb_host_url": Param('10.0.1.104', type=["null", "string"]),
         "collection_name": Param('a-test-collection', type=["null", "string"]),
         "image_collection_name": Param('multi-modal-rag', type=["null", "string"])
     }    
) as dag:

    # t1, and t2 are examples of tasks created by instantiating operators
    t1 = BashOperator(
        task_id="print_date",
        bash_command="date",
        dag=dag,
    )

    def create_emebeddings(**context):
        docs_folder = context["params"]["docs_folder"]
        chromadb_host = context["params"]["chromadb_host_url"]
        collection_name = context["params"]["collection_name"] 
        image_collection_name = context["params"]["image_collection_name"]  
        print(f"Indexing PDF from {docs_folder}")
        
        if(docs_folder):
            image_embedder = ImageEmbedder(doc_path=docs_folder,
                                           chromadb_host=chromadb_host,
                                           collection_name=collection_name, 
                                           embeddings=CLIPEmbeddings(model_name=CONST.IMAGE_MODEL_NAME))
            image_embedder.embedded()
    
    # Task: Scrape URLs and write to a file
    t2 = PythonOperator(
        task_id='parse_and_save_image_embedings',
        python_callable=create_emebeddings,
        dag=dag,
    )

    t1 >> [t2]