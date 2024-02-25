
import textwrap
from datetime import datetime, timedelta

# The DAG object; we'll need this to instantiate a DAG
from airflow.models.dag import DAG
from airflow.models.param import Param

# Operators; we need this to operate!
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

from dags.machine_learning.utils.sitemap_embedder import SitemapEmbedder

with DAG(
    "ml_url_embedder_dag",
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
    description="A simple DAG to process HTML content from provided sitemap.xml",
    schedule=None,
    catchup=False,
    tags=["machine_learning"],
    params={
         "url_list": Param("https://js.langchain.com/docs/get_started/introduction,https://js.langchain.com/docs/get_started/installation", type=["null", "string"]),
         "chromadb_host_url": Param('10.0.1.104', type=["null", "string"]),
         "chromadb_collection_name": Param('a-test-collection', type=["null", "string"])
     }    
) as dag:

    # t1, and t2 are examples of tasks created by instantiating operators
    t1 = BashOperator(
        task_id="print_date",
        bash_command="date",
        dag=dag,
    )

    def create_emebeddings(**context):
        url_list = context["params"]["url_list"]
        chromadb_host = context["params"]["chromadb_host_url"]
        collection_name = context["params"]["chromadb_collection_name"]  
        print(f"Indexing URLs {url_list}")
        
        if(url_list):
            scrapper = SitemapEmbedder(chromadb_host=chromadb_host,collection_name=collection_name)
            urls = url_list.split(",")
            for url in urls:
                print(f"Creating Embeddings for URLs {url}")
                documents = scrapper.parse_html_using_webloader(url=url)
                scrapper.save_documents_and_return_vectorstore(documents)  
                                                
    
    # Task: Scrape URLs and write to a file
    t2 = PythonOperator(
        task_id='parse_and_save_url_embedings',
        python_callable=create_emebeddings,
        dag=dag,
    )

    t1 >> [t2]