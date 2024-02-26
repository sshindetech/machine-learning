import logging

import chromadb

from langchain_community.vectorstores import Chroma

import machine_learning.utils.constants as CONST;

class ChromDBClient:
            
    def __init__(self, chromadb_host, collection_name):
        self.chromadb_host = chromadb_host
        self.collection_name = collection_name
        self.chroma_client = chromadb.HttpClient(host=chromadb_host, port=CONST.CHROMA_DB_PORT)
        logging.info(f"ChromDBClient initialized: {self.chromadb_host}")
        
    def delete_collection(self):
        logging.info(f"Deleting collection: {self.collection_name}")
        self.chroma_client.delete_collection(self.collection_name)        
         
         
class DocumentEmbeddingsClient(ChromDBClient):   
        
    def __init__(self, chromadb_host, collection_name):
        super().__init__(chromadb_host, collection_name)
                  
    def __init__(self, chromadb_host, collection_name, embeddings):
        super().__init__(chromadb_host, collection_name)
        self.embeddings = embeddings
        logging.info(f"DocumentEmbeddings initialized: {self.chromadb_host}")
                  
    def save_documents_and_return_vectorstore(self, documents):
        logging.info("START: save_documents_and_return_vectorstore")
        vectorStore = Chroma.from_documents(
            documents, 
            self.embeddings , 
            client=self.chroma_client, 
            collection_name=self.collection_name,
            collection_metadata = {
                "hnsw:space": "cosine",
            }
        )
        logging.info("END: save_documents_and_return_vectorstore")
        
        return vectorStore

