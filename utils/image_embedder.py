import logging

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders.image import UnstructuredImageLoader

from machine_learning.utils.chroma_client import DocumentEmbeddingsClient
import machine_learning.utils.constants as CONST;

class ImageEmbedder(DocumentEmbeddingsClient):
    def __init__(
        self, 
        embeddings,
        image_list,
        chromadb_host = CONST.CHROM_DB_HOST, 
        collection_name = CONST.CHROME_IMAGE_COLLECTION):
        super().__init__(chromadb_host, collection_name, embeddings)
        self.image_list = image_list 
    
    def embedded(self):
        image_uris = self.image_list
        
        textEmbeddingClient = DocumentEmbeddingsClient(
            chromadb_host=self.chromadb_host, 
            collection_name=CONST.CHROM_TEXT_COLLECTION,
            embeddings=HuggingFaceEmbeddings(model_name=CONST.TEXT_MODEL_NAME))
        
        for image_path in image_uris:
            # Embedded and add images to Vector Store
            logging.info("Embedding images from location: " + image_path)            
            # Save embedding in IMAGE collection for text search
            documents = [ 
                Document(page_content=image_path, metadata={ "photo_image_url": image_path})
            ]
            self.save_documents_and_return_vectorstore(documents)
            
            # Save embedding in TEXT collection for text search
            loader = UnstructuredImageLoader(image_path)
            textDocuments = loader.load()
            textEmbeddingClient.save_documents_and_return_vectorstore(textDocuments)             

