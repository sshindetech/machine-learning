import logging

from langchain_core.documents import Document

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.document_loaders.web_base import (WebBaseLoader,)
from langchain_community.embeddings import HuggingFaceEmbeddings

import os

import pypdfium2 as pdfium
from langchain_community.vectorstores import Chroma
from langchain_experimental.open_clip import OpenCLIPEmbeddings

from langchain_community.document_loaders.image import UnstructuredImageLoader


from machine_learning.utils.chroma_client import DocumentEmbeddingsClient

import machine_learning.utils.constants as CONST;

class ImageEmbedder(DocumentEmbeddingsClient):
    def __init__(
        self, 
        embeddings,
        doc_path = CONST.DOCUMENT_SOURCE,
        chromadb_host = CONST.CHROM_DB_HOST, 
        collection_name = CONST.CHROME_IMAGE_COLLECTION):
        super().__init__(chromadb_host, collection_name, embeddings)
        self.doc_path = doc_path 
            
    def __get_images_from_pdf(self, pdf_path, img_dump_path):
        """
        Extract images from each page of a PDF document and save as JPEG files.

        :param pdf_path: A string representing the path to the PDF file.
        :param img_dump_path: A string representing the path to dummp images.
        """
        pdf = pdfium.PdfDocument(pdf_path)
        n_pages = len(pdf)
        for page_number in range(n_pages):
            page = pdf.get_page(page_number)
            bitmap = page.render(scale=1, rotation=0, crop=(0, 0, 0, 0))
            pil_image = bitmap.to_pil()
            pil_image.save(f"{img_dump_path}/img_{page_number + 1}.jpg", format="JPEG")
    
    def embedded(self):
        # Load PDF
        doc_path = self.doc_path #Path(__file__).parent / "docs/DDOG_Q3_earnings_deck.pdf"                
        rel_img_dump_path = '/home/allquill/airflow/dags/machine_learning/dags/docs' #img_dump_path.relative_to(Path.cwd())
        
        pil_images = self.__get_images_from_pdf(doc_path, rel_img_dump_path)

        # Get image URIs
        image_uris = sorted(
            [
                os.path.join(rel_img_dump_path, image_name)
                for image_name in os.listdir(rel_img_dump_path)
                if image_name.endswith(".jpg")
            ]
        )

        # Embedded and add images to Vector Store
        logging.info("Embedding images from location: " + doc_path)
        
        textEmbeddingClient = DocumentEmbeddingsClient(
            chromadb_host=self.chromadb_host, 
            collection_name=CONST.CHROM_TEXT_COLLECTION,
            embeddings=HuggingFaceEmbeddings(model_name=CONST.TEXT_MODEL_NAME))
        
        for image_path in image_uris:
            # Save embedding in IMAGE collection for text search
            documents = [ 
                Document(page_content=image_path, metadata={ "photo_image_url": image_path})
            ]
            self.save_documents_and_return_vectorstore(documents) 
            
            # Save embedding in TEXT collection for text search
            loader = UnstructuredImageLoader(image_path)
            textDocuments = loader.load()
            textEmbeddingClient.save_documents_and_return_vectorstore(textDocuments)             

