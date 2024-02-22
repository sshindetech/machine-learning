import logging

import requests
from bs4 import BeautifulSoup

import chromadb

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.document_loaders.web_base import (WebBaseLoader,)
from langchain_community.embeddings import HuggingFaceEmbeddings

import os
from pathlib import Path

import pypdfium2 as pdfium
from langchain_community.vectorstores import Chroma
from langchain_experimental.open_clip import OpenCLIPEmbeddings

class ChromDBClient:
    
    def __init__(self):
        self.chromadb_host = '10.0.1.104', 
        self.collection_name = 'a-test-collection'
        print(f"ChromDBClient initialized: {self.chromadb_host}")
            
    def __init__(self, chromadb_host = '10.0.1.104', collection_name = 'a-test-collection'):
        self.chromadb_host = chromadb_host
        self.collection_name = collection_name
        self.chroma_client = chromadb.HttpClient(host=chromadb_host, port=8000)
        print(f"ChromDBClient initialized: {self.chromadb_host}")
        
    def delete_collection(self):
        logging.info(f"Deleting collection: {self.collection_name}")
        self.chroma_client.delete_collection(self.collection_name)        
         
    def save_and_return_vector_store(self, documents):
        vectorStore = Chroma.from_documents(
            documents, 
            HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"), 
            client=self.chroma_client, 
            collection_name=self.collection_name,
            collection_metadata = {
                "hnsw:space": "cosine",
            }
        )
        return vectorStore
    
    def save_and_return_image_vectorstore(self):        
        # Load embedding function
        print("Loading embedding function")  # noqa: T201
        embedding = OpenCLIPEmbeddings(model_name="ViT-H-14", checkpoint="laion2b_s32b_b79k")

        # Create chroma
        vectorstore_mmembd = Chroma(
            client=self.chroma_client, 
            collection_name="multi-modal-rag",
            embedding_function=embedding,
        )
                
        return vectorstore_mmembd    


class WebScraperEmbedder(ChromDBClient):
    
    def __init__(self):
        super.__init__()
        self.sitemap_url='https://www.netcentric.biz/sitemap.xml'
        self.max_url_to_process = 10,
                
            
    def __init__(self, sitemap_url='https://www.netcentric.biz/sitemap.xml', max_url_to_process = 10,
                chromadb_host = '10.0.1.104', collection_name = 'a-test-collection'):
        super().__init__(chromadb_host, collection_name)
        self.sitemap_url = sitemap_url
        self.max_url_to_process = max_url_to_process
        print(f"WebScraperEmbedder initialized: {self.sitemap_url}")
      
    def parse_html_using_webloader(self, url):
        loader = WebBaseLoader(url)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_overlap = 0,
        chunk_size = 500,
        )
        return text_splitter.split_documents(docs)
  
    def parse_sitemap_and_return_urls(self):
        page_urls = []
        response = requests.get(self.sitemap_url)

        if response.status_code == 200:
            sitemap_content = response.text

            soup = BeautifulSoup(sitemap_content, "lxml")
            urls = soup.find_all("loc")
            logging.info(f"The number of sitemaps are {len(urls)}")
            
            for url in urls:
                if(hasattr(url, 'contents')):
                    page_url = url.contents[0]
                    page_urls.append(page_url)
            return page_urls
        else:
            logging.info(f"Failed to fetch sitemap from {self.sitemap_url}")
            print(f"Failed to fetch sitemap from {self.sitemap_url}")  

    def parse_and_save_sitemap_embedings(self):    
        # sitemap_url = 'https://www.netcentric.biz/insights/2021/09/travel-marketing-automation'
        # documents = parse_html_using_webloader(sitemap_url)
        # save_and_return_vector_store(documents)
            
        urls = self.parse_sitemap_and_return_urls()
        
        count = 0
        
        for u in urls:
            if(count < self.max_url_to_process): #and u.find('/insights/') > -1)
                count = count + 1
                logging.info(f"Processing URL: {u}")
                documents = self.parse_html_using_webloader(u)
                self.save_and_return_vector_store(documents)
            else:
                break
        
        logging.info(f"** COMPLETED LOADING SITEMAP FROM: {self.sitemap_url}")

class ImageEmbedder(ChromDBClient):
    def __init__(self, doc_path = Path(__file__).parent / "docs/DDOG_Q3_earnings_deck.pdf",
                 chromadb_host = '10.0.1.104', collection_name = 'multi-modal-rag'):
        super().__init__(chromadb_host, collection_name)
        self.doc_path = doc_path 
            
    def get_images_from_pdf(self, pdf_path, img_dump_path):
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
        img_dump_path = Path(__file__).parent / "docs/"
        rel_doc_path = doc_path.relative_to(Path.cwd())
        rel_img_dump_path = img_dump_path.relative_to(Path.cwd())
        print("pdf index")  # noqa: T201
        
        pil_images = self.get_images_from_pdf(rel_doc_path, rel_img_dump_path)
        print("done")  # noqa: T201
        
        # vectorstore = Path(__file__).parent / "chroma_db_multi_modal"
        # re_vectorstore_path = vectorstore.relative_to(Path.cwd())

        vectorstore_mmembd = self.save_and_return_image_vectorstore()
        # Get image URIs
        image_uris = sorted(
            [
                os.path.join(rel_img_dump_path, image_name)
                for image_name in os.listdir(rel_img_dump_path)
                if image_name.endswith(".jpg")
            ]
        )

        # Add images
        print("Embedding images")  # noqa: T201
        vectorstore_mmembd.add_images(uris=image_uris)        

# WebScraperEmbedder().parse_and_save_sitemap_embedings()
        
# ChromDBClient().delete_collection()
