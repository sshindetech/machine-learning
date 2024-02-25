import logging

import requests
from bs4 import BeautifulSoup

from langchain_core.documents import Document

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.document_loaders.web_base import (WebBaseLoader,)
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import Chroma
from langchain_experimental.open_clip import OpenCLIPEmbeddings

from langchain_community.document_loaders.image import UnstructuredImageLoader

from dags.machine_learning.utils.chroma_client import DocumentEmbeddingsClient
import machine_learning.utils.constants as CONST;

class SitemapEmbedder(DocumentEmbeddingsClient):
                
    def __init__(
        self, sitemap_url = CONST.SITEMAP_DEFAULT, 
        max_url_to_process = CONST.SITEMAP_MAX_URL_TO_PROCESS,
        chromadb_host = CONST.CHROM_DB_HOST, 
        collection_name = CONST.CHROM_TEXT_COLLECTION,
        embeddings = HuggingFaceEmbeddings(model_name=CONST.TEXT_MODEL_NAME)):
        
        super().__init__(chromadb_host, collection_name, embeddings)
        self.sitemap_url = sitemap_url
        self.max_url_to_process = max_url_to_process
        logging.info(f"WebScraperEmbedder initialized: {self.sitemap_url}")
      
    def parse_html_using_webloader(self, url):
        loader = WebBaseLoader(url)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_overlap = 0,
            chunk_size = 500,
        )
        
        return text_splitter.split_documents(docs)
  
    def __parse_sitemap_and_return_urls(self):
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
        urls = self.__parse_sitemap_and_return_urls()
        count = 0
        
        for u in urls:
            if(count < self.max_url_to_process): #and u.find('/insights/') > -1)
                count = count + 1
                logging.info(f"Processing URL: {u}")
                documents = self.parse_html_using_webloader(u)
                self.save_documents_and_return_vectorstore(documents)
            else:
                break
        
        logging.info(f"** COMPLETED LOADING SITEMAP FROM: {self.sitemap_url}")
