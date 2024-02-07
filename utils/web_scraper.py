import logging

import requests
from bs4 import BeautifulSoup

import chromadb

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.document_loaders.web_base import (WebBaseLoader,)
from langchain_community.embeddings import HuggingFaceEmbeddings

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



# WebScraperEmbedder().parse_and_save_sitemap_embedings()
        
# ChromDBClient().delete_collection()
