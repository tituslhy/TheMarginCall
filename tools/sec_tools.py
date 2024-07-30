
#%%
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

import torch

from llama_index.core import (
    VectorStoreIndex,
    Document
)
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.postprocessor.longllmlingua import LongLLMLinguaPostprocessor

from sec_api import QueryApi
import requests
from bs4 import BeautifulSoup

import sys
__curdir__ = os.getcwd()

if "tools" in __curdir__:
    sys.path.append(os.path.join(
        __curdir__,
        "../src"
    ))
else:
    sys.path.append("./src")

from llamaindex_config import llm, embed_model, text_splitter

llm = llm
embed_model = embed_model
text_splitter = text_splitter
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#%%
class SECTool(BaseToolSpec):
    """Tools to read SEC10K reports"""
    
    spec_functions=[
        "search_10q_10k",
    ]
    
    def __init__(self, 
                 sec_api_key: str = os.getenv('SEC_API_KEY'),
                 cohere_api_key: str = os.getenv('COHERE_API_KEY'),
                 device: str = device
                 ):
        """Initialize SEC tool"""
        self.sec_api_key = sec_api_key
        self.cohere_api_key = cohere_api_key
        if self.sec_api_key is None:
            raise ValueError("SEC API key cannot be none")
        if self.cohere_api_key is None:
            raise ValueError("Cohere API key cannot be none")
        self.queryApi = QueryApi(api_key=self.sec_api_key)
        self.reranker = CohereRerank(top_n = 4, api_key = self.cohere_api_key)
        self.device = device
        self.prompt_compressor = LongLLMLinguaPostprocessor(
            instruction_str = "Given the context, please answer the final question",
            target_token = 300,
            rank_method = "longllmlingua",
            additional_compress_kwargs = {
                "condition_compare": True,
                "condition_in_question": "after",
                "context_budget": "+100",
                "reorder_context": "sort", #enables document reorder
                "dynamic_context_compression_ratio": 0.3,
            },
            model_name = "gpt2",
            device_map = self.device
        )

    @staticmethod
    def _download_form_html(url: str):
        """Function to download text from SEC website"""
        headers = {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'en-US,en;q=0.9,pt-BR;q=0.8,pt;q=0.7',
            'Cache-Control': 'max-age=0',
            'Dnt': '1',
            'Sec-Ch-Ua': '"Not_A Brand";v="8", "Chromium";v="120"',
            'Sec-Ch-Ua-Mobile': '?0',
            'Sec-Ch-Ua-Platform': '"macOS"',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Upgrade-Insecure-Requests': '1',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        return response.text
    
    def get_retriever_from_url(self, url: str, embed_model=embed_model):   
        """Creates an in-memory retriever from a URL"""
        text = self._download_form_html(url=url)
        soup = BeautifulSoup(text, 'html.parser')
        texts = soup.get_text()
        nodes = text_splitter.get_nodes_from_documents([Document(text=texts)])
        return VectorStoreIndex(nodes, embed_model=embed_model).as_retriever(
            similarity_top_k = 10
        )
    
    def return_contexts(self, url: str, question: str):
        """Retrieves and reranks nodes given a query string and a url 
        from an in-memory vector index"""
        retriever = self.get_retriever_from_url(url = url)
        nodes = retriever.retrieve(question)
        reranked_nodes = self.reranker.postprocess_nodes(
            nodes = nodes,
            query_str = question)
        refined_nodes = self.prompt_compressor.postprocess_nodes(
            nodes = reranked_nodes,
            query_str = question
        )
        contexts = "\n\n".join([n.get_content() for n in refined_nodes])
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        return contexts
    
    def search_10q_10k(
        self, 
        ticker: str, 
        question: str,
        tenq: bool = True):
        """
        Useful to search information from the latest 10-Q or 10-K forms of a
        given stock.
        args:
            ticker (str): ticker of interest
            query (str): the question of interest
            tenq (bool): Whether or not to search the 10-Q form
        """
        if tenq is True:
            query = {
                "query": {
                    "query_string": {
                        "query": f"ticker:{ticker} AND formType:\"10-Q\""
                    }
                },
                "from": "0",
                "size": "1",
                "sort": [{ "filedAt": { "order": "desc" }}]
            }
        else:
            query = {
            "query": {
                "query_string": {
                "query": f"ticker:{ticker} AND formType:\"10-K\""
                }
            },
            "from": "0",
            "size": "1",
            "sort": [{ "filedAt": { "order": "desc" }}]
            }
        filings = self.queryApi.get_filings(query)['filings']
        if len(filings) == 0:
            return "Sorry I couldn't find any filing for this stock, check if ticker is correct"
        link = filings[0]['linkToFilingDetails']
        return self.return_contexts(url=link, question=question)

def get_sec_tool():
    """Return SEC tool powered by SEC Edgar Filings API"""
    secTool = SECTool()
    return secTool.to_tool_list()
