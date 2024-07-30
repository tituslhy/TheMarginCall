import os
from prompt_optimization import load_index

import sys
__curdir__ = os.getcwd()
if "tools" in __curdir__ or "notebooks" in __curdir__:
    sys.path.append(os.path.join(
        __curdir__,
        "../src"
    ))
else:
    sys.path.append("./src")
from llamaindex_config import llm, embed_model

import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core.tools import (
    QueryEngineTool,
    ToolMetadata
)

client = qdrant_client.QdrantClient("http://localhost:6333")
aclient = qdrant_client.AsyncQdrantClient("http://localhost:6333")
vector_store = QdrantVectorStore(
    collection_name="investopedia",
    client=client,
    aclient=aclient,
    fastembed_sparse_model="Qdrant/bm42-all-minilm-l6-v2-attentions",
)

def get_rag_tools(qdrant_vector_store = vector_store,
                  llm = llm,
                  embed_model = embed_model,
                  similarity_top_k: int = 4,
                  sparse_top_k: int = 10):
    def load_index(qdrant_vector_store: QdrantVectorStore,
                embed_model):
        return VectorStoreIndex.from_vector_store(
            qdrant_vector_store,
            embed_model
        ) 
    index = load_index(qdrant_vector_store, embed_model)
    query_engine = index.as_query_engine(
        llm = llm,
        similarity_top_k=similarity_top_k, 
        sparse_top_k=sparse_top_k)
    query_engine_tool = [
        QueryEngineTool(
            query_engine=query_engine,
            metadata = ToolMetadata(
                name="investopedia_tool",
                description = (
                    "Provides conceptual information relating to the stock market, "
                    "stock market instruments such as options and futures, "
                    "as well as conceptual understanding for metrics used in "
                    "technical analysis and fundamental analysis."
                )
            )
        )
    ]
    return query_engine_tool

# from llama_index.core import PromptTemplate

# def get_rag_tools(prompt_path: str):
#     """Returns a query engine tool with a DSPy optimized prompt template"""
    
#     file = open(prompt_path, "r")
#     tmpl = PromptTemplate(file.read())
#     index = load_index()
#     query_engine = index.as_query_engine(
#         text_qa_tmpl = tmpl,
#         llm = llm,
#         embed_model=embed_model
#     )
#     query_engine_tool = [
#         QueryEngineTool(
#             query_engine=query_engine,
#             metadata = ToolMetadata(
#                 name="investopedia_tool",
#                 description = (
#                     "Provides conceptual information relating to the stock market, "
#                     "stock market instruments such as options and futures, "
#                     "as well as conceptual understanding for metrics used in "
#                     "technical analysis and fundamental analysis."
#                 )
#             )
#         )
#     ]
#     return query_engine_tool
