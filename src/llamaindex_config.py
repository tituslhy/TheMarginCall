#%%
import os
from dotenv import load_dotenv, find_dotenv
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.bedrock_converse import BedrockConverse
from llama_index.embeddings.bedrock import BedrockEmbedding

_ = load_dotenv(find_dotenv())

llm = BedrockConverse(
    model = "anthropic.claude-3-5-sonnet-20240620-v1:0",
    aws_access_key_id = os.environ["AWS_ACCESS_KEY"],
    aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"],
    region_name = "us-east-1"
)
embed_model = BedrockEmbedding(
    model = "amazon.titan-embed-text-v1",
    aws_access_key_id = os.environ["AWS_ACCESS_KEY"],
    aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"],
    aws_region_name = os.environ["AWS_DEFAULT_REGION"]
)
text_splitter = SentenceSplitter(
    chunk_size = 1024,
    chunk_overlap = 20
)
# %%
