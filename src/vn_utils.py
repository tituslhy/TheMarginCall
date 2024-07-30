import os
from dotenv import load_dotenv, find_dotenv
from vanna.openai.openai_chat import OpenAI_Chat
from openai import AzureOpenAI
from vanna.chromadb.chromadb_vector import ChromaDB_VectorStore
import yfinance as yf
from .db_utils import connect_sql

__curdir__ = os.getcwd()
if ("src" in __curdir__) or ("notebooks" in __curdir__):
    db_path = "../database/stocks.db"
else:
    db_path = "./database/stocks.db"

_ = load_dotenv(find_dotenv())


client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_GPT4O_DEPLOYMENT_NAME"],
    api_version=os.environ["AZURE_API_VERSION"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
)

class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self,
                             client = client,
                             config = config)

def get_long_name(ticker: str):
    return yf.Ticker(ticker).info['longName']

## Instantiate and initialize
vn = MyVanna(config={"model":"gpt4-o"})
vn.connect_to_sqlite(db_path)

_, tables = connect_sql(db_path)

for ticker in tables:
    vn.train(ddl=f"""
        CREATE TABLE IF NOT EXISTS {ticker.lower()} (
            id INT PRIMARY KEY,
            date DATETIME,
            open FLOAT,
            high FLOAT,
            low FLOAT,
            close FLOAT,
            volume INT,
        )
    """)
    vn.train(
        documentation = f"{get_long_name(ticker)}'s stock ticker is {ticker}."
    )