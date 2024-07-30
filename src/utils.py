from llama_index.readers.web import SimpleWebPageReader
import pandas as pd
import requests

## Utility functions ##
def rename_columns(ticker: str, df: pd.DataFrame) -> pd.DataFrame:
    """Helper function to post-process dataframe column names"""
    column_dict = {old_name: process_string(ticker = ticker,
                                            string_ = old_name) \
        for old_name in df.columns.values.tolist()}
    df.rename(columns=column_dict, inplace=True)
    return df

def process_string(ticker: str, 
                    string_: str) -> str:
    return f"{ticker}_{'_'.join(string_.lower().split(" "))}"


class CustomWebPageReader(SimpleWebPageReader):
    """
    Many websites, including Investopedia, require headers like User-Agent to be set in the request to return the correct content.
    To fix this, we'll modify the load_data method in the SimpleWebPageReader class to include appropriate headers.
    """
    
    def load_data(self, urls):
        """Edit the headers portion in the load_data method to be able to read .asp files"""
        
        if not isinstance(urls, list):
            raise ValueError("urls must be a list of strings.")
        documents = []
        
        ## This is the edit
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
        for url in urls:
            response = requests.get(url, headers=headers).text
            if self.html_to_text:
                import html2text
                response = html2text.html2text(response)

            metadata = None
            if self._metadata_fn is not None:
                metadata = self._metadata_fn(url)

            documents.append(Document(text=response, id_=url, metadata=metadata or {}))

        return documents