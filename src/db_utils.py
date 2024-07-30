#%%
import requests
import yfinance as yf
import pandas as pd
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    Float,
    DateTime,
    Integer
)
import sqlite3
from typing import Optional, Literal
from datetime import datetime
from dateutil.relativedelta import relativedelta

def get_ticker(company_name: str):
    yfinance = "https://query2.finance.yahoo.com/v1/finance/search"
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
    params = {"q": company_name, "quotes_count": 1, "country": "United States"}

    res = requests.get(url=yfinance, params=params, headers={'User-Agent': user_agent})
    data = res.json()

    company_code = data['quotes'][0]['symbol']
    return company_code
# %%

def get_start_date(years: int = 10):
    today = datetime.today()
    return (today-relativedelta(years=years)).strftime("%Y-%m-%d")

def get_stock_data(
    ticker: str,
    start_date: Optional[str] = None) -> pd.DataFrame:
    """Gets the daily historical prices and volume for a ticker across a specified period"""
    
    if start_date is not None:
        return yf.download(
            ticker,
            start = start_date,
            end = datetime.today().strftime("%Y-%m-%d")
            )
    return yf.download(
        ticker,
        start = get_start_date(),
        end = datetime.today().strftime("%Y-%m-%d")
    ) 

def connect_sql(db_path: str):
    conn = sqlite3.connect(db_path)
    sql_query = """SELECT name FROM sqlite_master WHERE type='table';"""
    cursor = conn.cursor()
    cursor.execute(sql_query)
    tables = cursor.fetchall()
    tables = [table[0] for table in tables]
    engine = create_engine(f"sqlite:///{db_path}")
    return engine, tables

def create_db_table(
    db_path: str, 
    company_name: str,
):
    engine, tables = connect_sql(db_path)
    metadata_obj = MetaData()
    table_name = get_ticker(company_name)
    table = Table(
        table_name,
        metadata_obj,
        Column("id", Integer, primary_key=True),
        Column("date", DateTime),
        Column("open", Float),
        Column("high", Float),
        Column("low", Float),
        Column("close", Float),
        Column("volume", Integer),
    )
    metadata_obj.create_all(engine)
    conn = sqlite3.connect(db_path)
    if table_name in tables: #update table
        start_date = pd.read_sql(
                f"""
                SELECT Date FROM {table_name}
                ORDER by Date DESC
                LIMIT 1
                """,
                conn
            )['Date'].values[0].split(" ")[0]
        data = get_stock_data(ticker = table_name,
                              start_date = start_date).reset_index()
        data.to_sql(
            table_name,
            conn,
            if_exists='append',
            index = False
        )
    else:
        data = get_stock_data(ticker = table_name).reset_index()
        data.to_sql(
            table_name,
            conn,
            index = False,
            if_exists = "replace"
        )