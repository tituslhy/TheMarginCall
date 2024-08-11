#%%
import warnings
warnings.filterwarnings('ignore')

from llama_index.core.tools.tool_spec.base import BaseToolSpec
import yfinance as yf
import pandas as pd
from typing import Optional, Literal, List, Union, Tuple, Dict

from forecaster import Forecaster

import os
import sys
__curdir__ = os.getcwd()

if ("tools" in __curdir__) or \
    ("agents" in __curdir__) or \
    ("tasks" in __curdir__):
    sys.path.append(os.path.join(
        __curdir__,
        "../src"
    ))
else:
    sys.path.append("./src")
from utils import rename_columns, process_string

#%%
class DataAnalysisTools(BaseToolSpec):
    """These tools are intended for general data analysis by agents. They allow for very general
    questions and specific questions by users."""
    
    spec_functions = [
        "get_stock_data",
        "get_min_or_max",
        "get_correlation_between_tickers",
        "get_rolling_average",
        "get_rolling_average_correl",
        "get_longest_uptrend",
        "get_longest_downtrend",
        "cagr",
        "get_statistics",
        "forecast"
    ]
    
    def __init__(self) -> None:
        """Initializes the Yahoo finance tool spec"""
        
    def get_stock_data(
        self, 
        ticker: str,
        period: Optional[
            Literal["1d",
                    "5d",
                    "1mo",
                    "3mo",
                    "6mo",
                    "1y",
                    "2y",
                    "5y",
                    "10y",
                    "ytd",
                    "max"]
            ] = "10y") -> pd.DataFrame:
        """Gets the daily historical prices and volume for a ticker across a specified 
        period"""
        return rename_columns(ticker = ticker, df = yf.Ticker(ticker).history(
            period=period
        ))
    
    def get_min_or_max(
        self, 
        ticker: str,
        field: Literal["Open", 
                        "High", 
                        "Low",
                        "Close",
                        "Volume",
                        "Dividends",
                        "Stock Splits"],
        period: Optional[
            Literal["1d",
                    "5d",
                    "1mo",
                    "3mo",
                    "6mo",
                    "1y",
                    "2y",
                    "5y",
                    "10y",
                    "ytd",
                    "max"]
            ] = "10y",
        get_max: Optional[bool] = True):
        """Gets min or max value of the field of interest"""
        df = self.get_stock_data(ticker=ticker, period=period)
        field = process_string(ticker=ticker, string_=field)
        if get_max is True:
            value = df[field].max()
        else:
            value = df[field].min()
        return df[df[field] == value]
    
    def get_correlation(
        self, 
        ticker: str,
        period: Optional[
            Literal["1d",
                    "5d",
                    "1mo",
                    "3mo",
                    "6mo",
                    "1y",
                    "2y",
                    "5y",
                    "10y",
                    "ytd",
                    "max"]
            ] = "10y"):
        """Computes correlation between all metrics for a specific ticker"""
        return self.get_stock_data(ticker=ticker, period=period).corr()
    
    def get_correlation_between_tickers(
        self,
        tickers: List[str],
        period: Optional[
                        Literal["1d",
                                "5d",
                                "1mo",
                                "3mo",
                                "6mo",
                                "1y",
                                "2y",
                                "5y",
                                "10y",
                                "ytd",
                                "max"]
                        ] = "10y"
    ):
        """Computes correlation of all metrics for all tickers"""
        df_fin = None
        for ticker in tickers:
            df = self.get_stock_data(ticker=ticker, period=period)
            if df_fin is None:
                df_fin = df
            else:
                df_fin = pd.merge(
                    df_fin, 
                    df, 
                    left_index = True,
                    right_index = True)
        return df_fin.corr()
    
    def get_rolling_average(
        self, 
        ticker: str,
        field: Literal["Open", 
                    "High", 
                    "Low",
                    "Close",
                    "Volume",
                    "Dividends",
                    "Stock Splits"], 
        n: Optional[int] = 30,
        period: Optional[
                    Literal["1d",
                            "5d",
                            "1mo",
                            "3mo",
                            "6mo",
                            "1y",
                            "2y",
                            "5y",
                            "10y",
                            "ytd",
                            "max"]
                    ] = "10y"):
        """Computes moving average for field of interest across a specified period"""
        df = self.get_stock_data(ticker=ticker, period=period)
        field = process_string(ticker=ticker, string_=field)
        return pd.DataFrame(df[field].rolling(n).mean())
    
    def get_rolling_average_correl(
        self, 
        tickers: List[str], 
        field: Literal["Open", 
                        "High", 
                        "Low",
                        "Close",
                        "Volume",
                        "Dividends",
                        "Stock Splits"], 
        n: Optional[int] = 30,
        period: Optional[
                    Literal["1d",
                            "5d",
                            "1mo",
                            "3mo",
                            "6mo",
                            "1y",
                            "2y",
                            "5y",
                            "10y",
                            "ytd",
                            "max"]
                    ] = "10y"
    ):
        """Gets correlation of rolling average for a specified field across a list of 
        tickers"""
        
        df_fin = None
        for ticker in tickers:
            df = self.get_rolling_average(
                ticker = ticker,
                field = field,
                n = n,
                period = period
            )
            if df_fin is None:
                df_fin = df
            else:
                df_fin = pd.merge(
                    df_fin, 
                    df,
                    left_index = True,
                    right_index = True)
        return df_fin.corr()
    
    def get_longest_uptrend(
        self,
        ticker: str
    ):
        """Computes longest stock price uptrend duration for ticker"""
        df = self.get_stock_data(ticker=ticker)
        df['uptrend_days'] = df[f'{ticker}_close'].diff().lt(0).cumsum()
        sizes = df.groupby('uptrend_days')[f'{ticker}_close'].transform('size')
        dates = df[sizes == sizes.max()].index 
        return dates, f"{(dates[-1] - dates[0]).days} days"
    
    def get_longest_downtrend(
        self,
        ticker: str
    ):
        """Computes longest stock price downtrend duration for ticker"""
        df = self.get_stock_data(ticker=ticker)
        df['downtrend_days'] = df[f'{ticker}_close'].diff().gt(0).cumsum()
        sizes=df.groupby('downtrend_days')[f'{ticker}_close'].transform('size')
        dates = df[sizes == sizes.max()].index 
        return dates, f"{(dates[-1] - dates[0]).days} days"
    
    def cagr(
        self,
        ticker: str,
        period: Optional[
                    Literal["1d",
                            "5d",
                            "1mo",
                            "3mo",
                            "6mo",
                            "1y",
                            "2y",
                            "5y",
                            "10y",
                            "ytd",
                            "max"]
                    ] = "10y"): 
        """Computes compounded annual growth rate of closing prices for specified 
        ticker"""
        df = self.get_stock_data(ticker=ticker, period=period)
        df['year'] = df.index.year
        trimmed = df.iloc[[0, -1]][['year',f'{ticker}_close']]
        n = trimmed['year'].iloc[-1] - trimmed['year'].iloc[0]
        start, end = trimmed[f'{ticker}_close'].iloc[-1], trimmed[f'{ticker}_close'].iloc[0]
        return f"{round(100*((end/start)**(1/n)-1), 1)}%"
    
    def get_statistics(
        self,
        ticker: str,
        period: Optional[
                    Literal["1d",
                            "5d",
                            "1mo",
                            "3mo",
                            "6mo",
                            "1y",
                            "2y",
                            "5y",
                            "10y",
                            "ytd",
                            "max"]
                    ] = "10y",
        grouping: Optional[
            Literal["year",
                    "quarter",
                    "month",
                    "week",
                    None]] = None
    ) -> Union[Tuple[float, float], pd.DataFrame]:
        """Gets descriptive statistics of data for grouping of interest"""
        df = self.get_stock_data(ticker=ticker, period=period)
        if grouping is None:
            return df.describe()
        df['year'] = df.index.year
        if grouping == 'year':
            return df.groupby(grouping).mean(), df.groupby(grouping).std()
        if grouping == 'quarter':
            df['quarter'] = df.index.quarter
            return df.groupby(["year", "quarter"]).mean(), df.groupby(["year", "quarter"]).std() 
        if grouping == 'month':
            df['month'] = df.index.month
            return df.groupby(["year", "month"]).mean(), df.groupby(["year", "month"]).std()
    
    def forecast(
        self,
        tickers: List[str],
    ) -> Dict[str, Dict[str, List[float]]]:
        """Develops a quick forecast of the stock prices for the next 3 months
        for a list of tickers at 95% confidence. Use this tool exclusively for
        forecasting future stock prices. 
        
        The models explored here are GARCH, ARCH and Naive forecasting. These models
        first forecast volatility and converts them into forecasted stock prices. The
        tool first uses an "autoARIMA-esque" approach to determine the best model (i.e.
        GARCH/ARCH and their specific hyperparameter configurations), then uses the best
        model from backtesting to forecast forward volatility and subsequently stock 
        prices.
        
        Volatility in this case is defined as logarthmic returns - simply just applying
        the natural logarithm on the ratio of the price of the current period and the
        previous period, where each period refers to the adjusted closing price at the
        monthly interval.s

        Args:
            tickers (List[str]): Tickers of interest
        Returns:
            Dict[str, List[float]]: A dictionary containing the forecasts
            at 95% confidence interval. Example output:
            {
                "Ticker": {
                    "GARCH(1,2)": [..., ..., ...],
                    "GARCH(1,2)-lo-95": [..., ..., ...],
                    "GARCH(1,2)-hi-95": [..., ..., ...],
                }
            }
            
            Whereby "lo-95" refers to the lower boundary  with 95% confidence, 
            and, "hi-95 refers to the high boundary with 95% confidence. The 
            values in the list are forecasted stock prices for the next 3 months.
        """
        forecaster = Forecaster()
        return forecaster(tickers = tickers)

def get_da_tools():
    da = DataAnalysisTools()
    return da.to_tool_list()  
