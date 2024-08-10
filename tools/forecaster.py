from typing import List, Optional, Dict
from datetime import datetime
import yfinance as yf
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from statsforecast.models import (
    GARCH, 
    ARCH, 
    Naive,
    _TS
)
from statsforecast import StatsForecast
from datasetsforecast.losses import mae
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

class Forecaster:
    """Runs a battery of statistical forecasting methods to forecast
    volatility of a stock"""
    
    def __init__(self):
        """Class constructor
        """
        self.models = [
            ARCH(1), 
            ARCH(2), 
            GARCH(1,1),
            GARCH(1,2),
            GARCH(2,2),
            GARCH(2,1),
            Naive()
        ]
        
    def post__init__(
        self, 
        tickers: List[str], 
        h: int = 3
    ):
        """Pre-processes results for statistical forecasting by calculating the
        logarthmic returns of stock prices and setting it as the target variable,
        and creating a cross validiation dataframe object.
        """
        self.tickers = [ticker.upper() for ticker in tickers]
        try:
            self.df = yf.download(
                tickers, 
                start = (
                    datetime.now() - relativedelta(years=10)
                ).strftime("%Y-%m-%d"), 
                end = datetime.today().strftime("%Y-%m-%d"), 
                interval='1mo') # use monthly prices
        except Exception as e:
            raise ValueError(e)
        if len(self.tickers)>1:
            ## Prepare dataframe
            self.df = self.df.loc[:, (['Adj Close'], self.tickers)]
            self.df.columns = self.df.columns.droplevel() # drop MultiIndex
            self.df = self.df.reset_index()
            
            ## Prepare target dataframe
            self.prices = self.df.melt(id_vars = 'Date')
            self.prices = self.prices.rename(
                columns={'Date': 'ds', 'Ticker': 'unique_id', 'value': 'y'}
            )
        
        else:
            self.df['Ticker'] = self.tickers[0]
            self.df = self.df.reset_index()
            self.prices = self.df[["Date", "Ticker", "Adj Close"]]
            self.prices = self.prices.rename(
                columns={'Date': 'ds', 'Ticker': 'unique_id', 'Adj Close': 'y'}
            )
            
        self.prices = self.prices[['unique_id', 'ds', 'y']]
        self.prices['rt'] = self.prices['y'].div(self.prices.groupby('unique_id')['y'].shift(1))
        self.prices['rt'] = np.log(self.prices['rt'])
        self.returns = self.prices[['unique_id', 'ds', 'rt']]
        self.returns = self.returns.rename(columns={'rt':'y'})
    
        self.sf = StatsForecast(
            df = self.returns, 
            models = self.models, 
            freq = 'MS',
            n_jobs = -1)
        
        ## Prepare cross-validation dataframe
        self.crossvalidation_df = self.sf.cross_validation(
            df = self.returns,
            h = h,
            step_size = 3,
            n_windows = 4
        )
        self.crossvalidation_df = self.crossvalidation_df.reset_index()
        self.crossvalidation_df.rename(columns = {'y' : 'actual'}, inplace = True)
    
    def evaluate(self):
        """Evaluates generated forecasts"""
        mae_cv = self.crossvalidation_df.groupby(['unique_id', 'cutoff']).apply(
            self.compute_cv_mae,
            models = (self.models)
        )
        self.mae = mae_cv.groupby('unique_id').mean()
        self.best_models = self.mae.idxmax(axis=1).reset_index().rename(columns={0: "best_model"})
    
    def forecast(
        self, h: int = 3, levels: Optional[List[int]] = [95]
    ):
        """Generates a forecast for the next h periods

        Args:
            h (int, optional): Forecasting horizon (number of periods to forecast). Defaults to 3.
            levels (Optional[List[int]], optional): Confidence level for prediction intervals. Defaults to [80,95].
        """
        self.forecasts = self.sf.forecast(h=h, level=levels)
        self.forecasts = forecasts.reset_index()
    
    def forecast_ticker(self, ticker: Optional[str] = None) -> Dict[str, List[float]]:
        """Returns a forecast for a ticker of interest

        Args:
            ticker (str): Stock ticker of interest. If this is none, we'll just compute the
            forecasts for all the tickers
        """
        if ticker:
            ticker = ticker.upper()
            filtered = self.forecasts[self.forecasts['unique_id'] == ticker]
            best_model = self.best_models[self.best_models['unique_id']==ticker]['best_model'].values[0]
            df = filtered[[col for col in filtered.columns if str(best_model) in col]]
            
            ## Get last price
            last_price = self.prices[self.prices['unique_id']==ticker].tail(1)['y'].values[0]
            
            return self.compute_forecast(
                last_price = last_price,
                df = df,
                ticker = ticker
            )
        else:
            returns = dict()
            for ticker in self.tickers:
                filtered = self.forecasts[self.forecasts['unique_id'] == ticker]
                best_model = self.best_models[self.best_models['unique_id']==ticker]['best_model'].values[0]
                df = filtered[[col for col in filtered.columns if str(best_model) in col]]
                
                ## Get last price
                last_price = self.prices[self.prices['unique_id']==ticker].tail(1)['y'].values[0]
                
                returns[ticker] = self.compute_forecast(
                    last_price = last_price,
                    df = df,
                    ticker = ticker
                )
            
            return returns
    
    def show_max(self):
        """Helper function to visually show the best forecasting model"""
        self.mae.style.highlight_max(color = 'lightblue', axis = 1)
    
    @staticmethod
    def compute_forecast(
        last_price: float, df: pd.DataFrame, ticker: str) -> float:
        """Static method to compute forecasted price from forecasted log return values
        """
        returns = defaultdict(list)
        last_value=last_price
        for col in df.columns:
            for vol in [v for v in df[col].values]:
                f = np.exp(vol) * last_value
                returns[col].append(
                    f
                )
                last_value=f

        return returns
    
    @staticmethod
    def compute_cv_mae(
        crossvalidation_df: pd.DataFrame,
        models: List[_TS]
    ):
        """Compute MAE for all models generated"""
        res = {}
        for mod in models: 
            res[mod] = mae(crossvalidation_df['actual'], crossvalidation_df[str(mod)])
        return pd.Series(res)
    
    def plot_post_init(self):
        """Plots forecast against actual test results.
        For debugging purposes"""
        try:
            StatsForecast.plot(
                self.returns, 
                self.crossvalidation_df.drop(['cutoff', 'actual'],
                    axis=1))
        except:
            self.post__init__()
            StatsForecast.plot(
                self.returns, 
                self.crossvalidation_df.drop(['cutoff', 'actual'],
                    axis=1))
    
    def __call__(
        self,
        tickers: List[str],
        ticker: Optional[str] = None,  
        h: Optional[int] = 3,
        levels: Optional[List[int]] = [95]
    ) -> Dict[str, List[float]]:
        """Super function to tie all the methods together.

        Args:
            ticker (Optional[str], optional): _description_. Defaults to None.
            h (Optional[int], optional): _description_. Defaults to 3.
            levels (Optional[List[int]], optional): _description_. Defaults to [95].

        Returns:
            Dictionary of forecasts
        """
        self.post__init__(tickers = tickers, h=h)
        self.evaluate()
        self.forecast(h=h, levels=levels)
        return self.forecast_ticker(ticker=ticker)