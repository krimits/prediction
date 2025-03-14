import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def load_crypto_data(symbol, timeframe):
    """
    Load cryptocurrency data from Yahoo Finance
    
    Args:
        symbol (str): Cryptocurrency symbol (e.g., 'BTC-USD')
        timeframe (str): Time period to fetch ('1mo', '3mo', '6mo', '1y')
        
    Returns:
        pandas.DataFrame: DataFrame containing the cryptocurrency data
    """
    try:
        end_date = datetime.now()
        
        # Calculate start date based on timeframe
        if timeframe == '1mo':
            start_date = end_date - timedelta(days=30)
            interval = '1h'
        elif timeframe == '3mo':
            start_date = end_date - timedelta(days=90)
            interval = '1h'
        elif timeframe == '6mo':
            start_date = end_date - timedelta(days=180)
            interval = '1d'
        elif timeframe == '1y':
            start_date = end_date - timedelta(days=365)
            interval = '1d'
        else:
            start_date = end_date - timedelta(days=90)  # Default to 3 months
            interval = '1h'
        
        # Download data from Yahoo Finance
        df = yf.download(
            symbol,
            start=start_date,
            end=end_date,
            interval=interval,
            progress=False
        )
        
        # Check if data is empty
        if df.empty:
            raise ValueError(f"No data found for {symbol} in the specified timeframe")
        
        # Reset index to make date a column
        df = df.reset_index()
        
        # Convert timezone-aware datetime to timezone-naive
        df['Date'] = df['Date'].dt.tz_localize(None)
        
        # Set Date as index
        df.set_index('Date', inplace=True)
        
        # Forward fill any missing values
        df.fillna(method='ffill', inplace=True)
        
        return df
    
    except Exception as e:
        raise Exception(f"Error loading data for {symbol}: {str(e)}")
