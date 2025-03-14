import pandas as pd
import numpy as np

def add_technical_indicators(df):
    """
    Add technical indicators to the dataframe
    
    Args:
        df (pandas.DataFrame): DataFrame with OHLCV data
        
    Returns:
        pandas.DataFrame: DataFrame with added technical indicators
    """
    try:
        # Make a copy to avoid modifying the original dataframe
        df = df.copy()
        
        # Moving Averages
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        
        # Relative Strength Index (RSI)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['Middle_Band'] = df['Close'].rolling(window=20).mean()
        df['STD'] = df['Close'].rolling(window=20).std()
        df['Upper_Band'] = df['Middle_Band'] + (df['STD'] * 2)
        df['Lower_Band'] = df['Middle_Band'] - (df['STD'] * 2)
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Fill NaN values
        df.fillna(method='bfill', inplace=True)
        
        return df
    
    except Exception as e:
        print(f"Error adding technical indicators: {str(e)}")
        return df
