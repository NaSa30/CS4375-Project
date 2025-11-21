# src/data_loader.py

import os
import yfinance as yf #using yfinance to download the stock data
import pandas as pd

def download_raw_data(tickers, start, end, folder="data/raw", filename="raw_data.parquet"):
    """
    Download the historical stock data for given tickers and save to Parquet.
    
    the args for this api pull include:
        tickers (list) which is a list of ticker symbols
        start (str) which has the Start 
        end (str) which has the end date
        folder (str) which is the path to the  folder to save file
        filename (str) whic stores the name of Parquet file
    """
    os.makedirs(folder, exist_ok=True)
    # debugging statements
    print(f"Downloading data for {len(tickers)} tickers from {start} to {end}...")
    
    # Download OHLCV data
    df = yf.download(tickers, start=start, end=end)[["Open", "High", "Low", "Close", "Volume"]]
    
    # Save to Parquet
    filepath = os.path.join(folder, filename) #dynamic path named
    df.to_parquet(filepath)
    print(f"Saved raw data to {filepath}")
    print(df.head())
    
    return filepath


if __name__ == "__main__":
    tickers = ['AAPL']  # Grabbing AAPL ticker for our purposes
    download_raw_data(tickers, start="2018-01-01", end="2022-12-31")
