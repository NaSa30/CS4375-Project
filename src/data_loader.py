# src/data_loader.py

import os
import yfinance as yf
import pandas as pd

def download_raw_data(tickers, start, end, folder="data/raw", filename="raw_data.parquet"):
    """
    Download historical stock data for given tickers and save to Parquet.
    
    Args:
        tickers (list): List of ticker symbols
        start (str): Start date YYYY-MM-DD
        end (str): End date YYYY-MM-DD
        folder (str): Folder to save file
        filename (str): Name of Parquet file
    """
    os.makedirs(folder, exist_ok=True)
    print(f"Downloading data for {len(tickers)} tickers from {start} to {end}...")
    
    # Download OHLCV data
    df = yf.download(tickers, start=start, end=end)[["Open", "High", "Low", "Close", "Volume"]]
    
    # Save to Parquet
    filepath = os.path.join(folder, filename)
    df.to_parquet(filepath)
    print(f"Saved raw data to {filepath}")
    print(df.head())
    
    return filepath


if __name__ == "__main__":
    tickers = ['AAPL']  # Grabbing AAPL ticker
    download_raw_data(tickers, start="2018-01-01", end="2022-12-31")
