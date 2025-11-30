# src/data_loader.py
import os
import yfinance as yf #using yfinance to download the stock data
import pandas as pd

def download_raw_data(ticker, start, end, folder="data/raw", filename="raw_data.csv"): 
    #download the raw stock data from yfinance and save it as a Parquet file.
    os.makedirs(folder, exist_ok=True)
    # debugging statements
    print(f"downloading data for {ticker} ticker from date:{start} to date:{end}")

    # download ticker data
    df = yf.download(ticker, start, end)[["Open", "High", "Low", "Close", "Volume"]]
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    # Save to csv
    filepath = os.path.join(folder, filename) #dynamic path named
    df.to_csv(filepath)
    print(f"{filepath}:location of raw data saved")
    print(df.head())
    
    return filepath

if __name__ == "__main__":
    ticker = 'AAPL'  # Grabbing AAPL ticker for our purposes
    download_raw_data(ticker, start="2018-01-01", end="2022-12-31")
