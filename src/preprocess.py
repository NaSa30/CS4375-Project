import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader


class StockDataset(Dataset):
    """
    the input for this function is thevsequence_length days of features of a particular stock
    the target is the next day's closing price
    """
    def __init__(self, df, sequence_length=30, feature_cols=None, target_col='Close', scaler=None):
        # intiialize the variables 
        self.sequence_length = sequence_length

        if feature_cols:
            self.feature_cols = feature_cols 
        else:
            self.feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

        self.target_col = target_col

        self.features = df[self.feature_cols].values.astype(np.float32)
        self.targets = df[self.target_col].values.astype(np.float32).reshape(-1, 1)

        # Fit the scaler if not provided i the dataset
        if scaler is None:
            self.scaler = MinMaxScaler() # use the min max scaler to scale the features between 0 and 1
            self.features = self.scaler.fit_transform(self.features)
        else:
            self.scaler = scaler
            self.features = self.scaler.transform(self.features)


    def __len__(self):
        return len(self.features) - self.sequence_length

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.sequence_length]
        y = self.targets[idx + self.sequence_length]
        return torch.tensor(x), torch.tensor(y)


def create_dataloaders(parquet_path, sequence_length=30, feature_cols=None, target_col='Close', batch_size=32, train_split=0.8, shuffle=True):
    """
    Load  the parquet file and create train/test PyTorch DataLoaders (initially they are parquet files)
    """
    df = pd.read_parquet(parquet_path)

    # Train/test split
    split_idx = int(len(df) * train_split)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx - sequence_length:]  # overlap for sequences

    # Create the datasets
    train_dataset = StockDataset(train_df, sequence_length, feature_cols, target_col)
    test_dataset = StockDataset(test_df, sequence_length, feature_cols, target_col, scaler=train_dataset.scaler)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_dataset.scaler


def save_preprocessed(df, scaler, filepath="data/preprocessed/preprocessed_data.parquet"):
    """
    save the scaled features and target as a parquet file. (and then later convert to csv if needed)
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Scale the features
    scaled_features = scaler.transform(df[['Open', 'High', 'Low', 'Close', 'Volume']])
    
    # build the DataFrame with scaled features while retaining the original index for time info
    scaled_df = pd.DataFrame(scaled_features, columns=['Open', 'High', 'Low', 'Close', 'Volume'], index=df.index)
    
    # Add the unscaled target column
    scaled_df['Close_Target'] = df['Close'].values
    
    scaled_df.to_parquet(filepath)
    print(f"Saved preprocessed data to {filepath}") # save it to the same directory 


if __name__ == "__main__":
    parquet_path = "data/raw/raw_data.parquet"  # Ensure data_loader.py was run first and then provide the correct path 
    train_loader, test_loader, scaler = create_dataloaders(parquet_path, sequence_length=30, batch_size=16) #optimal values for the code

    # load the entire raw dataframe again to save processed version
    full_df = pd.read_parquet(parquet_path)
    save_preprocessed(full_df, scaler)
    
    # for debugging purposes we print the shape of one batch
    for x, y in train_loader:
        print("X shape:", x.shape)
        print("Y shape:", y.shape)
        break
