import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(csv_path, window_len=30, featureColumns=None, target_col='Close', train_test_split=0.8):
    #the input for this function is thevwindow_len days of features of a particular stock. the target is the next day's closing price
    # add the feature columns
    if featureColumns is None:
        featureColumns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    #read data from csv and drop date...not important to have indexing stuff
    df = pd.read_csv(csv_path)
    if 'Date' in df.columns:
        df = df.drop(columns=['Date'])
    
    #split train and test with 80/20
    split_loc = int(len(df) * train_test_split)
    train_split = df.iloc[:split_loc]
    test_split = df.iloc[split_loc:]

    #scale the input features
    scaler_input = MinMaxScaler() #scale the features for performance
    train_f = train_split[featureColumns].values.astype(np.float32)
    train_f_scaled = scaler_input.fit_transform(train_f)
    test_f = test_split[featureColumns].values.astype(np.float32)
    test_f_scaled = scaler_input.transform(test_f)
    #scale the output target
    scaler_output = MinMaxScaler()
    train_t = train_split[[target_col]].values.astype(np.float32)
    train_t_scaled = scaler_output.fit_transform(train_t)
    test_t = test_split[[target_col]].values.astype(np.float32)
    test_t_scaled = scaler_output.transform(test_t)
    
    train_t = train_split[target_col].values.astype(np.float32)
    test_t = test_split[target_col].values.astype(np.float32)

    # make windowed sequences for the model
    X_train, y_train = sequence_maker(train_f_scaled, train_t_scaled, window_len)
    X_test, y_test = sequence_maker(test_f_scaled, test_t_scaled, window_len)
    
    return X_train, y_train, X_test, y_test, scaler_input, scaler_output


def sequence_maker(features, targets, window_len):
   #make the sequences based on each window
    X = []
    y = []
    n = len(features) - window_len
    for i in range(n):
        X.append(features[i:i + window_len])
        y.append(targets[i + window_len])
    
    return np.array(X), np.array(y).reshape(-1, 1)


def savePreppedCSV(df, scaler, filepath="data/preprocessed/preprocessed_data.csv"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df = df.drop(columns=['Date'], errors='ignore')
    
    # Scale the features
    scaled = scaler.transform(df[['Open', 'High', 'Low', 'Close', 'Volume']])
    scaled_df_new = pd.DataFrame(scaled, columns=['Open', 'High', 'Low', 'Close', 'Volume'],  index=df.index )


    scaled_df_new['Close_Target'] = df['Close'].values #predicitng column is close
    scaled_df_new.to_csv(filepath, index=False)
    print(f"Saved preprocessed data to {filepath}") #debugging


if __name__ == "__main__":
    csv_path = "data/raw/raw_data.csv"
    
    # Preprocess the data
    X_train, y_train, X_test, y_test, scaler = preprocess_data(csv_path, window_len=30, train_test_split=0.8 )
    
    entire_df = pd.read_csv(csv_path)
    savePreppedCSV(entire_df, scaler)