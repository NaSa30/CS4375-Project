import pandas as pd

# Load raw  file
df = pd.read_parquet("data/raw/raw_data.parquet")

# Save as CSV to file
df.to_csv("data/raw/raw_data.csv", index=False)

print("CSV file saved at data/raw/raw_data.csv")

# Load Parquet file
df = pd.read_parquet("data/preprocessed/preprocessed_data.parquet")

# Save as CSV to file
df.to_csv("data/preprocessed/preprocessed_data.csv", index=False)

print("CSV file saved at data/preprocessed/preprocessed_data.csv")
