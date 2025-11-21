import pandas as pd

# Load raw  file and then save to csv  (we also do the same for preprocxessed so the python code for lstm, rnn can run easily)
df = pd.read_parquet("data/raw/raw_data.parquet")
df.to_csv("data/raw/raw_data.csv", index=False)
print("CSV file saved at data/raw/raw_data.csv")

df = pd.read_parquet("data/preprocessed/preprocessed_data.parquet")
df.to_csv("data/preprocessed/preprocessed_data.csv", index=False)
print("CSV file saved at data/preprocessed/preprocessed_data.csv")
