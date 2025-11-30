import pandas as pd

#load the  file with raw data and then save to csv (we also do the same for preprocxessed so that the python code for lstm, rnn can be run easily)

df = pd.read_parquet("data/raw/raw_data.parquet")
df.to_csv("data/raw/raw_data.csv", index=False)
print("the csv file saved at data/raw/raw_data.csv")

df = pd.read_parquet("data/preprocessed/preprocessed_data.parquet")
df.to_csv("data/preprocessed/preprocessed_data.csv", index=False)
print("the csv file saved at data/preprocessed/preprocessed_data.csv")
