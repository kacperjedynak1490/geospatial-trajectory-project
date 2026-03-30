'''
I made sampled data:
 - 10k - for analysis and checking the validity of code
 - 100k - for ML, analysis to see more patterns at the end
 
You can find it in data_samples directory. I did it only for Parquet type,
but if it's needed I will add a CSV files too. Just uncomment last part of this code.
'''

import pandas as pd

#sample for 10k/100k rows from raw data
samples = {
    "10k": 10000,
    "100k": 100000
}

data = pd.read_parquet("./dane.parquet")
#print(data.head())

for key, item in samples.items():
    data_sampled = data.sample(item, random_state=42)
    data_sampled.to_parquet(f"./data_samples/data_{key}_raw.parquet", index=False)
    #data_sampled.to_csv(f"./data_samples/data_{key}_raw.csv", index=False)
