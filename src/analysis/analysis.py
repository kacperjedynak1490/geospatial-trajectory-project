'''
import pandas as pd

data = pd.read_csv('C:\\Users\\Acer\\Desktop\\Projekty\\geospatial-trajectory-project\\train.csv')

data.to_parquet("dane.parquet", engine="pyarrow")
'''