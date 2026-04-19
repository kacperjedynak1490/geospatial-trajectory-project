import pandas as pd

# loading data
input_path = 'data/raw/weather/hourly_weather.parquet'
print(f"Wczytywanie danych pogodowych z: {input_path}")
df = pd.read_parquet(input_path)

# rounding values to nearest integer
to_round = ['precipitation', 'wind_gusts_10m', 'is_day', 'weather_code', 'temperature_2m']
df[to_round] = df[to_round].round()

# choosing columns
choosed_columns = [
    'year', 'month', 'day', 'timezone', 'hour', 
    'precipitation', 'wind_gusts_10m', 'is_day', 'weather_code', 'temperature_2m'
]
df = df[choosed_columns]

# float to int
df[to_round] = df[to_round].astype(int)

# capital letters
df.columns = [kolumna.upper() for kolumna in df.columns]

# parquet file output
output = 'data/processed/weather.parquet'
print(f"file saved as: {output}")

df.to_parquet(output)

print("task completed succesfully")