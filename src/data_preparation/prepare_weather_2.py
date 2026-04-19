import pandas as pd

# loading data
sciezka_wejsciowa = 'data/raw/weather/hourly_weather.parquet'
print(f"Wczytywanie danych pogodowych z: {sciezka_wejsciowa}")
df = pd.read_parquet(sciezka_wejsciowa)

# rounding values to nearest integer
kolumny_do_zaokraglenia = ['precipitation', 'wind_gusts_10m', 'is_day', 'weather_code', 'temperature_2m']
df[kolumny_do_zaokraglenia] = df[kolumny_do_zaokraglenia].round()

# choosing columns
wybrane_kolumny = [
    'year', 'month', 'day', 'timezone', 'hour', 
    'precipitation', 'wind_gusts_10m', 'is_day', 'weather_code', 'temperature_2m'
]
df = df[wybrane_kolumny]

# float to int
df[kolumny_do_zaokraglenia] = df[kolumny_do_zaokraglenia].astype(int)

# capital letters
df.columns = [kolumna.upper() for kolumna in df.columns]

# csv file output
sciezka_wyjsciowa = 'data/raw/weather/weather.csv'
print(f"saving new CSV file to: {sciezka_wyjsciowa}")
df.to_csv(sciezka_wyjsciowa, index=False)

print("task completed successfully")