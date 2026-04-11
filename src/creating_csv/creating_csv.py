import pandas as pd

# loading data
print("wczytywanie danych")
df = pd.read_parquet('data/processed/taxi_prepared.parquet') 

# datetime processing
df['START_DATETIME'] = pd.to_datetime({
    'year': df['YEAR'],
    'month': df['MONTH'],
    'day': df['DAY'],
    'hour': df['HOUR'],
    'minute': df['MINUTE'],
    'second': df['SECOND']
})

# choosing columns for output
wybrane_kolumny = [
    'TRIP_ID',          # Identyfikator trasy
    'START_DATETIME',   # Data i godzina rozpoczęcia trasy
    'ACTUAL_DIST_KM',   # Długość trasy w km
    'TRIP_TIME_MIN',    # Czas trasy w minutach
    'START_LON',        # Długość geo punktu startu
    'START_LAT'         # Szerokość geo punktu startu
]

final_df = df[wybrane_kolumny]

nazwa_pliku_wyjsciowego = 'data/csv_general_files/general.csv'
final_df.to_csv(nazwa_pliku_wyjsciowego, index=False)

print(f"plik zapisano jako: {nazwa_pliku_wyjsciowego}")