"""
Script for preparing taxi data (Feature Engineering).
Based on the 'Taxi Trajectory' dataset from Kaggle (Porto, Portugal).

Tasks performed by this script:
-Cleaning: Removes rows where data is missing (MISSING_DATA = True).
-Standardization: Changes CALL_TYPE from letters (A, B, C) to numbers (1, 2, 3).
-Time Extraction: Extracts YEAR, MONTH, DAY, HOUR, MINUTE, SECOND, and WEEKDAY from the TIMESTAMP.
-Categorization: Adds part of the day (PARTDAY: 1-Morning, 2-Midday, 3-Afternoon, 4-Evening, 5-Night).
-Geometry: Parses the POLYLINE string to get start and end coordinates (START_LON/LAT, END_LON/LAT).
-Distance Calculation: Calculates actual driven distance (ACTUAL_DIST_KM) and straight-line distance (OPTIMAL_DIST_KM) using the Haversine formula.
-Time Calculation: Calculates total trip time in minutes (TRIP_TIME_MIN).
-Saving: Saves the final dataset as .parquet and .csv files for Machine Learning
"""


import pandas as pd
import json
import math
import numpy as np

#data

# --- CONFIGURATION -------------------
DEBUG_MODE = True  # False for full dataset, True for sample dataset  

if DEBUG_MODE:
    file_path = 'data/data_samples/data_100k_raw.parquet'
    output_path = 'data/data_samples/taxi_100k_prepared.parquet'
else:
    file_path = 'data/raw/dane.parquet'
    output_path = 'data/processed/taxi_prepared.parquet'
#--------------------------------------

df = pd.read_parquet(file_path)

#delete MISSING_DATA, MISSING_DATA column contains boolean values (False/True)
df = df[df['MISSING_DATA'] == False].copy() 
# change from A, B, C to 1, 2, 3 for CALL_TYPE
df['CALL_TYPE'] = df['CALL_TYPE'].map({'A': 1, 'B': 2, 'C': 3})


#TIMESTAMP
df['DATETIME'] = pd.to_datetime(df['TIMESTAMP'], unit='s')

#create columns: DAY, MONTH, YEAR, HOUR
df['YEAR'] = df['DATETIME'].dt.year
df['MONTH'] = df['DATETIME'].dt.month
df['DAY'] = df['DATETIME'].dt.day
df['HOUR'] = df['DATETIME'].dt.hour
df['MINUTE'] = df['DATETIME'].dt.minute
df['SECOND'] = df['DATETIME'].dt.second


# WEEKDAY (1-7, where monday is 1,... sunday is 7)
df['WEEKDAY'] = df['DATETIME'].dt.weekday + 1

#classification  PARTDAY (1 - morning, 2 - midday, 3 - afternoon, 4 - evening, 5 - night)
def get_partday(hour):
    if 6 <= hour < 11:      return 1        # morning
    elif 11 <= hour < 13:   return 2        # midday
    elif 13 <= hour < 17:   return 3        # afternoon
    elif 17 <= hour < 21:   return 4        # evening
    else: return 5                          # night

df['PARTDAY'] = df['HOUR'].apply(get_partday)

# POLYLINE and travel time
#POLYLINE is string, so it was changed to list
def parse_polyline(polyline_str):
    try:
        return json.loads(polyline_str)
    except:
        return []

df['POLYLINE_LIST'] = df['POLYLINE'].apply(parse_polyline)

# extract START and END coordinates
df['START_LON'] = df['POLYLINE_LIST'].apply(lambda x: x[0][0] if len(x) > 0 else None)
df['START_LAT'] = df['POLYLINE_LIST'].apply(lambda x: x[0][1] if len(x) > 0 else None)
df['END_LON'] = df['POLYLINE_LIST'].apply(lambda x: x[-1][0] if len(x) > 0 else None)
df['END_LAT'] = df['POLYLINE_LIST'].apply(lambda x: x[-1][1] if len(x) > 0 else None)

# trip time: (number_of_points - 1) * 15 seconds
df['TRIP_TIME_MIN'] = df['POLYLINE_LIST'].apply(lambda x: max(0, (len(x) - 1) * 15)/60) # in minutes

# distances (haversine  in km)
def calculate_actual_distance(polyline):
    if not polyline or len(polyline) < 2:
        return 0.0
    total_dist = 0.0
    R = 6371.0 
    for i in range(len(polyline) - 1):
        lon1, lat1 = polyline[i]
        lon2, lat2 = polyline[i+1]
        
        # Inline math is faster inside a loop than calling a separate function
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        
        a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        total_dist += R * c
    return total_dist

# vectorized haversine for OPTIMAL_DIST_KM (for large datasets)
def vectorized_haversine(lon1, lat1, lon2, lat2):
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

df['ACTUAL_DIST_KM'] = df['POLYLINE_LIST'].apply(calculate_actual_distance)
df['OPTIMAL_DIST_KM'] = vectorized_haversine(df['START_LON'], df['START_LAT'], df['END_LON'], df['END_LAT'])

# 6. Deviation Ratio
# np.where handles the division by zero natively and cleanly
df['DEVIATION_RATIO'] = np.where(
    (df['OPTIMAL_DIST_KM'].notnull()) & (df['OPTIMAL_DIST_KM'] > 0),
    df['ACTUAL_DIST_KM'] / df['OPTIMAL_DIST_KM'],
    1.0
)

# check   
print(df[['TRIP_ID', 'YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE', 'SECOND', 'PARTDAY', 'TRIP_TIME_MIN']].head())
print(df[['TRIP_ID', 'ACTUAL_DIST_KM', 'OPTIMAL_DIST_KM', 'DEVIATION_RATIO']].head())

#saving prepared data
columns_to_drop = ['POLYLINE', 'POLYLINE_LIST', 'DATETIME']
df_final = df.drop(columns=columns_to_drop, errors='ignore')
df_final.to_parquet(output_path)

#csv
#output_path_csv = 'data/data_samples/taxi_10k_prepared.csv'
#df_final.to_csv(output_path_csv, index=False)
