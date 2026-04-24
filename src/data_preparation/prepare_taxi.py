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
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString
import shapely.wkt
from src.visualisation.data_visualisation import creating_geometry
import osmnx as ox

#data

# --- CONFIGURATION -------------------
DEBUG_MODE = 2  # False for full dataset, True for sample dataset  
if (DEBUG_MODE == 1):
    file_path = 'data/data_samples/data_100k_raw.parquet'
    output_path = 'data/data_samples/taxi_100k_prepared.parquet'
elif (DEBUG_MODE == 2):
    file_path = 'data/data_samples/data_10k_raw.parquet'
    output_path = 'data/data_samples/taxi_10k_prepared.parquet'
else:
    file_path = 'data/raw/dane.parquet'
    output_path = 'data/processed/taxi_prepared.parquet'
#--------------------------------------

print("Loading data...")
df = pd.read_parquet(file_path)
geo_taxi = gpd.GeoDataFrame(df, geometry=df['POLYLINE'].apply(creating_geometry), crs='EPSG:4326')
geo_taxi['original_geometry'] = geo_taxi.geometry.copy()
porto_gdf = ox.geocode_to_gdf("Porto, Portugal")
porto_areas = pd.read_parquet('data/processed/area.parquet')
porto_areas_gdf = gpd.GeoDataFrame(porto_areas, geometry=porto_areas['POLYGON'].apply(lambda wkt: shapely.wkt.loads(wkt)), crs='EPSG:4326')
porto_areas_gdf = porto_areas_gdf.drop(columns=['POLYGON'])
#delete MISSING_DATA, MISSING_DATA column contains boolean values (False/True)
geo_taxi = geo_taxi[geo_taxi['MISSING_DATA'] == False].copy() 
geo_taxi = geo_taxi.drop(columns=['MISSING_DATA'])
print("Joining...")
geo_taxi = gpd.sjoin(geo_taxi, porto_gdf, predicate='within')
df = gpd.overlay(geo_taxi, porto_areas_gdf, how='intersection')
def multilinestring_to_linestring(geom):
    if geom is None or geom.is_empty:
        return geom
    elif geom.geom_type == 'MultiLineString':
        return geom.geoms[0]
    else:
        return geom
df['geometry'] = df['geometry'].apply(multilinestring_to_linestring)
# remove trips with very short geometry (probably errors)
def count_coords(geom):
    if geom is None or geom.is_empty:
        return 0
    elif geom.geom_type == 'LineString':
        return len(geom.coords)
    else:
        return 0
df = df[df['geometry'].apply(count_coords) > 2].copy()
df = df.reset_index(drop=True)
print(df.geometry.head(100))
print("Column manipulation...")
# change from A, B, C to 1, 2, 3 for CALL_TYPE
df['CALL_TYPE'] = df['CALL_TYPE'].map({'A': 1, 'B': 2, 'C': 3})

# trip time: (number_of_points - 1) * 15 seconds
def trip_time(geom):
    if geom is None or geom.is_empty:
        return 0
        
    count = 0
    if geom.geom_type == 'LineString':
        count = len(geom.coords)

    return max(0, (count - 1) * 15) / 60

df['TIME_IN_AREA'] = df['geometry'].apply(trip_time)

def start_in_area_time(row):
    orig_geom = row['original_geometry']
    inter_geom = row['geometry']
    
    if orig_geom is None or inter_geom is None or inter_geom.is_empty: #chceck for empty geometry
        return 0

    elif inter_geom.geom_type == 'LineString':
        start_coord = inter_geom.coords[0] # dont know if this is correct, probably not
    else:
        return 0
        
    start_pt = Point(start_coord)
    orig_coords = list(orig_geom.coords)
    
    for i in range(len(orig_coords) - 1): # searching for the point in the list
        segment = LineString([orig_coords[i], orig_coords[i+1]])
        if segment.distance(start_pt) < 1e-6:
            seconds = i * 15
            if segment.length == 0:
                segment_fraction = 0
            else:
                segment_fraction = segment.project(start_pt) / segment.length
            sekundy_dodatkowe = segment_fraction * 15
            return seconds + sekundy_dodatkowe
    return 0

df['SECONDS_TO_ADD'] = df.apply(start_in_area_time, axis=1)
# adding second number to original start time
df['TIMESTAMP'] = df['TIMESTAMP'] + df['SECONDS_TO_ADD']
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

# calculating starting and end points for optimal distance
def get_coords(geom):
    if geom is None or geom.is_empty:
        return None, None, None, None
    
    if geom.geom_type == 'LineString':
        start = geom.coords[0]
        end = geom.coords[-1]

    return start[0], start[1], end[0], end[1]

df['START_LON'], df['START_LAT'], df['END_LON'], df['END_LAT'] = zip(*df['geometry'].apply(get_coords))
df.to_crs(epsg=3857, inplace=True)  # project to metric CRS for accurate distance calculation
df['DIST_KM_IN_AREA'] = df.geometry.length / 10**3
start_points = gpd.GeoSeries.from_xy(df['START_LON'], df['START_LAT'], crs="EPSG:4326")
end_points = gpd.GeoSeries.from_xy(df['END_LON'], df['END_LAT'], crs="EPSG:4326")
df['OPTIMAL_DIST_KM'] = start_points.to_crs(epsg=3857).distance(end_points.to_crs(epsg=3857)) / 10**3
df.to_crs(epsg=4326, inplace=True)

# Deviation Ratio
df['DEVIATION_RATIO'] = np.where(
    (df['OPTIMAL_DIST_KM'].notnull()) & (df['OPTIMAL_DIST_KM'] > 0),
    df['DIST_KM_IN_AREA'] / df['OPTIMAL_DIST_KM'],
    1.0
)

df['SPEED'] = np.where(df['TIME_IN_AREA'] > 0, df['DIST_KM_IN_AREA'] / (df['TIME_IN_AREA'] / 60), np.nan)
# check   
print("Finished")
print("Data snippet")
print(df[['TRIP_ID', 'YEAR', 'MONTH', 'DAY', 'HOUR', 
          'MINUTE', 'SECOND', 'PARTDAY', 'TIME_IN_AREA']].head())
print(df[['TRIP_ID', 'DIST_KM_IN_AREA', 'DEVIATION_RATIO',
           'OPTIMAL_DIST_KM', 'geometry']].head())
print(df[['TRIP_ID', 'START_LON', 'START_LAT', 'END_LON',
           'END_LAT','SPEED','SECONDS_TO_ADD']].head())

df_norm = df[['TRIP_ID', 'YEAR', 'MONTH', 'DAY', 
                    'HOUR', 'MINUTE', 'SECOND', 'WEEKDAY',
                    'PARTDAY', 'geometry', 'TIME_IN_AREA',
                    'DIST_KM_IN_AREA', 'DEVIATION_RATIO',
                    'AREA_ID', 'SPEED']]

print("Final data snippet")
print(df_norm.head(20))
df_norm.to_parquet(output_path)

#csv
#output_path_csv = 'data/data_samples/taxi_10k_prepared.csv'
#df_norm.to_csv(output_path_csv, index=False)
