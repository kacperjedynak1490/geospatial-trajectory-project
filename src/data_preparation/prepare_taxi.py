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
import geopandas as gpd
from shapely.geometry import Point, LineString
import shapely.wkt
from src.visualisation.data_visualisation import creating_geometry
import osmnx as ox
import folium

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

geo_taxi = gpd.GeoDataFrame(df, geometry=df['POLYLINE'].apply(creating_geometry), crs='EPSG:4326')
porto_gdf = ox.geocode_to_gdf("Porto, Portugal")
geo_taxi = gpd.sjoin(geo_taxi, porto_gdf, predicate='within')
porto_areas = pd.read_parquet('data/processed/area.parquet')
porto_areas_gdf = gpd.GeoDataFrame(porto_areas, geometry=porto_areas['POLYGON'].apply(lambda wkt: shapely.wkt.loads(wkt)), crs='EPSG:4326')
porto_areas_gdf = porto_areas_gdf.drop(columns=['POLYGON'])

df = gpd.overlay(geo_taxi, porto_areas_gdf, how='intersection')
#print(df.head())


#delete MISSING_DATA, MISSING_DATA column contains boolean values (False/True)
df = df[df['MISSING_DATA'] == False].copy() 
# change from A, B, C to 1, 2, 3 for CALL_TYPE
df['CALL_TYPE'] = df['CALL_TYPE'].map({'A': 1, 'B': 2, 'C': 3})

# trip time: (number_of_points - 1) * 15 seconds
def trip_time(geom):
    if geom is None or geom.is_empty:
        return 0
        
    count = 0
    if geom.geom_type == 'LineString':
        count = len(geom.coords)
        
    #Trasa pocięta na fragmenty (MultiLineString)
    elif geom.geom_type == 'MultiLineString':
        for linia in geom.geoms:
            count += len(linia.coords)

    elif geom.geom_type == 'Point':
        count = 1

    return max(0, (count - 1) * 15) / 60

df['TIME_IN_AREA'] = df['geometry'].apply(trip_time)

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

df['POLYLINE_LIST'] = df.geometry.apply(parse_polyline)

# extract START and END coordinates
df['START_LON'] = df['POLYLINE_LIST'].apply(lambda x: x[0][0] if len(x) > 0 else None)
df['START_LAT'] = df['POLYLINE_LIST'].apply(lambda x: x[0][1] if len(x) > 0 else None)
df['END_LON'] = df['POLYLINE_LIST'].apply(lambda x: x[-1][0] if len(x) > 0 else None)
df['END_LAT'] = df['POLYLINE_LIST'].apply(lambda x: x[-1][1] if len(x) > 0 else None)
def get_coords(geom):
    if geom is None or geom.is_empty or geom.geom_type != 'LineString':
        return None, None, None, None
    
    start = geom.coords[0]
    end = geom.coords[-1]
    
    return start[0], start[1], end[0], end[1]

df['START_LON'], df['START_LAT'], df['END_LON'], df['END_LAT'] = zip(*df['geometry'].apply(get_coords))

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
    if np.any(pd.isnull([lon1, lat1, lon2, lat2])):
        return np.nan
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

#df['DIST_KM_IN_AREA'] = df['POLYLINE_LIST'].apply(calculate_actual_distance)
df.to_crs(epsg=3857, inplace=True)  # project to metric CRS for accurate distance calculation
df['DIST_KM_IN_AREA'] = df.geometry.length / 10**6 # approximate conversion from degrees to km at the equator
start_points = gpd.GeoSeries.from_xy(df['START_LON'], df['START_LAT'], crs="EPSG:3857")
end_points = gpd.GeoSeries.from_xy(df['END_LON'], df['END_LAT'], crs="EPSG:3857")
df['OPTIMAL_DIST_KM'] = start_points.distance(end_points) / 1000
#vectorized_haversine(df['START_LON'], df['START_LAT'], df['END_LON'], df['END_LAT'])
df.to_crs(epsg=4326, inplace=True)


# 6. Deviation Ratio
# np.where handles the division by zero natively and cleanly
df['DEVIATION_RATIO'] = np.where(
    (df['OPTIMAL_DIST_KM'].notnull()) & (df['OPTIMAL_DIST_KM'] > 0),
    df['DIST_KM_IN_AREA'] / df['OPTIMAL_DIST_KM'],
    1.0
)
df['SPEED'] = np.where(df['TIME_IN_AREA'] > 0, df['DIST_KM_IN_AREA'] / (df['TIME_IN_AREA'] / 60), np.nan)
# check   
print(df[['TRIP_ID', 'YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE', 'SECOND', 'PARTDAY', 'TIME_IN_AREA']].head())
print(df[['TRIP_ID', 'DIST_KM_IN_AREA', 'DEVIATION_RATIO', 'OPTIMAL_DIST_KM', 'geometry']].head())
print(df[['TRIP_ID', 'START_LON', 'START_LAT', 'END_LON', 'END_LAT','POLYLINE_LIST','SPEED']].head())
print(type(df['geometry']))
#saving prepared data
def visualizing():
    porto_gdf.to_crs(epsg=3857, inplace=True)    
    center_point = porto_gdf.geometry.centroid.iloc[0]
    porto_gdf.to_crs(epsg=4326, inplace=True)
    m = folium.Map(location=[center_point.y, center_point.x], zoom_start=12, tiles='CartoDB positron')
    folium.Choropleth(porto_gdf.geometry,columns=['AREA_ID']).add_to(m)
    folium.Choropleth(df.geometry.head(100),columns=['AREA_ID']).add_to(m)
    m.save("data/maps/taxi_map.html")

columns_to_drop = ['CALL_TYPE', 'ORIGIN_CALL','ORIGIN_STAND','TAXI_ID','TIMESTAMP','DAY_TYPE','MISSING_DATA','POLYLINE','DATETIME','START_LON', 'START_LAT', 'END_LON', 'END_LAT',
                   'OPTIMAL_DIST_KM','POLYLINE_LIST', 'DATETIME']
#TRIP_ID(int), YEAR(int), MONTH(int), DAY(int), HOUR(int), MINUTE(int), TIMEZONE(string),SECOND(int), WEEKDAY(int),
# PARTDAY(int), POLYLINE(xd nie wiem jaki typ object?), TIME_IN_AREA(int), DIST_KM_IN_AREA(float), DEVIATION_RATIO(float),
# AREA_ID(int),SPEED(float);
df_final = df.drop(columns=columns_to_drop, errors='ignore')
visualizing()
df_final.to_file(output_path.replace('.parquet', '.geojson'), driver='GeoJSON')
df_final.to_parquet(output_path)

#csv
#output_path_csv = 'data/data_samples/taxi_10k_prepared.csv'
#df_final.to_csv(output_path_csv, index=False)
