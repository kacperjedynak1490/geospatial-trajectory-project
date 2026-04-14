import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import json
from shapely.geometry import LineString, Point
import contextily
import osmnx as ox

#reading files do dataframes
taxi_10k = pd.read_parquet('data/data_samples/data_10k_raw.parquet')
taxi_100k = pd.read_parquet('data/data_samples/data_100k_raw.parquet')
porto_gdf = ox.geocode_to_gdf("Porto, Portugal")
weather_data = pd.read_parquet('data/raw/weather/hourly_weather.parquet')
taxi_10k_proc = pd.read_parquet('data/data_samples/taxi_10k_prepared.parquet')

# print(taxi_10k.head())
# print(taxi_100k.head())
# print(weather_data.head())

'''
to read geometry from the csv files we need special function to convert them to geodataframes later
'''
def creating_geometry(column):
    coords = json.loads(column)        
    if len(coords) >= 2:
        return LineString(coords)
    elif len(coords) == 1:
        return Point(coords[0])
    else:
        return None       

''' this function visualizes using geopandas, adds basemap, deletes axes, and so on
input type is a geodataframe, and the other one taht sets borders for the first one (usually our porto city)
'''
def visualize_taxi_trajectories(taxi_df, porto_gdf = porto_gdf, porto = True):
    taxi_porto = gpd.GeoDataFrame(taxi_df, geometry=taxi_df['POLYLINE'].apply(creating_geometry))
    taxi_porto = taxi_porto.set_crs("EPSG:4326")
    if porto:
        taxi_porto = gpd.sjoin(taxi_porto, porto_gdf, predicate='within')
    
    fig, ax = plt.subplots(figsize=(10, 10))
    taxi_porto.plot(ax=ax)
    ax.set_title('Taxi Trajectories')
    ax.set_axis_off()
    contextily.add_basemap(ax=ax, crs=taxi_porto.crs)
    if porto:
        porto_gdf.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=2)
    plt.show()

def visualize_taxi_raw():
    taxi_geo = gpd.GeoDataFrame(taxi_10k, geometry=taxi_10k['POLYLINE'].apply(creating_geometry))
    taxi_geo = taxi_geo.set_crs("EPSG:4326")
    taxi_geo.plot()
    contextily.add_basemap(ax=plt.gca(), crs=taxi_geo.crs)
    plt.show()
    
#print(taxi_10k_proc.head())
#print(taxi_10k_proc.columns)

taxi_10k_proc_geo_start = gpd.GeoDataFrame(taxi_10k_proc, geometry=gpd.points_from_xy(taxi_10k_proc['START_LON'], taxi_10k_proc['START_LAT']))
taxi_10k_proc_geo_end = gpd.GeoDataFrame(taxi_10k_proc, geometry=gpd.points_from_xy(taxi_10k_proc['END_LON'], taxi_10k_proc['END_LAT']))

taxi_10k_proc_geo_start = taxi_10k_proc_geo_start.set_crs("EPSG:4326")
taxi_10k_proc_geo_end = taxi_10k_proc_geo_end.set_crs("EPSG:4326")

'''
beacuse there can only be one geometry in geodataframe i separated for start point and endpoint
ideally we first filter by within porto then we dont really need geometry that much
'''
df_start = gpd.sjoin(taxi_10k_proc_geo_start, porto_gdf, predicate='within')
df_end = gpd.sjoin(taxi_10k_proc_geo_end, porto_gdf, predicate='within')

'''
function titles can vary because they're pretty random
they are also mostly crafted for Szymon's data but any row number works
'''

def visualize_boxplots(df):
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))
    axes[0].boxplot([df['TRIP_TIME_MIN']], labels=['TRIP_TIME_MIN'])
    axes[1].boxplot([df['DEVIATION_RATIO']], labels=['DEVIATION_RATIO'])
    axes[2].boxplot([df['ACTUAL_DIST_KM']], labels=['ACTUAL_DIST_KM'])
    plt.show()

def visualize_dependences(df, log=True):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].scatter(df['ACTUAL_DIST_KM'],df['TRIP_TIME_MIN'], alpha=0.5)
    axes[0].set_xlabel('ACTUAL_DIST_KM')
    axes[0].set_ylabel('TRIP_TIME_MIN')
    axes[0].set_title('TRIP_TIME_MIN vs ACTUAL_DIST_KM')
    
    axes[1].scatter(df['ACTUAL_DIST_KM'],df['TRIP_TIME_MIN'], alpha=0.5)
    axes[1].set_xlabel('ACTUAL_DIST_KM')
    axes[1].set_ylabel('TRIP_TIME_MIN')
    axes[1].set_xscale('log')
    axes[1].set_title('TRIP_TIME_MIN vs ACTUAL_DIST_KM')

    #this was messy, now better version
    # dates = pd.to_datetime(df[["YEAR", "MONTH", "DAY", "HOUR", "MINUTE", "SECOND"]].rename(columns=str.lower))
    # axes[1].scatter(dates, df['TRIP_TIME_MIN'], alpha=0.5)
    # axes[1].set_xlabel('TIME')
    # axes[1].set_ylabel('TRIP_TIME_MIN')
    grouped_months = df.groupby(['YEAR', 'MONTH']).size()
    positions = range(len(grouped_months))
    labels = [f"{year}-{month:02d}" for year, month in grouped_months.index]
    axes[2].bar(positions, grouped_months.values, color='purple')
    axes[2].set_xticks(positions)
    axes[2].set_xticklabels(labels, rotation=90, ha='right')
    axes[2].set_xlabel('TIME')
    axes[2].set_ylabel('COUNT')
    axes[2].set_title('Number of Trips per Month')
    plt.subplots_adjust(wspace=0.7)
    plt.show()

def visualize_barplots(df):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0,0].bar(df.groupby('YEAR').size().index, df.groupby('YEAR').size(), color='blue')
    axes[0,0].set_xticks([2013, 2014])
    axes[0,0].set_xlabel('YEAR')
    axes[0,0].set_ylabel('COUNT')
    axes[0,0].tick_params(axis='x', labelrotation=45)

    axes[0,1].bar(df.groupby('MONTH').size().index, df.groupby('MONTH').size(), color='green')
    axes[0,1].set_xticks(range(1, 13), labels=['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'])
    axes[0,1].set_xlabel('MONTH')
    axes[0,1].set_ylabel('COUNT')
    axes[0,1].tick_params(axis='x', labelrotation=45)

    axes[1,0].bar(df.groupby('WEEKDAY').size().index, df.groupby('WEEKDAY').size(), color='orange')
    axes[1,0].set_xticks(range(1, 8), labels=['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN'])
    axes[1,0].set_xlabel('WEEKDAY')
    axes[1,0].set_ylabel('COUNT')
    axes[1,0].tick_params(axis='x', labelrotation=45)
    
    axes[1,1].bar(df.groupby('PARTDAY').size().index, df.groupby('PARTDAY').size(), color='red')
    axes[1,1].set_xticks(range(1, 6), labels=['MORNING', 'MIDDAY', 'AFTERNOON', 'EVENING', 'NIGHT'])
    axes[1,1].set_xlabel('PARTDAY')
    axes[1,1].set_ylabel('COUNT')
    axes[1,1].tick_params(axis='x', labelrotation=45)

    plt.subplots_adjust(hspace=0.5)
    plt.show()
    
    # maybe i will correct later for now not working
def statistics(df):
    fig, axes = plt.subplots(6,6,figsize=(30, 20))
    for index,column in enumerate(df.columns):
        if df[column].dtype in ['int64', 'float64']:
            print(f"Statistics for {column}:")
            print(df[column].describe())
            print("\n")
        elif df[column].dtype == 'string':
            print(f"Value counts for {column}:")
            print(df[column].value_counts())
            axes[index//6, index%6].bar(df[column].unique(), df[column].value_counts().values)
            print("\n")
'''
this is strictly for weather data
'''

def weather_visu():
    fig, axes = plt.subplots(3, 2, figsize=(14, 7))
    dates = pd.to_datetime(weather_data[["year", "month", "day"]])
    axes[0,0].plot(dates, weather_data['temperature_2m'], c='b')
    axes[0,0].set_xlabel('TIME')
    axes[0,0].set_ylabel('TEMPERATURE_C')
    axes[0,0].set_title('Temperature over Time')
    axes[0,0].tick_params(axis='x', labelrotation=45)

    axes[0,1].plot(dates, weather_data['precipitation'], c='g')
    axes[0,1].set_xlabel('TIME')
    axes[0,1].set_ylabel('PRECIPITATION')
    axes[0,1].set_title('Precipitation over Time')
    axes[0,1].tick_params(axis='x', labelrotation=45)

    axes[1,0].plot(dates, weather_data['wind_gusts_10m'], c='r')
    axes[1,0].set_xlabel('TIME')
    axes[1,0].set_ylabel('WIND_GUSTS_10M')
    axes[1,0].set_title('Wind Gusts over Time')
    axes[1,0].tick_params(axis='x', labelrotation=45)

    axes[1,1].plot(dates, weather_data['relative_humidity_2m'], c='y')
    axes[1,1].set_xlabel('TIME')
    axes[1,1].set_ylabel('RELATIVE_HUMIDITY_2M')
    axes[1,1].set_title('Relative Humidity over Time')
    axes[1,1].tick_params(axis='x', labelrotation=45)

    axes[2,0].plot(dates, weather_data['rain'], c='m')
    axes[2,0].set_xlabel('TIME')
    axes[2,0].set_ylabel('RAIN')
    axes[2,0].set_title('Rain over Time')
    axes[2,0].tick_params(axis='x', labelrotation=45)
    
    axes[2,1].plot(dates, weather_data['weather_code'], c='c')
    axes[2,1].set_xlabel('TIME')
    axes[2,1].set_ylabel('WEATHER_CODE')
    axes[2,1].set_title('Weather Code over Time')
    axes[2,1].tick_params(axis='x', labelrotation=45)
    
    plt.subplots_adjust(hspace=1.2, wspace=0.3)
    plt.show()
    
'''
here are the calls
i double chcecked for different df if there are differences but apart from mapping i dont think they are significant
'''

#print(taxi_10k_proc.columns)
#print(taxi_10k_proc[['TRIP_TIME_MIN','ACTUAL_DIST_KM','OPTIMAL_DIST_KM','DEVIATION_RATIO']].describe())

#statistics(df_end)   # DONT USE DOESNT WORK         
#visualize_taxi_trajectories(taxi_10k, porto_gdf, False)
#visualize_taxi_trajectories(taxi_10k, porto_gdf, True)

#visualize_boxplots(taxi_10k_proc)
#visualize_boxplots(df_end)

#visualize_dependences(taxi_10k_proc)
#visualize_dependences(df_end,log=False)

#visualize_barplots(taxi_10k_proc)
#visualize_barplots(df_end)

#print(weather_data[['precipitation','rain','wind_gusts_10m','weather_code','temperature_2m','relative_humidity_2m']].describe())
#weather_visu()