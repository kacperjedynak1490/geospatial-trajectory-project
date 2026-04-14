import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
path_r="data/traffic/raw/"
path_e="data/traffic/processed/"

#Collecting Street Data
col=['osmid', 'lanes', 'maxspeed', 'highway', 'oneway', 'bridge', 'junction']
streets_gdf=gpd.read_file(f"{path_r}streets.geojson")[col+['geometry']]


#Data cleaning and transformation:
def cleaning_data(value):
    #if the value is a list: take only the first item
    if isinstance(value, (list, np.ndarray)):
        return value[0]
    return value

for column in streets_gdf.columns:
    streets_gdf[column]=streets_gdf[column].apply(cleaning_data)


#print(streets_gdf["highway"].unique())

#default values:
data_values = {
    'motorway':        [3, 120],
    'motorway_link':   [1, 60],
    'trunk':           [2, 100],
    'trunk_link':      [1, 50],
    'primary':         [2, 50],
    'primary_link':    [1, 40],
    'secondary':       [2, 50],
    'secondary_link':  [1, 40],
    'tertiary':        [1, 50],
    'tertiary_link':   [1, 30],
    'residential':     [1, 50],
    'living_street':   [1, 20],
    'unclassified':    [1, 50],
    'busway':          [1, 50],
    'road':            [1, 40],
    'crossing':        [0,0] #beacuse it is not a road
}

def fill_missing(row):
    highway=str(row['highway'])

    if highway in data_values:
        lanes,maxspeed=data_values[highway]

        if pd.isna(row['lanes']): row['lanes']=lanes
        if pd.isna(row['maxspeed']): row['maxspeed'] = maxspeed

    # bridge=1 | else=0
    row["bridge"]=1 if row["bridge"]=="yes" else 0


    #oneway=True | else=False
    #row["oneway"] = True if row["oneway"]=="yes" else False
    #in our data (from Python) there is already true/false
    if pd.isna(row['oneway']): row['oneway'] = False


    #roundabout=1 | circular=2 | else=0
    if  row["junction"]=="roundabout": row["junction"]=1
    elif row["junction"]=="circular": row["junction"]=2
    else: row["junction"]=0
    return row

streets=streets_gdf.apply(fill_missing, axis=1)

for column in streets_gdf.columns:
    if column in ["osmid", "lanes", "maxspeed", "bridge", "junction"]:
        streets[column] = streets[column].astype(int)
    elif column == "highway":
        streets[column] = streets[column].astype(str)
    elif column=="oneway":
        streets[column] = streets[column].astype(bool)


#streets.to_csv("streets.csv", index=False)
streets.to_file(f"{path_e}streets.geojson",driver='GeoJSON')

print(streets)


#data Visualisation
streets.plot()
plt.show()

