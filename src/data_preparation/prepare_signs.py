import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
path_r="data/traffic/raw/"
path_e="data/traffic/processed/"


#Collecting Street Data
files_points=[f"{path_r}traffic_lights.geojson",f"{path_r}give_way.geojson", f"{path_r}stop.geojson", f"{path_r}traffic_calming.geojson"]
cat=["traffic_light", "give_way", "stop", "traffic_calming"]
col=['@id','geometry' ]


gdf_list=[]

for i in range(len(files_points)):
    temp_gdf=gpd.read_file(files_points[i])[col]
    temp_gdf["category"]=cat[i]
    gdf_list.append(temp_gdf)



signs=pd.concat(gdf_list, ignore_index=True)
signs["@id"] = signs["@id"].astype(str).str.replace("node/", "")
print(signs)
signs=signs[["@id", "category", "geometry"]]
signs=gpd.GeoDataFrame(signs)
#signs.to_csv("signs.csv", index=False, header=True, sep=',')
signs.to_file(f"{path_e}signs.geojson", driver="GeoJSON")

signs.plot()
plt.show()
