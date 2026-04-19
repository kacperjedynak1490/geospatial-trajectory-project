import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely import Polygon
from shapely import wkt

#preparing area data
areas=pd.read_parquet("data/processed/area.parquet")
areas['geometry'] = areas['POLYGON'].apply(wkt.loads)
areas = gpd.GeoDataFrame(areas, geometry='geometry')
print(areas.shape[0])
#areas.plot(figsize=(10, 10), cmap='viridis', edgecolor='black')
#plt.show()

#preparing signs and streets data
signs=gpd.read_file("data/traffic/processed/signs.geojson")
streets=gpd.read_file("data/traffic/processed/streets.geojson")

areas.set_crs(signs.crs, inplace=True)
streets.set_crs(signs.crs, inplace=True)

#preparing data about junctions-making polygons for correct counting
junctions=streets[streets["junction"]!=0].copy()
junctions.geometry=junctions.buffer(0.0002)
junctions=junctions.dissolve()
junctions = junctions.explode(index_parts=False)

#junctions.plot(figsize=(10, 10), cmap='viridis', edgecolor='black')
#plt.show()

#checking the number of the items
#print(signs['category'].value_counts())

result = []

# first way: using full potential of geopandas:
temp_signs= gpd.sjoin(signs, areas, predicate='within', how='inner')
temp_streets = gpd.overlay(streets, areas, how='intersection')
temp_junctions = gpd.sjoin(junctions, areas, predicate='intersects', how='inner')
print(junctions.shape[0])

def signs_count(group):

    tmp= group['category'].value_counts()
    return pd.Series({'Area_traffic_lights': int(tmp.get('traffic_light', 0)),
                            'Area_traffic_calmigs': int(tmp.get('traffic_calming', 0)),
                            'Area_stop_signs': int(tmp.get('stop', 0)),
                            'Area_give_way_sign': int(tmp.get('give_way', 0))

        })

signs_stats=temp_signs.groupby('AREA_ID').apply(signs_count, include_groups=False).reset_index()

def streets_stat(group):
    if not group.empty:
        lanes=group['lanes'].mode()[0]
        maxspeed=group['maxspeed'].mode()[0]
        crossing=group['highway'].value_counts().get("crossing",0)
    else:
        lanes=None
        maxspeed=None
        crossing=None

    return pd.Series({'Area_lanes': lanes,
                        'Area_maxspeed': maxspeed,
                        'Area_crossings': crossing
    })

streets_stats=temp_streets.groupby('AREA_ID').apply(streets_stat,include_groups=False).reset_index()

junction=temp_junctions.groupby('AREA_ID').size().reset_index(name='Area_junctions')


#final data frame
df = areas[['AREA_ID']].merge(streets_stats, on='AREA_ID', how='left')

df = df.merge(signs_stats, on='AREA_ID', how='left')
df = df.merge(junction, on='AREA_ID', how='left')
df['Area_junctions']=df['Area_junctions'].fillna(0)
df['Area_junctions']=df['Area_junctions'].astype(int)

df.to_parquet("data/processed/traffic.parquet", index=False)



#second way: for
# for idx, row in areas.iterrows():
#     area= gpd.GeoDataFrame([row], geometry='geometry', crs=signs.crs)
#     temp_signs= gpd.sjoin(signs, area, predicate='within', how='inner')
#     temp_streets = gpd.overlay(streets, area, how='intersection')
#     temp_junctions = gpd.sjoin(junctions, area, predicate='intersects', how='inner')

#     if not temp_streets.empty and not temp_signs.empty:
#         lanes = temp_streets['lanes'].mode()[0]
#         maxspeed = temp_streets['maxspeed'].mode()[0]
#         crossing = temp_streets['highway'].value_counts().get('crossing', 0)

#         signs_count = temp_signs['category'].value_counts()
#         traffic_lights = signs_count.get('traffic_light', 0)
#         stop_sign = signs_count.get('stop', 0)
#         give_way = signs_count.get('give_way', 0)
#         calmings = signs_count.get('traffic_calming', 0)

#         junction=temp_junctions.shape[0]


#         result.append({'AREA_ID': [row['AREA_ID']],
#                                 'Area_lanes': lanes,
#                                 'Area_maxspeed': maxspeed,
#                                 'Area_crossings': crossing,
#                                 'Area_junctions':junction,
#                                 'Area_traffic_lights': traffic_lights,
#                                 'Area_traffic_calmigs': calmings,
#                                 'Area_stop_signs': stop_sign,
#                                 'Area_give_way_sign': give_way})


# df = pd.DataFrame(result)
# df['AREA_ID'] = df['AREA_ID'].apply(lambda x: x[0] if isinstance(x, list) else x)
# df['AREA_ID'].astype(str)
# print(df.shape[0])


