import json
import shapefile, csv
import pandas as pd

# i had different paths so check if it works for you
# DONT ADD EXTENSION TO THE NAME, IT HAS TO BE ADDED AUTOMATICALLY
train_shp = shapefile.Writer("raw_shapefile/raw_train", shapeType=shapefile.POLYLINE)
'''Starting shapfile creation
POLYLINE is line
POINT is single geo-point'''

# balancing number of points and data records
train_shp.autoBalance = 1

# create the field names and data type for each.
# to be edited if extra features are added to the dataset
train_shp.field("TRIP_ID", "N")
train_shp.field("CALL_TYPE", "C")
train_shp.field("ORIGIN_CALL", "N")
train_shp.field("ORIGIN_STAND", "N")
train_shp.field("TAXI_ID", "N")
train_shp.field("TIMESTAMP", "N")
train_shp.field("DAY_TYPE", "C")
train_shp.field("MISSING_DATA", "L")

# count the features for progress tracking
counter = 1

# access the CSV file
# if parquet file is used has to be changed
# Kacper says there is read_parquet in pandas, but I haven't tried it yet
'''with open('train.csv') as csvfile:
 reader = csv.reader(csvfile, delimiter=',')
 # skip the header
 next(reader, None)

#loop through each of the rows and assign the attributes to variables
for row in reader:
  trip_id = row[0]
  call_type = row[1]
  origin_call = row[2]
  origin_stand = row[3]
  taxi_id = row[4]
  timestamp = row[5]
  day_type = row[6]
  missing_data = row[7]
  coordinates = json.loads(row[8])  
  '''

# USE THIS FOR PARQUET
# didnt visualize so dont know if it works
parquet_df = pd.read_parquet('dane.parquet')
parquet_df = parquet_df.fillna(0)

for row in parquet_df.itertuples(index=False):
  trip_id = row.TRIP_ID
  call_type = row.CALL_TYPE
  origin_call = row.ORIGIN_CALL
  origin_stand = row.ORIGIN_STAND
  taxi_id = row.TAXI_ID
  timestamp = row.TIMESTAMP
  day_type = row.DAY_TYPE
  missing_data = row.MISSING_DATA
  coordinates = json.loads(row.POLYLINE)

# if data is missing - add null to keep the number of records the same
  if len(coordinates) == 0:
    train_shp.null()
    
  elif len(coordinates) == 1:
    # if only one point there cant be a line so we have to create it manually
    # we create a line from the same point to itself
    fake_line = [coordinates[0], coordinates[0]]
    train_shp.line([fake_line])
    
  else:
    # here there is normal line
    train_shp.line([coordinates])
  # add attribute data
  train_shp.record(trip_id, call_type, origin_call, origin_stand, taxi_id, timestamp, day_type, missing_data)

  # printintg progress to check if works
  print ("Feature " + str(counter) + " added to Shapefile.")
  counter = counter + 1

# save the Shapefile
train_shp.close()

# path has to be the same as shp file but with extension this time
prj_path = r"raw_shapefile/raw_train.prj"
# it fits this shp but not all
wgs84_wkt = 'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]'
#just saves to file
with open(prj_path, "w") as prj_file:
    prj_file.write(wgs84_wkt)
