# namieszałem trochę bo chciałem jedną funkcję
# sprawdze potem optymalizacje ale na razie działa
import json
from enum import StrEnum
import shapefile, csv
import pandas as pd

class LocationDataError(Exception):
  pass
class FileTypeError(Exception):
  pass
class DataError(Exception):
  pass

class Location(StrEnum):
  POLYLINE = "POLYLINE"
  POINT = "POINT"
class AvailableData(StrEnum):
  TAXI = "dane"
  SIGNS = "signs"

def shapefileAddRow(data_shp,row,data:AvailableData):
  if data == AvailableData.TAXI:
    trip_id = row[0]
    call_type = row[1]
    origin_call = row[2]
    origin_stand = row[3]
    taxi_id = row[4]
    timestamp = row[5]
    day_type = row[6]
    missing_data = row[7]
    coordinates = json.loads(row[8])  
    if len(coordinates) == 0:
      # if data is missing - add null to keep the number of records the same
      data_shp.null()
    
    elif len(coordinates) == 1:
      # if only one point there cant be a line so we have to create it manually
      # we create a line from the same point to itself
      fake_line = [coordinates[0], coordinates[0]]
      data_shp.line([fake_line])
    
    else:
      # here there is normal line
      data_shp.line([coordinates])
      # add attribute data
      data_shp.record(trip_id, call_type, origin_call, origin_stand, taxi_id, timestamp, day_type, missing_data)
  elif data == AvailableData.SIGNS:
    id = row[0]
    category = row[1]
    coordinates = (row[2], row[3])
    data_shp.point(float(coordinates[0]), float(coordinates[1]))
    data_shp.record(id, category)
  else:
    raise DataError("YOU SHOLDNT SEE THIS EVER\nThis data is not checked yet, please report to contributors (Jan Serzysko)")
  return


def shapefileConversion(filename_full:str, locationdata:Location, data:AvailableData):
  locationtype = getattr(shapefile, locationdata)
  filetype = filename_full.split(".")[-1]
  filename = filename_full.split(".")[0]
  if(locationdata != Location.POLYLINE and locationdata != Location.POINT):
    raise LocationDataError("locationdata has to be either POLYLINE or POINT")
  if(filetype != "csv" and filetype != "parquet"):
    raise FileTypeError("filetype has to be either csv or parquet")
  
  # DONT ADD EXTENSION TO THE NAME, IT HAS TO BE ADDED AUTOMATICALLY
  data_shp = shapefile.Writer(f"data/raw/raw_shapefile{filename}", shapeType=locationtype)
  '''Starting shapfile creation
  POLYLINE is line
  POINT is single geo-point'''

  # balancing number of points and data records
  data_shp.autoBalance = 1

  # create the field names and data type for each.
  # to be edited if extra features are added to the dataset
  if data == AvailableData.TAXI:
    data_shp.field("TRIP_ID", "N")
    data_shp.field("CALL_TYPE", "C")
    data_shp.field("ORIGIN_CALL", "N")
    data_shp.field("ORIGIN_STAND", "N")
    data_shp.field("TAXI_ID", "N")
    data_shp.field("TIMESTAMP", "N")
    data_shp.field("DAY_TYPE", "C")
    data_shp.field("MISSING_DATA", "L")
  elif data == AvailableData.SIGNS:
    data_shp.field("ID","C")
    data_shp.field("CATEGORY","C")
  else:
    raise DataError("This data is not checked yet, please report to contributors (Jan Serzysko)")

  # count the features for progress tracking
  counter = 1

  if filetype == "csv":
    with open(f'{filename_full}') as csvfile:
      reader = csv.reader(csvfile, delimiter=',')
      # skip the header
      next(reader, None)

      #loop through each of the rows and assign the attributes to variables
      for row in reader:
        shapefileAddRow(data_shp,row,data)
        print("Feature " + str(counter) + " added to Shapefile.")
        counter = counter + 1
   # didnt visualize so dont know if it works
  if filetype == "parquet":
    parquet_df = pd.read_parquet(filename_full)
    parquet_df = parquet_df.fillna(0)

    for row in parquet_df.itertuples(index=False):
      shapefileAddRow(data_shp,row,data)
      print("Feature " + str(counter) + " added to Shapefile.")
      counter = counter + 1
  # save the Shapefile
  data_shp.close()

  # path has to be the same as shp file but with extension this time
  prj_path = rf"data/raw/raw_shapefile/{filename}.prj"
  # it fits this shp but not all
  wgs84_wkt = 'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]'
  #just saves to file
  with open(prj_path, "w") as prj_file:
    prj_file.write(wgs84_wkt)
  return

# don't know if to just keep it as a module or manually call functions here
#shapefileConversion("dane.parquet", "POLYLINE",AvailableData.TAXI)
shapefileConversion("signs.csv", "POINT",AvailableData.SIGNS)