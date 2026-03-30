'''
That code takes data from Open Meteo API and saves it to json file.
At this moment it makes only request to API for Porto for historic date 01/07/2013
to 30/06/2014.
This code is copied from oficial Open Meteo website with some modifications.
I split date to year, month, day, hour and timezone and add most important columns.
'''
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
	"latitude": 41.1496,
	"longitude": -8.611,
	"start_date": "2013-07-01",
	"end_date": "2014-07-30",
	"hourly": ["precipitation", "rain", "wind_gusts_10m", "is_day", "weather_code", "temperature_2m", "relative_humidity_2m"],
	"timezone": "auto",
}
responses = openmeteo.weather_api(url, params = params)

# Process first location. Add a for-loop for multiple locations or weather models
response = responses[0]

# Process hourly data. The order of variables needs to be the same as requested.
hourly = response.Hourly()
hourly_precipitation = hourly.Variables(0).ValuesAsNumpy()
hourly_rain = hourly.Variables(1).ValuesAsNumpy()
hourly_wind_gusts_10m = hourly.Variables(2).ValuesAsNumpy()
hourly_is_day = hourly.Variables(3).ValuesAsNumpy()
hourly_weather_code = hourly.Variables(4).ValuesAsNumpy()
hourly_temperature_2m = hourly.Variables(5).ValuesAsNumpy()
hourly_relative_humidity_2m = hourly.Variables(6).ValuesAsNumpy()

hourly_data = {"date": pd.date_range(
	start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
	end =  pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
	freq = pd.Timedelta(seconds = hourly.Interval()),
	inclusive = "left"
)}

hourly_data["precipitation"] = hourly_precipitation
hourly_data["rain"] = hourly_rain
hourly_data["wind_gusts_10m"] = hourly_wind_gusts_10m
hourly_data["is_day"] = hourly_is_day
hourly_data["weather_code"] = hourly_weather_code
hourly_data["temperature_2m"] = hourly_temperature_2m
hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m

hourly_dataframe = pd.DataFrame(hourly_data)

hourly_dataframe['date'] = hourly_dataframe['date'].dt.tz_convert('Europe/Lisbon')
hourly_dataframe['year'] = hourly_dataframe['date'].dt.year
hourly_dataframe['month'] = hourly_dataframe['date'].dt.month
hourly_dataframe['day'] = hourly_dataframe['date'].dt.day
hourly_dataframe['hour'] = hourly_dataframe['date'].dt.hour
hourly_dataframe['timezone'] = hourly_dataframe['date'].dt.strftime('%Z')
hourly_dataframe['date'] = hourly_dataframe['date'].dt.tz_localize(None)

cols = ["year","month","day","hour","timezone","precipitation","rain","wind_gusts_10m","is_day","weather_code","temperature_2m","relative_humidity_2m"]
hourly_dataframe = hourly_dataframe[cols]

hourly_dataframe.to_parquet("./data/raw/weather/hourly_weather.parquet", index = False)
#hourly_dataframe.to_csv("./data/raw/weather/hourly_weather.csv", index = False)
