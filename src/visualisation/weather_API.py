import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

# this script will be simmilar to prepare_weather.py
# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

url = "https://api.open-meteo.com/v1/forecast"
params = {
	"latitude": 41.1496,
	"longitude": -8.611,
	"hourly": ["is_day", "temperature_2m", "relative_humidity_2m", "precipitation", "rain", "wind_gusts_10m", "weather_code"],
	"past_days": 1,
	"forecast_days": 7,
}
responses = openmeteo.weather_api(url, params = params)
response = responses[0]


#Process hourly data:
hourly = response.Hourly()

hourly_data = {"date": pd.date_range(
	start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
	end =  pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
	freq = pd.Timedelta(seconds = hourly.Interval()),
	inclusive = "left"
)}

hourly_data["is_day"] = hourly.Variables(0).ValuesAsNumpy()
hourly_data["temperature_2m"] = hourly.Variables(1).ValuesAsNumpy()
hourly_data["relative_humidity_2m"] = hourly.Variables(2).ValuesAsNumpy()
hourly_data["precipitation"] = hourly.Variables(3).ValuesAsNumpy()
hourly_data["rain"] = hourly.Variables(4).ValuesAsNumpy()
hourly_data["wind_gusts_10m"] = hourly.Variables(5).ValuesAsNumpy()
hourly_data["weather_code"] =  hourly.Variables(6).ValuesAsNumpy()


df = pd.DataFrame(hourly_data)

df['date'] = df['date'].dt.tz_convert('Europe/Lisbon')
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['hour'] = df['date'].dt.hour
df['timezone'] = df['date'].dt.strftime('%Z')

# Making final df for ml
# The code below will be simmilar to prepare_data_2.py, because the structure must be the same
cols = ['year', 'month', 'day','hour', 'timezone', 
    'precipitation', 'wind_gusts_10m', 'is_day', 'weather_code', 'temperature_2m']
df = df[cols]

# rounding values to nearest integer
round_col = ['precipitation', 'wind_gusts_10m', 'is_day', 'weather_code', 'temperature_2m']
df[round_col] = df[round_col].round().astype(int)

df.columns=[col.upper() for col in df.columns]
print(df)

df.to_parquet("data/visualization/weather/7day_weather.parquet")
