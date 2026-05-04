import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import joblib
import datetime
import shapely.wkt
import openmeteo_requests
import requests_cache
from retry_requests import retry

# page config
st.set_page_config(page_title="Porto Traffic Predictor", layout="wide", initial_sidebar_state="expanded")
st.title("🚗 Porto Traffic Prediction Dashboard")

# load xgboost model
@st.cache_resource
def load_model():
    return joblib.load('data/processed/xgboost_traffic_model_heatmap.pkl')

# load static spatial and traffic data
@st.cache_data
def load_static_data():
    area_df = pd.read_parquet('data/processed/area.parquet')
    traffic_df = pd.read_parquet('data/processed/traffic.parquet')
    
    # get original trained categories
    taxi_df = pd.read_parquet('data/processed/taxi_prepared_full.parquet', columns=['AREA_ID'])
    trained_categories = taxi_df['AREA_ID'].unique()
    
    # parse wkt polygons for pydeck
    def parse_wkt_to_pydeck(wkt_string):
        if pd.isna(wkt_string): return None
        poly = shapely.wkt.loads(wkt_string)
        return [list(poly.exterior.coords)]
        
    area_df['coordinates'] = area_df['POLYGON'].apply(parse_wkt_to_pydeck)
    
    # merge geometry with traffic features
    static_features = pd.merge(area_df, traffic_df, on='AREA_ID', how='left')
    return static_features, trained_categories

# fetch 8-day weather forecast
@st.cache_data(ttl=3600) 
def get_weather_forecast():
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 41.1496,
        "longitude": -8.611,
        "hourly": ["precipitation", "rain", "wind_gusts_10m", "is_day", "weather_code", "temperature_2m", "relative_humidity_2m"],
        "timezone": "Europe/Lisbon",
        "forecast_days": 8 
    }
    
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    hourly = response.Hourly()

    hourly_data = {"time": pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    )}
    
    hourly_data["precipitation"] = hourly.Variables(0).ValuesAsNumpy()
    hourly_data["rain"] = hourly.Variables(1).ValuesAsNumpy()
    hourly_data["wind_gusts_10m"] = hourly.Variables(2).ValuesAsNumpy()
    hourly_data["is_day"] = hourly.Variables(3).ValuesAsNumpy()
    hourly_data["weather_code"] = hourly.Variables(4).ValuesAsNumpy()
    hourly_data["temperature_2m"] = hourly.Variables(5).ValuesAsNumpy()
    hourly_data["relative_humidity_2m"] = hourly.Variables(6).ValuesAsNumpy()

    hourly_dataframe = pd.DataFrame(hourly_data)
    hourly_dataframe['time'] = hourly_dataframe['time'].dt.tz_convert('Europe/Lisbon').dt.tz_localize(None)
    return hourly_dataframe

# helper function for part of day classification
def get_partday(hour):
    if 6 <= hour < 11:      return 1        # morning
    elif 11 <= hour < 13:   return 2        # midday
    elif 13 <= hour < 17:   return 3        # afternoon
    elif 17 <= hour < 21:   return 4        # evening
    else: return 5                          # night

# initialize data
try:
    model = load_model()
    static_df, trained_categories = load_static_data() 
    weather_df = get_weather_forecast()
except Exception as e:
    st.error(f"Data loading error: {e}")
    st.stop()

# layout columns
col_controls, col_map = st.columns([1, 3])

# settings and weather info
with col_controls:
    st.header("⚙️ Settings")
    
    today = datetime.date.today()
    dates = [today + datetime.timedelta(days=i) for i in range(8)]
    
    selected_date = st.selectbox(
        "Select forecast day", 
        dates, 
        format_func=lambda x: "Today" if x == today else x.strftime('%Y-%m-%d (%A)')
    )
    
    selected_hour = st.slider("Select hour", 0, 23, 12, format="%02d:00")
    
    st.divider()
    st.subheader("🌦️ Weather Forecast")
    
    target_datetime = pd.to_datetime(f"{selected_date} {selected_hour:02d}:00:00")
    current_weather = weather_df[weather_df['time'] == target_datetime]
    
    if not current_weather.empty:
        w_data = current_weather.iloc[0]
        wc1, wc2 = st.columns(2)
        wc1.metric("Temperature", f"{w_data['temperature_2m']:.1f} °C")
        wc2.metric("Precipitation", f"{w_data['precipitation']:.1f} mm")
        
        wc3, wc4 = st.columns(2)
        wc3.metric("Wind gusts", f"{w_data['wind_gusts_10m']:.1f} km/h")
        
        # calculate partday string for ui
        partday_val = get_partday(selected_hour)
        if partday_val == 1: partday_str = "Morning"
        elif partday_val == 2: partday_str = "Midday"
        elif partday_val == 3: partday_str = "Afternoon"
        elif partday_val == 4: partday_str = "Evening"
        else: partday_str = "Night"
            
        wc4.metric("Part of day", partday_str)
    else:
        st.warning(f"No weather data for: {target_datetime}")

# prepare features for prediction
predict_df = static_df.copy()

# temporal features
predict_df['YEAR'] = selected_date.year
predict_df['MONTH'] = selected_date.month
predict_df['DAY'] = selected_date.day
predict_df['HOUR'] = selected_hour
predict_df['MINUTE'] = 0
predict_df['WEEKDAY'] = selected_date.weekday()
predict_df['PARTDAY'] = get_partday(selected_hour)

# weather features
if not current_weather.empty:
    for col in ["precipitation", "wind_gusts_10m", "is_day", "weather_code", "temperature_2m"]:
        predict_df[col.upper()] = w_data[col]
else:
    for col in ["PRECIPITATION", "WIND_GUSTS_10M", "IS_DAY", "WEATHER_CODE", "TEMPERATURE_2M"]:
        predict_df[col] = 0.0

# lag features (defaulted to 0 due to lack of live data)
predict_df['JAM_LEVEL_1H_AGO'] = 0.0
predict_df['JAM_LEVEL_1D_AGO'] = 0.0

# align categorical area_id with model
predict_df['AREA_ID'] = pd.Categorical(predict_df['AREA_ID'], categories=trained_categories)

# exact column order required by xgboost
expected_columns = [
    'YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE', 'WEEKDAY', 'PARTDAY', 
    'AREA_ID', 'PRECIPITATION', 'WIND_GUSTS_10M', 'IS_DAY', 'WEATHER_CODE', 
    'TEMPERATURE_2M', 'Area_lanes', 'Area_maxspeed', 'Area_crossings', 
    'Area_traffic_lights', 'Area_traffic_calmigs', 'Area_stop_signs', 
    'Area_give_way_sign', 'Area_junctions', 'JAM_LEVEL_1H_AGO', 'JAM_LEVEL_1D_AGO'
]

X_predict = predict_df[expected_columns]

# run inference
try:
    predict_df['PREDICTED_JAM'] = model.predict(X_predict)
except Exception as e:
    st.error(f"Prediction error: {e}")
    st.stop()

# render map
with col_map:
    # color mapping for jam levels
    def get_color(jam_level):
        if jam_level == 0: return [39, 174, 96, 100]    # green
        elif jam_level == 1: return [241, 196, 15, 100] # yellow
        elif jam_level == 2: return [230, 126, 34, 100] # orange
        elif jam_level == 3: return [192, 57, 43, 100]  # red
        return [128, 128, 128, 30]                    

    predict_df['color'] = predict_df['PREDICTED_JAM'].fillna(-1).apply(get_color)
    
    # summary metrics
    avg_jam = predict_df['PREDICTED_JAM'].mean()
    severe_jams = (predict_df['PREDICTED_JAM'] == 3).sum()
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Avg traffic level (0-3)", f"{avg_jam:.2f}" if not pd.isna(avg_jam) else "N/A")
    m2.metric("Severe jam zones (Level 3)", severe_jams)
    m3.markdown("**Legend:** 🟢 (0) Free flow | 🟡 (1) Slowdown | 🟠 (2) Jam | 🔴 (3) Severe jam")

    # pydeck layer
    polygon_layer = pdk.Layer(
        "PolygonLayer",
        predict_df,
        get_polygon="coordinates",
        get_fill_color="color",
        get_line_color=[255, 255, 255, 60],
        line_width_min_pixels=1,
        pickable=True,
        auto_highlight=True
    )

    view_state = pdk.ViewState(
        latitude=41.1496,
        longitude=-8.6110,
        zoom=11.5
    )

    tooltip = {
        "html": "<b>Predicted jam level:</b> <b style='font-size: 16px'>{PREDICTED_JAM}</b> <br/>"
                "Junctions: {Area_junctions} <br/>"
                "Max speed: {Area_maxspeed} km/h",
        "style": {"backgroundColor": "#2c3e50", "color": "white", "font-family": "sans-serif"}
    }

    deck = pdk.Deck(
        layers=[polygon_layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style='dark'
    )   

    st.pydeck_chart(deck)