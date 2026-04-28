'''
This file contains the code for training the traffic prediction model. 
I decided to use the XGBoost library, specifically the XGBClassifier, which performs exceptionally well on this type of tabular data.
Because we are dealing with time-series data, I use TimeSeriesSplit for cross-validation (if tuning).
I added two additional features (lag features) to the training set: JAM_LEVEL_1H_AGO and JAM_LEVEL_1D_AGO.
Instead of using shift(), which shifts by rows and ignores spatial/temporal gaps, I implemented a robust self-merge using 'timestamp' and 'AREA_ID'.

Dataset columns description:
taxi - TRIP_ID(int) PK, YEAR(int), MONTH(int), DAY(int), HOUR(int), MINUTE(int), SECOND(int), TIMEZONE(string), WEEKDAY(int), PARTDAY(int), POLYLINE(object), TIME_IN_AREA(int), DIST_KM_IN_AREA(float), DEVIATION_RATIO(float), AREA_ID(int) PK, SPEED(float)
area - AREA_ID(int) PK, POLYLINE(object)
weather - YEAR, MONTH, DAY, HOUR, TIMEZONE (PKs), PRECIPITATION(int), WIND_GUSTS_10M(int), IS_DAY(int), WEATHER_CODE(int), TEMPERATURE_2M(int)
traffic - AREA_ID(int) PK, AREA_AVG_LANES(int), AREA_DOMINANT_MAXSPEED(int), AREA_JUNCTIONS_COUNT(int), AREA_TRAFFIC_LIGHTS_COUNT(int)

Steps:
1. Load data into variables.
2. Merge datasets into one dataframe using LEFT JOIN.
3. Convert AREA_ID into a categorical type and sort data chronologically.
4. Add target variable JAM_LEVEL (4 classes based on DEVIATION_RATIO).
5. Calculate historical JAM_LEVEL (1 hour ago, 1 day ago) via self-merge.
6. Drop unnecessary columns before training.
7. Split data into X and y.
8. Define custom class weights to handle severe class imbalance.
9. Train the XGBoost model with optimized parameters for 4-class classification.
10. Evaluate the model and save it to a .pkl file.
'''

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import matplotlib.pyplot as plt

try:
    # Load data
    print("Loading data...")
    taxi = pd.read_parquet('data/data_samples/taxi_100k_prepared.parquet')
    weather = pd.read_parquet('data/processed/weather.parquet')
    traffic = pd.read_parquet('data/processed/traffic.parquet')
    area = pd.read_parquet('data/processed/area.parquet')
except Exception as e:
    print("Error loading data: ", e)
    raise e

# --- DATA PREPARATION ---
print("Merging datasets...")
data = taxi.merge(weather, on=['YEAR', 'MONTH', 'DAY', 'HOUR'], how='left')
data = data.merge(traffic, on='AREA_ID', how='left')
data = data.merge(area, on='AREA_ID', how='left')

# Sort data chronologically and set AREA_ID as category
data['AREA_ID'] = data['AREA_ID'].astype('category')
data.sort_values(by=['YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE', 'SECOND'], inplace=True)

print("Preparing target variable and lag features for 4-class classification...")

# --- TARGET VARIABLE CREATION (4 CLASSES) ---
conditions = [
    (data['DEVIATION_RATIO'] < 1.05),                                     # 0: GREEN (Free flow)
    (data['DEVIATION_RATIO'] >= 1.05) & (data['DEVIATION_RATIO'] < 1.3),  # 1: YELLOW (Slowdown)
    (data['DEVIATION_RATIO'] >= 1.3) & (data['DEVIATION_RATIO'] < 1.7),   # 2: ORANGE (Traffic Jam)
    (data['DEVIATION_RATIO'] >= 1.7)                                      # 3: RED (Severe Jam / Paralysis)
]
choices = [0, 1, 2, 3]
data['JAM_LEVEL'] = np.select(conditions, choices, default=0)

# --- FEATURE ENGINEERING: HISTORICAL LAGS ---
# 1. Create a helper DataFrame with historical data (median jam level per area per hour)
history = data.groupby(['YEAR', 'MONTH', 'DAY', 'HOUR', 'AREA_ID'])['JAM_LEVEL'].agg('median').reset_index()

# 2. Create a clean timestamp column for accurate time shifting
history['timestamp'] = pd.to_datetime(history[['YEAR', 'MONTH', 'DAY', 'HOUR']].assign(minute=0))

# 3. Prepare feature: 1 HOUR AGO
history_1h = history[['AREA_ID', 'timestamp', 'JAM_LEVEL']].copy()
history_1h['timestamp'] = history_1h['timestamp'] + pd.Timedelta(hours=1) # Shift time FORWARD to match future rows
history_1h.rename(columns={'JAM_LEVEL': 'JAM_LEVEL_1H_AGO'}, inplace=True)

# 4. Prepare feature: 1 DAY AGO
history_1d = history[['AREA_ID', 'timestamp', 'JAM_LEVEL']].copy()
history_1d['timestamp'] = history_1d['timestamp'] + pd.Timedelta(days=1) 
history_1d.rename(columns={'JAM_LEVEL': 'JAM_LEVEL_1D_AGO'}, inplace=True)

# 5. Merge historical features back into the main dataset
data['timestamp'] = pd.to_datetime(data[['YEAR', 'MONTH', 'DAY', 'HOUR']].assign(minute=0))
data = data.merge(history_1h, on=['AREA_ID', 'timestamp'], how='left')
data = data.merge(history_1d, on=['AREA_ID', 'timestamp'], how='left')

# 6. Crucial XGBoost step: Leave NaNs as floats (do NOT fill with 0) so the model learns from missing data
data['JAM_LEVEL_1H_AGO'] = data['JAM_LEVEL_1H_AGO'].astype(float)
data['JAM_LEVEL_1D_AGO'] = data['JAM_LEVEL_1D_AGO'].astype(float)

# --- DATA SPLITTING ---
print("Splitting data...")
cols_to_drop = [
    'TRIP_ID', 'POLYLINE', 'TIMEZONE', 'DEVIATION_RATIO', 
    'geometry', 'POLYGON', 'TIME_IN_AREA', 'DIST_KM_IN_AREA', 
    'SPEED', 'IS_JAM', 'JAM_LEVEL', 'timestamp', 'SECOND'
]

X = data.drop(columns=[c for c in cols_to_drop if c in data.columns], errors='ignore')
y = data['JAM_LEVEL']

# Chronological split (80% train, 20% test)
split_idx = int(len(data) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# --- MODEL TRAINING ---
# Custom weights (stepped) to handle imbalance without making the model panic
custom_weights = {
    0: 1.0,   # Free flow (baseline)
    1: 1.5,   # Slowdown
    2: 3.5,   # Traffic Jam
    3: 7.0    # Severe Jam 
}
sample_weights = y_train.map(custom_weights)

# Parameters tailored for 4-class classification to prevent overfitting
final_params = {
    'max_depth': 7,              # Shallower trees to generalize better
    'min_child_weight': 10,      # Higher threshold of evidence required for leaves
    'gamma': 1.0,                # Prunes branches that don't improve the loss
    'n_estimators': 500,
    'learning_rate': 0.01,
    'subsample': 0.6,
    'colsample_bytree': 0.6,
    'objective': 'multi:softprob',
    'num_class': 4               
}

final_model_heatmap = xgb.XGBClassifier(
    **final_params,
    eval_metric="mlogloss",
    random_state=60,
    enable_categorical=True,
    tree_method="hist",
    n_jobs=-1
)

print("Fitting the final model with custom class weights...")
final_model_heatmap.fit(X_train, y_train, sample_weight=sample_weights)

print("Saving the model...")
joblib.dump(final_model_heatmap, 'data/processed/xgboost_traffic_model_heatmap.pkl')

# --- EVALUATION ---
print("\n--- MODEL EVALUATION ---")
y_pred = final_model_heatmap.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print(f"\nAccuracy Score: {final_model_heatmap.score(X_test, y_test):.4f}")