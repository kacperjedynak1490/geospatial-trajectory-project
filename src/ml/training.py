'''
That file contains the code for training model. I decided to use XGBooost library, 
which have such a good function to classifiacation: XGBClassifier. I also use RandimizedSearchCV
to find the best parameters for model. At first the training should be on the sampled dataset to find
parameters, and then to the whole dataset. Because we have datetime date, I use TimeSeriesSplit
for cross-validation. I added two additional columns to training. This is IS_JAM_1H_AGO and IS_JAM_1D_AGO.
But the main problem with is that the shift() atribute is not working correctly, because
it shift by rows and not by hours. I will try to fix it.

Here is the description of columns in datasets:
taxi - 
TRIP_ID(int), - PK
YEAR(int), 
MONTH(int), 
DAY(int), 
HOUR(int), 
MINUTE(int), 
TIMEZONE(string),
SECOND(int), 
WEEKDAY(int), 
PARTDAY(int), 
POLYLINE(object), 
TIME_IN_AREA(int), 
DIST_KM_IN_AREA(float), 
DEVIATION_RATIO(float), 
AREA_ID(int), - PK
SPEED(float);

area - 
AREA_ID(int), - PK
POLYLINE(object)

weather -
YEAR(int), - PK
MONTH(int), - PK
DAY(int), -PK
TIMEZONE(string), -PK
HOUR(int), -PK
PRECIPITATION(int), 
WIND_GUSTS_10M(int), 
IS_DAY(int), 
WEATHER_CODE(int), 
TEMPERATURE_2M(int)

traffic - 
AREA_ID(int), - PK
AREA_AVG_LANES(int), 
AREA_DOMINANT_MAXSPEED(int), 
AREA_JUNCTIONS_COUNT(splitted into 3 types)(int), 
AREA_TRAFFIC_LIGHTS_COUNT(int)

X = without IS_JAM, DEVIATION_RATIO, TRIP_ID, POLYLINE, TIMEZONE
y = IS_JAM

Steps:
1. Load data to variables.
2. Merge data to one dataframe by particular columns (LEFT JOIN).
3. Then change AREA_ID into category and sort data by datetime.
4. Add target variable IS_JAM, which is 1 if DEVIATION_RATIO < 0.6 (maybe) and otheriwse 0.
5. Calculation of IS_JAM in the past.
6. Drop unnecesary columns for training.
7. Split data into X and y.
8. TimeSeriesSplit for RandomizedSearchCV.
9. Model for XGBoost XGBClassifier with parameters. (I will try find the best for this data).
10. Parameters for RandomizedSearchCV. (It should be fixed).
11.RandomizedSearchCV with parameters.
12. Fit RandomizedSearchCV.
13. Find best params and best score.
14. Find best model and plot importance of features. (Its likely to our next step - explain what have influence
on making jam in the area).
'''
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, classification_report
import shap
import matplotlib.pyplot as plt

#load data
taxi = pd.read_parquet('data/processed/taxi.parquet')
weather = pd.read_parquet('data/processed/weather.parquet')
traffic = pd.read_parquet('data/processed/traffic.parquet')
area = pd.read_parquet('data/processed/area.parquet')

#merge data
data = taxi.merge(weather, on=['YEAR', 'MONTH', 'DAY', 'HOUR', 'TIMEZONE'], how='left')
data = data.merge(traffic, on='AREA_ID', how='left')
data = data.merge(area, on='AREA_ID', how='left')

#sort data 
data['AREA_ID'] = data['AREA_ID'].astype('category')
data.sort_values(by=['YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE'], inplace=True)

#add target variable
data['IS_JAM'] = (data['DEVIATION_RATIO'] < 0.6).astype(int)

#calculate IS_JAM in the same AREA_ID one hour, one day ago
data['IS_JAM_1H_AGO'] = data.groupby('AREA_ID')['IS_JAM'].shift(1)
data['IS_JAM_1D_AGO'] = data.groupby('AREA_ID')['IS_JAM'].shift(24)

#drop unnecessary columns
print(data.columns)
data.drop(columns=['TRIP_ID', 'POLYLINE', 'TIMEZONE', 'DEVIATION_RATIO'], inplace=True)

#split data into X and y
X = data.drop(columns=['IS_JAM'])
y = data['IS_JAM']

#TimeSeriesSplit for RandomizedSearchCV
tscv = TimeSeriesSplit(n_splits=5, gap=2)

#model for XGBoost
xgb_model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=60,
    enable_categorical=True,
    tree_method="hist",
    n_jobs=-1
)

#Checking IS_JAM Ratio 0/1 to find scale_pos_weight
zeros = data['IS_JAM'].value_counts()[0]
ones = data['IS_JAM'].value_counts()[1]
scale_pos_weight = zeros / ones # that param should be roughly equal to ratio in params

#parameters for RandomizedSearchCV
params = {
    'n_estimators': np.arange(100, 1000, 100),
    'learning_rate': np.arange(0.01, 0.3, 0.01),
    'max_depth': np.arange(2, 8, 1),
    'subsample': np.arange(0.6, 0.9, 0.1),
    'colsample_bytree': np.arange(0.6, 1, 0.1),
    'gamma': np.arange(0, 2, 0.05),
    'scale_pos_weight': np.arange(1, 10, 1),
    'reg_alpha': np.arange(0, 1, 0.1),
    'reg_lambda': np.arange(0, 1, 0.1) 
}

#RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=params,
    n_iter=20,
    scoring='roc_auc',
    cv=tscv,
    n_jobs=-1,
    verbose=2
)

#fit RandomizedSearchCV
random_search.fit(X, y)

#best parameters with scores
print("Best parameters:", random_search.best_params_)
print("Best score:", random_search.best_score_)

#best model
best_model = random_search.best_estimator_

#plot of importance of features
xgb.plot_importance(best_model, max_num_features=10)
plt.show()

#plot of shap values (after whole dataset)
#explainer = shap.TreeExplainer(best_model)
#shap_values = explainer.shap_values(X)
#shap.summary_plot(shap_values, X)

#git restore --source origin/data_preparing -- ścieżka/do/pliku/data_preparing.py