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
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, classification_report
import shap
import joblib
import matplotlib.pyplot as plt

try:
    #load data
    print("Loading data...")
    taxi = pd.read_parquet('data/data_samples/taxi_100k_prepared.parquet')
    weather = pd.read_parquet('data/processed/weather.parquet')
    traffic = pd.read_parquet('data/processed/traffic.parquet')
    area = pd.read_parquet('data/processed/area.parquet')
except Exception as e:
    print("Error: ", e)
    raise e

#data preparation
#merge data
print("Merging data...")
data = taxi.merge(weather, on=['YEAR', 'MONTH', 'DAY', 'HOUR'], how='left')
data = data.merge(traffic, on='AREA_ID', how='left')
data = data.merge(area, on='AREA_ID', how='left')

#sort data 
data['AREA_ID'] = data['AREA_ID'].astype('category')
data.sort_values(by=['YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE'], inplace=True)

#print(data.columns)

DEBUG_MODE = 1
SEARCHING_BEST = 2
#0 for binary classification
#1 for multi classification
if DEBUG_MODE == 0:
    #TRAINING MODEL FOR BINARY CLASSIFICATION
    print("Training model for binary classification...")
    #add target variable
    max_speed = data['Area_maxspeed'].fillna(50).astype(float)
    data['IS_JAM'] = (data['SPEED'] < (0.3 * max_speed)).astype(int)
    print(data['IS_JAM'].value_counts(normalize=True))
    
    #drop unnecessary columns
    #print(data.columns)
    cols_to_drop = ['TRIP_ID', 'POLYLINE', 'TIMEZONE', 'DEVIATION_RATIO', 'geometry', 'POLYGON', 'TIME_IN_AREA', 'DIST_KM_IN_AREA', 'SPEED']
    data.drop(columns=cols_to_drop, errors='ignore', inplace=True)
    
    #split data into X and y
    X = data.drop(columns=['IS_JAM'], errors='ignore')
    y = data['IS_JAM']

    #Checking IS_JAM Ratio 0/1 to find scale_pos_weight
    zeros = data['IS_JAM'].value_counts()[0]
    ones = data['IS_JAM'].value_counts()[1]
    scale_pos_weight = zeros / ones # that param should be roughly equal to ratio in params
    print(f"Scale_pos_weight: {scale_pos_weight:.2f}")
    #TimeSeriesSplit 
    tscv = TimeSeriesSplit(n_splits=5, gap=2)

    #split into training and test data
    split_idx = int(len(data) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    if SEARCHING_BEST:
        #This code is for finding best parameters for XGBoost
        print("Searching for best parameters...")
        #model for XGBoost
        xgb_model_binary = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=60,
            enable_categorical=True,
            tree_method="hist",
            n_jobs=-1
        )

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
            estimator=xgb_model_binary,
            param_distributions=params,
            n_iter=20,
            scoring='roc_auc',
            cv=tscv,
            n_jobs=-1,
            verbose=2
        )

        #fit RandomizedSearchCV
        random_search.fit(X_train, y_train)

        #best parameters with scores
        print("Best parameters:", random_search.best_params_)
        print("Best score:", random_search.best_score_)

        #best model
        best_model = random_search.best_estimator_

        #plot of importance of features
        xgb.plot_importance(best_model, max_num_features=10)
        plt.show()

        #plot of shap values (after whole dataset)
        X_sample = X_test.sample(n=5000, random_state=42)
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_sample)
        shap.summary_plot(shap_values, X_sample)

        #AIC, RMSE, R^2

    else:
        print("Using predefined parameters")
    #FINAL MODEL FOR BINARY CLASSIFICATION
    #params for final model
    params = {
        'subsample': 0.6,
        'scale_pos_weight': 3.5,
        'reg_lambda': 0.4,
        'reg_alpha': 0.0,
        'n_estimators': 500,
        'max_depth': 9,
        'learning_rate': 0.01,
        'gamma': 1.75,
        'colsample_bytree': 0.6
    }

    #train the final model using best params
    final_model_binary = xgb.XGBClassifier(
        **params, 
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=60,
        enable_categorical=True,
        tree_method="hist",
        n_jobs=-1
    )

    print("Fitting the final model...")
    #fit the model
    final_model_binary.fit(X_train, y_train)

    #saving the model
    print("Saving the model...")
    joblib.dump(final_model_binary, 'data/processed/xgboost_traffic_model_binary.pkl')

    #predictions
    y_pred = final_model_binary.predict(X_test)

    #classification report
    print("Classification report:")
    print(classification_report(y_test, y_pred))

    #confusion matrix
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

elif DEBUG_MODE == 1:
    #TRAINING MODEL FOR MULTICLASSIFICATION
    print("Training model for multi classification...")
    cols_to_drop = [
        'TRIP_ID', 'POLYLINE', 'TIMEZONE', 'DEVIATION_RATIO', 
        'geometry', 'POLYGON', 'TIME_IN_AREA', 'DIST_KM_IN_AREA', 
        'SPEED', 'IS_JAM'
    ]

    #I decided to create 3 levels of jams, because for 4 the model isn't as acurately as we want.
    conditions = [
        (data['DEVIATION_RATIO'] < 1.3),
        (data['DEVIATION_RATIO'] >= 1.3) & (data['DEVIATION_RATIO'] < 30),
        (data['DEVIATION_RATIO'] >= 3.0)
    ]
    choices = [0, 1, 2]
    data['JAM_LEVEL'] = np.select(conditions, choices, default=0)

    X = data.drop(columns=cols_to_drop + ['JAM_LEVEL'], errors='ignore')
    y = data['JAM_LEVEL']

    #params to multiclassification
    final_params = {
        'max_depth': 9,
        'n_estimators': 500,
        'learning_rate': 0.01,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'objective': 'multi:softprob',
        'num_class': 3
    }

    final_model_heatmap = xgb.XGBClassifier(
        **final_params,
        eval_metric="mlogloss",
        random_state=60,
        enable_categorical=True,
        tree_method="hist",
        n_jobs=-1
    )

    #split into training and test data
    split_idx = int(len(data) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    #changing weights for better learning of model, because we have imbalance in classes
    weights = {
        0: 1,
        1: 9.3,
        2: 137
    }
    sample_weights = compute_sample_weight(class_weight=weights, y=y_train)
    print("Fitting the final model...")
    final_model_heatmap.fit(X_train, y_train, sample_weight=sample_weights)

    #saving the model
    print("Saving the model...")
    joblib.dump(final_model_heatmap, 'data/processed/xgboost_traffic_model_heatmap.pkl')

    #predictions
    y_pred = final_model_heatmap.predict(X_test)

    #classification report
    print("Classification report:")
    print(classification_report(y_test, y_pred))

    #confusion matrix
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

else:
    print("Wrong DEBUG_MODE value. Type 0 or 1.")
