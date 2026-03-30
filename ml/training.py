'''
What is needed?
Table with coulumns:
FROM TAXI DATA:
 - POLYLINE (I'm not sure but probably should be divided on rows, its hard to make model from this)
 - DAYTYPE (A, B, C) classification
    - A - normal day
    - B - holidays (e.g. Sunday, etc.)
    - C - trip started day before Bclass-day
 - PARTDAY (1, 2, 3, 4, 5) classification
    - 1 - morning (6:00 - 11:00)
    - 2 - midday (11:00 - 13:00)
    - 3 - afternoon (13:00 - 17:00)
    - 4 - evening (17:00 - 21:00)
    - 5 - night (21:00 - 6:00)
 - WEEKDAY (1-7)
 - MISSING_DATA (I don't really need this, but during transformation data we necessarily 
   drop TRUE_MISSING_DATA trips)
 - TIMESTAMP (It's a start date, so I think about divide into DAY,MONTH,YEAR)
    - DAY (numer of the day)
    - MONTH (number of the month)
    - YEAR (number of the year)
 - TAXI_ID (unique taxi driver id, that's meaningful for weird trajectory, to know whom)
 - TRIP_ID (just to know which trip is weird, but it is not required)
 - CALL_TYPE (1, 2, 3)
    - 1 - was dispatched from the central
    - 2 - was demanded directly to the taxi driver on a specific stand  
    - 3 - from random street

FROM WEATHER DATA:
 - WEATHERTYPE (pointed some different types)
    - 1 - SUNNY
    - 2 - WINDY
    - 3 - CLOUDY
    - 4 - RAINY
    - 5 - STORMY
 - SEASON (it could be useful to know which months are exactly in that season - 1,2,3,4)
    - 1 - SPRING
    - 2 - SUMMER
    - 3 - AUTUMN
    - 4 - WINTER

FROM EVENTS DATA:
 - POLYLINE (place, where)
 - DATE (of course divided into DAY, MONTH, YEAR)
 - RANGE (how crowded and important it was)
    - 1 - SMALL IMPACT ON
    - 2 - MEDIUM IMPACT ON
    - 3 - HIGH IMPACT ON

SOME INSTRUCTIONS TO POLYLINE IN TAXI DATA:
We can use explode() function to calculate ACTUAL_DIST etc. 
There are lists in string. We have to make a START_LAT, START_LON, END_LAT, END_LON,
ACTUAL_DIST, OPTIMAL_DIST and DEVIATION_RATIO (ACT/OPT) in type possible to read for computer.
Then during learning we just use this COLS.

I know that's much but for such a accurately model it's needed.
We can skip some points of this note.

INSTRUCTIONS FOR ML:
 - use XGBoost to make effective decision trees. LEARN NEEDED.
'''

from sklearn.model_selection import train_test_split

#X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=60)

