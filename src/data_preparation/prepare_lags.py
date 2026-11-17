import pandas as pd
import numpy as np

print("Loading full taxi data...")
df = pd.read_parquet('data/processed/taxi_prepared_full.parquet')

print("Calculating JAM_LEVEL exactly like in the training script...")
conditions = [
    (df['DEVIATION_RATIO'] < 1.05),                                     # 0: GREEN
    (df['DEVIATION_RATIO'] >= 1.05) & (df['DEVIATION_RATIO'] < 1.3),    # 1: YELLOW
    (df['DEVIATION_RATIO'] >= 1.3) & (df['DEVIATION_RATIO'] < 1.7),     # 2: ORANGE
    (df['DEVIATION_RATIO'] >= 1.7)                                      # 3: RED
]
choices = [0, 1, 2, 3]

df['JAM_LEVEL'] = np.select(conditions, choices, default=0)

print("Grouping and calculating historical lag features...")
lags_df = df.groupby(['AREA_ID', 'WEEKDAY', 'HOUR'])['JAM_LEVEL'].mean().reset_index()

output_path = 'data/processed/lag_history.parquet'
lags_df.to_parquet(output_path)

print(f"Success! Lag history saved to: {output_path}")