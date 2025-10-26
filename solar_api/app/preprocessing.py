# app/preprocessing.py
import pandas as pd
import numpy as np

def preprocess_new_data(df_raw: pd.DataFrame):
    df = df_raw.copy()

    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].ffill().bfill()

    for col in num_cols:
        if (df[col] < 0).any():
            df[col] = np.clip(df[col], 0, None)

    return df
