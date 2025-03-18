import numpy as np
import pandas as pd

def detect_continuous_columns(df):
    continuous_columns = []
    for column in df.columns:
        unique_values = df[column].nunique()
        total_values = len(df[column])

        if unique_values / total_values > 0.5:
            continuous_columns.append(column)
    return continuous_columns

def detect_columns_type(df,threshold=10):
    categorical_columns = []
    num_columns = df.describe().columns.tolist()
    continuous_columns = list(num_columns)
    for column in num_columns :
        unique_values = len(df[column].unique())
        if unique_values < threshold:
            continuous_columns.remove(column)
            
    categorical_columns = list(set(df.columns)-set(continuous_columns))
    return continuous_columns,categorical_columns