import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from scipy import stats


def remove_selected_columns(df, columns_remove):
    """Remove selected columns safely."""
    return df.drop(columns=columns_remove, errors='ignore')


def remove_rows_with_missing_data(df, columns):
    """Remove rows with missing data. Always return df."""
    if columns:
        df = df.dropna(subset=columns)
    return df


def fill_missing_data(df, columns, method):
    """Fill missing data using mean/median/mode."""
    for column in columns:
        if method == 'mean':
            df[column].fillna(df[column].mean(), inplace=True)
        elif method == 'median':
            df[column].fillna(df[column].median(), inplace=True)
        elif method == 'mode':
            df[column].fillna(df[column].mode().iloc[0], inplace=True)
    return df


def one_hot_encode(df, columns):
    """Perform One-Hot Encoding."""
    return pd.get_dummies(df, columns=columns, prefix=columns, drop_first=False)


def label_encode(df, columns):
    """Perform Label Encoding."""
    le = LabelEncoder()
    for col in columns:
        df[col] = le.fit_transform(df[col].astype(str))
    return df


def standard_scale(df, columns):
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df


def min_max_scale(df, columns, feature_range=(0, 1)):
    scaler = MinMaxScaler(feature_range=feature_range)
    df[columns] = scaler.fit_transform(df[columns])
    return df


def detect_outliers_iqr(df, column_name):
    """Detect outliers using IQR."""
    data = df[column_name]
    q25, q50, q75 = np.percentile(data, [25, 50, 75])
    iqr = q75 - q25
    lower = q25 - 1.5 * iqr
    upper = q75 + 1.5 * iqr
    outliers = sorted([x for x in data if x < lower or x > upper])
    return outliers


def detect_outliers_zscore(df, column_name, threshold=3):
    """Detect outliers using Z-Score."""
    data = df[column_name]
    z = np.abs(stats.zscore(data))
    return [data[i] for i in range(len(data)) if z[i] > threshold]


def remove_outliers(df, column_name, outliers):
    """Remove rows that contain outlier values."""
    return df[~df[column_name].isin(outliers)]


def transform_outliers(df, column_name, outliers):
    """Replace outliers with median of non-outliers."""
    non_outliers = df[~df[column_name].isin(outliers)]
    median_value = non_outliers[column_name].median()
    df.loc[df[column_name].isin(outliers), column_name] = median_value
    return df
