import numpy as np
import pandas as pd


def clean(df: pd.DataFrame, impute: bool = False) -> pd.DataFrame:
    # Calculate the percentage of missing values in each column
    missing_percentage = df.isnull().sum() / len(df) * 100

    # Identify columns with more than 90% missing values
    columns_to_drop = missing_percentage[missing_percentage > 90].index

    # Drop the columns with more than 90% missing values
    df_cleaned = df.drop(columns=columns_to_drop, errors='ignore')

    if impute:
        # Impute missing numerical values with the median
        numerical_cols = df_cleaned.select_dtypes(include=np.number).columns
        for col in numerical_cols:
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())

        # Impute missing categorical values with the mode
        categorical_cols = df_cleaned.select_dtypes(exclude=np.number).columns
        for col in categorical_cols:
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])

    return df_cleaned