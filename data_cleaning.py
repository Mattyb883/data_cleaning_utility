import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import stats

def clean_dataset(df, 
                  missing_threshold=0.5, 
                  scaling_method=None, 
                  outlier_method=None, 
                  encode_categoricals=True, 
                  date_columns=None, 
                  max_unique_for_encoding=10):
    """
    Clean a dataset by applying standard preprocessing steps.
    Parameters:
        df (pd.DataFrame): DataFrame to clean.
        missing_threshold (float): Drop columns with missing % over this threshold.
        scaling_method (str): 'standard' or 'minmax' for scaling numerical data; None skips scaling.
        outlier_method (str): 'IQR' or 'zscore' for handling outliers; None skips outlier handling.
        encode_categoricals (bool): If True, applies one-hot encoding to categorical columns.
        date_columns (list): List of columns to parse as dates for date-based feature extraction. If None, tries to auto-detect.
        max_unique_for_encoding (int): Maximum unique values in a column for one-hot encoding. Columns exceeding this are skipped.
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    
    # Lowercase column headers for consistency
    df.columns = df.columns.str.lower()
    
    # Auto-detect date columns if not provided
    if date_columns is None:
        date_columns = [col for col in df.columns if 'date' in col.lower()]

    # Drop columns exceeding the missing data threshold
    df = df.dropna(thresh=len(df) * (1 - missing_threshold), axis=1)

    # Fill missing values
    df = df.apply(lambda col: col.fillna(col.mode()[0]) if col.dtype == 'object' else col.fillna(col.median()))
    
    # Parse and format specified date columns to ISO 8601 format (YYYY-MM-DD)
    for date_col in date_columns:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce', utc=True).dt.strftime('%Y-%m-%d')
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Standardize text data, applying only to object (string) columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.lower().str.strip()
    
    # Handle outliers if specified, only for numeric columns
    if outlier_method == 'IQR':
        num_cols = df.select_dtypes(include=np.number)
        Q1 = num_cols.quantile(0.25)
        Q3 = num_cols.quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((num_cols < (Q1 - 1.5 * IQR)) | (num_cols > (Q3 + 1.5 * IQR))).any(axis=1)]
    elif outlier_method == 'zscore':
        num_cols = df.select_dtypes(include=np.number)
        df = df[(np.abs(stats.zscore(num_cols)) < 3).all(axis=1)]

    # Scale numerical data if specified
    if scaling_method:
        scaler = StandardScaler() if scaling_method == 'standard' else MinMaxScaler()
        num_cols = df.select_dtypes(include=np.number).columns
        df[num_cols] = scaler.fit_transform(df[num_cols])

    # Encode categorical data if specified
    if encode_categoricals:
        # Limit encoding to columns with unique values below the threshold
        categorical_cols = [col for col in df.select_dtypes(include='object').columns 
                            if df[col].nunique() <= max_unique_for_encoding]
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    return df
