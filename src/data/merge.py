import pandas as pd

def set_index(df: pd.DataFrame, index_col: str = 'COUNTYFP') -> pd.DataFrame:
    """Set index of dataframe to specified geographic unit
    (default: 'COUNTYFP'), and 'year', 'month' for time-varying
    dataframes"""
    if 'year' in df.columns and 'month' in df.columns:
        df.set_index([index_col, 'year', 'month'], inplace=True) 
    else:
        df.set_index(index_col, inplace=True)
