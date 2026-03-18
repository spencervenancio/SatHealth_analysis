import pandas as pd
from src.data.loaders import load_fips_crosswalk

def set_index(df: pd.DataFrame, index_col: str = 'COUNTYFP') -> pd.DataFrame:
    """Set index of dataframe to specified geographic unit
    (default: 'COUNTYFP'), and 'year', 'month' for time-varying
    dataframes"""
    if 'year' in df.columns and 'month' in df.columns:
        df.set_index([index_col, 'year', 'month'], inplace=True) 
    else:
        df.set_index(index_col, inplace=True)

def merge_datasets(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge list of dataframes on their indices using outer join"""
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = merged_df.merge(df, left_index=True, right_index=True, how='outer')
    return merged_df

def cbsa_to_counties(crosswalk_df: pd.DataFrame) -> dict[str, list[str]]:
    """Create dictionary mapping CBSA codes to lists of COUNTYFP codes"""
    cw = crosswalk_df[['cbsacode', 'fipscountycode']].rename(columns={'fipscountycode': 'COUNTYFP', 'cbsacode': 'CBSAFP'})
    return (
        cw.groupby('CBSAFP')['COUNTYFP']
          .apply(list)
          .to_dict()
    )
    
if __name__ == "__main__":
    pass