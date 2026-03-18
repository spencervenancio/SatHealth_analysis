import pandas as pd
from src.data.loaders import load_fips_crosswalk, load_icd

        
def set_index(df: pd.DataFrame, index_col: str = 'COUNTYFP') -> pd.DataFrame:
    """Set index of dataframe to specified geographic unit
    (default: 'COUNTYFP'), and 'year', 'month' for time-varying
    dataframes"""
    if 'year' in df.columns and 'month' in df.columns:
        return df.set_index([index_col, 'year', 'month'])
    return df.set_index(index_col)

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

def expand_cbsa_to_county(df: pd.DataFrame, cbsa_to_counties: dict) -> pd.DataFrame:
    df = df.copy()   
    df['COUNTYFP'] = df.index.map(cbsa_to_counties)
    df = df.explode('COUNTYFP')
    return df
    
if __name__ == "__main__":
    # Example usage
    crosswalk_df = load_fips_crosswalk()
    cbsa_county_mapping = cbsa_to_counties(crosswalk_df)   
    icd1 = load_icd(1)
    icd1 = icd1.set_index('CBSAFP')
    icd1_expanded = expand_cbsa_to_county(icd1, cbsa_county_mapping))
    print(icd1_expanded.info())
    