import pandas as pd
        
def set_index(df: pd.DataFrame, index_col: str = 'COUNTYFP') -> None:
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
        merged_df = merged_df.merge(df, 
                                    left_index=True, 
                                    right_index=True, 
                                    how='outer')
    return merged_df

def cbsa_to_counties(
    crosswalk_df: pd.DataFrame
    ) -> dict[str, list[str]]:
    """Create dictionary mapping CBSA codes to lists of COUNTYFP codes"""
    cw = crosswalk_df[['cbsacode', 'fipscountycode']]
    cw = cw.rename(columns={'fipscountycode': 'COUNTYFP', 
                            'cbsacode': 'CBSAFP'})
    return (
        cw.groupby('CBSAFP')['COUNTYFP']
          .apply(list)
          .to_dict()
    )

def expand_cbsa_to_county(df: pd.DataFrame, 
                          cbsa_to_counties: dict) -> pd.DataFrame:
    df = df.copy()   
    df['COUNTYFP'] = df['CBSAFP'].map(cbsa_to_counties)
    df = df.explode('COUNTYFP').reset_index(drop=True)
    return df

def monthly_to_annual(df: pd.DataFrame, merge_col: str = 'COUNTYFP') -> pd.DataFrame:
    """Aggregate monthly data to annual by taking mean across months 
    for nonadditive variables,and sum across months for additive variables"""
    add_cols = [col for col in df.columns if '_sum' in col]
    non_add_cols = [col for col in df.columns if '_sum' not in col]
    
    grouped_df = df.groupby([merge_col, 'year'])
    
    anual = (grouped_df[add_cols].sum()
               .join(
                   grouped_df[non_add_cols].mean()
                   ))
    return anual

    
if __name__ == "__main__":
    pass
    