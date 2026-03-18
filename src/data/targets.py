import pandas as pd

def create_target(
    icd_df: pd.DataFrame, 
    code: str = 'F32', # ICD code for depression 
    code_str: str = 'depr', 
    index: str | list[str] = ['COUNTYFP', 'year']
    ) -> pd.DataFrame:
    target_col = f"{code_str}_prev"
    df = icd_df[icd_df['code'] == code][[
        'year', 'prevalence'
    ]]
    df = df.rename(columns={'prevalence': target_col}).reset_index()
    return df.set_index(index)