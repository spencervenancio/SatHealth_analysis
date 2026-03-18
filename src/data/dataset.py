from pathlib import Path

from loguru import logger
import typer

from src.config import PROCESSED_DATA_DIR
from src.data.loaders import load_airquality, load_climate, load_fips_crosswalk, load_greenery, load_icd, load_landcover, load_sdi
from src.data.merge import cbsa_to_counties, expand_cbsa_to_county, merge_datasets, monthly_to_annual, set_index
from src.data.targets import create_target

app = typer.Typer()


@app.command()
def main(
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
):
    
    # Load datasets
    aq = load_airquality()
    cl = load_climate()
    gr = load_greenery()
    lc = load_landcover()
    sdi = load_sdi()
    icd3 = load_icd(3)
    cw = load_fips_crosswalk()
    dfs = [aq, cl, gr, lc, sdi]
    
    # Add COUNTYFP to ICD3
    cbsa_county_mapping = cbsa_to_counties(cw)
    icd3_expanded = expand_cbsa_to_county(icd3, cbsa_county_mapping)
    dfs.append(icd3_expanded)
    
    for df in dfs:
        set_index(df)
    
    # Split into time-varying and static datasets
    time_varying_dfs = [aq, cl, gr]
    static_dfs = [lc, sdi]

    time_varying_df = merge_datasets(time_varying_dfs)
    static_df = merge_datasets(static_dfs)
    
    # Aggregate time-varying data to annual level
    annual_tvdf = monthly_to_annual(time_varying_df)
    
    # Create target variable
    target = create_target(icd3_expanded)
    
    # Merge target with annual time-varying data    
    annual_and_target = annual_tvdf.merge(target, left_index=True, right_index=True, how='inner')
    
    # Merge with static data
    final_df = annual_and_target.merge(static_df, left_index=True, right_index=True, how='left')

    final_df.to_csv(output_path)

    logger.success("Processing dataset complete.")
    
    
if __name__ == "__main__":
    app()
