from src.config import RAW_DATA_DIR, COUNTY_DATA_DIR, SATHEALTH_DATA_DIR
import pandas as pd

# Load county-level data
airquality = pd.read_csv(COUNTY_DATA_DIR / 'airquality.csv')
climate = pd.read_csv(COUNTY_DATA_DIR / 'climate.csv')
greenery = pd.read_csv(COUNTY_DATA_DIR / 'greenery.csv')
landcover = pd.read_csv(COUNTY_DATA_DIR / 'landcover.csv')
sdi = pd.read_csv(COUNTY_DATA_DIR / 'sdi.csv')

# Load ICD data
icd1 = pd.read_csv(SATHEALTH_DATA_DIR / 'icdl1_prev_ohio.csv')
icd2 = pd.read_csv(SATHEALTH_DATA_DIR / 'icdl2_prev_ohio.csv')
icd3 = pd.read_csv(SATHEALTH_DATA_DIR / 'icdl3_prev_ohio.csv')

# Load CBSA to FIPS mapping
cbsa_fips = pd.read_csv(RAW_DATA_DIR / 'cbsa_to_fips.csv')
cbsa_county_mapping = cbsa_fips[['cbsacode', 'fipscountycode']] .rename(columns={'fipscountycode': 'COUNTYFP'})

# Set indices 
time_varying_dfs = [airquality, climate, greenery]
static_dfs = [landcover, sdi]

for df in time_varying_dfs:
    df.set_index(['COUNTYFP', 'year', 'month'], inplace=True)
    
for df in static_dfs:
    df.set_index('COUNTYFP', inplace=True)

# =========================================================================
if __name__ == "__main__":
    print("Data loaded successfully!")
# =========================================================================
