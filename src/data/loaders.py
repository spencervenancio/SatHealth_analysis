from src.config import RAW_DATA_DIR, COUNTY_DATA_DIR, SATHEALTH_DATA_DIR
import pandas as pd

# Load county-level data
def load_airquality():
    return pd.read_csv(COUNTY_DATA_DIR / 'airquality.csv')

def load_climate():
    return pd.read_csv(COUNTY_DATA_DIR / 'climate.csv')

def load_greenery():
    return pd.read_csv(COUNTY_DATA_DIR / 'greenery.csv')

def load_landcover():
    return pd.read_csv(COUNTY_DATA_DIR / 'landcover.csv')

def load_sdi():
    return pd.read_csv(COUNTY_DATA_DIR / 'sdi.csv')

# Load ICD data
def load_icd(level: int):
    if level == 1:
        return pd.read_csv(SATHEALTH_DATA_DIR / 'icdl1_prev_ohio.csv')
    elif level == 2:
        return pd.read_csv(SATHEALTH_DATA_DIR / 'icdl2_prev_ohio.csv')
    elif level == 3:
        return pd.read_csv(SATHEALTH_DATA_DIR / 'icdl3_prev_ohio.csv')

# Load crosswalk data
def load_fips_crosswalk():
    return pd.read_csv(RAW_DATA_DIR / 'cbsa_to_fips.csv')


