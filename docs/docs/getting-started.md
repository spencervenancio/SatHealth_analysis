Getting started
===============


## Downloading the Data

Download the [SatHealth DataSet](https://aimed-sathealth.net/). It should be named `sathealth_dataset/` and placed in `data/raw/`.

Download the `.csv` version of the CBSA to FIP Crosswalk from the [National Bureau of Economic Research](https://www.nber.org/research/data/census-core-based-statistical-area-cbsa-federal-information-processing-series-fips-county-crosswalk). Rename the file `cbsa_to_fips.csv`, and place it in `data/external/`

## Installing the Dependencies

Install dependencies by running in your terminal:

```bash
$ make requirements
$ conda activate SatHealth
```

## Building the Dataset

```bash
$ make data
```
--------