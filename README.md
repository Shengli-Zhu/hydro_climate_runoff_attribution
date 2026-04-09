# Quantifying and Attributing Climate Change Impacts on Runoff across Different Hydroclimatic Regimes

**Geo Modeling Final Project**

A comparative study of Saudi Arabia, Italy, and Bangladesh using ERA5-Land reanalysis data, XGBoost machine learning, and SHAP explainability framework.

## Study Areas

| Country | Climate | Annual Precip | Runoff Regime |
|---------|---------|---------------|---------------|
| Saudi Arabia | Extreme arid (BWh/BWk) | < 100 mm | Flash flood driven |
| Italy | Semi-arid to sub-humid (Csa/Cfa) | 500–1500 mm | Seasonal, strong N-S gradient |
| Bangladesh | Tropical monsoon (Am) | > 2000 mm | Monsoon driven |

## Project Structure

```
hydro_climate_runoff_attribution/
├── gee/
│   └── era5land_extraction.js       # GEE script: export monthly + annual GeoTIFF stacks
├── data/
│   ├── GEE_ERA5Land_Monthly/        # Downloaded GeoTIFFs (39 files, one per var/country)
│   ├── GEE_ERA5Land_Annual/         # Downloaded annual GeoTIFFs (15 files, for trend maps)
│   └── netcdf/                      # Converted NetCDF files (3 files, one per country)
├── scripts/
│   ├── 00_convert_to_netcdf.py      # One-time: merge GeoTIFFs -> compressed NetCDF
│   ├── utils_load.py                # Shared data loading utilities (used by 01-03)
│   ├── 01_water_balance.py          # Water balance analysis: P = ET + R + ΔS + ε
│   ├── 02_trend_analysis.py         # Mann-Kendall + Sen's slope (pixel-level + regional)
│   └── 03_xgboost_shap.py           # XGBoost regression + SHAP attribution
├── figures/                         # Output figures
├── results/                         # Output tables / CSV summaries
└── requirements.txt
```

## Workflow

### Step 1 — GEE Data Extraction

Open `gee/era5land_extraction.js` in the [GEE Code Editor](https://code.earthengine.google.com/).

First verify the data availability (run the `print` lines, check the console):

- `Total images` should be **312** (2000-01 to 2025-12)
- `system:index sample` should be `"200001"`

Then submit all export tasks (54 total) from the **Tasks** tab. Submit Bangladesh first (smallest), then Italy, then Saudi Arabia. Outputs go to two Google Drive folders:

- `GEE_ERA5Land_Monthly/` — 39 GeoTIFFs, 312 bands each (~1 GB)
- `GEE_ERA5Land_Annual/`  — 15 GeoTIFFs, 26 bands each (~80 MB)

### Step 2 — Download and Convert

Download the GeoTIFFs from Google Drive into:

- `data/GEE_ERA5Land_Monthly/` (monthly stacks)
- `data/GEE_ERA5Land_Annual/` (annual stacks)

Then run the one-time conversion to compressed NetCDF (~225 MB total):

```bash
D:/anaconda3/envs/hydroclimate/python.exe scripts/00_convert_to_netcdf.py
```

### Step 3 — Analysis

```bash
D:/anaconda3/envs/hydroclimate/python.exe scripts/01_water_balance.py
D:/anaconda3/envs/hydroclimate/python.exe scripts/02_trend_analysis.py
D:/anaconda3/envs/hydroclimate/python.exe scripts/03_xgboost_shap.py
```

## Data

- **Source**: ERA5-Land Monthly Aggregated (`ECMWF/ERA5_LAND/MONTHLY_AGGR`)
- **Resolution**: 0.1° (~11 km)
- **Period**: 2000-01 to 2025-12 (312 months)
- **Variables**: precipitation, evapotranspiration, surface/subsurface runoff, total soil water storage (4 layers), 2m temperature, dewpoint temperature, net radiation (SW/LW), wind speed, soil temperature, surface pressure

## Methods

- **Water Balance**: P = ET + R + ΔS + ε (monthly, pixel-level)
- **Trend Analysis**: Mann-Kendall test + Sen's slope (pixel-level spatial maps + regional time series)
- **Attribution**: XGBoost regression + SHAP (SHapley Additive exPlanations), trained per country on ~1–115 million pixel-month samples
