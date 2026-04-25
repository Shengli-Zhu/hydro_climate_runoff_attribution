# Climate Controls on Runoff Across Hydroclimatic Regimes

**Geo-Environmental Modeling and Analysis Final Project | KAUST**

A comparative study of Saudi Arabia (arid), Italy (transitional), and Bangladesh (humid) using ERA5-Land reanalysis data (1950–2025), XGBoost machine learning, and SHAP explainability framework.

## Study Areas

| Country | Climate (Köppen) | Annual P (ERA5) | R/P | XGBoost R² (annual) | KGE (annual) |
|---------|-----------------|-----------------|-----|---------------------|--------------|
| Saudi Arabia | Hyper-arid (BWh/BWk) | ~67 mm | 0.03 | 0.59 | 0.69 |
| Italy | Mediterranean–temperate (Csa/Cfa) | ~1047 mm | 0.41 | 0.97 | 0.96 |
| Bangladesh | Tropical monsoon (Am) | ~2316 mm | 0.56 | 0.98 | 0.98 |

## Project Structure

```
hydro_climate_runoff_attribution/
├── gee/
│   ├── era5land_extraction.js       # GEE: export ERA5-Land monthly + annual GeoTIFF stacks
│   └── merit_dem_extraction.js      # GEE: export MERIT DEM for study area maps
├── data/
│   ├── GEE_ERA5Land_Monthly/        # Monthly GeoTIFFs (39 files, 911 bands each)
│   ├── GEE_ERA5Land_Annual/         # Annual GeoTIFFs (15 files, 76 bands each)
│   ├── DEM/                         # MERIT DEM GeoTIFFs (3 files, one per country)
│   └── netcdf/                      # Converted NetCDF files (3 files, one per country)
├── scripts/
│   ├── 00_convert_to_netcdf.py      # One-time: merge monthly GeoTIFFs → compressed NetCDF
│   ├── utils_load.py                # Shared data loading utilities
│   ├── 01_water_balance.py          # Water balance + spatial distribution maps
│   ├── 02_trend_analysis.py         # Mann-Kendall + Sen's slope (regional + pixel-level)
│   └── 03_xgboost_shap.py           # XGBoost (monthly + annual) + SHAP attribution
├── figures/                         # Output figures (fig01–fig13b, 18 total)
├── results/                         # Output CSVs (metrics, SHAP importance, trends)
├── report/                          # LaTeX report and supplement
│   ├── final_report_temp.tex        # 10-page main report (annual model only)
│   ├── final_supplement_temp.tex    # 4-page supplement (monthly model justification)
│   ├── references.bib               # Bibliography
│   └── figures_fp/                  # Local figure directory for the report
├── Research Plan.md                 # Detailed research plan (English)
├── README.md                        # This file
└── environment.yml                  # Conda environment definition
```

## Workflow

### Step 1 — ERA5-Land Data Extraction (GEE)

Open `gee/era5land_extraction.js` in the [GEE Code Editor](https://code.earthengine.google.com/) and submit all export tasks (54 total). Submit Bangladesh first (smallest), then Italy, then Saudi Arabia.

Outputs go to two Google Drive folders:
- `GEE_ERA5Land_Monthly/` — 39 GeoTIFFs, **911 bands** each (1950-02 to 2025-12)
- `GEE_ERA5Land_Annual/`  — 15 GeoTIFFs, **76 bands** each (annual totals/means per year)

Download and place in `data/GEE_ERA5Land_Monthly/` and `data/GEE_ERA5Land_Annual/`.

### Step 2 — MERIT DEM Extraction (GEE, optional for fig01)

Open `gee/merit_dem_extraction.js` in GEE and run 3 export tasks (one per country). Download outputs to `data/DEM/` as `Saudi_DEM.tif`, `Italy_DEM.tif`, `Bangladesh_DEM.tif`.

### Step 3 — Setup environment

```bash
conda env create -f environment.yml
conda activate hydroclimate
```

### Step 4 — Convert to NetCDF

```bash
python scripts/00_convert_to_netcdf.py
```

Merges 39 monthly GeoTIFFs into 3 compressed NetCDF files (~250 MB total) in `data/netcdf/`.

### Step 5 — Analysis

```bash
python scripts/01_water_balance.py   # fig01–fig05
python scripts/02_trend_analysis.py  # fig06–fig08
python scripts/03_xgboost_shap.py    # fig09–fig13b
```

## Data

| Item | Detail |
|------|--------|
| ERA5-Land dataset | `ECMWF/ERA5_LAND/MONTHLY_AGGR` |
| Spatial resolution | 0.1° (~11 km) |
| Period | 1950-02 to 2025-12 (911 months) |
| DEM dataset | `MERIT/DEM/v1_0_3` (Yamazaki et al. 2017), 90 m |
| Variables | P, ET, R (surface + subsurface), soil water (4 layers), T, Td, net radiation (SW/LW), wind speed, soil temperature, surface pressure |

## Methods

| Method | Details |
|--------|---------|
| Water balance | P = ET + R + ΔS + ε (monthly, national mean) |
| Trend analysis | Mann-Kendall test + Sen's slope; annual + seasonal; regional and pixel-level |
| ML model | XGBoost regression (max 500 trees, depth 5, lr 0.1), trained per country on pixel-level data |
| Train/val/test split | train ≤ 1996 / val 1997–2004 / test ≥ 2005; KGE-based early stopping (patience 20) |
| Attribution | SHAP TreeExplainer; primary results from annual model on full pixel-year test set |
| Evaluation | R², RMSE, NSE, KGE |

## Key Results

**Water balance**: ET/P ranges from 1.11 (Saudi Arabia, ET > P due to known ERA5-Land overestimation in hyperarid regions) to 0.46 (Bangladesh).

**Trends**: All three countries show significant warming. Bangladesh shows significant decreasing P and R trends (−4.1 and −3.7 mm/yr). Italy shows significant ET increase (+0.88 mm/yr).

**SHAP attribution** (annual model):
- Saudi Arabia: All features have low SHAP magnitude (|SHAP| < 5 mm/yr); annual runoff is driven by stochastic extreme events that annual climate means cannot capture
- Italy: Precipitation dominant (|SHAP| ~270 mm/yr); soil water change ranks #3
- Bangladesh: Precipitation overwhelmingly dominant (|SHAP| ~500 mm/yr); antecedent soil water ranks #2 (saturation-excess feedback)

**Cross-regime gradient**: Precipitation SHAP magnitude increases by ~250× from Saudi Arabia to Bangladesh, paralleling a rise in machine-learning predictability from R² = 0.59 to 0.98 — a quantitative SHAP-based traverse of the Budyko water-limited to energy-limited continuum.

## Output Figures

| Figure | Content |
|--------|---------|
| fig01 | Study area maps with MERIT DEM terrain fill |
| fig02 | Multi-year mean P/ET/R spatial distribution (3×3 panel) |
| fig03 | Annual water balance time series (P/ET/R, 3 countries) |
| fig04 | Water balance structure bar chart (ET/P, R/P) |
| fig05 | Annual runoff coefficient (R/P) inter-annual variability |
| fig06 | Annual trend time series (P/ET/R/T, Mann-Kendall) |
| fig07 | Trend heatmap (Sen's slope × season × country) |
| fig08 | Pixel-level Sen's slope spatial maps (P/R/T, 3×3) |
| fig09 / fig09b | Monthly / annual XGBoost: predicted vs observed |
| fig10 / fig10b | Monthly / annual SHAP temporal evolution |
| fig11 / fig11b | Monthly / annual SHAP beeswarm summary |
| fig12 / fig12b | Monthly / annual feature importance comparison |
| fig13 / fig13b | Monthly / annual precipitation SHAP dependence plot |

The report uses only the annual figures (fig09b, fig11b, fig12b, fig13b) for physical interpretation; monthly figures are kept in the supplement to justify the annual focus.

## Environment

```bash
conda env create -f environment.yml
conda activate hydroclimate
```

Key packages: `xarray`, `rioxarray`, `rasterio`, `xgboost`, `shap`, `pymannkendall`, `cartopy`, `matplotlib`
