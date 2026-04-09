# Quantifying and Attributing Climate Change Impacts on Runoff across Different Hydroclimatic Regimes

**Geo Modeling Final Project**

A comparative study of Saudi Arabia, Italy, and Bangladesh using ERA5-Land reanalysis data, XGBoost machine learning, and SHAP explainability framework.

## Study Areas

| Country | Climate | Annual Precip | Runoff Regime |
|---------|---------|---------------|---------------|
| Saudi Arabia | Extreme arid (BWh/BWk) | < 100 mm | Flash flood driven |
| Italy | Semi-arid to sub-humid (Csa/Cfa) | 500-1500 mm | Seasonal, N-S gradient |
| Bangladesh | Tropical monsoon (Am) | > 2000 mm | Monsoon driven |

## Project Structure

```
hydro_climate_runoff_attribution/
├── gee/                        # Google Earth Engine scripts
│   └── era5land_extraction.js  # Data extraction & export
├── data/                       # CSV data from GEE export
├── scripts/
│   ├── 01_water_balance.py     # Water balance analysis (P = ET + R + ΔS + ε)
│   ├── 02_trend_analysis.py    # Mann-Kendall + Sen's slope trends
│   └── 03_xgboost_shap.py     # XGBoost modeling + SHAP attribution
├── figures/                    # Output figures
├── results/                    # Output tables/CSV
└── requirements.txt
```

## Workflow

1. **Data extraction**: Run `gee/era5land_extraction.js` in [Google Earth Engine Code Editor](https://code.earthengine.google.com/) to export monthly ERA5-Land data for three countries
2. **Place CSV files** in `data/` directory
3. **Run analysis scripts** sequentially:
   ```bash
   pip install -r requirements.txt
   cd scripts
   python 01_water_balance.py
   python 02_trend_analysis.py
   python 03_xgboost_shap.py
   ```

## Methods

- **Water Balance**: P = ET + R + ΔS + ε
- **Trend Analysis**: Mann-Kendall test + Sen's slope (annual & seasonal)
- **Attribution**: XGBoost regression + SHAP (SHapley Additive exPlanations)

## Data Source

- ERA5-Land Monthly Aggregated (`ECMWF/ERA5_LAND/MONTHLY_AGGR`)
- Spatial resolution: 0.1° (~11 km)
- Temporal coverage: 2000-01 to 2025-03
