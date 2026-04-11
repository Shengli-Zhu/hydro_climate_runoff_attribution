# Research Plan: Quantifying and Attributing Climate Change Impacts on Runoff across Different Hydroclimatic Regimes

**A Comparative Study of Saudi Arabia, Italy, and Bangladesh Using ERA5-Land Reanalysis Data, XGBoost, and SHAP**

---

## 1. Overview

This project investigates how climate change influences runoff generation across three countries that span the global aridity gradient: Saudi Arabia (extreme arid), Italy (semi-arid to sub-humid transitional), and Bangladesh (tropical humid). We address a fundamental question in hydroclimatology: **does the mechanism by which climate variables control runoff change systematically from dry to wet climates?**

To answer this, we combine three complementary approaches:

1. **Water balance analysis** — diagnose how precipitation is partitioned into evapotranspiration, runoff, and soil water storage
2. **Trend analysis** — detect significant changes in climate and hydrological variables over 1950–2025 using non-parametric Mann-Kendall tests
3. **Machine learning attribution** — use XGBoost regression with SHAP (SHapley Additive exPlanations) values to quantify the contribution of each climate variable to runoff, both at the monthly and annual timescale

All data come from the ERA5-Land reanalysis dataset (ECMWF), accessed via Google Earth Engine, covering February 1950 to December 2025 (911 months). Terrain data for study area visualisation comes from MERIT DEM (Yamazaki et al., 2017).

---

## 2. Scientific Background and Motivation

### 2.1 Why these three countries?

The three study areas sit at three distinct positions along the aridity spectrum:

| Country | Annual P (ERA5) | Aridity | Runoff Regime |
|---------|----------------|---------|---------------|
| Saudi Arabia | ~67 mm/yr | Hyper-arid (BWh/BWk) | Episodic flash floods; ET ≈ P |
| Italy | ~1047 mm/yr | Semi-arid → Sub-humid (Csa/Cfa) | Seasonal; strong N–S gradient |
| Bangladesh | ~2316 mm/yr | Tropical monsoon (Am) | Monsoon-driven; sustained high flows |

Placing them together lets us observe **how the dominant controls on runoff shift** as water availability increases. In water-limited systems (Saudi Arabia), runoff is controlled almost entirely by precipitation pulses and nearly all precipitation is consumed by evapotranspiration. In energy-limited systems (Bangladesh), precipitation far exceeds atmospheric demand, so temperature and radiation become relatively more important in modulating how much precipitation converts to runoff.

### 2.2 Why ERA5-Land?

ERA5-Land is the state-of-the-art land surface reanalysis product from ECMWF (Muñoz-Sabater et al., 2021). It is produced by replaying the ECMWF TESSEL land surface model forced by ERA5 atmospheric fields, and provides physically consistent estimates of the full water and energy balance at 0.1° (~11 km) resolution from 1950 to near-present. Advantages relevant to this study:

- Complete spatial and temporal coverage — no gauge network required
- Internally consistent P, ET, R, and soil moisture — closed-form water balance possible
- Long record (76 years, 911 months) enables robust trend detection

**Known limitation**: In hyper-arid regions like Saudi Arabia, ERA5-Land TESSEL tends to overestimate actual ET. Our data show ET/P = 1.11 for Saudi Arabia, which is physically impossible in a closed system without external water sources. This is acknowledged and discussed as a model limitation.

### 2.3 Why XGBoost + SHAP?

Traditional methods (regression, correlation, Budyko framework) can identify which variables are statistically associated with runoff, but cannot quantify the contribution of each variable to individual predictions in a model-agnostic way. SHAP resolves this:

- **SHAP is model-specific but locally exact**: for each sample (pixel-year or pixel-month), SHAP decomposes the model prediction into additive contributions from each feature, satisfying local accuracy, missingness, and consistency axioms (Lundberg & Lee, 2017)
- **XGBoost + TreeExplainer**: the combination allows exact (not approximate) SHAP computation for tree-based models, making it computationally feasible on millions of samples
- **Physical interpretability**: a positive SHAP value for precipitation means "this level of precipitation, in this pixel-month, causes the model to predict higher runoff than average"

---

## 3. Research Questions and Hypotheses

**RQ1**: How have runoff and related climate variables changed over 1950–2025 across the three countries, both regionally and at the pixel level?

**RQ2**: How do the water balance components (P, ET, R, ΔS) differ structurally across the aridity gradient, and how have they evolved over time?

**RQ3**: Which climate variables are the dominant controls on runoff in each country, and does this dominance shift systematically along the aridity gradient?

### Hypotheses

**H1**: The predictability of monthly runoff from monthly climate means increases from arid to humid (Saudi Arabia R² << Italy R² < Bangladesh R²), because arid-zone runoff is driven by stochastic extreme events that are not captured by monthly averages.

**H2**: In annual models (which remove within-year noise), precipitation emerges as the dominant SHAP feature in all three countries, but its absolute contribution increases dramatically from Saudi Arabia to Bangladesh, reflecting the increasing efficiency of precipitation-to-runoff conversion.

**H3**: Italy's monthly model will show soil temperature as the top feature (a proxy for seasonality — cold winters produce runoff, hot summers suppress it through evapotranspiration), but this effect disappears in the annual model once seasonal variability is aggregated out.

---

## 4. Data

### 4.1 ERA5-Land (Climate Variables)

- **GEE Dataset ID**: `ECMWF/ERA5_LAND/MONTHLY_AGGR`
- **Period**: 1950-02 to 2025-12 (911 months)
- **Resolution**: 0.1° (~11,132 m native scale in GEE export)
- **Extraction script**: `gee/era5land_extraction.js`

#### Variables extracted and their unit conversions

| Variable | GEE Band | Raw Unit | Converted Unit | Use |
|----------|----------|----------|---------------|-----|
| Precipitation | `total_precipitation_sum` | m | mm | Water balance P |
| Evapotranspiration | `total_evaporation_sum` | m (negative) | mm (×−1000) | Water balance ET |
| Surface runoff | `surface_runoff_sum` | m | mm | Component of R |
| Subsurface runoff | `sub_surface_runoff_sum` | m | mm | Component of R |
| Total runoff | surface + subsurface | — | mm | Target variable |
| Soil water layer 1 (0–7 cm) | `volumetric_soil_water_layer_1` | m³/m³ | mm (×70) | ΔS calculation |
| Soil water layer 2 (7–28 cm) | `volumetric_soil_water_layer_2` | m³/m³ | mm (×210) | ΔS calculation |
| Soil water layer 3 (28–100 cm) | `volumetric_soil_water_layer_3` | m³/m³ | mm (×720) | ΔS calculation |
| Soil water layer 4 (100–289 cm) | `volumetric_soil_water_layer_4` | m³/m³ | mm (×1890) | ΔS calculation |
| 2 m temperature | `temperature_2m` | K | °C (−273.15) | XGBoost feature |
| 2 m dewpoint temperature | `dewpoint_temperature_2m` | K | °C | XGBoost feature |
| Net solar radiation | `surface_net_solar_radiation_sum` | J/m² | J/m² | XGBoost feature |
| Net thermal radiation | `surface_net_thermal_radiation_sum` | J/m² | J/m² | XGBoost feature |
| Wind speed | u + v components | m/s | m/s (√u²+v²) | XGBoost feature |
| Soil temperature L1 | `soil_temperature_level_1` | K | °C | XGBoost feature |
| Surface pressure | `surface_pressure` | Pa | Pa | XGBoost feature |

**Total soil water storage**: S = S₁ + S₂ + S₃ + S₄ (mm equivalent depth)
**Soil water storage change**: ΔS = S(t) − S(t−1)

### 4.2 MERIT DEM (Terrain)

- **GEE Dataset ID**: `MERIT/DEM/v1_0_3`
- **Resolution**: 3 arc-seconds (~90 m)
- **Use**: Study area map (fig01) — terrain fill for each country's subplot
- **Extraction script**: `gee/merit_dem_extraction.js`
- **Local paths**: `data/DEM/Saudi_DEM.tif`, `Italy_DEM.tif`, `Bangladesh_DEM.tif`
- **Note**: Loaded with 1500-pixel downsampling on read to avoid memory issues at 90 m resolution

### 4.3 Country Boundaries

All country boundaries are from the FAO GAUL 2015 dataset (`FAO/GAUL/2015/level0`), accessed directly in GEE during export. Each GeoTIFF is clipped to the exact country polygon (not a bounding rectangle).

---

## 5. Data Processing Pipeline

```
GEE Export (gee/era5land_extraction.js)
    ├── Monthly GeoTIFF stacks (39 files × 911 bands)
    │   → data/GEE_ERA5Land_Monthly/
    └── Annual GeoTIFF stacks (15 files × 76 bands)
        → data/GEE_ERA5Land_Annual/
             ↓
scripts/00_convert_to_netcdf.py
    → data/netcdf/Saudi.nc, Italy.nc, Bangladesh.nc (~250 MB total)
      (zlib-compressed, float32, dims: time × lat × lon)
             ↓
scripts/utils_load.py
    ├── load_country_dataset(name)          → xarray Dataset (time, lat, lon)
    ├── load_country_mean_timeseries(name)  → pandas DataFrame (911 rows, national mean)
    ├── load_annual_geotiff(name, var)      → numpy 3D array (76 years × h × w)
    └── load_pixel_dataframe(name)          → pandas DataFrame (millions of pixel-months)
```

### Why two formats (NetCDF + GeoTIFF)?

- **NetCDF** (monthly): used for national-mean time series and pixel-level monthly DataFrames. Compressed and indexed for fast access.
- **Annual GeoTIFF stacks**: used directly for pixel-level trend mapping. Keeping them separate from NetCDF avoids mixing temporal resolutions and allows simple rasterio-based reading.

---

## 6. Methods

### 6.1 Water Balance Analysis (`scripts/01_water_balance.py`)

The water balance equation applied monthly, at the national mean scale:

$$P = ET + R + \Delta S + \varepsilon$$

where ε is the residual (closure error) capturing model inconsistencies and unmeasured processes.

**Computed quantities:**
- Annual runoff coefficient: RC = R_annual / P_annual
- ET/P ratio and R/P ratio (multi-year mean)
- Annual aggregation: flux variables (P, ET, R) are summed; state variables (T) are averaged

**Key finding — Saudi Arabia closure error**: ET/P = 1.11 (ET > P), which is physically impossible in a system without external water sources. This reflects a known ERA5-Land TESSEL bias in hyperarid regions, possibly exacerbated by the model capturing evaporation from fossil groundwater aquifers. This is treated as a data limitation and discussed explicitly.

**Output figures:**
- fig01: Study area maps (MERIT DEM terrain fill, horizontal colorbar)
- fig02: Multi-year mean P/ET/R spatial distribution (3 rows × 3 columns)
- fig03: Annual water balance time series (P/ET/R, 3 countries × 76 years)
- fig04: Water balance structure bar chart (ET/P, R/P, three countries)
- fig05: Annual runoff coefficient (R/P) inter-annual variability

### 6.2 Trend Analysis (`scripts/02_trend_analysis.py`)

#### Regional trend analysis

For each country's national-mean annual time series (P, ET, R, T_annual):

- **Mann-Kendall test** (`pymannkendall`): non-parametric test for monotonic trend; reports Kendall's τ and p-value. Advantage over linear regression: robust to non-normality and outliers, no assumption of constant variance.
- **Sen's slope** (`scipy.stats.theilslopes`): median of all pairwise slopes — robust estimator of the trend magnitude (mm/yr or °C/yr).
- Applied at both annual and seasonal (DJF/MAM/JJA/SON) scales to detect seasonal asymmetry.

**Key results:**
- All three countries: significant warming trend
- Bangladesh: significant decreasing P (−4.1 mm/yr, p<0.05) and R (−3.7 mm/yr, p<0.05)
- Italy: significant ET increase (+0.88 mm/yr, p<0.001); no significant R trend
- Saudi Arabia: no significant P or ET trend; weak R decrease (p<0.01)

#### Pixel-level spatial trend maps

For each country and each variable (P, R, T), Sen's slope and Kendall τ p-value are computed **independently at every grid cell** from the 76-band annual GeoTIFF stacks:

```python
for each pixel:
    slope, _, _, _ = scipy.stats.theilslopes(values_76yr)
    tau, pval = scipy.stats.kendalltau(range(76), values_76yr)
```

Significant pixels (p < 0.05) are rendered at full opacity; non-significant pixels at 40% opacity. This reveals spatial heterogeneity hidden by national averages — e.g., Italy's contrasting N–S trend patterns.

**Output figures:**
- fig06: Regional annual trend time series (P/ET/R/T with significance markers)
- fig07: Trend heatmap (Sen's slope by variable × season, 3 countries)
- fig08: Pixel-level Sen's slope spatial maps (P/R/T, 3 rows × 3 columns)

### 6.3 XGBoost + SHAP Attribution (`scripts/03_xgboost_shap.py`)

#### 6.3.1 Data preparation

The monthly NetCDF is "unrolled" into a pixel-level DataFrame where each row is one pixel-month:

| Column | Description |
|--------|-------------|
| pixel_id | unique (lat, lon) identifier |
| year, month | temporal index |
| P_mm, T_C, Td_C, … | 8 climate features |
| R_mm | target variable |
| dS | soil water change (previous month) |

**Sample sizes:**
| Country | Pixel-months | Pixel-years (annual model) |
|---------|-------------|--------------------------|
| Saudi Arabia | ~15.8 million | ~1.32 million |
| Italy | ~3.1 million | ~0.26 million |
| Bangladesh | ~1.3 million | ~0.11 million |

**Train/test split** (temporal, not random):
- Training: year ≤ 2004 (~73% of data, 1950–2004)
- Testing: year ≥ 2005 (~27% of data, 2005–2025)

This ordering prevents data leakage — the model cannot see future years during training.

#### 6.3.2 Monthly model

**XGBoost hyperparameters** (fixed; grid search infeasible at million-sample scale):
```python
xgb.XGBRegressor(
    n_estimators=200, max_depth=5, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, random_state=42
)
```

**Evaluation metrics:**
- **R²**: fraction of variance explained
- **RMSE**: root mean squared error (mm/month)
- **NSE** (Nash-Sutcliffe Efficiency): standard hydrological performance metric; NSE = 1 − [Σ(obs−sim)² / Σ(obs−mean_obs)²]; equivalent to R² for the test set
- **KGE** (Kling-Gupta Efficiency): KGE = 1 − √[(r−1)² + (α−1)² + (β−1)²], decomposing into correlation (r), variability ratio (α = σ_sim/σ_obs), and bias ratio (β = μ_sim/μ_obs)

**Monthly model results:**

| Country | R² | RMSE (mm/month) | NSE | KGE |
|---------|----|-----------------|-----|-----|
| Saudi Arabia | 0.334 | 1.79 | 0.334 | 0.521 |
| Italy | 0.816 | 25.17 | 0.816 | 0.862 |
| Bangladesh | 0.936 | 35.67 | 0.936 | 0.944 |

#### 6.3.3 Annual model

Monthly data are aggregated to annual pixel-year records:
- Flux variables (P, ET, R, Rn_sw, Rn_lw): **annual sum**
- State/intensive variables (T, Td, Wind, Ts): **annual mean**
- Soil water change (dS): **annual sum**

A separate XGBoost model is trained on these annual records with 8 corresponding annual features. The same temporal split applies (year ≤ 2004 / year ≥ 2005).

**Annual model results:**

| Country | R² | RMSE (mm/year) | NSE |
|---------|----|----------------|-----|
| Saudi Arabia | 0.559 | 9.14 | 0.559 |
| Italy | 0.970 | 76.43 | 0.970 |
| Bangladesh | 0.984 | 86.67 | 0.984 |

The dramatic R² improvement (e.g., Saudi Arabia: 0.334 → 0.559) confirms that annual aggregation removes within-year noise that obscures the climate–runoff relationship in monthly data.

#### 6.3.4 SHAP analysis

**What is a SHAP value?**

For a given prediction $\hat{y}$ (e.g., predicted annual runoff = 850 mm/yr for a specific pixel-year), the SHAP decomposition is:

$$\hat{y} = \phi_0 + \phi_{P} + \phi_{T} + \phi_{Td} + \phi_{Rn\_sw} + \phi_{Rn\_lw} + \phi_{Wind} + \phi_{Ts} + \phi_{dS}$$

where:
- $\phi_0$ = base value (mean prediction across all training samples)
- $\phi_i$ = SHAP value for feature $i$ — the marginal contribution of that feature to moving the prediction away from the base value

Positive SHAP: the feature value pushes the prediction *above* the mean. Negative SHAP: pushes it *below* the mean.

**Computation:**
- Monthly model: 5000 random samples from the test set, using `shap.TreeExplainer` (exact SHAP for tree models, not approximate)
- Annual model: all test-set pixel-years

**Temporal SHAP evolution**: For each year, the mean SHAP value across all pixels is computed. This shows how the "importance weight" of each climate driver has changed over 1950–2025. The monthly temporal plot uses a 3-year rolling mean to reduce noise; the annual temporal plot has one point per year (no smoothing needed).

---

## 7. Results and Interpretation

### 7.1 Water Balance Structure

The three countries sit at fundamentally different points on the Budyko curve:

- **Saudi Arabia**: Nearly all precipitation is consumed by (or attributed to) ET; runoff coefficient RC = 3.2%. The ERA5-Land ET > P anomaly is discussed as a model artefact.
- **Italy**: Roughly 40% of precipitation generates runoff (RC = 41.2%). Strong spatial gradient — wetter Alpine north vs. drier Mediterranean south.
- **Bangladesh**: Over half of precipitation becomes runoff (RC = 56.2%), reflecting chronic soil saturation under monsoon conditions.

### 7.2 Trend Patterns

**Temperature**: All three countries show statistically significant warming, consistent with global climate change. Saudi Arabia shows the strongest warming in absolute terms (Sen's slope +0.029°C/yr).

**Bangladesh precipitation and runoff**: Both show significant decreasing trends (P: −4.1 mm/yr, R: −3.7 mm/yr), suggesting a possible weakening of monsoon intensity over 1950–2025, consistent with some regional climate projections.

**Italy ET increase**: Significant ET increase (+0.88 mm/yr, p<0.001) without a corresponding R decrease likely reflects warming-driven potential evapotranspiration increase being partially offset by unchanged precipitation.

**Spatial heterogeneity (fig08)**: The pixel-level maps reveal patterns hidden in national averages. Italy shows a clear north–south contrast in P and R trends. Bangladesh shows stronger declining trends in the northeast (Sylhet region) than the southwest delta. Saudi Arabia's sparse significant pixels cluster along the western Hijaz/Asir mountains.

### 7.3 SHAP Attribution — Key Findings

#### Monthly model (fig11, fig12)

**Saudi Arabia** (R² = 0.334): The model struggles to predict monthly runoff from monthly climate means. Dewpoint temperature ranks above precipitation as the most "important" feature — a physically implausible result that arises because the model picks up on the seasonal temperature/humidity cycle rather than the episodic precipitation–runoff relationship. This confirms that Saudi Arabia's runoff is driven by stochastic extreme events, and monthly climate averages contain insufficient information.

**Italy** (R² = 0.816): Soil temperature is the top feature (mean |SHAP| ≈ 56 mm/month). This is not directly physical — rather, soil temperature acts as a **seasonal proxy**: low values in winter correspond to low evapotranspiration and snowmelt runoff (positive SHAP), while high summer values correspond to high evapotranspiration suppressing runoff (negative SHAP). The model has learned the seasonal cycle through soil temperature rather than through precipitation.

**Bangladesh** (R² = 0.936): Precipitation is overwhelmingly the top feature (mean |SHAP| ≈ 73 mm/month), with all other features playing secondary roles. The model has correctly learned that month-to-month runoff variation is driven primarily by monsoon rainfall.

#### Annual model (fig11b, fig12b)

Once seasonal noise is removed by annual aggregation, a cleaner picture emerges:

**All three countries**: Precipitation is the top feature in the annual model — consistent with the physical understanding that inter-annual runoff variability is driven primarily by inter-annual precipitation variability.

**Saudi Arabia**: Even in the annual model, all features have very small SHAP magnitudes (|SHAP| < 5 mm/yr), confirming that annual climate averages cannot explain annual runoff variability in an extreme arid environment. This is itself a key finding.

**Italy**: Precipitation SHAP (~270 mm/yr) is now clearly dominant. Soil temperature falls to a secondary role — confirming that its monthly importance was capturing seasonality, not a true annual-scale control.

**Bangladesh**: Precipitation SHAP (~500 mm/yr) dwarfs all other features. Antecedent soil water change (dS) ranks second, reflecting the **saturation excess mechanism**: in years where antecedent soil moisture is already high, the same amount of precipitation generates disproportionately more runoff.

#### Cross-regime gradient (fig12b — most important figure for RQ3)

Reading the annual bar chart from Saudi Arabia → Italy → Bangladesh:

1. **Precipitation SHAP magnitude increases by two orders of magnitude** (~2 → ~270 → ~500 mm/yr). Each unit of precipitation contributes increasingly more to runoff as the climate gets wetter — consistent with the Budyko framework where water-limited systems convert little precipitation to runoff.

2. **Model predictability (R²) increases from 0.56 → 0.97 → 0.98**, confirming that in humid climates, annual runoff is almost entirely determined by annual precipitation, while in arid climates, additional factors (extreme event timing, infiltration capacity) introduce irreducible unpredictability.

3. **The transition from monthly to annual scale removes seasonality effects** (Italy soil temperature artifact), suggesting that monthly SHAP results should be interpreted cautiously in seasonally-driven climates.

#### Precipitation–SHAP dependence plots (fig13b)

These plots show the non-linear relationship between annual precipitation amount and its SHAP contribution, coloured by temperature:

- **Saudi Arabia**: Highly scattered — no consistent relationship, reinforcing the stochastic nature of arid runoff.
- **Italy**: A threshold-like response — below ~500 mm/yr, precipitation adds little to runoff (all consumed by ET); above this threshold, each additional millimetre of precipitation generates substantial runoff. High-temperature years (red) tend to have lower SHAP at the same precipitation level, suggesting temperature-driven ET suppresses precipitation-to-runoff conversion.
- **Bangladesh**: Near-linear relationship between precipitation and its SHAP contribution. High-temperature points (red) again show slightly lower SHAP, but the effect is smaller than in Italy because the system is more firmly in the energy-limited regime.

---

## 8. Discussion Points

### 8.1 Monthly vs. annual SHAP: a methodological lesson

The discrepancy between monthly and annual SHAP rankings in Italy (soil temperature dominates monthly, precipitation dominates annually) illustrates a general principle: **SHAP values reflect the information structure of the dataset, not just physical causal pathways**. When a dataset has strong seasonal structure, the model will learn to use seasonal proxies (soil temperature, radiation) as shortcuts. Annual aggregation removes this confound and reveals the true inter-annual controls.

### 8.2 ERA5-Land limitations in arid regions

The Saudi Arabia ET > P result is a known issue with land surface models in hyperarid environments. TESSEL may be overestimating bare soil evaporation, or capturing evaporation from deep fossil aquifers not balanced by modern precipitation. Users of ERA5-Land in arid regions should treat ET estimates with caution.

### 8.3 What SHAP cannot tell us

SHAP quantifies the association between climate variables and runoff **as learned by XGBoost**. It does not prove causation. In particular:
- Multi-collinear features (e.g., temperature and net radiation are correlated) may have their importance split across them in arbitrary ways
- SHAP values are specific to this model trained on this dataset — they may differ from process-based model attributions

### 8.4 Spatial heterogeneity

By training a single model per country, we assume that the climate–runoff relationship is stationary across space within each country. This is a significant simplification — Italy's Mediterranean south and Alpine north have fundamentally different runoff processes. Future work could apply clustered or spatially-stratified SHAP analysis.

---

## 9. Repository Structure

```
hydro_climate_runoff_attribution/
├── gee/
│   ├── era5land_extraction.js        # GEE: ERA5-Land monthly + annual export (54 tasks)
│   └── merit_dem_extraction.js       # GEE: MERIT DEM export for 3 countries
├── data/
│   ├── GEE_ERA5Land_Monthly/         # 39 GeoTIFFs, 911 bands each
│   ├── GEE_ERA5Land_Annual/          # 15 GeoTIFFs, 76 bands each
│   ├── DEM/                          # MERIT DEM GeoTIFFs (3 files)
│   └── netcdf/                       # Compressed NetCDF (3 files, ~250 MB)
├── scripts/
│   ├── 00_convert_to_netcdf.py       # GeoTIFF → NetCDF conversion (run once)
│   ├── utils_load.py                 # Shared loading functions
│   ├── 01_water_balance.py           # Water balance + spatial distribution
│   ├── 02_trend_analysis.py          # Trend analysis (regional + pixel-level)
│   └── 03_xgboost_shap.py            # XGBoost (monthly + annual) + SHAP
├── figures/                          # All output figures (fig01–fig13b)
├── results/                          # Output CSVs (metrics, SHAP importance, trends)
├── 研究计划_zh.md                    # Research plan (Chinese)
├── 研究计划_en.md                    # Research plan (English, this file)
└── README.md                         # Quick-start guide
```

### Running the analysis

```bash
# Step 1: Run GEE scripts in GEE Code Editor, download outputs to data/
# Step 2: Convert to NetCDF
D:/anaconda3/envs/hydroclimate/python.exe scripts/00_convert_to_netcdf.py

# Step 3: Analysis (run in order)
D:/anaconda3/envs/hydroclimate/python.exe scripts/01_water_balance.py   # ~2 min
D:/anaconda3/envs/hydroclimate/python.exe scripts/02_trend_analysis.py  # ~5 min
D:/anaconda3/envs/hydroclimate/python.exe scripts/03_xgboost_shap.py    # ~30–60 min
```

---

## 10. Output Figures Reference

| Figure | Content | Script |
|--------|---------|--------|
| fig01 | Study area maps with MERIT DEM terrain | 01 |
| fig02 | Multi-year mean P/ET/R spatial distribution (3×3) | 01 |
| fig03 | Annual water balance time series (1950–2025) | 01 |
| fig04 | Water balance structure bar chart (ET/P, R/P) | 01 |
| fig05 | Annual runoff coefficient (R/P) inter-annual variability | 01 |
| fig06 | Regional annual trend time series with significance markers | 02 |
| fig07 | Trend heatmap (Sen's slope by season, 4 variables, 3 countries) | 02 |
| fig08 | Pixel-level Sen's slope spatial maps (P/R/T, 3×3) | 02 |
| fig09 | Monthly XGBoost: predicted vs observed runoff (scatter) | 03 |
| fig09b | Annual XGBoost: predicted vs observed runoff (scatter) | 03 |
| fig10 | Monthly SHAP temporal evolution (3-yr rolling mean) | 03 |
| fig10b | Annual SHAP temporal evolution (one point per year) | 03 |
| fig11 | Monthly SHAP beeswarm summary (3 countries) | 03 |
| fig11b | Annual SHAP beeswarm summary (3 countries) | 03 |
| fig12 | Monthly feature importance comparison — bar chart | 03 |
| fig12b | Annual feature importance comparison — bar chart | 03 |
| fig13 | Monthly precipitation SHAP dependence plot (colour = T) | 03 |
| fig13b | Annual precipitation SHAP dependence plot (colour = T) | 03 |

---

## 11. Key References

1. Muñoz-Sabater, J., et al. (2021). ERA5-Land: a state-of-the-art global reanalysis dataset for land applications. *Earth System Science Data*, 13, 4349–4383.
2. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.
3. Lundberg, S. M., et al. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*, 2, 56–67.
4. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of KDD 2016*.
5. Yamazaki, D., et al. (2017). A high-accuracy map of global terrain elevations. *Geophysical Research Letters*, 44, 5844–5853. (MERIT DEM)
6. Gupta, H. V., Kling, H., Yilmaz, K. K., & Martinez, G. F. (2009). Decomposition of the mean squared error and NSE: Implications for improving hydrological modelling. *Journal of Hydrology*, 377, 80–91. (KGE)
7. Budyko, M. I. (1974). *Climate and Life*. Academic Press.
8. Nash, J. E., & Sutcliffe, J. V. (1970). River flow forecasting through conceptual models. *Journal of Hydrology*, 10, 282–290. (NSE)
9. Mann, H. B. (1945). Nonparametric tests against trend. *Econometrica*, 13, 245–259.
10. Sen, P. K. (1968). Estimates of the regression coefficient based on Kendall's tau. *Journal of the American Statistical Association*, 63, 1379–1389.
