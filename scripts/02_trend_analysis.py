"""
02_trend_analysis.py
Trend Analysis for Regional-Mean Time Series

Methods: Mann-Kendall test + Sen's slope estimator
Scales: Annual and seasonal (DJF, MAM, JJA, SON)
Outputs:
  - Trend summary table (CSV)
  - Fig.4-style trend visualization (regional mean bar charts)
"""

import sys
import pandas as pd
import numpy as np
import pymannkendall as mk
import matplotlib.pyplot as plt
import os
import rioxarray
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from utils_load import load_country_mean_timeseries

# ============================================================================
# Configuration
# ============================================================================
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

COUNTRIES = {
    'Saudi_Arabia': {'label': 'Saudi Arabia (Arid)', 'color': '#E74C3C'},
    'Italy': {'label': 'Italy (Transition)', 'color': '#3498DB'},
    'Bangladesh': {'label': 'Bangladesh (Humid)', 'color': '#2ECC71'},
}

COUNTRY_NC_NAME = {
    'Saudi_Arabia': 'Saudi',
    'Italy': 'Italy',
    'Bangladesh': 'Bangladesh',
}

SEASONS = {
    'DJF': [12, 1, 2],
    'MAM': [3, 4, 5],
    'JJA': [6, 7, 8],
    'SON': [9, 10, 11],
}

VARIABLES = {
    'P_mm': {'label': 'Precipitation (mm)', 'unit': 'mm/yr'},
    'ET_mm': {'label': 'Evapotranspiration (mm)', 'unit': 'mm/yr'},
    'R_mm': {'label': 'Runoff (mm)', 'unit': 'mm/yr'},
    'T_C': {'label': 'Temperature (°C)', 'unit': '°C/yr'},
}


# ============================================================================
# Data Loading
# ============================================================================
def load_country_data(country_name):
    """Load spatially averaged monthly time series from NetCDF."""
    nc_name = COUNTRY_NC_NAME.get(country_name, country_name)
    return load_country_mean_timeseries(nc_name)


# ============================================================================
# Annual and Seasonal Aggregation
# ============================================================================
def compute_annual_series(df, var):
    """Compute annual totals (P, ET, R) or annual means (T)."""
    if var == 'T_C':
        return df.groupby('year')[var].mean()
    else:
        return df.groupby('year')[var].sum()


def compute_seasonal_series(df, var, season_months):
    """Compute seasonal totals/means for a given variable."""
    # For DJF, assign Dec to the following year's winter
    df_s = df.copy()
    if set(season_months) == {12, 1, 2}:
        df_s['season_year'] = df_s['year']
        df_s.loc[df_s['month'] == 12, 'season_year'] = df_s['year'] + 1
        year_col = 'season_year'
    else:
        year_col = 'year'

    mask = df_s['month'].isin(season_months)
    subset = df_s[mask]

    if var == 'T_C':
        series = subset.groupby(year_col)[var].mean()
    else:
        series = subset.groupby(year_col)[var].sum()

    # Filter to complete years
    series = series[series.index >= 2001]  # exclude partial first year for DJF
    series = series[series.index <= 2025]
    return series


# ============================================================================
# Mann-Kendall + Sen's Slope
# ============================================================================
def mk_trend_test(series, alpha=0.05):
    """
    Perform Mann-Kendall trend test with Sen's slope.
    Returns dict with test results.
    """
    values = series.dropna().values
    if len(values) < 10:
        return {'trend': 'insufficient data', 'p': np.nan,
                'slope': np.nan, 'intercept': np.nan, 'significant': False}

    result = mk.original_test(values, alpha=alpha)
    return {
        'trend': result.trend,
        'p': result.p,
        'z': result.z,
        'tau': result.Tau,
        'slope': result.slope,       # Sen's slope (unit/year)
        'intercept': result.intercept,
        'significant': result.p < alpha,
    }


# ============================================================================
# Batch Trend Analysis
# ============================================================================
def analyze_trends(data_dict):
    """Run MK trend tests for all countries, variables, and time scales."""
    results = []

    for country, info in COUNTRIES.items():
        df = data_dict[country]

        for var, var_info in VARIABLES.items():
            # Annual trend
            annual = compute_annual_series(df, var)
            mk_res = mk_trend_test(annual)
            results.append({
                'Country': info['label'],
                'Variable': var_info['label'],
                'Scale': 'Annual',
                'Trend': mk_res['trend'],
                'Sen_Slope': mk_res['slope'],
                'p_value': mk_res['p'],
                'Tau': mk_res.get('tau', np.nan),
                'Significant': mk_res['significant'],
            })

            # Seasonal trends
            for season_name, season_months in SEASONS.items():
                seasonal = compute_seasonal_series(df, var, season_months)
                mk_res = mk_trend_test(seasonal)
                results.append({
                    'Country': info['label'],
                    'Variable': var_info['label'],
                    'Scale': season_name,
                    'Trend': mk_res['trend'],
                    'Sen_Slope': mk_res['slope'],
                    'p_value': mk_res['p'],
                    'Tau': mk_res.get('tau', np.nan),
                    'Significant': mk_res['significant'],
                })

    return pd.DataFrame(results)


# ============================================================================
# Visualization
# ============================================================================
def plot_annual_trends(data_dict):
    """
    Plot annual time series with Sen's slope trend line for each variable.
    Layout: 4 rows (variables) x 3 columns (countries).
    """
    fig, axes = plt.subplots(4, 3, figsize=(18, 16), sharex=True)

    for j, (country, info) in enumerate(COUNTRIES.items()):
        df = data_dict[country]

        for i, (var, var_info) in enumerate(VARIABLES.items()):
            ax = axes[i, j]
            annual = compute_annual_series(df, var)
            years = annual.index.values
            values = annual.values

            # MK test
            mk_res = mk_trend_test(annual)

            # Plot data
            ax.plot(years, values, 'o-', color=info['color'],
                    markersize=3, linewidth=1)

            # Plot trend line
            if mk_res['slope'] is not np.nan:
                trend_line = mk_res['intercept'] + mk_res['slope'] * np.arange(len(years))
                ax.plot(years, trend_line, '--', color='black', linewidth=1.2)

            # Significance marker
            sig_text = '**' if mk_res['significant'] else 'ns'
            slope_text = f"Sen={mk_res['slope']:.3f}/yr ({sig_text})"
            ax.text(0.02, 0.95, slope_text, transform=ax.transAxes,
                    fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            if i == 0:
                ax.set_title(info['label'], fontsize=12, fontweight='bold')
            if j == 0:
                ax.set_ylabel(f"{var_info['label'].split(' (')[0]}\n({var_info['unit']})",
                              fontsize=9)

            ax.grid(True, alpha=0.3)

    axes[-1, 1].set_xlabel('Year', fontsize=12)
    plt.suptitle('Annual Trends: Mann-Kendall + Sen\'s Slope (1950-2025)',
                 fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig06_annual_trends.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print('Saved: fig06_annual_trends.png')


def plot_trend_heatmap(trend_df):
    """
    Heatmap of Sen's slopes across countries, variables, and seasons.
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    scales = ['Annual', 'DJF', 'MAM', 'JJA', 'SON']
    scale_labels = ['Annual', 'Winter\n(DJF)', 'Spring\n(MAM)', 'Summer\n(JJA)', 'Autumn\n(SON)']

    for idx, (var, var_info) in enumerate(VARIABLES.items()):
        ax = axes[idx]
        subset = trend_df[trend_df['Variable'] == var_info['label']]

        # Build matrix: rows=countries, cols=scales
        matrix = []
        sig_matrix = []
        country_labels = []
        for country_info in COUNTRIES.values():
            row = []
            sig_row = []
            country_labels.append(country_info['label'].split(' (')[0])
            for scale in scales:
                match = subset[(subset['Country'] == country_info['label']) &
                               (subset['Scale'] == scale)]
                if len(match) > 0:
                    row.append(match.iloc[0]['Sen_Slope'])
                    sig_row.append(match.iloc[0]['Significant'])
                else:
                    row.append(np.nan)
                    sig_row.append(False)
            matrix.append(row)
            sig_matrix.append(sig_row)

        matrix = np.array(matrix)
        sig_matrix = np.array(sig_matrix)

        # Color mapping
        vmax = np.nanmax(np.abs(matrix)) if not np.all(np.isnan(matrix)) else 1
        im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto',
                       vmin=-vmax, vmax=vmax)

        # Annotations
        for ii in range(matrix.shape[0]):
            for jj in range(matrix.shape[1]):
                val = matrix[ii, jj]
                if not np.isnan(val):
                    marker = '*' if sig_matrix[ii, jj] else ''
                    ax.text(jj, ii, f'{val:.3f}{marker}',
                            ha='center', va='center', fontsize=8)

        ax.set_xticks(range(len(scales)))
        ax.set_xticklabels(scale_labels, rotation=0, fontsize=8)
        ax.set_yticks(range(len(country_labels)))
        ax.set_yticklabels(country_labels, fontsize=9, rotation=90, va='center')
        ax.set_title(var_info['label'], fontsize=11, fontweight='bold')

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("Sen's Slope Trends (* = significant at α=0.05, 1950–2025)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig07_trend_heatmap.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print('Saved: fig07_trend_heatmap.png')


# ============================================================================
# Pixel-Level Spatial Trend Analysis
# ============================================================================
ANNUAL_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'GEE_ERA5Land_Annual')

TREND_VARS = {
    'P_mm':  {'label': 'Precipitation trend (mm/yr)', 'cmap': 'BrBG'},
    'R_mm':  {'label': 'Runoff trend (mm/yr)',         'cmap': 'PuOr'},
    'T_C':   {'label': 'Temperature trend (°C/yr)',    'cmap': 'RdBu_r'},
}



def load_annual_geotiff(country_nc_name, var):
    """
    Load annual GeoTIFF stack for one country/variable.
    Returns (data_3d, lons, lats) where data_3d has shape (n_years, nlat, nlon).
    """
    fpath = os.path.join(ANNUAL_DIR, f'{country_nc_name}_Annual_{var}.tif')
    da = rioxarray.open_rasterio(fpath, masked=True)   # (band, y, x)
    data = da.values.astype(float)   # (n_years, nlat, nlon)
    lons = da.x.values
    lats = da.y.values
    return data, lons, lats


def compute_pixel_trends(data_3d):
    """
    Compute pixel-level Sen's slope and Mann-Kendall p-value.
    data_3d: (n_years, nlat, nlon)
    Returns slopes and pvals arrays of shape (nlat, nlon).
    """
    n_years, nlat, nlon = data_3d.shape
    years = np.arange(n_years, dtype=float)
    slopes = np.full((nlat, nlon), np.nan)
    pvals  = np.full((nlat, nlon), np.nan)

    for i in range(nlat):
        for j in range(nlon):
            ts = data_3d[:, i, j]
            if np.isnan(ts).any():
                continue
            res = stats.theilslopes(ts, years)
            slopes[i, j] = res.slope
            _, pval = stats.kendalltau(years, ts)
            pvals[i, j] = pval

    return slopes, pvals


def plot_spatial_trend_maps():
    """
    Pixel-level Sen's slope spatial trend maps for P, R, T.
    Layout: 3 rows (variables) x 3 columns (countries).
    Significant pixels (p<0.05) are shown with full opacity; others are lighter.
    """
    fig, axes = plt.subplots(
        3, 3, figsize=(16, 13),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )

    for j, (country, info) in enumerate(COUNTRIES.items()):
        nc_name = COUNTRY_NC_NAME[country]

        for i, (var, var_info) in enumerate(TREND_VARS.items()):
            ax = axes[i, j]
            print(f'  Computing pixel trends: {nc_name} / {var}...')

            data_3d, lons, lats = load_annual_geotiff(nc_name, var)
            slopes, pvals = compute_pixel_trends(data_3d)

            # Symmetric color scale
            vmax = np.nanpercentile(np.abs(slopes), 98)
            if vmax == 0 or np.isnan(vmax):
                vmax = 1.0

            # Plot all pixels (non-significant dimmed via alpha trick using two layers)
            im = ax.pcolormesh(lons, lats, slopes,
                               cmap=var_info['cmap'], vmin=-vmax, vmax=vmax,
                               transform=ccrs.PlateCarree(), alpha=0.4)
            # Overlay significant pixels at full opacity
            sig_slopes = np.where(pvals < 0.05, slopes, np.nan)
            ax.pcolormesh(lons, lats, sig_slopes,
                          cmap=var_info['cmap'], vmin=-vmax, vmax=vmax,
                          transform=ccrs.PlateCarree(), alpha=1.0)

            ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.gridlines(linewidth=0.3, alpha=0.5)
            ax.set_aspect('auto')

            plt.colorbar(im, ax=ax, orientation='vertical',
                         fraction=0.046, pad=0.04)

            if i == 0:
                ax.set_title(info['label'], fontsize=11, fontweight='bold')

    # Row labels via fig.text — cartopy axes ignore set_ylabel
    row_label_ys = [0.78, 0.49, 0.20]
    for i, var_info in enumerate(TREND_VARS.values()):
        fig.text(0.095, row_label_ys[i], var_info['label'],
                 fontsize=9, fontweight='bold',
                 ha='right', va='center', rotation=90)

    plt.suptitle(
        'Pixel-Level Sen\'s Slope (1950–2025)\nOpaque = significant (p<0.05), '
        'Transparent = non-significant',
        fontsize=13, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    fig.subplots_adjust(left=0.10)  # must be after tight_layout
    plt.savefig(os.path.join(FIG_DIR, 'fig08_spatial_trend_maps.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print('Saved: fig08_spatial_trend_maps.png')


# ============================================================================
# Main
# ============================================================================
def main():
    print('=' * 60)
    print('Trend Analysis (Mann-Kendall + Sen\'s Slope)')
    print('=' * 60)

    # Load data
    print('\nLoading data...')
    data = {}
    for name in COUNTRIES:
        data[name] = load_country_data(name)

    # Run trend analysis
    print('Running Mann-Kendall tests...')
    trend_df = analyze_trends(data)

    # Save results
    trend_df.to_csv(os.path.join(RESULTS_DIR, 'trend_analysis_results.csv'),
                    index=False, float_format='%.5f')
    print('\nSaved: trend_analysis_results.csv')

    # Display key results
    print('\n--- Annual Trend Summary ---')
    annual_only = trend_df[trend_df['Scale'] == 'Annual']
    print(annual_only[['Country', 'Variable', 'Sen_Slope', 'p_value', 'Significant']]
          .to_string(index=False))

    # Visualizations
    print('\nGenerating figures...')
    plot_annual_trends(data)
    plot_trend_heatmap(trend_df)
    print('\nComputing pixel-level trend maps (may take a few minutes)...')
    plot_spatial_trend_maps()

    print('\nTrend analysis complete!')


if __name__ == '__main__':
    main()
