"""
01_water_balance.py
Water Balance Analysis for Saudi Arabia, Italy, and Bangladesh

Computes: P = ET + R + ΔS + ε
Outputs:
  - Fig.1: Study area map (3-country locations)
  - Fig.2: Monthly water balance time series (3-country comparison)
  - Fig.3: Multi-year mean water balance structure (bar chart)
  - Fig.5: Annual runoff coefficient (R/P) inter-annual variation
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature

sys.path.insert(0, os.path.dirname(__file__))
from utils_load import load_country_dataset, load_country_mean_timeseries

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

# Map script country keys to NetCDF file prefixes
COUNTRY_NC_NAME = {
    'Saudi_Arabia': 'Saudi',
    'Italy': 'Italy',
    'Bangladesh': 'Bangladesh',
}

# ============================================================================
# Data Loading
# ============================================================================
def load_country_data(country_name):
    """Load spatially averaged monthly time series from NetCDF."""
    nc_name = COUNTRY_NC_NAME.get(country_name, country_name)
    return load_country_mean_timeseries(nc_name)


def load_all_countries():
    """Load data for all three countries."""
    data = {}
    for name in COUNTRIES:
        data[name] = load_country_data(name)
    return data


# ============================================================================
# Water Balance Computation
# ============================================================================
def compute_water_balance(df):
    """
    Compute water balance components.

    P = ET + R + ΔS + ε
    where ε = P - ET - R - ΔS (residual/closure error)
    """
    df = df.copy()

    # Soil water storage change (mm/month)
    df['dS'] = df['S_mm'].diff()

    # Residual (closure error)
    df['residual'] = df['P_mm'] - df['ET_mm'] - df['R_mm'] - df['dS']

    # Monthly runoff coefficient
    df['RC'] = df['R_mm'] / df['P_mm'].replace(0, np.nan)

    # Year and month columns for aggregation
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

    return df


# ============================================================================
# Annual Aggregation
# ============================================================================
def compute_annual(df):
    """Compute annual totals for P, ET, R and mean for T."""
    annual = df.groupby('year').agg(
        P_annual=('P_mm', 'sum'),
        ET_annual=('ET_mm', 'sum'),
        R_annual=('R_mm', 'sum'),
        dS_annual=('dS', 'sum'),
        T_annual=('T_C', 'mean'),
    ).reset_index()

    annual['residual_annual'] = (annual['P_annual'] - annual['ET_annual']
                                  - annual['R_annual'] - annual['dS_annual'])
    annual['RC_annual'] = annual['R_annual'] / annual['P_annual'].replace(0, np.nan)

    # Ratios
    annual['ET_P_ratio'] = annual['ET_annual'] / annual['P_annual'].replace(0, np.nan)
    annual['R_P_ratio'] = annual['R_annual'] / annual['P_annual'].replace(0, np.nan)

    return annual


# ============================================================================
# Visualization
# ============================================================================
def plot_study_area():
    """
    Fig.1: Study area map showing the three countries on a world map.
    """
    study_regions = {
        'Saudi_Arabia': {
            'label': 'Saudi Arabia\n(Arid)',
            'color': '#E74C3C',
            'center': (45.0, 24.0),   # lon, lat
            'extent': [36, 56, 16, 33],
        },
        'Italy': {
            'label': 'Italy\n(Transition)',
            'color': '#3498DB',
            'center': (12.5, 42.5),
            'extent': [6, 19, 36, 48],
        },
        'Bangladesh': {
            'label': 'Bangladesh\n(Humid)',
            'color': '#2ECC71',
            'center': (90.5, 23.5),
            'extent': [88, 93, 20, 27],
        },
    }

    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # Background features
    ax.add_feature(cfeature.OCEAN, facecolor='#D6EAF8', zorder=0)
    ax.add_feature(cfeature.LAND, facecolor='#F0F0F0', zorder=1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor='#AAAAAA', zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='#888888', zorder=2)

    # Highlight each country bounding box
    for info in study_regions.values():
        ext = info['extent']
        lon0, lon1, lat0, lat1 = ext
        ax.add_patch(mpatches.Rectangle(
            (lon0, lat0), lon1 - lon0, lat1 - lat0,
            linewidth=1.5, edgecolor=info['color'],
            facecolor=info['color'], alpha=0.5,
            transform=ccrs.PlateCarree(), zorder=3
        ))
        # Label
        ax.text(
            info['center'][0], info['center'][1], info['label'],
            transform=ccrs.PlateCarree(), fontsize=9, fontweight='bold',
            color='black', ha='center', va='center', zorder=4,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none')
        )

    ax.set_extent([-20, 120, -10, 60], crs=ccrs.PlateCarree())
    ax.gridlines(draw_labels=True, linewidth=0.4, color='gray',
                 alpha=0.5, linestyle='--')

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor=info['color'], alpha=0.6,
                       label=info['label'].replace('\n', ' '))
        for info in study_regions.values()
    ]
    ax.legend(handles=legend_handles, loc='lower left', fontsize=10,
              framealpha=0.9)

    ax.set_title('Study Areas: Saudi Arabia, Italy, Bangladesh (ERA5-Land, 2000–2025)',
                 fontsize=13, fontweight='bold', pad=10)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig1_study_area.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print('Saved: fig1_study_area.png')


def plot_spatial_distribution():
    """
    Fig.1b: Spatial distribution of multi-year mean P, ET, R for each country.
    Layout: 3 rows (P/ET/R) x 3 columns (countries).
    """
    SPATIAL_VARS = {
        'P_mm':  {'label': 'Precipitation\n(mm/month)', 'cmap': 'Blues'},
        'ET_mm': {'label': 'Evapotranspiration\n(mm/month)', 'cmap': 'YlOrRd'},
        'R_mm':  {'label': 'Runoff\n(mm/month)', 'cmap': 'Greens'},
    }
    var_labels = list(SPATIAL_VARS.values())

    fig, axes = plt.subplots(
        3, 3, figsize=(15, 12),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )
    # Tighten column spacing; leave left margin for row labels
    fig.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.02,
                        wspace=0.15, hspace=0.12)

    for j, (country, info) in enumerate(COUNTRIES.items()):
        nc_name = COUNTRY_NC_NAME[country]
        ds = load_country_dataset(nc_name)
        mean_ds = ds.mean(dim='time', skipna=True)

        for i, (var, var_info) in enumerate(SPATIAL_VARS.items()):
            ax = axes[i, j]
            data_arr = mean_ds[var].values
            lons = mean_ds['lon'].values
            lats = mean_ds['lat'].values

            vmin, vmax = np.nanpercentile(data_arr, [2, 98])
            im = ax.pcolormesh(lons, lats, data_arr,
                               cmap=var_info['cmap'], vmin=vmin, vmax=vmax,
                               transform=ccrs.PlateCarree())
            ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.gridlines(linewidth=0.3, alpha=0.5)

            plt.colorbar(im, ax=ax, orientation='vertical',
                         fraction=0.046, pad=0.04, label='mm/month')

            if i == 0:
                ax.set_title(info['label'], fontsize=11, fontweight='bold')

    # Row labels via fig.text() — cartopy axes ignore set_ylabel
    row_ys = [0.79, 0.50, 0.20]   # approximate vertical center of each row
    for i, var_info in enumerate(var_labels):
        fig.text(0.01, row_ys[i], var_info['label'],
                 fontsize=10, fontweight='bold',
                 ha='left', va='center', rotation=90)

    fig.suptitle('Multi-Year Mean Spatial Distribution (2000–2025)',
                 fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(FIG_DIR, 'fig1b_spatial_distribution.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print('Saved: fig1b_spatial_distribution.png')


def plot_water_balance_timeseries(data_dict):
    """
    Fig.2: Monthly P, ET, R time series for three countries.
    Three-row subplot layout, one country per row.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    for ax, (name, info) in zip(axes, COUNTRIES.items()):
        df = data_dict[name]
        ax.plot(df['date'], df['P_mm'], label='P', color='#2980B9', linewidth=0.8)
        ax.plot(df['date'], df['ET_mm'], label='ET', color='#E74C3C', linewidth=0.8)
        ax.plot(df['date'], df['R_mm'], label='R', color='#27AE60', linewidth=0.8)
        ax.fill_between(df['date'], 0, df['P_mm'], alpha=0.15, color='#2980B9')
        ax.set_ylabel('mm/month')
        ax.set_title(info['label'], fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Date')
    plt.suptitle('Monthly Water Balance Components (2000-2025)', fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig2_water_balance_timeseries.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print('Saved: fig2_water_balance_timeseries.png')


def plot_water_balance_structure(annual_dict):
    """
    Fig.3: Multi-year mean water balance structure comparison (stacked bar).
    Shows ET/P and R/P ratios for three countries.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    country_labels = []
    et_ratios = []
    r_ratios = []
    ds_ratios = []
    residual_ratios = []

    for name, info in COUNTRIES.items():
        annual = annual_dict[name]
        P_mean = annual['P_annual'].mean()
        ET_mean = annual['ET_annual'].mean()
        R_mean = annual['R_annual'].mean()
        dS_mean = annual['dS_annual'].mean()

        country_labels.append(info['label'])
        et_ratios.append(ET_mean / P_mean * 100)
        r_ratios.append(R_mean / P_mean * 100)
        ds_ratios.append(abs(dS_mean) / P_mean * 100)
        residual_ratios.append(max(0, (P_mean - ET_mean - R_mean - dS_mean) / P_mean * 100))

    x = np.arange(len(country_labels))
    width = 0.5

    bars_et = ax.bar(x, et_ratios, width, label='ET/P (%)', color='#E74C3C', alpha=0.85)
    bars_r = ax.bar(x, r_ratios, width, bottom=et_ratios, label='R/P (%)',
                    color='#2ECC71', alpha=0.85)

    # Add value labels on bars
    for i, (et, r) in enumerate(zip(et_ratios, r_ratios)):
        ax.text(i, et / 2, f'{et:.1f}%', ha='center', va='center',
                fontsize=10, fontweight='bold', color='white')
        ax.text(i, et + r / 2, f'{r:.1f}%', ha='center', va='center',
                fontsize=10, fontweight='bold', color='white')

    ax.set_xticks(x)
    ax.set_xticklabels(country_labels, fontsize=11)
    ax.set_ylabel('Percentage of Precipitation (%)', fontsize=12)
    ax.set_title('Multi-Year Mean Water Balance Structure', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 120)

    # Add absolute values table below
    table_data = []
    for name in COUNTRIES:
        annual = annual_dict[name]
        table_data.append([
            f"{annual['P_annual'].mean():.1f}",
            f"{annual['ET_annual'].mean():.1f}",
            f"{annual['R_annual'].mean():.1f}",
        ])

    table = ax.table(cellText=table_data,
                     rowLabels=[COUNTRIES[n]['label'] for n in COUNTRIES],
                     colLabels=['P (mm/yr)', 'ET (mm/yr)', 'R (mm/yr)'],
                     loc='bottom', bbox=[0.0, -0.35, 1.0, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig3_water_balance_structure.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print('Saved: fig3_water_balance_structure.png')


def plot_runoff_coefficient(annual_dict):
    """
    Fig.5: Annual runoff coefficient (R/P) inter-annual variation.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    for name, info in COUNTRIES.items():
        annual = annual_dict[name]
        ax.plot(annual['year'], annual['RC_annual'],
                marker='o', markersize=4, linewidth=1.5,
                label=info['label'], color=info['color'])

    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Runoff Coefficient (R/P)', fontsize=12)
    ax.set_title('Annual Runoff Coefficient (2000-2025)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig5_runoff_coefficient.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print('Saved: fig5_runoff_coefficient.png')


# ============================================================================
# Results Export
# ============================================================================
def export_results(data_dict, annual_dict):
    """Export water balance summary tables to CSV."""
    # Monthly data with water balance
    for name in COUNTRIES:
        df = data_dict[name]
        cols = ['date', 'P_mm', 'ET_mm', 'R_mm', 'S_mm', 'dS', 'residual', 'RC']
        df[cols].to_csv(
            os.path.join(RESULTS_DIR, f'{name}_monthly_water_balance.csv'),
            index=False
        )

    # Annual summary comparison
    summary_rows = []
    for name, info in COUNTRIES.items():
        annual = annual_dict[name]
        summary_rows.append({
            'Country': info['label'],
            'P_mean_mm_yr': annual['P_annual'].mean(),
            'ET_mean_mm_yr': annual['ET_annual'].mean(),
            'R_mean_mm_yr': annual['R_annual'].mean(),
            'RC_mean': annual['RC_annual'].mean(),
            'ET_P_ratio': annual['ET_P_ratio'].mean(),
            'R_P_ratio': annual['R_P_ratio'].mean(),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        os.path.join(RESULTS_DIR, 'water_balance_summary.csv'),
        index=False, float_format='%.3f'
    )
    print('Saved: water_balance_summary.csv')
    print(summary_df.to_string(index=False))


# ============================================================================
# Main
# ============================================================================
def main():
    print('=' * 60)
    print('Water Balance Analysis')
    print('=' * 60)

    # Load data
    print('\nLoading data...')
    data = load_all_countries()

    # Compute water balance
    print('Computing water balance...')
    for name in COUNTRIES:
        data[name] = compute_water_balance(data[name])

    # Annual aggregation
    annual = {}
    for name in COUNTRIES:
        annual[name] = compute_annual(data[name])

    # Visualizations
    print('\nGenerating figures...')
    plot_study_area()
    plot_spatial_distribution()
    plot_water_balance_timeseries(data)
    plot_water_balance_structure(annual)
    plot_runoff_coefficient(annual)

    # Export results
    print('\nExporting results...')
    export_results(data, annual)

    print('\nWater balance analysis complete!')


if __name__ == '__main__':
    main()
