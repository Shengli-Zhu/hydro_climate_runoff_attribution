"""
02_trend_analysis.py
Trend Analysis for Regional-Mean Time Series

Methods: Mann-Kendall test + Sen's slope estimator
Scales: Annual and seasonal (DJF, MAM, JJA, SON)
Outputs:
  - Trend summary table (CSV)
  - Fig.4-style trend visualization (regional mean bar charts)
"""

import pandas as pd
import numpy as np
import pymannkendall as mk
import matplotlib.pyplot as plt
import os

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
    """Load monthly ERA5-Land CSV for one country."""
    filepath = os.path.join(DATA_DIR, f'{country_name}_ERA5Land_Monthly.csv')
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m')
    df = df.sort_values('date').reset_index(drop=True)
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

    for col in VARIABLES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


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
    series = series[series.index <= 2024]
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
                ax.set_ylabel(var_info['unit'], fontsize=10)

            ax.grid(True, alpha=0.3)

    axes[-1, 1].set_xlabel('Year', fontsize=12)
    plt.suptitle('Annual Trends: Mann-Kendall + Sen\'s Slope (2000-2024)',
                 fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig4_annual_trends.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print('Saved: fig4_annual_trends.png')


def plot_trend_heatmap(trend_df):
    """
    Heatmap of Sen's slopes across countries, variables, and seasons.
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)

    scales = ['Annual', 'DJF', 'MAM', 'JJA', 'SON']

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
        ax.set_xticklabels(scales, rotation=45, fontsize=9)
        ax.set_yticks(range(len(country_labels)))
        ax.set_yticklabels(country_labels, fontsize=9)
        ax.set_title(var_info['label'], fontsize=11, fontweight='bold')

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("Sen's Slope Trends (* = significant at α=0.05)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig4_trend_heatmap.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print('Saved: fig4_trend_heatmap.png')


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

    print('\nTrend analysis complete!')


if __name__ == '__main__':
    main()
