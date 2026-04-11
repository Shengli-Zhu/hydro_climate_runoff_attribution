"""
03_xgboost_shap.py
XGBoost Regression + SHAP Attribution Analysis

For each country:
  - Train XGBoost to predict monthly runoff (R) from climate features
  - Temporal train/test split (80/20)
  - TimeSeriesSplit cross-validation for hyperparameter tuning
  - SHAP analysis for feature attribution
Outputs:
  - Fig.6: Predicted vs Observed scatter (3 panels)
  - Fig.7: SHAP Summary Plots (3 panels)
  - Fig.8: SHAP Bar Plot (3-country comparison)
  - Fig.9: SHAP Dependence Plot for precipitation (3 panels)
  - Model performance metrics CSV
"""

import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
import shap
import matplotlib.pyplot as plt
import warnings
import os

sys.path.insert(0, os.path.dirname(__file__))
from utils_load import load_pixel_dataframe

warnings.filterwarnings('ignore')

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

FEATURE_NAMES = ['P_mm', 'T_C', 'Td_C', 'Rn_sw', 'Rn_lw', 'Wind', 'Ts_C', 'S_prev']
FEATURE_LABELS = {
    'P_mm': 'Precipitation',
    'T_C': 'Temperature',
    'Td_C': 'Dewpoint Temp',
    'Rn_sw': 'Net SW Radiation',
    'Rn_lw': 'Net LW Radiation',
    'Wind': 'Wind Speed',
    'Ts_C': 'Soil Temperature',
    'S_prev': 'Antecedent Soil Water',
}

RANDOM_STATE = 42

FEATURE_NAMES_ANNUAL = ['P_annual', 'T_annual', 'Td_annual',
                         'Rn_sw_annual', 'Rn_lw_annual',
                         'Wind_annual', 'Ts_annual', 'dS_annual']

FEATURE_LABELS_ANNUAL = {
    'P_annual':     'Precipitation',
    'T_annual':     'Temperature',
    'Td_annual':    'Dewpoint Temp',
    'Rn_sw_annual': 'Net SW Radiation',
    'Rn_lw_annual': 'Net LW Radiation',
    'Wind_annual':  'Wind Speed',
    'Ts_annual':    'Soil Temperature',
    'dS_annual':    'Soil Water Change',
}


# ============================================================================
# Data Loading & Feature Engineering
# ============================================================================
def load_and_prepare(country_name):
    """Load pixel-level data from NetCDF and prepare features for ML."""
    nc_name = COUNTRY_NC_NAME.get(country_name, country_name)
    df = load_pixel_dataframe(nc_name)  # already has S_prev, dS
    df = df.dropna(subset=FEATURE_NAMES + ['R_mm']).reset_index(drop=True)
    return df


def aggregate_to_annual(df):
    """Aggregate pixel-level monthly df to annual pixel df."""
    agg = df.groupby(['pixel_id', 'lat', 'lon', 'year']).agg(
        P_annual=('P_mm',  'sum'),
        ET_annual=('ET_mm', 'sum'),
        R_annual=('R_mm',  'sum'),
        T_annual=('T_C',   'mean'),
        Td_annual=('Td_C', 'mean'),
        Rn_sw_annual=('Rn_sw', 'sum'),
        Rn_lw_annual=('Rn_lw', 'sum'),
        Wind_annual=('Wind', 'mean'),
        Ts_annual=('Ts_C', 'mean'),
        dS_annual=('dS',   'sum'),
    ).reset_index()
    return agg.dropna(subset=FEATURE_NAMES_ANNUAL + ['R_annual'])


# ============================================================================
# Model Training
# ============================================================================
def train_xgboost(df, country_name, tune_hyperparams=True):
    """
    Train XGBoost model with temporal split.
    Returns: model, X_test, y_test, y_pred, metrics
    """
    X = df[FEATURE_NAMES].values
    y = df['R_mm'].values
    dates = df['time'].values

    # Year-based temporal split: 1950-2004 train (~73%), 2005-2025 test (~27%)
    train_mask = df['year'] <= 2004
    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]
    dates_test = dates[~train_mask]

    if tune_hyperparams:
        # Hyperparameter tuning with GridSearchCV
        # Note: tune_hyperparams=False recommended for large datasets (~1M+ rows)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.05, 0.1, 0.2],
            'subsample': [0.8],
            'colsample_bytree': [0.8],
        }

        base_model = xgb.XGBRegressor(random_state=RANDOM_STATE)
        grid_search = GridSearchCV(
            base_model, param_grid, cv=3,
            scoring='r2', n_jobs=-1, verbose=0
        )
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        print(f'  {country_name} best params: {best_params}')
    else:
        model = xgb.XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE
        )
        model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    nse = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2)

    # KGE: Kling-Gupta Efficiency
    r = np.corrcoef(y_test, y_pred)[0, 1]
    alpha = y_pred.std() / (y_test.std() + 1e-10)
    beta = y_pred.mean() / (y_test.mean() + 1e-10)
    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

    metrics = {'R2': r2, 'RMSE': rmse, 'NSE': nse, 'KGE': kge}
    print(f'  {country_name}: R2={r2:.3f}, RMSE={rmse:.2f}, NSE={nse:.3f}, KGE={kge:.3f}')

    return model, X_train, X_test, y_test, y_pred, dates_test, metrics


def train_xgboost_annual(df_annual, country_name):
    """Train XGBoost on annual pixel data. Train: ≤2004, Test: ≥2005."""
    X = df_annual[FEATURE_NAMES_ANNUAL].values
    y = df_annual['R_annual'].values
    train_mask = df_annual['year'] <= 2004
    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]

    model = xgb.XGBRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    nse  = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2)
    r    = np.corrcoef(y_test, y_pred)[0, 1]
    alpha = y_pred.std() / (y_test.std() + 1e-10)
    beta  = y_pred.mean() / (y_test.mean() + 1e-10)
    kge   = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

    metrics = {'R2': r2, 'RMSE': rmse, 'NSE': nse, 'KGE': kge}
    print(f'  {country_name} Annual: R2={r2:.3f}, RMSE={rmse:.2f}, NSE={nse:.3f}')
    return model, X_test, y_test, y_pred, metrics


# ============================================================================
# SHAP Analysis
# ============================================================================
def compute_shap(model, X_test, max_samples=5000):
    """Compute SHAP values using TreeExplainer (subsampled for speed)."""
    explainer = shap.TreeExplainer(model)
    if len(X_test) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_test), max_samples, replace=False)
        X_sample = X_test[idx]
    else:
        X_sample = X_test
    shap_values = explainer.shap_values(X_sample)
    return explainer, shap_values, X_sample


# ============================================================================
# Visualization
# ============================================================================
def plot_predicted_vs_observed(results):
    """
    Fig.6: Predicted vs Observed scatter plots (3 panels).
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for ax, (country, info) in zip(axes, COUNTRIES.items()):
        res = results[country]
        y_test = res['y_test']
        y_pred = res['y_pred']
        metrics = res['metrics']

        ax.scatter(y_test, y_pred, alpha=0.6, s=20, color=info['color'],
                   edgecolors='white', linewidth=0.3)

        # 1:1 line
        lims = [min(y_test.min(), y_pred.min()),
                max(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, 'k--', linewidth=1, alpha=0.7)

        ax.set_xlabel('Observed R (mm/month)', fontsize=11)
        ax.set_ylabel('Predicted R (mm/month)', fontsize=11)
        ax.set_title(info['label'], fontsize=12, fontweight='bold')

        # Metrics text
        text = f"R²={metrics['R2']:.3f}\nRMSE={metrics['RMSE']:.2f}\nNSE={metrics['NSE']:.3f}\nKGE={metrics['KGE']:.3f}"
        ax.text(0.05, 0.95, text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

    plt.suptitle('XGBoost: Predicted vs Observed Monthly Runoff',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig09_predicted_vs_observed.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print('Saved: fig09_predicted_vs_observed.png')


def plot_annual_predicted_vs_observed(annual_results):
    """Fig.9b: Annual predicted vs observed scatter (3 panels)."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for ax, (country, info) in zip(axes, COUNTRIES.items()):
        res = annual_results[country]
        y_test = res['y_test']
        y_pred = res['y_pred']
        metrics = res['metrics']

        ax.scatter(y_test, y_pred, alpha=0.6, s=20, color=info['color'],
                   edgecolors='white', linewidth=0.3)
        lims = [min(y_test.min(), y_pred.min()),
                max(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, 'k--', linewidth=1, alpha=0.7)
        ax.set_xlabel('Observed R (mm/year)', fontsize=11)
        ax.set_ylabel('Predicted R (mm/year)', fontsize=11)
        ax.set_title(info['label'], fontsize=12, fontweight='bold')
        text = (f"R2={metrics['R2']:.3f}\nRMSE={metrics['RMSE']:.2f}\n"
                f"NSE={metrics['NSE']:.3f}\nKGE={metrics['KGE']:.3f}")
        ax.text(0.05, 0.95, text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

    plt.suptitle('Annual XGBoost: Predicted vs Observed Annual Runoff',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig09b_annual_predicted_vs_observed.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print('Saved: fig09b_annual_predicted_vs_observed.png')


def plot_shap_summary(results):
    """
    Fig.7: SHAP Summary (beeswarm) plots, 3 panels.
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for ax, (country, info) in zip(axes, COUNTRIES.items()):
        res = results[country]
        plt.sca(ax)
        shap.summary_plot(
            res['shap_values'],
            pd.DataFrame(res['X_shap'], columns=FEATURE_NAMES),
            feature_names=[FEATURE_LABELS[f] for f in FEATURE_NAMES],
            show=False,
            plot_size=None,
        )
        ax.set_title(info['label'], fontsize=12, fontweight='bold')
        ax.set_ylabel('Features', fontsize=10)

    plt.suptitle('SHAP Summary: Feature Contributions to Runoff',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig11_shap_summary.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print('Saved: fig11_shap_summary.png')


def plot_shap_bar_comparison(results):
    """
    Fig.8: SHAP feature importance bar chart (3-country side-by-side).
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    importance_data = {}
    for country, info in COUNTRIES.items():
        shap_vals = results[country]['shap_values']
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        importance_data[info['label']] = mean_abs_shap

    x = np.arange(len(FEATURE_NAMES))
    width = 0.25

    for i, (label, values) in enumerate(importance_data.items()):
        color = list(COUNTRIES.values())[i]['color']
        ax.barh(x + i * width, values, width, label=label, color=color, alpha=0.85)

    ax.set_yticks(x + width)
    ax.set_yticklabels([FEATURE_LABELS[f] for f in FEATURE_NAMES], fontsize=11)
    ax.set_xlabel('Mean |SHAP value| (mm/month)', fontsize=12)
    ax.set_ylabel('Features', fontsize=11)
    ax.set_title('Feature Importance Comparison Across Countries',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig12_shap_bar_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print('Saved: fig12_shap_bar_comparison.png')


def plot_shap_dependence_precipitation(results):
    """
    Fig.9: SHAP Dependence plot for Precipitation across 3 countries.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    p_idx = FEATURE_NAMES.index('P_mm')

    for ax, (country, info) in zip(axes, COUNTRIES.items()):
        res = results[country]
        X_df = pd.DataFrame(res['X_shap'], columns=FEATURE_NAMES)
        shap_vals = res['shap_values']

        scatter = ax.scatter(
            X_df['P_mm'], shap_vals[:, p_idx],
            c=X_df['T_C'], cmap='RdYlBu_r', s=20, alpha=0.7,
            edgecolors='white', linewidth=0.3
        )
        ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
        ax.set_xlabel('Precipitation (mm/month)', fontsize=11)
        ax.set_ylabel('SHAP value for Precipitation', fontsize=11)
        ax.set_title(info['label'], fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.colorbar(scatter, ax=ax, label='Temperature (°C)', shrink=0.8)

    plt.suptitle('SHAP Dependence: Precipitation Effect on Runoff',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig13_shap_dependence_P.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print('Saved: fig13_shap_dependence_P.png')



# ============================================================================
# Temporal SHAP Analysis
# ============================================================================
def compute_shap_temporal(model, df):
    """
    Full-data per-year SHAP: use all pixel-months in each year,
    return dict {year: mean_shap_vector (8,)}.
    No subsampling — TreeExplainer forward-pass is fast.
    """
    explainer = shap.TreeExplainer(model)
    results_by_year = {}
    for year, grp in df.groupby('year'):
        X_samp = grp[FEATURE_NAMES].values
        sv = explainer.shap_values(X_samp)
        results_by_year[year] = sv.mean(axis=0)
    return results_by_year


def plot_shap_temporal(temporal_by_country):
    """
    Fig.10: Annual mean SHAP values over 1950-2025, one panel per country.
    Raw annual values shown as thin lines; 3-year rolling mean as thick lines.
    temporal_by_country: {country: {year: mean_shap_vector}}
    """
    colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    for ax, (country, info) in zip(axes, COUNTRIES.items()):
        shap_by_year = temporal_by_country[country]
        years = sorted(shap_by_year.keys())
        shap_matrix = np.array([shap_by_year[y] for y in years])

        for i, feat in enumerate(FEATURE_NAMES):
            c = colors[i % len(colors)]
            # 3-year rolling mean (monthly model)
            rolled = pd.Series(shap_matrix[:, i], index=years).rolling(3, center=True).mean()
            ax.plot(years, rolled.values,
                    label=FEATURE_LABELS[feat],
                    color=c, linewidth=2.0)

        ax.axhline(y=0, color='black', linewidth=0.5, linestyle='-', alpha=0.4)
        ax.set_ylabel('Mean SHAP (mm/month)', fontsize=10)
        ax.set_title(info['label'], fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=7, ncol=2)

    axes[-1].set_xlabel('Year', fontsize=11)
    plt.suptitle('Temporal Evolution of SHAP Feature Contributions (Annual Mean, 1950–2025)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig10_shap_temporal.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print('Saved: fig10_shap_temporal.png')


def plot_annual_shap_temporal(annual_results):
    """
    Fig.10b: Annual SHAP temporal evolution using annual XGBoost model.
    One data point per year — no rolling mean needed.
    annual_results: {country: {'df_annual': df, 'model': model}}
    """
    colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    for ax, (country, info) in zip(axes, COUNTRIES.items()):
        df_annual = annual_results[country]['df_annual']
        model     = annual_results[country]['model']
        explainer = shap.TreeExplainer(model)

        shap_by_year = {}
        for year, grp in df_annual.groupby('year'):
            sv = explainer.shap_values(grp[FEATURE_NAMES_ANNUAL].values)
            shap_by_year[year] = sv.mean(axis=0)

        years       = sorted(shap_by_year.keys())
        shap_matrix = np.array([shap_by_year[y] for y in years])

        for i, feat in enumerate(FEATURE_NAMES_ANNUAL):
            c = colors[i % len(colors)]
            ax.plot(years, shap_matrix[:, i], 'o-',
                    label=FEATURE_LABELS_ANNUAL[feat],
                    color=c, linewidth=1.5, markersize=3)

        ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.4)
        ax.set_ylabel('Mean SHAP (mm/year)', fontsize=10)
        ax.set_title(info['label'], fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=7, ncol=2)

    axes[-1].set_xlabel('Year', fontsize=11)
    plt.suptitle('Annual SHAP: Climate Drivers of Annual Runoff (1950–2025)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig10b_annual_shap_temporal.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print('Saved: fig10b_annual_shap_temporal.png')


def plot_annual_shap_summary(annual_results):
    """Fig.11b: SHAP beeswarm for annual model (3 panels)."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for ax, (country, info) in zip(axes, COUNTRIES.items()):
        res = annual_results[country]
        plt.sca(ax)
        shap.summary_plot(
            res['shap_values'],
            pd.DataFrame(res['X_shap'], columns=FEATURE_NAMES_ANNUAL),
            feature_names=[FEATURE_LABELS_ANNUAL[f] for f in FEATURE_NAMES_ANNUAL],
            show=False,
            plot_size=None,
        )
        ax.set_title(info['label'], fontsize=12, fontweight='bold')
        ax.set_ylabel('Features', fontsize=10)

    plt.suptitle('Annual SHAP Summary: Feature Contributions to Annual Runoff',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig11b_annual_shap_summary.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print('Saved: fig11b_annual_shap_summary.png')


def plot_annual_shap_bar_comparison(annual_results):
    """Fig.12b: Annual SHAP feature importance bar chart (3-country)."""
    fig, ax = plt.subplots(figsize=(12, 7))

    importance_data = {}
    for country, info in COUNTRIES.items():
        shap_vals = annual_results[country]['shap_values']
        importance_data[info['label']] = np.abs(shap_vals).mean(axis=0)

    x = np.arange(len(FEATURE_NAMES_ANNUAL))
    width = 0.25

    for i, (label, values) in enumerate(importance_data.items()):
        color = list(COUNTRIES.values())[i]['color']
        ax.barh(x + i * width, values, width, label=label, color=color, alpha=0.85)

    ax.set_yticks(x + width)
    ax.set_yticklabels([FEATURE_LABELS_ANNUAL[f] for f in FEATURE_NAMES_ANNUAL], fontsize=11)
    ax.set_xlabel('Mean |SHAP value| (mm/year)', fontsize=12)
    ax.set_ylabel('Features', fontsize=11)
    ax.set_title('Annual Feature Importance Comparison Across Countries',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig12b_annual_shap_bar.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print('Saved: fig12b_annual_shap_bar.png')


def plot_annual_shap_dependence_precipitation(annual_results):
    """Fig.13b: Annual SHAP dependence plot for P_annual (3 panels)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    p_idx = FEATURE_NAMES_ANNUAL.index('P_annual')
    t_idx = FEATURE_NAMES_ANNUAL.index('T_annual')

    for ax, (country, info) in zip(axes, COUNTRIES.items()):
        res = annual_results[country]
        X_df = pd.DataFrame(res['X_shap'], columns=FEATURE_NAMES_ANNUAL)
        shap_vals = res['shap_values']

        scatter = ax.scatter(
            X_df['P_annual'], shap_vals[:, p_idx],
            c=X_df['T_annual'], cmap='RdYlBu_r', s=20, alpha=0.7,
            edgecolors='white', linewidth=0.3
        )
        ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
        ax.set_xlabel('Annual Precipitation (mm/year)', fontsize=11)
        ax.set_ylabel('SHAP value for Annual Precipitation', fontsize=11)
        ax.set_title(info['label'], fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Temperature (°C)', shrink=0.8)

    plt.suptitle('Annual SHAP Dependence: Precipitation Effect on Annual Runoff',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig13b_annual_shap_dependence_P.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print('Saved: fig13b_annual_shap_dependence_P.png')


# ============================================================================
# Results Export
# ============================================================================
def export_metrics(results):
    """Export model performance metrics to CSV."""
    rows = []
    for country, info in COUNTRIES.items():
        m = results[country]['metrics']
        rows.append({
            'Country': info['label'],
            'R2': m['R2'],
            'RMSE': m['RMSE'],
            'NSE': m['NSE'],
            'KGE': m['KGE'],
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, 'xgboost_metrics.csv'),
              index=False, float_format='%.4f')
    print('Saved: xgboost_metrics.csv')
    print(df.to_string(index=False))


def export_shap_importance(results):
    """Export mean |SHAP| values for each country."""
    rows = []
    for country, info in COUNTRIES.items():
        shap_vals = results[country]['shap_values']
        mean_abs = np.abs(shap_vals).mean(axis=0)
        for i, feat in enumerate(FEATURE_NAMES):
            rows.append({
                'Country': info['label'],
                'Feature': FEATURE_LABELS[feat],
                'Mean_Abs_SHAP': mean_abs[i],
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, 'shap_importance.csv'),
              index=False, float_format='%.4f')
    print('Saved: shap_importance.csv')


# ============================================================================
# Main
# ============================================================================
def main():
    print('=' * 60)
    print('XGBoost + SHAP Attribution Analysis')
    print('=' * 60)

    results = {}
    temporal_by_country = {}

    for country, info in COUNTRIES.items():
        print(f'\n--- {info["label"]} ---')
        print('Loading and preparing data...')
        df = load_and_prepare(country)
        print(f'  Samples: {len(df)} pixel-months')

        print('Training XGBoost...')
        model, X_train, X_test, y_test, y_pred, dates_test, metrics = \
            train_xgboost(df, country, tune_hyperparams=False)

        print('Computing SHAP values (test set)...')
        explainer, shap_values, X_shap = compute_shap(model, X_test)

        results[country] = {
            'model': model,
            'X_train': X_train,
            'X_test': X_test,
            'X_shap': X_shap,
            'y_test': y_test,
            'y_pred': y_pred,
            'dates_test': dates_test,
            'metrics': metrics,
            'explainer': explainer,
            'shap_values': shap_values,
        }

        # Temporal SHAP (full dataset, all pixels per year)
        print('Computing temporal SHAP (full dataset, per year)...')
        temporal_by_country[country] = compute_shap_temporal(model, df)

    # Annual model
    print('\n' + '=' * 60)
    print('Annual XGBoost + SHAP')
    annual_results = {}
    for country, info in COUNTRIES.items():
        print(f'\n--- {info["label"]} (Annual) ---')
        df = load_and_prepare(country)
        df_annual = aggregate_to_annual(df)
        print(f'  Annual samples: {len(df_annual)} pixel-years')
        model_a, X_test_a, y_test_a, y_pred_a, metrics_a = \
            train_xgboost_annual(df_annual, country)
        print('Computing annual SHAP values...')
        _, shap_values_a, X_shap_a = compute_shap(model_a, X_test_a)
        annual_results[country] = {
            'df_annual':   df_annual,
            'model':       model_a,
            'metrics':     metrics_a,
            'X_shap':      X_shap_a,
            'shap_values': shap_values_a,
            'y_test':      y_test_a,
            'y_pred':      y_pred_a,
        }

    # Visualizations
    print('\n' + '=' * 60)
    print('Generating figures...')
    plot_predicted_vs_observed(results)              # fig09
    plot_annual_predicted_vs_observed(annual_results) # fig09b
    plot_shap_temporal(temporal_by_country)          # fig10
    plot_annual_shap_temporal(annual_results)             # fig10b
    plot_shap_summary(results)                            # fig11
    plot_annual_shap_summary(annual_results)              # fig11b
    plot_shap_bar_comparison(results)                     # fig12
    plot_annual_shap_bar_comparison(annual_results)       # fig12b
    plot_shap_dependence_precipitation(results)           # fig13
    plot_annual_shap_dependence_precipitation(annual_results)  # fig13b

    # Export results
    print('\nExporting results...')
    export_metrics(results)
    export_shap_importance(results)

    print('\nXGBoost + SHAP analysis complete!')


if __name__ == '__main__':
    main()
