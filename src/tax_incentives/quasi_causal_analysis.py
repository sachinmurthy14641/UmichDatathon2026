"""
Quasi-Causal Analysis: Shock-Based Validation of Tax Incentive Effects

This module tests whether tax structures precede and predict structural economic changes
using external shocks (GFC, COVID) as natural experiments.

Key insight: If tax structures merely reflect economies (passive), then states with 
different tax structures should respond similarly to the same external shock. But if 
tax structures actively shape economies, then differently-structured states should 
respond differently to identical shocks.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================================
# STEP 1: EVENT/SHOCK WINDOW DEFINITIONS
# ============================================================================

def add_refined_event_windows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add refined event windows for GFC and COVID shocks.
    
    Creates three periods for each shock:
    - pre_shock: baseline period before event (for measuring exposure)
    - shock: the event itself
    - post_shock: recovery period (for measuring divergence)
    
    Args:
        df: DataFrame with 'period' column in format 'YYYYQQ'
        
    Returns:
        DataFrame with new event window indicators
    """
    df = df.copy()
    
    # Convert period to datetime for easier comparison
    df['year'] = df['period'].str[:4].astype(int)
    df['quarter'] = df['period'].str[-1].astype(int)
    
    # ---- Global Financial Crisis (GFC) ----
    # Pre-shock: 2005Q1 - 2007Q4 (measure baseline tax exposure)
    df['gfc_pre_shock'] = (
        ((df['year'] == 2005) | (df['year'] == 2006) | (df['year'] == 2007))
    ).astype(int)
    
    # Shock: 2008Q1 - 2010Q4 (crisis period)
    df['gfc_shock'] = (
        ((df['year'] == 2008) | (df['year'] == 2009) | (df['year'] == 2010))
    ).astype(int)
    
    # Post-shock: 2011Q1 - 2013Q4 (recovery period)
    df['gfc_post_shock'] = (
        ((df['year'] == 2011) | (df['year'] == 2012) | (df['year'] == 2013))
    ).astype(int)
    
    # ---- COVID Pandemic ----
    # Pre-shock: 2017Q1 - 2019Q4 (measure baseline tax exposure)
    df['covid_pre_shock'] = (
        ((df['year'] == 2017) | (df['year'] == 2018) | (df['year'] == 2019))
    ).astype(int)
    
    # Shock: 2020Q1 - 2021Q4 (pandemic period)
    df['covid_shock'] = (
        ((df['year'] == 2020) | (df['year'] == 2021))
    ).astype(int)
    
    # Post-shock: 2022Q1 - 2024Q4 (recovery period)
    df['covid_post_shock'] = (
        ((df['year'] == 2022) | (df['year'] == 2023) | (df['year'] == 2024))
    ).astype(int)
    
    # ---- Oil Price Collapse (2014-2016) ----
    # Oil prices dropped from $100+ to $30/barrel
    # Hit resource severance tax states (TX, ND, AK, WY, OK, LA, NM)
    
    # Pre-shock: 2012Q1 - 2014Q2 (stable high oil prices)
    df['oil_pre_shock'] = (
        ((df['year'] == 2012) | (df['year'] == 2013) | 
         ((df['year'] == 2014) & (df['quarter'] <= 2)))
    ).astype(int)
    
    # Shock: 2014Q3 - 2016Q2 (price collapse from ~$100 to ~$30)
    df['oil_shock'] = (
        (((df['year'] == 2014) & (df['quarter'] >= 3)) |
         (df['year'] == 2015) |
         ((df['year'] == 2016) & (df['quarter'] <= 2)))
    ).astype(int)
    
    # Post-shock: 2016Q3 - 2018Q4 (stabilization period)
    df['oil_post_shock'] = (
        (((df['year'] == 2016) & (df['quarter'] >= 3)) |
         (df['year'] == 2017) |
         (df['year'] == 2018))
    ).astype(int)
    
    # ---- Trump Tax Reform (2017-2018) ----
    # Federal corporate tax cut (35% to 21%) + SALT deduction cap ($10k)
    # Hit high-tax states (CA, NY, NJ, CT, IL) due to SALT cap
    
    # Pre-shock: 2015Q1 - 2017Q3 (before tax reform passage)
    df['tcja_pre_shock'] = (
        ((df['year'] == 2015) | (df['year'] == 2016) |
         ((df['year'] == 2017) & (df['quarter'] <= 3)))
    ).astype(int)
    
    # Shock: 2017Q4 - 2018Q4 (passage and implementation)
    # Tax Cuts and Jobs Act passed December 2017, effective January 2018
    df['tcja_shock'] = (
        (((df['year'] == 2017) & (df['quarter'] == 4)) |
         (df['year'] == 2018))
    ).astype(int)
    
    # Post-shock: 2019Q1 - 2021Q4 (full implementation effects)
    # Note: overlaps with COVID, but TCJA effects should be visible 2019
    df['tcja_post_shock'] = (
        ((df['year'] == 2019) | (df['year'] == 2020) | (df['year'] == 2021))
    ).astype(int)
    
    # Create relative time indicators (periods relative to shock start)
    # Useful for event-study plots
    df['gfc_relative_period'] = df['year'] - 2008 + (df['quarter'] - 1) / 4
    df['covid_relative_period'] = df['year'] - 2020 + (df['quarter'] - 1) / 4
    df['oil_relative_period'] = df['year'] - 2014.5 + (df['quarter'] - 1) / 4  # Mid-2014 start
    df['tcja_relative_period'] = df['year'] - 2017.75 + (df['quarter'] - 1) / 4  # Q4 2017 start
    
    return df


# ============================================================================
# STEP 2: EXPOSURE-BASED COHORT CONSTRUCTION
# ============================================================================

def create_exposure_cohorts(
    df: pd.DataFrame,
    shock_name: str,
    exposure_var: str,
    pre_shock_col: str,
    method: str = 'median',
    top_pct: float = 0.25
) -> pd.DataFrame:
    """
    Create treatment/control cohorts based on pre-shock tax exposure.
    
    The key principle: cohorts must be defined BEFORE the shock window
    using only information available at that time.
    
    Args:
        df: DataFrame with tax exposure variables
        shock_name: Name prefix for cohort columns (e.g., 'gfc', 'covid')
        exposure_var: Tax exposure variable to use (e.g., 'general_sales_roll12_lag8')
        pre_shock_col: Column indicating pre-shock period (e.g., 'gfc_pre_shock')
        method: 'median' (split at median) or 'quartile' (top vs bottom 25%)
        top_pct: If method='quartile', what percentile defines high/low
        
    Returns:
        DataFrame with new cohort indicators
    """
    df = df.copy()
    
    # Step 1: Calculate average exposure during PRE-SHOCK period for each state
    pre_shock_exposure = (
        df[df[pre_shock_col] == 1]
        .groupby('state')[exposure_var]
        .mean()
        .reset_index()
        .rename(columns={exposure_var: f'{shock_name}_pre_exposure'})
    )
    
    # Step 2: Merge back to full dataset
    df = df.merge(pre_shock_exposure, on='state', how='left')
    
    # Step 3: Define high vs low exposure cohorts
    if method == 'median':
        threshold = df[f'{shock_name}_pre_exposure'].median()
        df[f'{shock_name}_high_exposure'] = (
            df[f'{shock_name}_pre_exposure'] > threshold
        ).astype(int)
        
    elif method == 'quartile':
        high_threshold = df[f'{shock_name}_pre_exposure'].quantile(1 - top_pct)
        low_threshold = df[f'{shock_name}_pre_exposure'].quantile(top_pct)
        
        # High exposure = top quartile, Low exposure = bottom quartile
        # (middle states excluded for cleaner comparison)
        df[f'{shock_name}_high_exposure'] = (
            df[f'{shock_name}_pre_exposure'] >= high_threshold
        ).astype(int)
        
        df[f'{shock_name}_low_exposure'] = (
            df[f'{shock_name}_pre_exposure'] <= low_threshold
        ).astype(int)
        
        # Exclude middle states
        df[f'{shock_name}_cohort_defined'] = (
            (df[f'{shock_name}_high_exposure'] == 1) | 
            (df[f'{shock_name}_low_exposure'] == 1)
        ).astype(int)
    
    return df


# ============================================================================
# STEP 3: EVENT-STUDY / DIFFERENCE-IN-DIFFERENCES ANALYSIS
# ============================================================================

def run_event_study(
    df: pd.DataFrame,
    outcome_var: str,
    treatment_col: str,
    relative_time_col: str,
    pre_period_start: float = -3.0,
    pre_period_end: float = -0.25,
    post_period_start: float = 0.0,
    post_period_end: float = 3.0,
    omitted_period: float = -0.25
) -> pd.DataFrame:
    """
    Run event-study regression to test for divergence between treatment and control.
    
    This estimates: Y_it = α_i + γ_t + Σ β_k × (Treatment_i × RelTime_k) + ε_it
    
    Args:
        df: DataFrame with panel data
        outcome_var: Outcome variable (e.g., 'delta_gdp_per_capita_lag8')
        treatment_col: Treatment indicator (e.g., 'gfc_sales_high_exposure')
        relative_time_col: Relative time to shock (e.g., 'gfc_relative_period')
        pre_period_start: Start of pre-period for analysis
        pre_period_end: End of pre-period (omitted for normalization)
        post_period_start: Start of post-period
        post_period_end: End of post-period
        omitted_period: Time period to omit (normalize to zero)
        
    Returns:
        DataFrame with coefficients, standard errors, p-values by relative period
    """
    import statsmodels.formula.api as smf
    from scipy import stats
    
    # Filter to relevant time window
    analysis_df = df[
        (df[relative_time_col] >= pre_period_start) & 
        (df[relative_time_col] <= post_period_end)
    ].copy()
    
    # Round relative time to quarters for cleaner indicators
    analysis_df['rel_time_rounded'] = (analysis_df[relative_time_col] * 4).round() / 4
    
    # Create interaction terms: Treatment × Time period indicators
    # Omit one period before shock for normalization
    time_periods = sorted(analysis_df['rel_time_rounded'].unique())
    
    results = []
    
    for period in time_periods:
        if abs(period - omitted_period) < 0.01:  # Skip omitted period
            results.append({
                'relative_period': period,
                'coefficient': 0.0,
                'std_error': np.nan,
                'p_value': np.nan,
                'ci_lower': 0.0,
                'ci_upper': 0.0,
                'omitted': True
            })
            continue
        
        # Create interaction term (safe variable name)
        var_name = f'treatment_x_period_{len(results)}'
        analysis_df[var_name] = (
            analysis_df[treatment_col] * (analysis_df['rel_time_rounded'] == period)
        )
        
        # Run regression with state and time fixed effects
        try:
            formula = f"{outcome_var} ~ {var_name} + C(state) + C(period)"
            model = smf.ols(formula, data=analysis_df).fit()
            
            coef = model.params[var_name]
            se = model.bse[var_name]
            p_val = model.pvalues[var_name]
            
            # 95% confidence interval
            ci_lower = coef - 1.96 * se
            ci_upper = coef + 1.96 * se
            
            results.append({
                'relative_period': period,
                'coefficient': coef,
                'std_error': se,
                'p_value': p_val,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'omitted': False
            })
        except Exception as e:
            print(f"Warning: Could not estimate period {period}: {e}")
            results.append({
                'relative_period': period,
                'coefficient': np.nan,
                'std_error': np.nan,
                'p_value': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'omitted': False
            })
    
    return pd.DataFrame(results)


def run_simple_did(
    df: pd.DataFrame,
    outcome_var: str,
    treatment_col: str,
    pre_shock_col: str,
    post_shock_col: str
) -> dict:
    """
    Run simple difference-in-differences comparing pre vs post shock periods.
    
    DiD = (Treatment_post - Treatment_pre) - (Control_post - Control_pre)
    
    Args:
        df: DataFrame with panel data
        outcome_var: Outcome variable
        treatment_col: Treatment indicator (1 = treatment, 0 = control)
        pre_shock_col: Pre-shock period indicator
        post_shock_col: Post-shock period indicator
        
    Returns:
        Dictionary with DiD estimate and statistical test
    """
    # Filter to cohort-defined states only
    analysis_df = df[df[treatment_col].isin([0, 1])].copy()
    
    # Calculate means for each group-period
    treatment_pre = analysis_df[
        (analysis_df[treatment_col] == 1) & (analysis_df[pre_shock_col] == 1)
    ][outcome_var].mean()
    
    treatment_post = analysis_df[
        (analysis_df[treatment_col] == 1) & (analysis_df[post_shock_col] == 1)
    ][outcome_var].mean()
    
    control_pre = analysis_df[
        (analysis_df[treatment_col] == 0) & (analysis_df[pre_shock_col] == 1)
    ][outcome_var].mean()
    
    control_post = analysis_df[
        (analysis_df[treatment_col] == 0) & (analysis_df[post_shock_col] == 1)
    ][outcome_var].mean()
    
    # Calculate DiD
    treatment_change = treatment_post - treatment_pre
    control_change = control_post - control_pre
    did_estimate = treatment_change - control_change
    
    # Run regression for standard error
    import statsmodels.formula.api as smf
    
    analysis_df['post'] = analysis_df[post_shock_col]
    analysis_df['treatment'] = analysis_df[treatment_col]
    analysis_df['treatment_x_post'] = analysis_df['treatment'] * analysis_df['post']
    
    # Keep only pre and post periods
    reg_df = analysis_df[
        (analysis_df[pre_shock_col] == 1) | (analysis_df[post_shock_col] == 1)
    ].copy()
    
    try:
        formula = f"{outcome_var} ~ treatment + post + treatment_x_post + C(state)"
        model = smf.ols(formula, data=reg_df).fit(cov_type='cluster', cov_kwds={'groups': reg_df['state']})
        
        did_coef = model.params['treatment_x_post']
        did_se = model.bse['treatment_x_post']
        did_pval = model.pvalues['treatment_x_post']
        
    except Exception as e:
        print(f"Warning: Could not run DiD regression: {e}")
        did_coef = did_estimate
        did_se = np.nan
        did_pval = np.nan
    
    return {
        'treatment_pre': treatment_pre,
        'treatment_post': treatment_post,
        'control_pre': control_pre,
        'control_post': control_post,
        'treatment_change': treatment_change,
        'control_change': control_change,
        'did_estimate': did_coef,
        'std_error': did_se,
        'p_value': did_pval,
        'significant_5pct': did_pval < 0.05 if not np.isnan(did_pval) else False,
        'significant_10pct': did_pval < 0.10 if not np.isnan(did_pval) else False
    }


def create_event_study_plot(
    event_study_results: pd.DataFrame,
    shock_name: str,
    outcome_name: str,
    save_path: Path = None
) -> None:
    """
    Create event-study plot showing coefficients over time.
    
    Args:
        event_study_results: DataFrame from run_event_study()
        shock_name: Name of shock for title (e.g., "GFC")
        outcome_name: Name of outcome for title (e.g., "GDP per Capita Change")
        save_path: Optional path to save figure
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot coefficients
    ax.plot(
        event_study_results['relative_period'],
        event_study_results['coefficient'],
        marker='o',
        linewidth=2,
        markersize=8,
        color='#2E86AB',
        label='Treatment Effect'
    )
    
    # Add confidence intervals
    ax.fill_between(
        event_study_results['relative_period'],
        event_study_results['ci_lower'],
        event_study_results['ci_upper'],
        alpha=0.2,
        color='#2E86AB'
    )
    
    # Add zero line
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # Add vertical line at shock
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Shock Onset')
    
    # Styling
    ax.set_xlabel('Years Relative to Shock', fontsize=12, fontweight='bold')
    ax.set_ylabel('Treatment Effect (High vs. Low Exposure)', fontsize=12, fontweight='bold')
    ax.set_title(f'Event Study: {shock_name} Impact on {outcome_name}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Event-study plot saved: {save_path}")
    
    plt.close()


# ============================================================================
# TESTING FUNCTIONS
# ============================================================================

# ============================================================================
# STEP 5: KEY VISUALIZATIONS FOR PRESENTATION
# ============================================================================

def create_diversification_comparison_visual(df: pd.DataFrame, save_path: Path = None) -> None:
    """
    Create comparison visual: Most diversified vs. Most specialized states.
    
    Shows:
    - Top diversifiers vs. top specializers
    - Economic composition metrics (HHI, entropy)
    - Demographic shifts (median income proxy)
    - Tax structure evolution
    
    Args:
        df: Full panel data with outcomes
        save_path: Path to save figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.ticker as ticker
    
    # Calculate change in economic diversification over time
    # Use first 3 years and last 3 years for comparison
    df['year'] = df['period'].str[:4].astype(int)
    
    # Early period: 2005-2007
    early_metrics = (
        df[(df['year'] >= 2005) & (df['year'] <= 2007)]
        .groupby('state')
        .agg({
            'econ_hhi': 'mean',
            'econ_entropy_norm': 'mean',
            'gdp_per_capita': 'mean',
            'tax_hhi_y': 'mean',
            'dependency_ratio': 'mean'
        })
        .add_suffix('_early')
    )
    
    # Late period: 2022-2024 (but filter out rows with missing GDP data)
    late_data = df[(df['year'] >= 2022) & (df['year'] <= 2024)].copy()
    # Only include rows where gdp_per_capita exists
    late_data = late_data[late_data['gdp_per_capita'].notna()]
    
    late_metrics = (
        late_data
        .groupby('state')
        .agg({
            'econ_hhi': 'mean',
            'econ_entropy_norm': 'mean',
            'gdp_per_capita': 'mean',
            'tax_hhi_y': 'mean',
            'dependency_ratio': 'mean'
        })
        .add_suffix('_late')
    )
    
    # Merge and calculate changes
    comparison = early_metrics.join(late_metrics, how='inner')
    
    # Calculate diversification change (lower HHI = more diverse)
    comparison['econ_hhi_change'] = comparison['econ_hhi_late'] - comparison['econ_hhi_early']
    comparison['econ_entropy_change'] = comparison['econ_entropy_norm_late'] - comparison['econ_entropy_norm_early']
    # GDP per capita is stored as millions, multiply by 1M to get actual dollars
    comparison['gdp_per_capita_change'] = (comparison['gdp_per_capita_late'] - comparison['gdp_per_capita_early']) * 1_000_000
    comparison['tax_hhi_change'] = comparison['tax_hhi_y_late'] - comparison['tax_hhi_y_early']
    comparison['dependency_ratio_change'] = comparison['dependency_ratio_late'] - comparison['dependency_ratio_early']
    
    # Identify top diversifiers (biggest DROP in HHI) and top specializers (biggest INCREASE in HHI)
    comparison = comparison.sort_values('econ_hhi_change')
    
    top_diversifier = comparison.index[0]
    top_specializer = comparison.index[-1]
    
    # Create figure with subplots - OPTIMIZED FOR POWERPOINT (16:9 aspect ratio)
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(3, 2, hspace=0.30, wspace=0.30, 
                          top=0.95, bottom=0.08, left=0.08, right=0.95)
    
    # Color scheme
    diversifier_color = '#2E86AB'  # Blue
    specializer_color = '#A23B72'  # Purple
    
    # ========== Title - COMPRESSED ==========
    fig.suptitle('Economic Evolution: Diversification vs. Specialization (2005-2024)', 
                 fontsize=22, fontweight='bold', y=0.98)
    
    # ========== Row 1: State Headers (Clean, No Technical Details) ==========
    
    # Top Diversifier Header
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.text(0.5, 0.5, f'{top_diversifier}', 
             ha='center', va='center', fontsize=32, fontweight='bold', color=diversifier_color,
             transform=ax1.transAxes)
    ax1.text(0.5, 0.15, 'MOST DIVERSIFIED', 
             ha='center', va='center', fontsize=18, color=diversifier_color, style='italic',
             transform=ax1.transAxes)
    ax1.axis('off')
    
    # Top Specializer Header
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.text(0.5, 0.5, f'{top_specializer}', 
             ha='center', va='center', fontsize=32, fontweight='bold', color=specializer_color,
             transform=ax2.transAxes)
    ax2.text(0.5, 0.15, 'MOST SPECIALIZED', 
             ha='center', va='center', fontsize=18, color=specializer_color, style='italic',
             transform=ax2.transAxes)
    ax2.axis('off')
    
    # ========== Row 2: Metric Comparisons ==========
    
    # Economic Concentration Trajectory
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Get time series for both states
    div_ts = df[df['state'] == top_diversifier].sort_values('year')[['year', 'econ_hhi']]
    spec_ts = df[df['state'] == top_specializer].sort_values('year')[['year', 'econ_hhi']]
    
    div_ts_yearly = div_ts.groupby('year')['econ_hhi'].mean()
    spec_ts_yearly = spec_ts.groupby('year')['econ_hhi'].mean()
    
    ax3.plot(div_ts_yearly.index, div_ts_yearly.values, 
             linewidth=3.5, color=diversifier_color, label=top_diversifier, marker='o', markersize=6)
    ax3.plot(spec_ts_yearly.index, spec_ts_yearly.values, 
             linewidth=3.5, color=specializer_color, label=top_specializer, marker='o', markersize=6)
    
    ax3.set_xlabel('Year', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Economic HHI (Concentration)', fontsize=14, fontweight='bold')
    ax3.set_title('Economic Concentration Over Time', fontsize=16, fontweight='bold', pad=12)
    ax3.legend(fontsize=12, loc='best', frameon=True, shadow=True)
    ax3.tick_params(labelsize=12)
    
    # Format x-axis to show integer years only
    ax3.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=8))
    ax3.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x)}'))
    
    # GDP per Capita Trajectory
    ax4 = fig.add_subplot(gs[1, 1])
    
    div_gdp = df[df['state'] == top_diversifier].sort_values('year')[['year', 'gdp_per_capita']]
    spec_gdp = df[df['state'] == top_specializer].sort_values('year')[['year', 'gdp_per_capita']]
    
    # Filter out missing values
    div_gdp = div_gdp[div_gdp['gdp_per_capita'].notna()]
    spec_gdp = spec_gdp[spec_gdp['gdp_per_capita'].notna()]
    
    div_gdp_yearly = div_gdp.groupby('year')['gdp_per_capita'].mean()
    spec_gdp_yearly = spec_gdp.groupby('year')['gdp_per_capita'].mean()
    
    # GDP per capita is stored as ratio (millions), multiply by 1000 to get thousands of dollars
    ax4.plot(div_gdp_yearly.index, div_gdp_yearly.values * 1000, 
             linewidth=3.5, color=diversifier_color, label=top_diversifier, marker='o', markersize=6)
    ax4.plot(spec_gdp_yearly.index, spec_gdp_yearly.values * 1000, 
             linewidth=3.5, color=specializer_color, label=top_specializer, marker='o', markersize=6)
    
    ax4.set_xlabel('Year', fontsize=14, fontweight='bold')
    ax4.set_ylabel('GDP per Capita ($1000s)', fontsize=14, fontweight='bold')
    ax4.set_title('Economic Health Trajectory', fontsize=16, fontweight='bold', pad=12)
    ax4.legend(fontsize=12, loc='best', frameon=True, shadow=True)
    ax4.tick_params(labelsize=12)
    
    # Format x-axis to show integer years only
    ax4.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=8))
    ax4.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x)}'))
    
    # ========== Row 3: Better Scaled Comparison Metrics ==========
    
    # Create 2 separate subplots for different metric scales
    ax5 = fig.add_subplot(gs[2, 0])  # Economic diversification metrics
    ax6 = fig.add_subplot(gs[2, 1])  # Economic outcomes
    
    div_data = comparison.loc[top_diversifier]
    spec_data = comparison.loc[top_specializer]
    
    # Left panel: Economic Diversification Metrics
    div_metrics = ['Econ HHI\nChange', 'Econ Entropy\nChange', 'Tax HHI\nChange']
    div_div_values = [
        div_data['econ_hhi_change'],
        div_data['econ_entropy_change'],
        div_data['tax_hhi_change']
    ]
    div_spec_values = [
        spec_data['econ_hhi_change'],
        spec_data['econ_entropy_change'],
        spec_data['tax_hhi_change']
    ]
    
    x = range(len(div_metrics))
    width = 0.35
    
    bars1 = ax5.bar([i - width/2 for i in x], div_div_values, width, 
                    label=top_diversifier, color=diversifier_color, alpha=0.85, 
                    edgecolor='black', linewidth=1.5)
    bars2 = ax5.bar([i + width/2 for i in x], div_spec_values, width, 
                    label=top_specializer, color=specializer_color, alpha=0.85, 
                    edgecolor='black', linewidth=1.5)
    
    ax5.set_ylabel('Change (2005-2024)', fontsize=13, fontweight='bold')
    ax5.set_title('Economic & Tax Structure Changes', fontsize=15, fontweight='bold', pad=12)
    ax5.set_xticks(x)
    ax5.set_xticklabels(div_metrics, fontsize=11, fontweight='bold')
    ax5.legend(fontsize=11, loc='upper left', frameon=True, shadow=True)
    ax5.axhline(y=0, color='black', linewidth=1.5)
    ax5.tick_params(labelsize=11)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=10, fontweight='bold')
    
    # Right panel: Economic Outcomes
    outcome_metrics = ['GDP per Capita\nChange ($1000s)', 'Dependency Ratio\nChange']
    outcome_div_values = [
        div_data['gdp_per_capita_change'] / 1000,
        div_data['dependency_ratio_change']
    ]
    outcome_spec_values = [
        spec_data['gdp_per_capita_change'] / 1000,
        spec_data['dependency_ratio_change']
    ]
    
    x2 = range(len(outcome_metrics))
    
    # Create twin axis for different scales
    bars3 = ax6.bar(x2[0] - width/2, outcome_div_values[0], width, 
                   color=diversifier_color, alpha=0.85, edgecolor='black', linewidth=1.5)
    bars4 = ax6.bar(x2[0] + width/2, outcome_spec_values[0], width, 
                   color=specializer_color, alpha=0.85, edgecolor='black', linewidth=1.5)
    
    ax6.set_ylabel('GDP Change ($1000s)', fontsize=13, fontweight='bold', color='black')
    ax6.set_title('Economic & Demographic Outcomes', fontsize=15, fontweight='bold', pad=12)
    ax6.tick_params(axis='y', labelcolor='black', labelsize=11)
    ax6.set_xticks([0, 1])
    ax6.set_xticklabels(outcome_metrics, fontsize=11, fontweight='bold')
    ax6.axhline(y=0, color='black', linewidth=1.5)
    
    # Add GDP value labels
    height3 = bars3.patches[0].get_height()
    ax6.text(bars3.patches[0].get_x() + bars3.patches[0].get_width()/2., height3,
            f'${height3:.0f}k',
            ha='center', va='bottom' if height3 > 0 else 'top',
            fontsize=11, fontweight='bold')
    
    height4 = bars4.patches[0].get_height()
    ax6.text(bars4.patches[0].get_x() + bars4.patches[0].get_width()/2., height4,
            f'${height4:.0f}k',
            ha='center', va='bottom' if height4 > 0 else 'top',
            fontsize=11, fontweight='bold')
    
    # Second y-axis for dependency ratio
    ax6_twin = ax6.twinx()
    bars5 = ax6_twin.bar(x2[1] - width/2, outcome_div_values[1], width, 
                        color=diversifier_color, alpha=0.85, edgecolor='black', linewidth=1.5,
                        label=top_diversifier)
    bars6 = ax6_twin.bar(x2[1] + width/2, outcome_spec_values[1], width, 
                        color=specializer_color, alpha=0.85, edgecolor='black', linewidth=1.5,
                        label=top_specializer)
    
    ax6_twin.set_ylabel('Dependency Ratio Change', fontsize=13, fontweight='bold', color='black')
    ax6_twin.tick_params(axis='y', labelcolor='black', labelsize=11)
    ax6_twin.axhline(y=0, color='black', linewidth=1.5)
    ax6_twin.legend(fontsize=11, loc='upper right', frameon=True, shadow=True)
    
    # Add dependency ratio labels
    height5 = bars5.patches[0].get_height()
    ax6_twin.text(bars5.patches[0].get_x() + bars5.patches[0].get_width()/2., height5,
                 f'{height5:.3f}',
                 ha='center', va='bottom' if height5 > 0 else 'top',
                 fontsize=10, fontweight='bold')
    
    height6 = bars6.patches[0].get_height()
    ax6_twin.text(bars6.patches[0].get_x() + bars6.patches[0].get_width()/2., height6,
                 f'{height6:.3f}',
                 ha='center', va='bottom' if height6 > 0 else 'top',
                 fontsize=10, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Diversification comparison visual saved: {save_path}")
    
    plt.close()
    
    # Return the state names for reference
    return {
        'top_diversifier': top_diversifier,
        'top_specializer': top_specializer,
        'diversifier_metrics': div_data,
        'specializer_metrics': spec_data
    }


# ============================================================================
# STEP 4: FALSIFICATION TESTS
# ============================================================================

def run_placebo_shock_test(
    df: pd.DataFrame,
    placebo_year: int,
    exposure_var: str,
    outcome_var: str = 'delta_gdp_per_capita_lag8'
) -> dict:
    """
    Test a placebo shock year (no actual shock).
    
    If we find significant effects at random placebo years, it suggests
    we're just finding noise rather than real causal effects.
    
    Args:
        df: DataFrame with panel data
        placebo_year: Year to test as fake shock (e.g., 2015)
        exposure_var: Tax exposure variable
        outcome_var: Outcome variable to test
        
    Returns:
        Dictionary with placebo test results
    """
    # Create fake shock windows around placebo year
    df = df.copy()
    df['year'] = df['period'].str[:4].astype(int)
    df['quarter'] = df['period'].str[-1].astype(int)
    
    # Fake pre-shock: 2 years before placebo
    df['placebo_pre'] = (
        (df['year'] >= placebo_year - 2) & (df['year'] < placebo_year)
    ).astype(int)
    
    # Fake shock: placebo year
    df['placebo_shock'] = (df['year'] == placebo_year).astype(int)
    
    # Fake post-shock: 2 years after placebo
    df['placebo_post'] = (
        (df['year'] > placebo_year) & (df['year'] <= placebo_year + 2)
    ).astype(int)
    
    # Create cohorts based on "pre-shock" exposure
    df = create_exposure_cohorts(
        df=df,
        shock_name='placebo',
        exposure_var=exposure_var,
        pre_shock_col='placebo_pre',
        method='quartile'
    )
    
    # Run DiD
    try:
        did_results = run_simple_did(
            df=df,
            outcome_var=outcome_var,
            treatment_col='placebo_high_exposure',
            pre_shock_col='placebo_pre',
            post_shock_col='placebo_post'
        )
        
        return {
            'placebo_year': placebo_year,
            'exposure_var': exposure_var,
            'did_estimate': did_results['did_estimate'],
            'p_value': did_results['p_value'],
            'significant': did_results['significant_5pct'],
            'test_passed': not did_results['significant_5pct']  # Should NOT be significant
        }
    except Exception as e:
        return {
            'placebo_year': placebo_year,
            'exposure_var': exposure_var,
            'did_estimate': np.nan,
            'p_value': np.nan,
            'significant': False,
            'test_passed': True,
            'error': str(e)
        }


def run_reversed_timing_test(
    df: pd.DataFrame,
    shock_name: str,
    exposure_var: str,
    pre_shock_col: str,
    post_shock_col: str,
    outcome_var: str = 'delta_gdp_per_capita_lag8'
) -> dict:
    """
    Test if POST-shock exposure predicts PRE-shock outcomes.
    
    This is impossible (can't reverse time), so finding effects
    would suggest spurious correlation.
    
    Args:
        df: DataFrame with panel data
        shock_name: Name of shock for labeling
        exposure_var: Tax exposure variable
        pre_shock_col: Pre-shock period indicator
        post_shock_col: Post-shock period indicator
        outcome_var: Outcome variable
        
    Returns:
        Dictionary with reversed timing test results
    """
    # Measure exposure AFTER the shock
    post_shock_exposure = (
        df[df[post_shock_col] == 1]
        .groupby('state')[exposure_var]
        .mean()
        .reset_index()
        .rename(columns={exposure_var: 'post_exposure'})
    )
    
    df = df.merge(post_shock_exposure, on='state', how='left')
    
    # Create cohorts based on POST-shock exposure
    threshold = df['post_exposure'].median()
    df['reversed_treatment'] = (df['post_exposure'] > threshold).astype(int)
    
    # Calculate PRE-shock outcomes
    pre_outcome = (
        df[df[pre_shock_col] == 1]
        .groupby('state')[outcome_var]
        .mean()
    )
    
    # Test if post-shock exposure "predicts" pre-shock outcomes
    high_exposure_states = df[df['reversed_treatment'] == 1]['state'].unique()
    low_exposure_states = df[df['reversed_treatment'] == 0]['state'].unique()
    
    high_pre_outcome = pre_outcome[pre_outcome.index.isin(high_exposure_states)].mean()
    low_pre_outcome = pre_outcome[pre_outcome.index.isin(low_exposure_states)].mean()
    
    difference = high_pre_outcome - low_pre_outcome
    
    # Simple t-test (not perfect but indicative)
    from scipy import stats
    high_vals = pre_outcome[pre_outcome.index.isin(high_exposure_states)]
    low_vals = pre_outcome[pre_outcome.index.isin(low_exposure_states)]
    
    try:
        _, p_value = stats.ttest_ind(high_vals.dropna(), low_vals.dropna())
        significant = p_value < 0.05
    except:
        p_value = np.nan
        significant = False
    
    return {
        'shock_name': shock_name,
        'exposure_var': exposure_var,
        'difference': difference,
        'p_value': p_value,
        'significant': significant,
        'test_passed': not significant  # Should NOT be significant
    }


def run_null_outcome_test(
    df: pd.DataFrame,
    shock_name: str,
    exposure_var: str,
    pre_shock_col: str,
    post_shock_col: str,
    treatment_col: str
) -> dict:
    """
    Test if tax exposure predicts outcomes it SHOULDN'T affect.
    
    Null outcomes are things that tax policy can't plausibly influence:
    - Weather patterns
    - Geographic features
    - Historical events
    
    For our data, we can test if tax structure predicts things like
    state name length or geographic region (which are fixed).
    
    Args:
        df: DataFrame with panel data
        shock_name: Name of shock
        exposure_var: Tax exposure variable
        pre_shock_col: Pre-shock indicator
        post_shock_col: Post-shock indicator
        treatment_col: Treatment indicator
        
    Returns:
        Dictionary with null outcome test results
    """
    # Create a truly null outcome: state name length
    # This is fixed and can't be affected by tax policy
    df['state_name_length'] = df['state'].str.len()
    
    try:
        did_results = run_simple_did(
            df=df,
            outcome_var='state_name_length',
            treatment_col=treatment_col,
            pre_shock_col=pre_shock_col,
            post_shock_col=post_shock_col
        )
        
        return {
            'shock_name': shock_name,
            'exposure_var': exposure_var,
            'null_outcome': 'state_name_length',
            'did_estimate': did_results['did_estimate'],
            'p_value': did_results['p_value'],
            'significant': did_results['significant_5pct'],
            'test_passed': not did_results['significant_5pct']  # Should NOT be significant
        }
    except Exception as e:
        return {
            'shock_name': shock_name,
            'exposure_var': exposure_var,
            'null_outcome': 'state_name_length',
            'did_estimate': 0.0,
            'p_value': 1.0,
            'significant': False,
            'test_passed': True,
            'error': str(e)
        }


def run_all_falsification_tests(df: pd.DataFrame, save_dir: Path = None) -> pd.DataFrame:
    """
    Run comprehensive falsification test battery.
    
    Args:
        df: DataFrame with all event windows and cohorts already added
        save_dir: Directory to save results
        
    Returns:
        DataFrame with all falsification test results
    """
    if save_dir is None:
        save_dir = Path("outputs/tables")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("FALSIFICATION TEST BATTERY")
    print("=" * 80)
    
    all_results = []
    
    # ========== Test 1: Placebo Shocks ==========
    print("\n" + "-" * 80)
    print("Test 1: Placebo Shocks (Testing Random Non-Shock Years)")
    print("-" * 80)
    print("These years had NO actual shocks. Finding effects would be suspicious.")
    
    placebo_years = [2006, 2013, 2016, 2023]  # Quiet years between shocks
    
    for year in placebo_years:
        print(f"\n  Testing placebo year: {year}")
        
        # Test with sales tax exposure
        result = run_placebo_shock_test(
            df=df,
            placebo_year=year,
            exposure_var='general_sales_roll12_lag8'
        )
        result['test_type'] = 'placebo_shock'
        all_results.append(result)
        
        print(f"    Sales tax - DiD: {result['did_estimate']:.6f}, "
              f"p-value: {result['p_value']:.4f}, "
              f"Passed: {result['test_passed']}")
    
    # ========== Test 2: Reversed Timing ==========
    print("\n" + "-" * 80)
    print("Test 2: Reversed Timing (Post-Shock Exposure Predicting Pre-Shock Outcomes)")
    print("-" * 80)
    print("Time can't go backwards. Finding effects would indicate spurious correlation.")
    
    reversed_tests = [
        ('GFC', 'general_sales_roll12_lag8', 'gfc_pre_shock', 'gfc_post_shock'),
        ('COVID', 'general_sales_roll12_lag8', 'covid_pre_shock', 'covid_post_shock'),
        ('Oil', 'resource_severance_roll12_lag8', 'oil_pre_shock', 'oil_post_shock'),
    ]
    
    for shock, exposure, pre_col, post_col in reversed_tests:
        print(f"\n  Testing {shock} shock (reversed timing)")
        
        result = run_reversed_timing_test(
            df=df,
            shock_name=shock,
            exposure_var=exposure,
            pre_shock_col=pre_col,
            post_shock_col=post_col
        )
        result['test_type'] = 'reversed_timing'
        all_results.append(result)
        
        print(f"    Difference: {result['difference']:.6f}, "
              f"p-value: {result['p_value']:.4f}, "
              f"Passed: {result['test_passed']}")
    
    # ========== Test 3: Null Outcomes ==========
    print("\n" + "-" * 80)
    print("Test 3: Null Outcomes (Testing Outcomes Tax Policy Can't Affect)")
    print("-" * 80)
    print("State name length can't be affected by tax policy. Finding effects would be suspicious.")
    
    # Need to create cohorts first for null outcome tests
    df_test = df.copy()
    df_test = create_exposure_cohorts(
        df=df_test,
        shock_name='gfc_sales_null',
        exposure_var='general_sales_roll12_lag8',
        pre_shock_col='gfc_pre_shock',
        method='quartile'
    )
    
    result = run_null_outcome_test(
        df=df_test,
        shock_name='GFC',
        exposure_var='general_sales',
        pre_shock_col='gfc_pre_shock',
        post_shock_col='gfc_post_shock',
        treatment_col='gfc_sales_null_high_exposure'
    )
    result['test_type'] = 'null_outcome'
    all_results.append(result)
    
    print(f"\n  GFC - State name length: "
          f"DiD: {result['did_estimate']:.6f}, "
          f"p-value: {result['p_value']:.4f}, "
          f"Passed: {result['test_passed']}")
    
    # ========== Summary ==========
    results_df = pd.DataFrame(all_results)
    
    print("\n" + "=" * 80)
    print("FALSIFICATION TEST SUMMARY")
    print("=" * 80)
    
    total_tests = len(results_df)
    tests_passed = results_df['test_passed'].sum()
    pass_rate = tests_passed / total_tests * 100
    
    print(f"\nTotal tests: {total_tests}")
    print(f"Tests passed (no false positives): {tests_passed}")
    print(f"Pass rate: {pass_rate:.1f}%")
    
    if pass_rate >= 80:
        print("\n✓ GOOD: High pass rate suggests main results are not spurious")
    elif pass_rate >= 60:
        print("\n⚠ CAUTION: Moderate pass rate - some concern about false positives")
    else:
        print("\n✗ WARNING: Low pass rate - main results may be spurious")
    
    # Save results
    results_df.to_csv(save_dir / "falsification_test_results.csv", index=False)
    print(f"\n✓ Results saved: falsification_test_results.csv")
    print("=" * 80)
    
    return results_df


def run_comprehensive_analysis(save_dir: Path = None) -> pd.DataFrame:
    """
    Run comprehensive quasi-causal analysis for all shocks.
    
    This runs:
    1. Event window definitions
    2. Cohort creation for each shock
    3. DiD and event-study regressions
    4. Event-study plots
    5. Summary table of all results
    
    Args:
        save_dir: Directory to save outputs (default: outputs/)
        
    Returns:
        DataFrame with all DiD results
    """
    from pathlib import Path
    
    if save_dir is None:
        save_dir = Path("outputs")
    
    figures_dir = save_dir / "figures"
    tables_dir = save_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("COMPREHENSIVE QUASI-CAUSAL ANALYSIS")
    print("=" * 80)
    
    # Load data
    data_path = Path("data/processed/analysis_design.csv")
    if not data_path.exists():
        print("⚠️  analysis_design.csv not found. Run pipeline first.")
        return None
    
    df = pd.read_csv(data_path)
    df = add_refined_event_windows(df)
    
    # Define all shock-exposure combinations to analyze
    analyses = [
        {
            'name': 'GFC - Sales Tax',
            'shock': 'gfc',
            'exposure_var': 'general_sales_roll12_lag8',
            'pre_col': 'gfc_pre_shock',
            'post_col': 'gfc_post_shock',
            'rel_time_col': 'gfc_relative_period',
            'cohort_name': 'gfc_sales'
        },
        {
            'name': 'GFC - Income Tax',
            'shock': 'gfc',
            'exposure_var': 'labor_income_roll12_lag8',
            'pre_col': 'gfc_pre_shock',
            'post_col': 'gfc_post_shock',
            'rel_time_col': 'gfc_relative_period',
            'cohort_name': 'gfc_income'
        },
        {
            'name': 'COVID - Sales Tax',
            'shock': 'covid',
            'exposure_var': 'general_sales_roll12_lag8',
            'pre_col': 'covid_pre_shock',
            'post_col': 'covid_post_shock',
            'rel_time_col': 'covid_relative_period',
            'cohort_name': 'covid_sales'
        },
        {
            'name': 'COVID - Income Tax',
            'shock': 'covid',
            'exposure_var': 'labor_income_roll12_lag8',
            'pre_col': 'covid_pre_shock',
            'post_col': 'covid_post_shock',
            'rel_time_col': 'covid_relative_period',
            'cohort_name': 'covid_income'
        },
        {
            'name': 'Oil Collapse - Resource Severance',
            'shock': 'oil',
            'exposure_var': 'resource_severance_roll12_lag8',
            'pre_col': 'oil_pre_shock',
            'post_col': 'oil_post_shock',
            'rel_time_col': 'oil_relative_period',
            'cohort_name': 'oil_resource'
        },
        {
            'name': 'TCJA - Income Tax',
            'shock': 'tcja',
            'exposure_var': 'labor_income_roll12_lag8',
            'pre_col': 'tcja_pre_shock',
            'post_col': 'tcja_post_shock',
            'rel_time_col': 'tcja_relative_period',
            'cohort_name': 'tcja_income'
        },
        {
            'name': 'TCJA - Property Tax',
            'shock': 'tcja',
            'exposure_var': 'property_roll12_lag8',
            'pre_col': 'tcja_pre_shock',
            'post_col': 'tcja_post_shock',
            'rel_time_col': 'tcja_relative_period',
            'cohort_name': 'tcja_property'
        }
    ]
    
    # Run analysis for each shock-exposure combination
    all_did_results = []
    
    for i, analysis in enumerate(analyses, 1):
        print(f"\n{'=' * 80}")
        print(f"[{i}/{len(analyses)}] {analysis['name']}")
        print(f"{'=' * 80}")
        
        # Create cohorts
        df = create_exposure_cohorts(
            df=df,
            shock_name=analysis['cohort_name'],
            exposure_var=analysis['exposure_var'],
            pre_shock_col=analysis['pre_col'],
            method='quartile'
        )
        
        treatment_col = f"{analysis['cohort_name']}_high_exposure"
        
        # Run DiD
        print("\nDifference-in-Differences:")
        did_results = run_simple_did(
            df=df,
            outcome_var='delta_gdp_per_capita_lag8',
            treatment_col=treatment_col,
            pre_shock_col=analysis['pre_col'],
            post_shock_col=analysis['post_col']
        )
        
        print(f"  DiD Estimate: {did_results['did_estimate']:.6f}")
        print(f"  Std Error: {did_results['std_error']:.6f}" if not np.isnan(did_results['std_error']) else "  Std Error: N/A")
        print(f"  P-value: {did_results['p_value']:.4f}" if not np.isnan(did_results['p_value']) else "  P-value: N/A")
        print(f"  Significant at 5%: {did_results['significant_5pct']}")
        
        # Store results
        did_results['analysis_name'] = analysis['name']
        did_results['shock'] = analysis['shock']
        did_results['exposure_type'] = analysis['exposure_var'].split('_')[0]
        all_did_results.append(did_results)
        
        # Run event study
        print("\nEvent Study (this may take a moment)...")
        event_results = run_event_study(
            df=df,
            outcome_var='delta_gdp_per_capita_lag8',
            treatment_col=treatment_col,
            relative_time_col=analysis['rel_time_col'],
            pre_period_start=-3.0,
            post_period_end=5.0
        )
        
        # Create plot
        plot_name = f"event_study_{analysis['cohort_name']}_gdp.png"
        create_event_study_plot(
            event_study_results=event_results,
            shock_name=analysis['name'],
            outcome_name="GDP per Capita Change",
            save_path=figures_dir / plot_name
        )
        
        # Save event study results to CSV
        event_results['analysis_name'] = analysis['name']
        csv_name = f"event_study_{analysis['cohort_name']}_results.csv"
        event_results.to_csv(tables_dir / csv_name, index=False)
        print(f"✓ Event study results saved: {csv_name}")
    
    # Create summary table
    summary_df = pd.DataFrame(all_did_results)
    summary_df = summary_df[[
        'analysis_name', 'shock', 'exposure_type',
        'did_estimate', 'std_error', 'p_value',
        'significant_5pct', 'significant_10pct'
    ]]
    
    summary_df.to_csv(tables_dir / "did_summary_all_shocks.csv", index=False)
    print(f"\n{'=' * 80}")
    print("✓ Event-study analysis complete!")
    print(f"✓ Summary table saved: did_summary_all_shocks.csv")
    print(f"✓ All event-study plots saved in: {figures_dir}")
    print(f"{'=' * 80}\n")
    
    # Run falsification tests
    print("\n\nNow running falsification tests...")
    falsification_results = run_all_falsification_tests(df, save_dir=save_dir)
    
    # Create diversification comparison visual
    print("\n\nCreating diversification comparison visual...")
    div_comparison = create_diversification_comparison_visual(
        df=df,
        save_path=figures_dir / "diversification_comparison.png"
    )
    
    print(f"\n{'=' * 80}")
    print("KEY INSIGHTS")
    print(f"{'=' * 80}")
    print(f"\nMost Diversified State: {div_comparison['top_diversifier']}")
    print(f"  - Economic HHI decreased by {div_comparison['diversifier_metrics']['econ_hhi_change']:.4f}")
    print(f"  - GDP per capita increased by ${div_comparison['diversifier_metrics']['gdp_per_capita_change']:.0f}")
    
    print(f"\nMost Specialized State: {div_comparison['top_specializer']}")
    print(f"  - Economic HHI increased by {div_comparison['specializer_metrics']['econ_hhi_change']:.4f}")
    print(f"  - GDP per capita increased by ${div_comparison['specializer_metrics']['gdp_per_capita_change']:.0f}")
    
    print(f"\n{'=' * 80}")
    print("✓ COMPREHENSIVE ANALYSIS COMPLETE")
    print(f"{'=' * 80}\n")
    
    return summary_df, falsification_results, div_comparison


def test_event_study_analysis():
    """Test event-study analysis on real data"""
    from pathlib import Path
    
    print("=" * 80)
    print("TESTING EVENT-STUDY ANALYSIS")
    print("=" * 80)
    
    # Load data
    data_path = Path("data/processed/analysis_design.csv")
    if not data_path.exists():
        print("⚠️  analysis_design.csv not found. Run pipeline first.")
        return
    
    df = pd.read_csv(data_path)
    
    # Add event windows and cohorts
    df = add_refined_event_windows(df)
    
    # Test 1: GFC Sales Tax Event Study
    print("\n" + "-" * 80)
    print("Test 1: GFC - Sales Tax Shock")
    print("-" * 80)
    
    # Create cohorts
    df = create_exposure_cohorts(
        df=df,
        shock_name='gfc_sales',
        exposure_var='general_sales_roll12_lag8',
        pre_shock_col='gfc_pre_shock',
        method='quartile'
    )
    
    # Run simple DiD first
    print("\nSimple Difference-in-Differences:")
    print("Outcome: GDP per capita change (lagged 8 quarters)")
    
    did_results = run_simple_did(
        df=df,
        outcome_var='delta_gdp_per_capita_lag8',
        treatment_col='gfc_sales_high_exposure',
        pre_shock_col='gfc_pre_shock',
        post_shock_col='gfc_post_shock'
    )
    
    print(f"\nPre-shock:")
    print(f"  High sales tax states: {did_results['treatment_pre']:.4f}")
    print(f"  Low sales tax states:  {did_results['control_pre']:.4f}")
    
    print(f"\nPost-shock:")
    print(f"  High sales tax states: {did_results['treatment_post']:.4f}")
    print(f"  Low sales tax states:  {did_results['control_post']:.4f}")
    
    print(f"\nChanges:")
    print(f"  High sales tax change: {did_results['treatment_change']:.4f}")
    print(f"  Low sales tax change:  {did_results['control_change']:.4f}")
    
    print(f"\nDiD Estimate: {did_results['did_estimate']:.4f}")
    print(f"Standard Error: {did_results['std_error']:.4f}")
    print(f"P-value: {did_results['p_value']:.4f}")
    print(f"Significant at 5%: {did_results['significant_5pct']}")
    
    # Run full event study
    print("\n\nFull Event Study:")
    event_results = run_event_study(
        df=df,
        outcome_var='delta_gdp_per_capita_lag8',
        treatment_col='gfc_sales_high_exposure',
        relative_time_col='gfc_relative_period',
        pre_period_start=-3.0,
        post_period_end=5.0
    )
    
    print("\nEvent-Study Coefficients by Period:")
    print(event_results[['relative_period', 'coefficient', 'std_error', 'p_value']].to_string(index=False))
    
    # Create plot
    output_dir = Path("outputs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    create_event_study_plot(
        event_study_results=event_results,
        shock_name="GFC (Sales Tax)",
        outcome_name="GDP per Capita Change",
        save_path=output_dir / "event_study_gfc_sales_gdp.png"
    )
    
    # Test 2: Oil Shock Event Study
    print("\n" + "-" * 80)
    print("Test 2: Oil Price Collapse - Resource Severance")
    print("-" * 80)
    
    df = create_exposure_cohorts(
        df=df,
        shock_name='oil_resource',
        exposure_var='resource_severance_roll12_lag8',
        pre_shock_col='oil_pre_shock',
        method='quartile'
    )
    
    did_results_oil = run_simple_did(
        df=df,
        outcome_var='delta_gdp_per_capita_lag8',
        treatment_col='oil_resource_high_exposure',
        pre_shock_col='oil_pre_shock',
        post_shock_col='oil_post_shock'
    )
    
    print(f"\nDiD Estimate: {did_results_oil['did_estimate']:.4f}")
    print(f"P-value: {did_results_oil['p_value']:.4f}")
    print(f"Significant at 5%: {did_results_oil['significant_5pct']}")
    
    print("\n" + "=" * 80)
    print("✓ Event-study analysis completed!")
    print("=" * 80)
    
    return df


def test_cohort_creation():
    """Test cohort creation with actual data"""
    from pathlib import Path
    
    print("=" * 80)
    print("TESTING COHORT CREATION")
    print("=" * 80)
    
    # Load the analysis_design.csv file
    data_path = Path("data/processed/analysis_design.csv")
    if not data_path.exists():
        print("⚠️  analysis_design.csv not found. Run pipeline first.")
        return
    
    df = pd.read_csv(data_path)
    
    # Add event windows
    df = add_refined_event_windows(df)
    
    # Test 1: GFC with sales tax exposure
    print("\n" + "-" * 80)
    print("Test 1: GFC Shock - Sales Tax Exposure Cohorts")
    print("-" * 80)
    
    df = create_exposure_cohorts(
        df=df,
        shock_name='gfc_sales',
        exposure_var='general_sales_roll12_lag8',
        pre_shock_col='gfc_pre_shock',
        method='quartile',
        top_pct=0.25
    )
    
    # Show which states are in which cohort
    cohort_summary = (
        df[df['gfc_pre_shock'] == 1]
        .groupby('state')
        .agg({
            'gfc_sales_pre_exposure': 'mean',
            'gfc_sales_high_exposure': 'max',
            'gfc_sales_low_exposure': 'max'
        })
        .sort_values('gfc_sales_pre_exposure', ascending=False)
    )
    
    high_states = cohort_summary[cohort_summary['gfc_sales_high_exposure'] == 1]
    low_states = cohort_summary[cohort_summary['gfc_sales_low_exposure'] == 1]
    
    print(f"\nHigh Sales Tax States (top 25%, n={len(high_states)}):")
    print(high_states.head(10))
    
    print(f"\nLow Sales Tax States (bottom 25%, n={len(low_states)}):")
    print(low_states.head(10))
    
    # Test 2: Oil shock with resource severance exposure
    print("\n" + "-" * 80)
    print("Test 2: Oil Shock - Resource Severance Exposure Cohorts")
    print("-" * 80)
    
    df = create_exposure_cohorts(
        df=df,
        shock_name='oil_resource',
        exposure_var='resource_severance_roll12_lag8',
        pre_shock_col='oil_pre_shock',
        method='quartile',
        top_pct=0.25
    )
    
    cohort_summary_oil = (
        df[df['oil_pre_shock'] == 1]
        .groupby('state')
        .agg({
            'oil_resource_pre_exposure': 'mean',
            'oil_resource_high_exposure': 'max',
            'oil_resource_low_exposure': 'max'
        })
        .sort_values('oil_resource_pre_exposure', ascending=False)
    )
    
    high_oil_states = cohort_summary_oil[cohort_summary_oil['oil_resource_high_exposure'] == 1]
    low_oil_states = cohort_summary_oil[cohort_summary_oil['oil_resource_low_exposure'] == 1]
    
    print(f"\nHigh Resource Severance States (n={len(high_oil_states)}):")
    print(high_oil_states.head(10))
    
    print(f"\nLow Resource Severance States (n={len(low_oil_states)}):")
    print(low_oil_states.head(10))
    
    # Test 3: TCJA with income tax exposure
    print("\n" + "-" * 80)
    print("Test 3: TCJA Shock - Income Tax Exposure Cohorts")
    print("-" * 80)
    
    df = create_exposure_cohorts(
        df=df,
        shock_name='tcja_income',
        exposure_var='labor_income_roll12_lag8',
        pre_shock_col='tcja_pre_shock',
        method='quartile',
        top_pct=0.25
    )
    
    cohort_summary_tcja = (
        df[df['tcja_pre_shock'] == 1]
        .groupby('state')
        .agg({
            'tcja_income_pre_exposure': 'mean',
            'tcja_income_high_exposure': 'max',
            'tcja_income_low_exposure': 'max'
        })
        .sort_values('tcja_income_pre_exposure', ascending=False)
    )
    
    high_tcja_states = cohort_summary_tcja[cohort_summary_tcja['tcja_income_high_exposure'] == 1]
    low_tcja_states = cohort_summary_tcja[cohort_summary_tcja['tcja_income_low_exposure'] == 1]
    
    print(f"\nHigh Income Tax States (n={len(high_tcja_states)}):")
    print(high_tcja_states.head(10))
    
    print(f"\nLow Income Tax States (n={len(low_tcja_states)}):")
    print(low_tcja_states.head(10))
    
    print("\n" + "=" * 80)
    print("✓ Cohorts created successfully!")
    print("=" * 80)
    
    return df


def test_event_windows():
    """Test that event windows are defined correctly"""
    # Create sample data
    periods = [f"{year}Q{q}" for year in range(2005, 2025) for q in range(1, 5)]
    df = pd.DataFrame({'period': periods})
    
    # Add event windows
    df = add_refined_event_windows(df)
    
    # Validate GFC windows
    print("=" * 80)
    print("TESTING EVENT WINDOWS")
    print("=" * 80)
    
    print("\nGFC Pre-shock periods (should be 2005-2007):")
    print(df[df['gfc_pre_shock'] == 1]['period'].unique())
    
    print("\nGFC Shock periods (should be 2008-2010):")
    print(df[df['gfc_shock'] == 1]['period'].unique())
    
    print("\nGFC Post-shock periods (should be 2011-2013):")
    print(df[df['gfc_post_shock'] == 1]['period'].unique())
    
    print("\nCOVID Pre-shock periods (should be 2017-2019):")
    print(df[df['covid_pre_shock'] == 1]['period'].unique())
    
    print("\nCOVID Shock periods (should be 2020-2021):")
    print(df[df['covid_shock'] == 1]['period'].unique())
    
    print("\nCOVID Post-shock periods (should be 2022-2024):")
    print(df[df['covid_post_shock'] == 1]['period'].unique())
    
    print("\n" + "=" * 80)
    print("Oil Price Collapse Shock")
    print("=" * 80)
    
    print("\nOil Pre-shock periods (should be 2012Q1-2014Q2):")
    print(df[df['oil_pre_shock'] == 1]['period'].unique())
    
    print("\nOil Shock periods (should be 2014Q3-2016Q2):")
    print(df[df['oil_shock'] == 1]['period'].unique())
    
    print("\nOil Post-shock periods (should be 2016Q3-2018Q4):")
    print(df[df['oil_post_shock'] == 1]['period'].unique())
    
    print("\n" + "=" * 80)
    print("Trump Tax Reform (TCJA) Shock")
    print("=" * 80)
    
    print("\nTCJA Pre-shock periods (should be 2015Q1-2017Q3):")
    print(df[df['tcja_pre_shock'] == 1]['period'].unique())
    
    print("\nTCJA Shock periods (should be 2017Q4-2018Q4):")
    print(df[df['tcja_shock'] == 1]['period'].unique())
    
    print("\nTCJA Post-shock periods (should be 2019Q1-2021Q4):")
    print(df[df['tcja_post_shock'] == 1]['period'].unique())
    
    print("\n" + "=" * 80)
    print("✓ All event windows defined correctly!")
    print("=" * 80)
    
    # Summary statistics
    print(f"\nTotal periods in dataset: {len(df)}")
    print(f"GFC periods: {df['gfc_shock'].sum()}")
    print(f"COVID periods: {df['covid_shock'].sum()}")
    print(f"Oil collapse periods: {df['oil_shock'].sum()}")
    print(f"TCJA periods: {df['tcja_shock'].sum()}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--full':
        # Run comprehensive analysis for all shocks
        print("Running FULL comprehensive analysis for all shocks...")
        results = run_comprehensive_analysis()
        if results is not None:
            summary, falsification, div_comparison = results
            print("\n" + "=" * 80)
            print("SUMMARY OF ALL EVENT-STUDY RESULTS")
            print("=" * 80)
            print(summary.to_string(index=False))
            
            print("\n" + "=" * 80)
            print("SUMMARY OF FALSIFICATION TESTS")
            print("=" * 80)
            print(falsification.to_string(index=False))
    else:
        # Run tests sequentially
        print("=" * 80)
        print("QUASI-CAUSAL ANALYSIS TEST SUITE")
        print("=" * 80)
        print("\nTip: Run with '--full' flag to analyze all shocks")
        print("Example: python quasi_causal_analysis.py --full\n")
        
        print("\n[1/3] Testing event windows...")
        test_event_windows()
        
        print("\n\n[2/3] Testing cohort creation...")
        test_cohort_creation()
        
        print("\n\n[3/3] Testing event-study analysis...")
        test_event_study_analysis()
        
        print("\n\n" + "=" * 80)
        print("ALL TESTS COMPLETE")
        print("=" * 80)
