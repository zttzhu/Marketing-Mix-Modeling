# %%
"""
Marketing Mix Modeling (MMM) Script
====================================
This script implements a comprehensive MMM workflow using:
1. Robyn-style methodology with adstock and saturation transformations
2. Baseline OLS model for comparison
3. Model validation, diagnostics, and budget optimization

"""

# %%
# ============================================================================
# CONFIGURATION
# ============================================================================
# Model configuration parameters
TRAIN_TEST_SPLIT = 0.8  # Proportion of data for training
OPTIMIZATION_MAXITER = 30  # Maximum iterations for hyperparameter optimization
OPTIMIZATION_POPSIZE = 10  # Population size for differential evolution
ADSTOCK_TYPE = 'geometric'  # 'geometric' or 'weibull'
OPTIMIZE_HYPERPARAMS = True  # Whether to optimize hyperparameters
USE_LOG_LOG_MODEL = True  # Use log-log transformation: log(sales) ~ log(transformed_impressions)
USE_IMPRESSIONS = True  # Use impressions (mdip_*) instead of spend (mdsp_*)

# Log-Log Model Interpretation:
# ==============================
# When USE_LOG_LOG_MODEL = True:
#   Model form: log(sales) = β₀ + β₁*log(f(impressions₁)) + β₂*log(f(impressions₂)) + ...
#   where f(x) = saturation(adstock(x))
#
# Interpretation:
#   - Coefficients represent elasticities
#   - β = 0.5 means: 1% increase in transformed impressions → 0.5% increase in sales
#   - Captures multiplicative relationships and diminishing returns naturally
#   - Better for modeling percentage changes and multiplicative effects
#
# When USE_LOG_LOG_MODEL = False (Linear):
#   Model form: sales = β₀ + β₁*f(impressions₁) + β₂*f(impressions₂) + ...
#   - Coefficients represent absolute contributions
#   - β = 1000 means: 1 unit increase in transformed impressions → 1000 increase in sales

print("="*80)
print("MARKETING MIX MODELING - CONFIGURATION")
print("="*80)
print(f"Train/Test Split: {TRAIN_TEST_SPLIT}")
print(f"Adstock Type: {ADSTOCK_TYPE}")
print(f"Optimize Hyperparameters: {OPTIMIZE_HYPERPARAMS}")
print(f"Log-Log Model: {USE_LOG_LOG_MODEL}")
print(f"Use Impressions: {USE_IMPRESSIONS}")
print("="*80)

# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plot
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy.optimize import differential_evolution, minimize
from scipy import stats
import warnings
import statsmodels.api as sm
import os
from datetime import datetime
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', '{:.2f}'.format)

# %%
mmm_data = pd.read_csv("data.csv")
# %%
mmm_data
# %%
# Check to see if there are any NAs
mmm_data.columns[mmm_data.isna().sum()>0] # No NAs
# %%
# understand data
mmm_data.describe()
mmm_data.columns
# Data is at weekly level and end at Sunday
mmm_data['wk_strt_dt'] = pd.to_datetime(mmm_data['wk_strt_dt'])
mmm_data['wk_strt_dt'].dt.dayofweek.unique()
# %%
# 1. Media Variables 
# media impression columns
mdip_col = [col for col in mmm_data.columns if "mdip_" in col]
# media spend columns
mdsp_col = [col for col in mmm_data.columns if "mdsp_" in col]

# 2. Base Variables
# macro economics variables 
me_col = [col for col in mmm_data.columns if "me_" in col]
# Holiday Variables
hol_col = [col for col in mmm_data.columns if "hldy_" in col]
# Seasonal Variables
sea_col = [col for col in mmm_data.columns if "seas_" in col]
# Store count Variables
st_col = [col for col in mmm_data.columns if "st_" in col]
# Discount
dis_col = [col for col in mmm_data.columns if "mkrdn_" in col]
base_vars = me_col+hol_col+sea_col+st_col+dis_col

# 3. Sales Variables
sales_col = ['sales']
# %% 
# EDA 
# plot sales overtime. Observe seasonal pattern - Reach the peak by the end of each year
plot.figure(figsize=(16,8))
plot.plot(mmm_data['wk_strt_dt'],mmm_data['sales'])

# %%
# Heatmap between impressions and sales - sales has the strongest correlation with mdip_vidtr
plot.figure(figsize=(10,8))
sns.heatmap(mmm_data[mdip_col+sales_col].corr(),square=True,annot=True,vmax=1,vmin=-1,cmap="RdBu")
# %%
# Spend and sales: mdsp_vidtr has the strongest correlation with sales
sns.heatmap(mmm_data[mdsp_col+sales_col].corr(),square=True,annot=True,vmax=1,vmin=-1,cmap="RdBu")
# %%
# Draw distribution
sns.histplot(mmm_data[sales_col])
sns.pairplot(mmm_data[mdip_col+sales_col],x_vars= mdip_col,y_vars=sales_col)

# %%
# Function to show media spend analysis
def show_media_spend(data, date_col='wk_strt_dt', spend_cols=None, plot_type='all', figsize=(16, 10)):
    """
    Display spend analysis for different media channels.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The MMM dataset containing media spend columns
    date_col : str, default='wk_strt_dt'
        Name of the date column
    spend_cols : list, optional
        List of media spend column names. If None, will auto-detect columns with 'mdsp_' prefix
    plot_type : str, default='all'
        Type of visualization: 'all', 'time_series', 'summary', 'comparison', 'total', 'distribution'
    figsize : tuple, default=(16, 10)
        Figure size for plots
    
    Returns:
    --------
    pandas.DataFrame : Summary statistics of media spend
    """
    # Auto-detect spend columns if not provided
    if spend_cols is None:
        spend_cols = [col for col in data.columns if "mdsp_" in col]
    
    if len(spend_cols) == 0:
        print("No media spend columns found. Please check column names.")
        return None
    
    # Extract channel names (remove 'mdsp_' prefix)
    channel_names = [col.replace('mdsp_', '') for col in spend_cols]
    
    # Create summary statistics
    summary_stats = pd.DataFrame({
        'Channel': channel_names,
        'Total_Spend': [data[col].sum() for col in spend_cols],
        'Mean_Spend': [data[col].mean() for col in spend_cols],
        'Median_Spend': [data[col].median() for col in spend_cols],
        'Std_Spend': [data[col].std() for col in spend_cols],
        'Min_Spend': [data[col].min() for col in spend_cols],
        'Max_Spend': [data[col].max() for col in spend_cols],
        'Pct_of_Total': [data[col].sum() / sum([data[c].sum() for c in spend_cols]) * 100 
                         for col in spend_cols]
    })
    
    # Display summary statistics
    if plot_type in ['all', 'summary', 'total']:
        print("\n" + "="*80)
        print("MEDIA SPEND SUMMARY STATISTICS")
        print("="*80)
        print(summary_stats.to_string(index=False))
        print("\n")
    
    # Time series plot
    if plot_type in ['all', 'time_series']:
        plot.figure(figsize=figsize)
        num_channels = len(spend_cols)
        rows = (num_channels + 1) // 2
        cols = 2 if num_channels > 1 else 1
        
        for idx, (col, channel) in enumerate(zip(spend_cols, channel_names), 1):
            plot.subplot(rows, cols, idx)
            plot.plot(data[date_col], data[col], linewidth=2)
            plot.title(f'Spend Over Time: {channel.upper()}', fontsize=12, fontweight='bold')
            plot.xlabel('Date', fontsize=10)
            plot.ylabel('Spend', fontsize=10)
            plot.grid(True, alpha=0.3)
            plot.xticks(rotation=45)
        
        plot.tight_layout()
        plot.show()
    
    # Comparison plot (all channels together)
    if plot_type in ['all', 'comparison']:
        plot.figure(figsize=figsize)
        for col, channel in zip(spend_cols, channel_names):
            plot.plot(data[date_col], data[col], label=channel.upper(), linewidth=2, alpha=0.8)
        plot.title('Media Spend Comparison Across All Channels', fontsize=14, fontweight='bold')
        plot.xlabel('Date', fontsize=12)
        plot.ylabel('Spend', fontsize=12)
        plot.legend(loc='best', fontsize=10)
        plot.grid(True, alpha=0.3)
        plot.xticks(rotation=45)
        plot.tight_layout()
        plot.show()
    
    # Total spend by channel (bar chart)
    if plot_type in ['all', 'total']:
        plot.figure(figsize=(12, 6))
        summary_stats_sorted = summary_stats.sort_values('Total_Spend', ascending=False)
        bars = plot.bar(summary_stats_sorted['Channel'], summary_stats_sorted['Total_Spend'], 
                       color=sns.color_palette("husl", len(summary_stats_sorted)))
        plot.title('Total Spend by Media Channel', fontsize=14, fontweight='bold')
        plot.xlabel('Media Channel', fontsize=12)
        plot.ylabel('Total Spend', fontsize=12)
        plot.xticks(rotation=45, ha='right')
        plot.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plot.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:,.0f}',
                     ha='center', va='bottom', fontsize=9)
        
        plot.tight_layout()
        plot.show()
        
        # Pie chart for percentage distribution
        plot.figure(figsize=(10, 8))
        plot.pie(summary_stats_sorted['Total_Spend'], 
                labels=summary_stats_sorted['Channel'],
                autopct='%1.1f%%',
                startangle=90,
                colors=sns.color_palette("husl", len(summary_stats_sorted)))
        plot.title('Media Spend Distribution (Percentage)', fontsize=14, fontweight='bold')
        plot.tight_layout()
        plot.show()
    
    # Distribution plot
    if plot_type in ['all', 'distribution']:
        plot.figure(figsize=figsize)
        num_channels = len(spend_cols)
        rows = (num_channels + 1) // 2
        cols = 2 if num_channels > 1 else 1
        
        for idx, (col, channel) in enumerate(zip(spend_cols, channel_names), 1):
            plot.subplot(rows, cols, idx)
            sns.histplot(data[col], kde=True, bins=30)
            plot.title(f'Spend Distribution: {channel.upper()}', fontsize=12, fontweight='bold')
            plot.xlabel('Spend', fontsize=10)
            plot.ylabel('Frequency', fontsize=10)
            plot.grid(True, alpha=0.3)
        
        plot.tight_layout()
        plot.show()
    
    return summary_stats

# %%
# ============================================================================
# END-TO-END MMM WORKFLOW USING ROBYN METHODOLOGY
# ============================================================================

# %%
# Enhanced Adstock Functions (Robyn-style: Geometric and Weibull)
def adstock_geometric(x, theta):
    """
    Geometric adstock transformation (Robyn-style).
    
    Parameters:
    -----------
    x : array-like
        Input media variable
    theta : float
        Decay parameter (0 < theta < 1), higher = longer carryover
    
    Returns:
    --------
    array : Adstock transformed variable
    """
    x = np.array(x)
    adstock = np.zeros_like(x)
    for i in range(len(x)):
        if i == 0:
            adstock[i] = x[i]
        else:
            adstock[i] = x[i] + theta * adstock[i-1]
    return adstock

def adstock_weibull(x, shape, scale):
    """
    Weibull adstock transformation (Robyn-style).
    
    Parameters:
    -----------
    x : array-like
        Input media variable
    shape : float
        Shape parameter (k)
    scale : float
        Scale parameter (lambda)
    
    Returns:
    --------
    array : Adstock transformed variable
    """
    x = np.array(x)
    n = len(x)
    weights = np.zeros(n)
    for i in range(n):
        if i == 0:
            weights[i] = 1
        else:
            weights[i] = np.exp(-((i / scale) ** shape))
    
    adstock = np.zeros_like(x)
    for i in range(n):
        for j in range(i + 1):
            adstock[i] += x[j] * weights[i - j]
    return adstock

# %%
# Enhanced Saturation Function (Hill function - Robyn-style)
def saturation_hill(x, alpha, gamma):
    """
    Hill function for saturation transformation (Robyn-style).
    
    Parameters:
    -----------
    x : array-like
        Adstock transformed media variable
    alpha : float
        Half-saturation point (inflection point)
    gamma : float
        Shape parameter (steepness of curve)
    
    Returns:
    --------
    array : Saturated media variable
    """
    x = np.array(x)
    # Avoid division by zero
    x = np.maximum(x, 1e-10)
    return (x ** gamma) / (x ** gamma + alpha ** gamma)

# %%
# MMM Model Class (Robyn-style)
class MMMModel:
    """
    Marketing Mix Model following Robyn methodology.
    Uses Ridge regression with adstock and saturation transformations.
    """
    
    def __init__(self, alpha=1.0, normalize=True, positive=True):
        """
        Initialize MMM model.
        
        Parameters:
        -----------
        alpha : float
            Ridge regularization parameter
        normalize : bool
            Whether to normalize features
        """
        self.alpha = alpha
        self.normalize = normalize
        # Enforce non-negative media coefficients to avoid impossible negative ROAS
        # (media variables are expected to have non-negative incremental contribution).
        self.positive = positive
        self.model = Ridge(alpha=alpha, positive=positive)
        self.scaler = StandardScaler() if normalize else None
        self.feature_names = []
        self.coefficients = None
        self.intercept = None
        
    def fit(self, X, y):
        """
        Fit the MMM model.
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Target variable (sales/revenue)
        """
        if self.scaler:
            X = self.scaler.fit_transform(X)
        self.model.fit(X, y)
        self.coefficients = self.model.coef_
        self.intercept = self.model.intercept_
        return self
    
    def predict(self, X):
        """Predict using the fitted model."""
        if self.scaler:
            X = self.scaler.transform(X)
        return self.model.predict(X)
    
    def score(self, X, y):
        """Calculate R² score."""
        return self.model.score(X, y) if self.scaler is None else \
               self.model.score(self.scaler.transform(X), y)

# %%
# Hyperparameter Optimization
class MMMOptimizer:
    """
    Optimize hyperparameters for adstock and saturation transformations.
    Uses evolutionary algorithm similar to Robyn's Nevergrad approach.
    """
    
    def __init__(self, adstock_type='geometric'):
        """
        Initialize optimizer.
        
        Parameters:
        -----------
        adstock_type : str
            'geometric' or 'weibull'
        """
        self.adstock_type = adstock_type
        self.best_params = None
        self.best_score = None
        
    def objective_function(self, params, media_data, target, base_vars_data=None, 
                          alpha_range=(0.1, 10.0)):
        """
        Objective function for hyperparameter optimization.
        Minimizes NRMSE (Normalized Root Mean Squared Error).
        
        Parameters:
        -----------
        params : array
            Hyperparameters [adstock_param, saturation_alpha, saturation_gamma, ridge_alpha]
        media_data : dict
            Dictionary of media channel data
        target : array
            Target variable
        base_vars_data : array, optional
            Baseline variables
        alpha_range : tuple
            Range for saturation alpha (half-saturation point)
        
        Returns:
        --------
        float : NRMSE score
        """
        try:
            # Extract parameters
            if self.adstock_type == 'geometric':
                theta = params[0]
                # Ensure theta is in valid range
                theta = np.clip(theta, 0.01, 0.99)
            else:  # weibull
                shape = params[0]
                scale = params[1]
                shape = np.clip(shape, 0.1, 10.0)
                scale = np.clip(scale, 0.1, 20.0)
            
            sat_alpha = params[-2] if self.adstock_type == 'geometric' else params[-2]
            sat_gamma = params[-1]
            ridge_alpha = params[-3] if self.adstock_type == 'geometric' else params[-3]
            
            # Clip saturation parameters
            sat_alpha = np.clip(sat_alpha, alpha_range[0], alpha_range[1])
            sat_gamma = np.clip(sat_gamma, 0.1, 3.0)
            ridge_alpha = np.clip(ridge_alpha, 0.01, 100.0)
            
            # Transform media variables
            X_list = []
            for channel, data in media_data.items():
                # Apply adstock
                if self.adstock_type == 'geometric':
                    adstocked = adstock_geometric(data, theta)
                else:
                    adstocked = adstock_weibull(data, shape, scale)
                
                # Apply saturation
                saturated = saturation_hill(adstocked, sat_alpha, sat_gamma)
                X_list.append(saturated.reshape(-1, 1))
            
            # Combine features
            X_media = np.hstack(X_list)
            
            # Add baseline variables if provided
            if base_vars_data is not None:
                X = np.hstack([X_media, base_vars_data])
            else:
                X = X_media
            
            # Fit model
            model = MMMModel(alpha=ridge_alpha, normalize=True)
            model.fit(X, target)
            y_pred = model.predict(X)
            
            # Calculate NRMSE
            rmse = np.sqrt(mean_squared_error(target, y_pred))
            nrmse = rmse / (np.max(target) - np.min(target) + 1e-10)
            
            return nrmse
            
        except Exception as e:
            return 1e10  # Return large penalty for invalid parameters
    
    def optimize(self, media_data, target, base_vars_data=None, 
                 bounds=None, maxiter=50, popsize=15):
        """
        Optimize hyperparameters using differential evolution.
        
        Parameters:
        -----------
        media_data : dict
            Dictionary of media channel data {channel_name: data_array}
        target : array
            Target variable
        base_vars_data : array, optional
            Baseline variables
        bounds : list, optional
            Parameter bounds
        maxiter : int
            Maximum iterations
        popsize : int
            Population size
        
        Returns:
        --------
        dict : Best parameters
        """
        # Default bounds
        if bounds is None:
            if self.adstock_type == 'geometric':
                # [theta, ridge_alpha, sat_alpha, sat_gamma]
                bounds = [(0.01, 0.99), (0.01, 100.0), (0.1, 10.0), (0.1, 3.0)]
            else:  # weibull
                # [shape, scale, ridge_alpha, sat_alpha, sat_gamma]
                bounds = [(0.1, 10.0), (0.1, 20.0), (0.01, 100.0), (0.1, 10.0), (0.1, 3.0)]
        
        # Optimize
        result = differential_evolution(
            self.objective_function,
            bounds,
            args=(media_data, target, base_vars_data),
            maxiter=maxiter,
            popsize=popsize,
            seed=42,
            polish=True
        )
        
        self.best_params = result.x
        self.best_score = result.fun
        
        # Format results
        if self.adstock_type == 'geometric':
            return {
                'theta': result.x[0],
                'ridge_alpha': result.x[1],
                'saturation_alpha': result.x[2],
                'saturation_gamma': result.x[3],
                'nrmse': result.fun
            }
        else:
            return {
                'shape': result.x[0],
                'scale': result.x[1],
                'ridge_alpha': result.x[2],
                'saturation_alpha': result.x[3],
                'saturation_gamma': result.x[4],
                'nrmse': result.fun
            }

# %%
# Pareto Frontier Analysis (Robyn-style)
class ParetoFrontier:
    """
    Analyze Pareto frontier for model selection.
    Balances model fit (NRMSE) vs. decomposition quality (RSSD).
    """
    
    def __init__(self):
        self.models = []
        self.pareto_models = []
        
    def add_model(self, model_id, nrmse, rssd, params, model_obj):
        """Add a model candidate to the frontier."""
        self.models.append({
            'model_id': model_id,
            'nrmse': nrmse,
            'rssd': rssd,
            'params': params,
            'model': model_obj
        })
    
    def calculate_rssd(self, y_true, y_pred, y_decomp):
        """
        Calculate Root Sum of Squared Decomposition (RSSD).
        Measures how well model explains decomposition.
        """
        return np.sqrt(np.sum((y_true - y_decomp) ** 2))
    
    def find_pareto_frontier(self):
        """Find Pareto-optimal models."""
        if len(self.models) == 0:
            return []
        
        # Sort by NRMSE
        sorted_models = sorted(self.models, key=lambda x: x['nrmse'])
        
        pareto = []
        for model in sorted_models:
            is_dominated = False
            for pareto_model in pareto:
                # Check if dominated (worse in both metrics)
                if (pareto_model['nrmse'] <= model['nrmse'] and 
                    pareto_model['rssd'] <= model['rssd']):
                    is_dominated = True
                    break
            if not is_dominated:
                # Remove models dominated by this one
                pareto = [m for m in pareto if not 
                         (model['nrmse'] <= m['nrmse'] and model['rssd'] <= m['rssd'])]
                pareto.append(model)
        
        self.pareto_models = pareto
        return pareto
    
    def plot_pareto_frontier(self):
        """Visualize Pareto frontier."""
        if len(self.models) == 0:
            print("No models to plot")
            return
        
        plot.figure(figsize=(10, 6))
        
        # Plot all models
        nrmse_all = [m['nrmse'] for m in self.models]
        rssd_all = [m['rssd'] for m in self.models]
        plot.scatter(nrmse_all, rssd_all, alpha=0.5, label='All Models', s=50)
        
        # Plot Pareto frontier
        if len(self.pareto_models) > 0:
            nrmse_pareto = [m['nrmse'] for m in self.pareto_models]
            rssd_pareto = [m['rssd'] for m in self.pareto_models]
            plot.scatter(nrmse_pareto, rssd_pareto, color='red', 
                        s=100, marker='*', label='Pareto Frontier', zorder=5)
            
            # Annotate Pareto models
            for i, model in enumerate(self.pareto_models):
                plot.annotate(f"Model {model['model_id']}", 
                            (model['nrmse'], model['rssd']),
                            xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plot.xlabel('NRMSE (Lower is Better)', fontsize=12)
        plot.ylabel('RSSD (Lower is Better)', fontsize=12)
        plot.title('Pareto Frontier: Model Selection', fontsize=14, fontweight='bold')
        plot.legend()
        plot.grid(True, alpha=0.3)
        plot.tight_layout()
        plot.show()

# %%
# Response Curves Generator
def generate_response_curves(model, media_data, base_vars_data, 
                            channel_name, params, n_points=100):
    """
    Generate response curves for a media channel (Robyn-style).
    
    Parameters:
    -----------
    model : MMMModel
        Fitted MMM model
    media_data : dict
        All media channel data
    base_vars_data : array
        Baseline variables
    channel_name : str
        Name of channel to generate curve for
    params : dict
        Optimized hyperparameters
    n_points : int
        Number of points for curve
    
    Returns:
    --------
    tuple : (spend_range, response_values)
    """
    # Get original spend data for the channel
    original_spend = media_data[channel_name]
    spend_min = np.min(original_spend)
    spend_max = np.max(original_spend)
    spend_range = np.linspace(spend_min, spend_max, n_points)
    
    # Get adstock and saturation parameters
    if 'theta' in params:
        adstock_param = params['theta']
        adstock_func = lambda x: adstock_geometric(x, adstock_param)
    else:
        adstock_param = (params['shape'], params['scale'])
        adstock_func = lambda x: adstock_weibull(x, params['shape'], params['scale'])
    
    sat_alpha = params['saturation_alpha']
    sat_gamma = params['saturation_gamma']
    
    # Calculate response for each spend level
    responses = []
    for spend in spend_range:
        # Create modified media data with this spend level
        modified_media = media_data.copy()
        modified_media[channel_name] = np.full(len(original_spend), spend)
        
        # Transform
        X_list = []
        for ch, data in modified_media.items():
            adstocked = adstock_func(data)
            saturated = saturation_hill(adstocked, sat_alpha, sat_gamma)
            X_list.append(saturated.reshape(-1, 1))
        
        X_media = np.hstack(X_list)
        if base_vars_data is not None:
            X = np.hstack([X_media, base_vars_data])
        else:
            X = X_media
        
        # Predict (use mean prediction)
        pred = model.predict(X)
        responses.append(np.mean(pred))
    
    return spend_range, np.array(responses)

# %%
# Budget Allocator (Robyn-style)
class BudgetAllocator:
    """
    Optimize budget allocation across media channels.
    """
    
    def __init__(self, model, media_data, base_vars_data, params):
        """
        Initialize budget allocator.
        
        Parameters:
        -----------
        model : MMMModel
            Fitted MMM model
        media_data : dict
            Media channel data
        base_vars_data : array
            Baseline variables
        params : dict
            Model hyperparameters
        """
        self.model = model
        self.media_data = media_data
        self.base_vars_data = base_vars_data
        self.params = params
        self.channel_names = list(media_data.keys())

    def _transform_media(self, media_dict):
        """
        Transform a dict of raw media series into the model feature matrix X.
        Uses the allocator's channel order to keep column alignment stable.
        """
        X_list = []
        for ch in self.channel_names:
            series = np.array(media_dict[ch], dtype=float)
            if 'theta' in self.params:
                adstocked = adstock_geometric(series, self.params['theta'])
            else:
                adstocked = adstock_weibull(series, self.params['shape'], self.params['scale'])
            saturated = saturation_hill(
                adstocked,
                self.params['saturation_alpha'],
                self.params['saturation_gamma']
            )
            X_list.append(saturated.reshape(-1, 1))

        X_media = np.hstack(X_list)
        if self.base_vars_data is not None:
            return np.hstack([X_media, self.base_vars_data])
        return X_media
        
    def allocate_budget(self, total_budget, channel_bounds=None, 
                       objective='max_response'):
        """
        Allocate budget to maximize response or achieve target efficiency.
        
        Parameters:
        -----------
        total_budget : float
            Total budget to allocate
        channel_bounds : dict, optional
            {channel: (min_share, max_share)} constraints
        objective : str
            'max_response' or 'target_roas'
        
        Returns:
        --------
        dict : Optimal allocation
        """
        n_channels = len(self.channel_names)
        
        # Default bounds:
        # Use a *feasible* box constraint (sum of mins must be <= 100%).
        # If you want business constraints, pass channel_bounds explicitly.
        if channel_bounds is None:
            channel_bounds = {ch: (0.0, 1.0) for ch in self.channel_names}
        
        # Set up optimization bounds
        bounds = []
        for ch in self.channel_names:
            min_share, max_share = channel_bounds.get(ch, (0.05, 0.50))
            bounds.append((total_budget * min_share, total_budget * max_share))
        
        # Objective function (SLSQP with equality constraint)
        # We scale each channel's *historical weekly pattern* to match a candidate annual budget.
        # This avoids unrealistic "constant weekly spend" and produces non-trivial optima.
        current_spend_by_ch = np.array([np.sum(self.media_data[ch]) for ch in self.channel_names], dtype=float)
        current_spend_by_ch = np.where(current_spend_by_ch <= 0, 1e-10, current_spend_by_ch)

        def objective_func(budgets):
            budgets = np.array(budgets, dtype=float)

            scaled_media = {}
            for i, ch in enumerate(self.channel_names):
                scale = budgets[i] / current_spend_by_ch[i]
                scaled_media[ch] = np.array(self.media_data[ch], dtype=float) * scale

            X = self._transform_media(scaled_media)
            pred = self.model.predict(X)
            return -float(np.sum(pred))

        # Start from historical spend shares (more stable than equal split)
        hist_share = current_spend_by_ch / np.sum(current_spend_by_ch)
        initial_guess = (hist_share * total_budget).tolist()

        constraints = ({'type': 'eq', 'fun': lambda b: np.sum(b) - total_budget},)
        result = minimize(
            objective_func,
            x0=initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )

        # Format results (ensure exact budget sum after optimizer tolerance)
        optimal_budgets = np.array(result.x, dtype=float)
        optimal_budgets = optimal_budgets / np.sum(optimal_budgets) * total_budget
        
        allocation = {}
        for i, ch in enumerate(self.channel_names):
            allocation[ch] = {
                'budget': optimal_budgets[i],
                'share': optimal_budgets[i] / total_budget,
                'roas': self._calculate_roas(ch, optimal_budgets[i])
            }
        
        return allocation
    
    def _calculate_roas(self, channel, budget):
        """
        Calculate historical ROAS (past performance) for a channel.

        This uses a standard MMM counterfactual (leave-one-channel-out):
        - pred_all: model predictions with all channels as observed historically
        - pred_wo:  model predictions with this channel set to 0 (others unchanged)
        Contribution = sum(pred_all - pred_wo)
        ROAS = Contribution / historical spend of the channel
        
        Parameters:
        -----------
        channel : str
            Channel name
        budget : float
            Budget allocated to channel
        
        Returns:
        --------
        float : ROAS value
        """
        channel_spend = float(np.sum(self.media_data[channel]))
        if channel_spend <= 0:
            return 0.0

        media_all = {ch: np.array(self.media_data[ch], dtype=float) for ch in self.channel_names}
        media_wo = {ch: np.array(self.media_data[ch], dtype=float) for ch in self.channel_names}
        media_wo[channel] = np.zeros_like(media_wo[channel])

        pred_all = self.model.predict(self._transform_media(media_all))
        pred_wo = self.model.predict(self._transform_media(media_wo))

        contribution = float(np.sum(pred_all - pred_wo))
        return contribution / channel_spend

# %%
# End-to-End MMM Workflow Function
def run_robyn_mmm(data, date_col='wk_strt_dt', dep_var='sales',
                  media_spend_cols=None, base_vars=None,
                  adstock_type='geometric', trials=5, 
                  optimize_hyperparams=True, train_test_split=0.8,
                  use_log_transform=False, use_impressions=False):
    """
    Run end-to-end MMM workflow following Robyn methodology.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input data
    date_col : str
        Date column name
    dep_var : str
        Dependent variable (sales/revenue)
    media_spend_cols : list, optional
        Media spend/impression columns (auto-detected if None)
    base_vars : list, optional
        Baseline variables
    adstock_type : str
        'geometric' or 'weibull'
    trials : int
        Number of optimization trials
    optimize_hyperparams : bool
        Whether to optimize hyperparameters
    train_test_split : float
        Proportion of data for training (default 0.8)
    use_log_transform : bool
        If True, uses log-log model: log(sales) ~ log(transformed_media)
    use_impressions : bool
        If True, uses impression columns (mdip_*) instead of spend (mdsp_*)
    
    Returns:
    --------
    dict : Complete MMM results
    """
    print("="*80)
    print("ROBYN-STYLE MMM WORKFLOW")
    print("="*80)
    print(f"Model Type: {'Log-Log' if use_log_transform else 'Linear'}")
    print(f"Media Variable: {'Impressions (mdip_*)' if use_impressions else 'Spend (mdsp_*)'}")
    
    # 1. Prepare data
    print("\n[1/8] Preparing data...")
    if media_spend_cols is None:
        # Auto-detect media columns based on use_impressions flag
        if use_impressions:
            media_spend_cols = [col for col in data.columns if "mdip_" in col]
        else:
            media_spend_cols = [col for col in data.columns if "mdsp_" in col]
    
    if base_vars is None:
        base_vars = [col for col in data.columns if any(prefix in col 
                   for prefix in ['me_', 'hldy_', 'seas_', 'st_', 'mkrdn_'])]
    
    # Chronological train/test split
    split_point = int(len(data) * train_test_split)
    train_data = data.iloc[:split_point].copy()
    test_data = data.iloc[split_point:].copy()
    
    # Extract media data for training
    media_data_train = {}
    prefix_to_remove = 'mdip_' if use_impressions else 'mdsp_'
    for col in media_spend_cols:
        channel_name = col.replace(prefix_to_remove, '')
        media_data_train[channel_name] = train_data[col].values
    
    # Extract target and baseline for training
    target_train = train_data[dep_var].values
    base_vars_array_train = train_data[base_vars].values if len(base_vars) > 0 else None
    
    # Extract test data
    media_data_test = {}
    for col in media_spend_cols:
        channel_name = col.replace(prefix_to_remove, '')
        media_data_test[channel_name] = test_data[col].values
    target_test = test_data[dep_var].values
    base_vars_array_test = test_data[base_vars].values if len(base_vars) > 0 else None
    
    # Apply log transformation if enabled
    if use_log_transform:
        print("  - Applying log transformation to sales and media variables...")
        target_train_original = target_train.copy()
        target_test_original = target_test.copy()
        target_train = np.log1p(target_train)  # log(1 + sales) to handle zeros
        target_test = np.log1p(target_test)
        # Note: Media variables will be log-transformed AFTER adstock/saturation
    
    print(f"  - Media channels: {list(media_data_train.keys())}")
    print(f"  - Baseline variables: {len(base_vars)}")
    print(f"  - Training data points: {len(train_data)}")
    print(f"  - Test data points: {len(test_data)}")
    
    # 2. Optimize hyperparameters (on training data only)
    print("\n[2/8] Optimizing hyperparameters...")
    if optimize_hyperparams:
        optimizer = MMMOptimizer(adstock_type=adstock_type)
        best_params = optimizer.optimize(
            media_data_train, target_train, base_vars_array_train,
            maxiter=30, popsize=10
        )
        print(f"  - Best NRMSE (train): {best_params['nrmse']:.4f}")
        print(f"  - Parameters: {best_params}")
    else:
        # Use default parameters
        if adstock_type == 'geometric':
            best_params = {'theta': 0.5, 'ridge_alpha': 1.0, 
                          'saturation_alpha': 1.0, 'saturation_gamma': 1.0}
        else:
            best_params = {'shape': 1.0, 'scale': 1.0, 'ridge_alpha': 1.0,
                          'saturation_alpha': 1.0, 'saturation_gamma': 1.0}
    
    # 3. Transform media variables (training)
    print("\n[3/8] Transforming media variables...")
    X_list_train = []
    for channel, channel_data in media_data_train.items():
        # Apply adstock
        if adstock_type == 'geometric':
            adstocked = adstock_geometric(channel_data, best_params['theta'])
        else:
            adstocked = adstock_weibull(channel_data, 
                                      best_params['shape'], 
                                      best_params['scale'])
        
        # Apply saturation
        saturated = saturation_hill(adstocked, 
                                   best_params['saturation_alpha'],
                                   best_params['saturation_gamma'])
        
        # Apply log transformation if enabled (AFTER adstock and saturation)
        if use_log_transform:
            saturated = np.log1p(saturated)  # log(1 + x) to handle zeros
        
        X_list_train.append(saturated.reshape(-1, 1))
    
    X_media_train = np.hstack(X_list_train)
    if base_vars_array_train is not None:
        X_train = np.hstack([X_media_train, base_vars_array_train])
    else:
        X_train = X_media_train
    
    # Transform test data
    X_list_test = []
    for channel, channel_data in media_data_test.items():
        if adstock_type == 'geometric':
            adstocked = adstock_geometric(channel_data, best_params['theta'])
        else:
            adstocked = adstock_weibull(channel_data, 
                                      best_params['shape'], 
                                      best_params['scale'])
        saturated = saturation_hill(adstocked, 
                                   best_params['saturation_alpha'],
                                   best_params['saturation_gamma'])
        
        # Apply log transformation if enabled
        if use_log_transform:
            saturated = np.log1p(saturated)
        
        X_list_test.append(saturated.reshape(-1, 1))
    
    X_media_test = np.hstack(X_list_test)
    if base_vars_array_test is not None:
        X_test = np.hstack([X_media_test, base_vars_array_test])
    else:
        X_test = X_media_test
    
    # 4. Fit model
    print("\n[4/8] Fitting MMM model...")
    model = MMMModel(alpha=best_params['ridge_alpha'], normalize=True)
    model.fit(X_train, target_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Transform predictions back from log space if needed
    if use_log_transform:
        y_pred_train_original = np.expm1(y_pred_train)  # exp(x) - 1 to reverse log1p
        y_pred_test_original = np.expm1(y_pred_test)
        # Keep log-space predictions for metrics in log space
        y_pred_train_log = y_pred_train
        y_pred_test_log = y_pred_test
        # For evaluation, use original scale
        y_pred_train = y_pred_train_original
        y_pred_test = y_pred_test_original
        target_train_for_metrics = target_train_original
        target_test_for_metrics = target_test_original
    else:
        target_train_for_metrics = target_train
        target_test_for_metrics = target_test
    
    # Calculate metrics (training)
    rmse_train = np.sqrt(mean_squared_error(target_train_for_metrics, y_pred_train))
    nrmse_train = rmse_train / (np.max(target_train_for_metrics) - np.min(target_train_for_metrics) + 1e-10)
    r2_train = r2_score(target_train_for_metrics, y_pred_train)
    mae_train = mean_absolute_error(target_train_for_metrics, y_pred_train)
    
    # Calculate metrics (test)
    rmse_test = np.sqrt(mean_squared_error(target_test_for_metrics, y_pred_test))
    nrmse_test = rmse_test / (np.max(target_test_for_metrics) - np.min(target_test_for_metrics) + 1e-10)
    r2_test = r2_score(target_test_for_metrics, y_pred_test)
    mae_test = mean_absolute_error(target_test_for_metrics, y_pred_test)
    mape_test = np.mean(np.abs((target_test_for_metrics - y_pred_test) / (target_test_for_metrics + 1e-10))) * 100
    
    print(f"\n  Training Metrics:")
    print(f"    - R²: {r2_train:.4f}")
    print(f"    - NRMSE: {nrmse_train:.4f}")
    print(f"    - RMSE: {rmse_train:.2f}")
    print(f"    - MAE: {mae_train:.2f}")
    print(f"\n  Test Metrics:")
    print(f"    - R²: {r2_test:.4f}")
    print(f"    - NRMSE: {nrmse_test:.4f}")
    print(f"    - RMSE: {rmse_test:.2f}")
    print(f"    - MAE: {mae_test:.2f}")
    print(f"    - MAPE: {mape_test:.2f}%")
    
    # 5. Contribution Decomposition
    print("\n[5/8] Calculating contribution decomposition...")
    
    def decompose_contributions(model, X_media, X_base, media_names, base_names):
        """Decompose predictions into baseline and media contributions."""
        # Baseline contribution (intercept + control variables)
        if X_base is not None and len(X_base.shape) > 1:
            n_base = X_base.shape[1]
            base_contrib = model.intercept + np.sum(X_base * model.coefficients[-n_base:], axis=1)
        else:
            base_contrib = np.full(len(X_media), model.intercept)
        
        # Media contributions by channel
        media_contribs = {}
        n_media = len(media_names)
        for i, channel in enumerate(media_names):
            media_contribs[channel] = X_media[:, i] * model.coefficients[i]
        
        # Total prediction
        total_pred = base_contrib + np.sum([media_contribs[ch] for ch in media_names], axis=0)
        
        return {
            'baseline': base_contrib,
            'media': media_contribs,
            'total': total_pred
        }
    
    # Decompose for training and test
    decomposition_train = decompose_contributions(
        model, X_media_train, base_vars_array_train, 
        list(media_data_train.keys()), base_vars
    )
    decomposition_test = decompose_contributions(
        model, X_media_test, base_vars_array_test,
        list(media_data_test.keys()), base_vars
    )
    
    # Print contribution summary
    print("\n  Average Contributions (Training):")
    print(f"    Baseline: {np.mean(decomposition_train['baseline']):.2f}")
    for channel in media_data_train.keys():
        print(f"    {channel}: {np.mean(decomposition_train['media'][channel]):.2f}")
    
    # 6. Model diagnostics and residual analysis
    print("\n[6/8] Generating model diagnostics and residual analysis...")
    
    # Plot actual vs predicted (train and test)
    plot.figure(figsize=(16, 6))
    
    plot.subplot(1, 3, 1)
    plot.scatter(target_train_for_metrics, y_pred_train, alpha=0.6, label='Train', s=30)
    plot.scatter(target_test_for_metrics, y_pred_test, alpha=0.6, label='Test', s=30, marker='^')
    plot.plot([min(target_train_for_metrics.min(), target_test_for_metrics.min()), 
               max(target_train_for_metrics.max(), target_test_for_metrics.max())], 
             [min(target_train_for_metrics.min(), target_test_for_metrics.min()), 
              max(target_train_for_metrics.max(), target_test_for_metrics.max())], 
             'r--', lw=2, label='Perfect Prediction')
    plot.xlabel('Actual Sales', fontsize=11)
    plot.ylabel('Predicted Sales', fontsize=11)
    plot.title(f'Actual vs Predicted\nTrain R²={r2_train:.3f}, Test R²={r2_test:.3f}', 
               fontsize=12, fontweight='bold')
    plot.legend()
    plot.grid(True, alpha=0.3)
    
    # Residuals plot
    residuals_train = target_train_for_metrics - y_pred_train
    residuals_test = target_test_for_metrics - y_pred_test
    
    plot.subplot(1, 3, 2)
    plot.scatter(y_pred_train, residuals_train, alpha=0.6, label='Train', s=30)
    plot.scatter(y_pred_test, residuals_test, alpha=0.6, label='Test', s=30, marker='^')
    plot.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plot.xlabel('Predicted Sales', fontsize=11)
    plot.ylabel('Residuals', fontsize=11)
    plot.title('Residuals vs Predicted', fontsize=12, fontweight='bold')
    plot.legend()
    plot.grid(True, alpha=0.3)
    
    # Time series plot
    plot.subplot(1, 3, 3)
    all_dates = pd.concat([train_data[date_col], test_data[date_col]])
    all_actual = np.concatenate([target_train_for_metrics, target_test_for_metrics])
    all_pred = np.concatenate([y_pred_train, y_pred_test])
    plot.plot(all_dates, all_actual, label='Actual', linewidth=2, alpha=0.7)
    plot.plot(all_dates, all_pred, label='Predicted', linewidth=2, alpha=0.7)
    plot.axvline(x=train_data[date_col].iloc[-1], color='g', linestyle='--', 
                linewidth=1, label='Train/Test Split')
    plot.xlabel('Date', fontsize=11)
    plot.ylabel('Sales', fontsize=11)
    plot.title('Time Series: Actual vs Predicted', fontsize=12, fontweight='bold')
    plot.legend()
    plot.xticks(rotation=45)
    plot.grid(True, alpha=0.3)
    plot.tight_layout()
    plot.show()
    
    # Q-Q plot for residuals
    plot.figure(figsize=(12, 5))
    
    plot.subplot(1, 2, 1)
    stats.probplot(residuals_train, dist="norm", plot=plot)
    plot.title('Q-Q Plot: Training Residuals', fontsize=12, fontweight='bold')
    plot.grid(True, alpha=0.3)
    
    plot.subplot(1, 2, 2)
    stats.probplot(residuals_test, dist="norm", plot=plot)
    plot.title('Q-Q Plot: Test Residuals', fontsize=12, fontweight='bold')
    plot.grid(True, alpha=0.3)
    plot.tight_layout()
    plot.show()
    
    # 7. Response curves
    print("\n[7/8] Generating response curves...")
    response_curves = {}
    n_channels = len(media_data_train)
    rows = (n_channels + 1) // 2
    cols = 2 if n_channels > 1 else 1
    
    plot.figure(figsize=(14, 4 * rows))
    for idx, channel in enumerate(media_data_train.keys(), 1):
        spend_range, responses = generate_response_curves(
            model, media_data_train, base_vars_array_train, channel, best_params
        )
        response_curves[channel] = (spend_range, responses)
        
        plot.subplot(rows, cols, idx)
        plot.plot(spend_range, responses, linewidth=2)
        plot.xlabel('Spend', fontsize=10)
        plot.ylabel('Response (Sales)', fontsize=10)
        plot.title(f'Response Curve: {channel.upper()}', fontsize=11, fontweight='bold')
        plot.grid(True, alpha=0.3)
    
    plot.tight_layout()
    plot.show()
    
    # 8. Budget allocation example
    print("\n[8/8] Budget allocation example...")
    allocator = BudgetAllocator(model, media_data_train, base_vars_array_train, best_params)
    total_budget = sum([train_data[col].sum() for col in media_spend_cols])
    allocation = allocator.allocate_budget(total_budget)
    
    print("\nOptimal Budget Allocation:")
    print("-" * 80)
    print(f"{'Channel':20s} | {'Budget':>15s} | {'Share':>8s} | {'ROAS':>10s}")
    print("-" * 80)
    for channel, alloc in allocation.items():
        print(f"{channel:20s} | ${alloc['budget']:>13,.0f} | {alloc['share']*100:>6.1f}% | {alloc['roas']:>9.2f}")
    
    # Compile results
    results = {
        'model': model,
        'params': best_params,
        'use_log_transform': use_log_transform,
        'use_impressions': use_impressions,
        'metrics': {
            'train': {
                'r2': r2_train,
                'nrmse': nrmse_train,
                'rmse': rmse_train,
                'mae': mae_train
            },
            'test': {
                'r2': r2_test,
                'nrmse': nrmse_test,
                'rmse': rmse_test,
                'mae': mae_test,
                'mape': mape_test
            }
        },
        'predictions': {
            'train': y_pred_train,
            'test': y_pred_test
        },
        'actual': {
            'train': target_train_for_metrics,
            'test': target_test_for_metrics
        },
        'decomposition': {
            'train': decomposition_train,
            'test': decomposition_test
        },
        'response_curves': response_curves,
        'budget_allocation': allocation,
        'feature_names': list(media_data_train.keys()) + base_vars,
        'coefficients': model.coefficients,
        'train_data': train_data,
        'test_data': test_data
    }
    
    print("\n" + "="*80)
    print("MMM WORKFLOW COMPLETE")
    print("="*80)
    
    return results

# %%
# Run the end-to-end MMM workflow
print("\nStarting Robyn-style MMM workflow...")
mmm_results = run_robyn_mmm(
    mmm_data,
    date_col='wk_strt_dt',
    dep_var='sales',
    media_spend_cols=mdip_col if USE_IMPRESSIONS else mdsp_col,
    base_vars=base_vars,
    adstock_type=ADSTOCK_TYPE,
    trials=5,
    optimize_hyperparams=OPTIMIZE_HYPERPARAMS,
    train_test_split=TRAIN_TEST_SPLIT,
    use_log_transform=USE_LOG_LOG_MODEL,
    use_impressions=USE_IMPRESSIONS
)

# %%
# ============================================================================
# BASELINE MMM USING OLS (Alternative approach for comparison)
# ============================================================================
# Baseline MMM using OLS on raw spend and control variables
# IMPROVEMENT: Now uses ALL media channels, ALL controls, adstock, and Ridge regression

def geometric_adstock_simple(x, decay=0.5):
    """
    Apply geometric adstock transformation for lagged marketing effects.
    
    Parameters:
    -----------
    x : array-like
        Media variable to transform
    decay : float
        Decay rate (0-1). Higher = longer carryover effect
    
    Returns:
    --------
    array : Adstocked variable
    """
    adstocked = np.zeros_like(x, dtype=float)
    adstocked[0] = x[0]
    for i in range(1, len(x)):
        adstocked[i] = x[i] + decay * adstocked[i-1]
    return adstocked

# Use ALL media channels and control variables (not just subset)
media_cols_baseline = mdsp_col  # All media spend columns
control_cols_baseline = base_vars  # All control variables

print("\n" + "="*80)
print("IMPROVED BASELINE OLS MMM")
print("="*80)
print(f"Media channels ({len(media_cols_baseline)}): {[col.replace('mdsp_', '') for col in media_cols_baseline]}")
print(f"Control variables ({len(control_cols_baseline)}): {len(control_cols_baseline)} features")
print(f"Using adstock transformation with decay=0.5")
print(f"Using Ridge regression with regularization")

# Apply adstock transformation to media variables
model_data = mmm_data.copy()
adstock_decay = 0.5  # Can be optimized later

for col in media_cols_baseline:
    model_data[f'{col}_adstock'] = geometric_adstock_simple(model_data[col].values, decay=adstock_decay)

# Chronological train/test split (same split as Robyn model)
split_point = int(len(model_data) * TRAIN_TEST_SPLIT)
train_baseline = model_data.iloc[:split_point]
test_baseline = model_data.iloc[split_point:]

# Prepare features: adstocked media + all controls
adstock_feature_cols = [f'{col}_adstock' for col in media_cols_baseline]
feature_cols_all = adstock_feature_cols + control_cols_baseline

# Create feature matrices
X_train_baseline = train_baseline[feature_cols_all].values
y_train_baseline = train_baseline['sales'].values
X_test_baseline = test_baseline[feature_cols_all].values
y_test_baseline_actual = test_baseline['sales'].values

# Standardize features (important for Ridge regression)
scaler_baseline = StandardScaler()
X_train_baseline_scaled = scaler_baseline.fit_transform(X_train_baseline)
X_test_baseline_scaled = scaler_baseline.transform(X_test_baseline)

# Fit Ridge regression (with regularization to prevent overfitting)
# Ridge is better than OLS when you have many correlated features
baseline_ridge_model = Ridge(alpha=100.0)  # Regularization strength
baseline_ridge_model.fit(X_train_baseline_scaled, y_train_baseline)

# Predictions
y_pred_baseline_train = baseline_ridge_model.predict(X_train_baseline_scaled)
y_pred_baseline = baseline_ridge_model.predict(X_test_baseline_scaled)

# Calculate metrics for both train and test
r2_baseline_train = r2_score(y_train_baseline, y_pred_baseline_train)
rmse_baseline_train = np.sqrt(mean_squared_error(y_train_baseline, y_pred_baseline_train))
mae_baseline_train = mean_absolute_error(y_train_baseline, y_pred_baseline_train)
mape_baseline_train = np.mean(np.abs((y_train_baseline - y_pred_baseline_train) / (y_train_baseline + 1e-10))) * 100

r2_baseline = r2_score(y_test_baseline_actual, y_pred_baseline)
rmse_baseline = np.sqrt(mean_squared_error(y_test_baseline_actual, y_pred_baseline))
mae_baseline = mean_absolute_error(y_test_baseline_actual, y_pred_baseline)
mape_baseline = np.mean(np.abs((y_test_baseline_actual - y_pred_baseline) / (y_test_baseline_actual + 1e-10))) * 100

print("\n" + "-"*80)
print("IMPROVED BASELINE MODEL RESULTS")
print("-"*80)
print(f"\nTraining Set Metrics:")
print(f"  R²:   {r2_baseline_train:.4f}")
print(f"  RMSE: {rmse_baseline_train:,.0f}")
print(f"  MAE:  {mae_baseline_train:,.0f}")
print(f"  MAPE: {mape_baseline_train:.2f}%")

print(f"\nTest Set Metrics:")
print(f"  R²:   {r2_baseline:.4f}")
print(f"  RMSE: {rmse_baseline:,.0f}")
print(f"  MAE:  {mae_baseline:,.0f}")
print(f"  MAPE: {mape_baseline:.2f}%")

# Feature importance (top 10 most important features)
feature_importance = pd.DataFrame({
    'Feature': feature_cols_all,
    'Coefficient': baseline_ridge_model.coef_
})
feature_importance['Abs_Coefficient'] = np.abs(feature_importance['Coefficient'])
feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False)

print(f"\nTop 10 Most Important Features:")
print(feature_importance[['Feature', 'Coefficient']].head(10).to_string(index=False))

# Plot actual vs predicted sales
plot.figure(figsize=(14,7))

# Subplot 1: Test set
plot.subplot(1, 2, 1)
plot.plot(test_baseline['wk_strt_dt'], y_test_baseline_actual, label='Actual', linewidth=2, alpha=0.7)
plot.plot(test_baseline['wk_strt_dt'], y_pred_baseline, label='Predicted', linewidth=2, alpha=0.7)
plot.legend()
plot.xlabel('Date', fontsize=11)
plot.ylabel('Sales', fontsize=11)
plot.title(f'Test Set: Improved Baseline Model (R² = {r2_baseline:.3f})', 
           fontsize=12, fontweight='bold')
plot.grid(True, alpha=0.3)
plot.xticks(rotation=45)

# Subplot 2: Residuals
plot.subplot(1, 2, 2)
residuals = y_test_baseline_actual - y_pred_baseline
plot.scatter(y_pred_baseline, residuals, alpha=0.5)
plot.axhline(y=0, color='r', linestyle='--', linewidth=2)
plot.xlabel('Predicted Sales', fontsize=11)
plot.ylabel('Residuals', fontsize=11)
plot.title('Residual Plot (Test Set)', fontsize=12, fontweight='bold')
plot.grid(True, alpha=0.3)

plot.tight_layout()
plot.show()

# Additional diagnostic: Actual vs Predicted scatter
plot.figure(figsize=(8, 8))
plot.scatter(y_test_baseline_actual, y_pred_baseline, alpha=0.6)
plot.plot([y_test_baseline_actual.min(), y_test_baseline_actual.max()], 
          [y_test_baseline_actual.min(), y_test_baseline_actual.max()], 
          'r--', lw=2, label='Perfect Prediction')
plot.xlabel('Actual Sales', fontsize=12)
plot.ylabel('Predicted Sales', fontsize=12)
plot.title(f'Actual vs Predicted (Test Set, R² = {r2_baseline:.3f})', 
           fontsize=13, fontweight='bold')
plot.legend()
plot.grid(True, alpha=0.3)
plot.tight_layout()
plot.show()

# %%
# ============================================================================
# MODEL COMPARISON
# ============================================================================
print("\n" + "="*80)
print("MODEL COMPARISON: Robyn MMM vs Baseline OLS")
print("="*80)

comparison_df = pd.DataFrame({
    'Metric': ['R²', 'RMSE', 'MAE', 'MAPE (%)'],
    'Robyn MMM (Test)': [
        mmm_results['metrics']['test']['r2'],
        mmm_results['metrics']['test']['rmse'],
        mmm_results['metrics']['test']['mae'],
        mmm_results['metrics']['test']['mape']
    ],
    'Baseline OLS (Test)': [
        r2_baseline,
        rmse_baseline,
        mae_baseline,
        mape_baseline
    ]
})

comparison_df['Improvement'] = ((comparison_df['Baseline OLS (Test)'] - 
                                  comparison_df['Robyn MMM (Test)']) / 
                                 comparison_df['Baseline OLS (Test)'] * 100)
comparison_df['Improvement'] = comparison_df['Improvement'].apply(
    lambda x: f"{x:.1f}%" if not pd.isna(x) else "N/A"
)

# For R², improvement is positive if Robyn is higher
comparison_df.loc[0, 'Improvement'] = (
    f"{(mmm_results['metrics']['test']['r2'] - r2_baseline) / r2_baseline * 100:.1f}%"
    if r2_baseline > 0 else "N/A"
)

print("\n" + comparison_df.to_string(index=False))
print("\n" + "="*80)

# %%
# ============================================================================
# MEDIA CHANNEL METRICS SUMMARY BY YEAR
# ============================================================================
print("\n" + "="*80)
print("CALCULATING MEDIA CHANNEL METRICS BY YEAR")
print("="*80)

def calculate_channel_metrics_by_year(data, mmm_results, date_col='wk_strt_dt', 
                                     dep_var='sales', media_spend_cols=None, 
                                     media_impression_cols=None):
    """
    Calculate comprehensive metrics for each media channel by year.
    
    Metrics calculated:
    - Spend: Total spend per channel per year
    - Impressions: Total impressions per channel per year
    - ROAS: Return on Ad Spend (Incremental Revenue / Spend)
    - CPM: Cost Per Mille (Spend / Impressions * 1000)
    - Effectiveness: Incremental Revenue per Impression
    - Due-to Contribution: Media contribution from MMM counterfactual (leave-one-channel-out)
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Full dataset with dates, spend, impressions, and sales
    mmm_results : dict
        Results from run_robyn_mmm function
    date_col : str
        Date column name
    dep_var : str
        Dependent variable (sales)
    media_spend_cols : list
        Media spend column names
    media_impression_cols : list
        Media impression column names
    
    Returns:
    --------
    pandas.DataFrame : Summary table with metrics by channel and year
    """
    # Auto-detect columns if not provided
    if media_spend_cols is None:
        media_spend_cols = [col for col in data.columns if "mdsp_" in col]
    if media_impression_cols is None:
        media_impression_cols = [col for col in data.columns if "mdip_" in col]
    
    # Ensure date is datetime
    data = data.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    data['year'] = data[date_col].dt.year
    
    # Get model and parameters
    model = mmm_results['model']
    params = mmm_results['params']
    
    # Get all data (train + test) for model decomposition
    # We'll use this for calculating due-to contributions
    all_data = pd.concat([mmm_results['train_data'], mmm_results['test_data']])
    all_data[date_col] = pd.to_datetime(all_data[date_col])
    all_data['year'] = all_data[date_col].dt.year
    
    # Align data indices - we need to match rows between full data and train+test
    # For spend/impressions, use full data; for decomposition, use train+test data
    # Create a mapping based on dates
    all_data['date_key'] = all_data[date_col].dt.strftime('%Y-%m-%d')
    data['date_key'] = data[date_col].dt.strftime('%Y-%m-%d')
    
    # Get decomposition for all data
    # We need to transform all data and get contributions
    media_data_all = {}
    for col in media_spend_cols:
        channel_name = col.replace('mdsp_', '')
        media_data_all[channel_name] = all_data[col].values
    
    # Transform all media data
    X_list_all = []
    for channel, channel_data in media_data_all.items():
        if 'theta' in params:
            adstocked = adstock_geometric(channel_data, params['theta'])
        else:
            adstocked = adstock_weibull(channel_data, params['shape'], params['scale'])
        saturated = saturation_hill(adstocked, 
                                   params['saturation_alpha'],
                                   params['saturation_gamma'])
        X_list_all.append(saturated.reshape(-1, 1))
    
    X_media_all = np.hstack(X_list_all)
    
    # Get base variables
    base_vars_list = mmm_results['feature_names'][len(media_data_all):] if \
        len(mmm_results['feature_names']) > len(media_data_all) else []
    
    if len(base_vars_list) > 0:
        base_vars_array_all = all_data[base_vars_list].values
        X_all = np.hstack([X_media_all, base_vars_array_all])
    else:
        base_vars_array_all = None
        X_all = X_media_all
    
    # ------------------------------------------------------------------------
    # Compute per-week channel contributions via leave-one-channel-out
    # contribution_c[t] = pred_all[t] - pred_without_channel_c[t]
    # This is more stable/meaningful than coef * transformed_x when features are scaled.
    # ------------------------------------------------------------------------
    media_channel_names = [col.replace('mdsp_', '') for col in media_spend_cols]

    def _transform_X_from_media_dict(media_dict):
        """Build X using the same channel order as media_spend_cols."""
        X_list = []
        for ch in media_channel_names:
            series = np.array(media_dict[ch], dtype=float)
            if 'theta' in params:
                adstocked = adstock_geometric(series, params['theta'])
            else:
                adstocked = adstock_weibull(series, params['shape'], params['scale'])
            saturated = saturation_hill(
                adstocked,
                params['saturation_alpha'],
                params['saturation_gamma']
            )
            X_list.append(saturated.reshape(-1, 1))
        X_media = np.hstack(X_list)
        if base_vars_array_all is not None:
            return np.hstack([X_media, base_vars_array_all])
        return X_media

    # Build media dict in the proper order
    media_all = {ch: np.array(all_data[f"mdsp_{ch}"].values, dtype=float) for ch in media_channel_names}
    pred_all = model.predict(_transform_X_from_media_dict(media_all))

    contrib_series = {}
    for ch in media_channel_names:
        media_wo = {k: np.array(v, dtype=float) for k, v in media_all.items()}
        media_wo[ch] = np.zeros_like(media_wo[ch])
        pred_wo = model.predict(_transform_X_from_media_dict(media_wo))
        contrib_series[ch] = (pred_all - pred_wo)
    
    # Calculate metrics by channel and year
    summary_rows = []
    
    for year in sorted(data['year'].unique()):
        # Use full data for spend and impressions
        year_data_full = data[data['year'] == year].copy()
        
        # Use train+test data for contribution series (only if year exists in train+test)
        year_mask_all = all_data['year'] == year
        year_indices = np.where(year_mask_all)[0] if year_mask_all.any() else np.array([])
        
        for i, col in enumerate(media_spend_cols):
            channel_name = col.replace('mdsp_', '')
            
            # Get corresponding impression column
            imp_col = col.replace('mdsp_', 'mdip_')
            if imp_col not in media_impression_cols:
                # Try to find matching impression column
                imp_col = None
                for imp in media_impression_cols:
                    if imp.replace('mdip_', '') == channel_name:
                        imp_col = imp
                        break
            
            # Calculate spend from full data
            spend = year_data_full[col].sum()
            
            # Calculate impressions from full data
            if imp_col and imp_col in year_data_full.columns:
                impressions = year_data_full[imp_col].sum()
            else:
                impressions = np.nan
            
            # Due-to contribution via leave-one-channel-out (sum over weeks in that year)
            if len(year_indices) > 0 and channel_name in contrib_series:
                due_to = float(np.sum(contrib_series[channel_name][year_indices]))
            else:
                due_to = 0.0
            
            # Calculate ROAS (Incremental Revenue / Spend)
            # ROAS = Due-to / Spend
            if spend > 0:
                roas = due_to / spend
            else:
                roas = 0.0
            
            # Calculate CPM (Cost Per Mille = Spend / Impressions * 1000)
            if impressions > 0 and not np.isnan(impressions):
                cpm = (spend / impressions) * 1000
            else:
                cpm = np.nan
            
            # Calculate Effectiveness (Incremental Revenue per Impression)
            # Effectiveness = Due-to / Impressions
            if impressions > 0 and not np.isnan(impressions):
                effectiveness = due_to / impressions
            else:
                effectiveness = np.nan
            
            summary_rows.append({
                'Year': year,
                'Channel': channel_name,
                'Spend': spend,
                'Impressions': impressions if not np.isnan(impressions) else 0,
                'ROAS': roas,
                'CPM': cpm if not np.isnan(cpm) else 0,
                'Effectiveness': effectiveness if not np.isnan(effectiveness) else 0,
                'Due-to Contribution': due_to
            })
    
    summary_df = pd.DataFrame(summary_rows)
    
    return summary_df

# Calculate metrics
channel_metrics = calculate_channel_metrics_by_year(
    mmm_data,
    mmm_results,
    date_col='wk_strt_dt',
    dep_var='sales',
    media_spend_cols=mdsp_col,
    media_impression_cols=mdip_col
)

# Format and display summary table
print("\n" + "="*80)
print("MEDIA CHANNEL METRICS SUMMARY BY YEAR")
print("="*80)

# Format the table for better readability
formatted_metrics = channel_metrics.copy()
formatted_metrics['Spend'] = formatted_metrics['Spend'].apply(lambda x: f"${x:,.0f}")
formatted_metrics['Impressions'] = formatted_metrics['Impressions'].apply(
    lambda x: f"{x:,.0f}" if not pd.isna(x) and x > 0 else "N/A"
)
formatted_metrics['ROAS'] = formatted_metrics['ROAS'].apply(lambda x: f"{x:.2f}")
formatted_metrics['CPM'] = formatted_metrics['CPM'].apply(
    lambda x: f"${x:.2f}" if not pd.isna(x) and x > 0 else "N/A"
)
formatted_metrics['Effectiveness'] = formatted_metrics['Effectiveness'].apply(
    lambda x: f"${x:.4f}" if not pd.isna(x) and x != 0 else "N/A"
)
formatted_metrics['Due-to Contribution'] = formatted_metrics['Due-to Contribution'].apply(
    lambda x: f"${x:,.0f}"
)

# Display by year
for year in sorted(channel_metrics['Year'].unique()):
    year_data = formatted_metrics[formatted_metrics['Year'] == year]
    print(f"\n{'='*100}")
    print(f"YEAR {year}")
    print(f"{'='*100}")
    print(year_data[['Channel', 'Spend', 'Impressions', 'ROAS', 'CPM', 
                      'Effectiveness', 'Due-to Contribution']].to_string(index=False))

# Also create a pivot table for easier comparison
print("\n" + "="*100)
print("SUMMARY TABLE - ALL YEARS")
print("="*100)

# Create pivot tables for key metrics
pivot_spend = channel_metrics.pivot_table(
    index='Channel', columns='Year', values='Spend', aggfunc='sum', fill_value=0
)
pivot_roas = channel_metrics.pivot_table(
    index='Channel', columns='Year', values='ROAS', aggfunc='mean', fill_value=0
)
pivot_due_to = channel_metrics.pivot_table(
    index='Channel', columns='Year', values='Due-to Contribution', aggfunc='sum', fill_value=0
)

print("\nTotal Spend by Channel and Year:")
print("-" * 100)
print(pivot_spend.applymap(lambda x: f"${x:,.0f}"))

print("\nAverage ROAS by Channel and Year:")
print("-" * 100)
print(pivot_roas.applymap(lambda x: f"{x:.2f}"))

print("\nTotal Due-to Contribution by Channel and Year:")
print("-" * 100)
print(pivot_due_to.applymap(lambda x: f"${x:,.0f}"))

# Calculate totals across all years
print("\n" + "="*100)
print("TOTALS ACROSS ALL YEARS")
print("="*100)
totals = channel_metrics.groupby('Channel').agg({
    'Spend': 'sum',
    'Impressions': 'sum',
    'Due-to Contribution': 'sum'
}).reset_index()

totals['ROAS'] = totals['Due-to Contribution'] / totals['Spend']
totals['CPM'] = (totals['Spend'] / totals['Impressions'] * 1000).replace([np.inf, -np.inf], np.nan)
totals['Effectiveness'] = (totals['Due-to Contribution'] / totals['Impressions']).replace(
    [np.inf, -np.inf], np.nan
)

totals_display = totals.copy()
totals_display['Spend'] = totals_display['Spend'].apply(lambda x: f"${x:,.0f}")
totals_display['Impressions'] = totals_display['Impressions'].apply(lambda x: f"{x:,.0f}")
totals_display['ROAS'] = totals_display['ROAS'].apply(lambda x: f"{x:.2f}")
totals_display['CPM'] = totals_display['CPM'].apply(
    lambda x: f"${x:.2f}" if not pd.isna(x) else "N/A"
)
totals_display['Effectiveness'] = totals_display['Effectiveness'].apply(
    lambda x: f"${x:.4f}" if not pd.isna(x) else "N/A"
)
totals_display['Due-to Contribution'] = totals_display['Due-to Contribution'].apply(
    lambda x: f"${x:,.0f}"
)

print("\n" + totals_display.to_string(index=False))
print("\n" + "="*100)

# Save to CSV for further analysis
channel_metrics.to_csv('media_channel_metrics_by_year.csv', index=False)
print("\nMetrics saved to 'media_channel_metrics_by_year.csv'")

# %%
