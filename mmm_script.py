# %%
print("Hello Marketing Mix Modeling")
# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plot
import prophet as Prophet
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy.optimize import differential_evolution, minimize
from scipy.stats import pearsonr
import warnings
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
# Plot the time series plot for sales. Not seeing any seasonal pattern heres
trend_sales = mmm_data[['wk_strt_dt','sales']].rename(columns = {'wk_strt_dt':'ds','sales':'y'})
#trend_sales['y'] = trend_sales['y'].apply(lambda x: np.log(x + 1)) 
# Fit model
model = Prophet.Prophet(yearly_seasonality=True, weekly_seasonality=True)
model.fit(trend_sales)
sales_fit = model.predict(model.make_future_dataframe(periods=0))
# %%
# Extract and plot --> The trend is going down straightly
trend = sales_fit[['ds','trend']]
plot.plot(trend['ds'],trend['trend'],label = 'trend')
# %%
# Generate yearly column to double check --> Sales raised from 2015-2017 but in the trend chart 
# it is straightly going down
mmm_data['year'] = mmm_data['wk_strt_dt'].dt.year
mmm_data.groupby('year')['sales'].sum()
# %%
# Seasonality component
seasonal = sales_fit[['ds','yearly']]
plot.plot(seasonal['ds'],seasonal['yearly'],label = 'seasonal')
# %%
# create adstock function
def adstock(support,half_life):
    decay = np.exp(np.log(0.5) / half_life)
    num_period = support.shape[0]
    adstock = np.zeros(num_period)
    for i in range(num_period):
        if i ==0 :
            adstock[i] = (1.0 - decay) * support[i]
        else:
            adstock[i] = (1.0 - decay) * support[i] + decay * adstock[i-1]
    return adstock
# %%
# Create Adstock variables
for col in mdip_col:
    new_col = 'ad_' + col
    mmm_data[new_col] = adstock(mmm_data[col],1)
# %%
# check to see if it is working correctly
check_column = ['mdip_dm','ad_mdip_dm']
mmm_data[check_column]
# %%
# create s-curve function
# Use hill function to capture, need to define half-saturation point and shape parameter
# Input x is adstock, s is shape parameter and a is half saturation 
def scrv_transformation(adstock,a,s):
    num_period = adstock.shape[0]
    scrv = np.zeros(num_period)
    for i in range(num_period):
        scrv[i] = 1/(1+(adstock[i]/a)**(-s))
    return scrv
# %%
# Apply s-curve transformation to all adstock variables
adstock_col = [col for col in mmm_data.columns if "ad_" in col]
for col in adstock_col:
    scrv_cols = 'scrv_'+col
    mmm_data[scrv_cols] = scrv_transformation(mmm_data[col],0.5,1)
# %%
check_column_scrv = ['mdip_dm','ad_mdip_dm','scrv_ad_mdip_dm']
mmm_data[check_column_scrv]
# %%
adstock = 2431942.50
a = 2431942
s = 0.5
1/(1+(adstock/a)**(-s))
# %%
(adstock/a)**(-s)
# a here is the half saturation impression, so need to check how to get the half saturation of each media channel
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
    
    def __init__(self, alpha=1.0, normalize=True):
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
        self.model = Ridge(alpha=alpha, normalize=normalize)
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
        
        # Default bounds: each channel gets 5% to 50% of budget
        if channel_bounds is None:
            channel_bounds = {ch: (0.05, 0.50) for ch in self.channel_names}
        
        # Set up optimization bounds
        bounds = []
        for ch in self.channel_names:
            min_share, max_share = channel_bounds.get(ch, (0.05, 0.50))
            bounds.append((total_budget * min_share, total_budget * max_share))
        
        # Objective function
        def objective_func(budgets):
            # Ensure sum equals total budget
            budgets = np.array(budgets)
            budgets = budgets / np.sum(budgets) * total_budget
            
            # Calculate response
            X_list = []
            for i, ch in enumerate(self.channel_names):
                # Use budget as spend level (assuming weekly average)
                n_weeks = len(self.media_data[ch])
                weekly_spend = budgets[i] / n_weeks
                spend_array = np.full(n_weeks, weekly_spend)
                
                # Transform
                if 'theta' in self.params:
                    adstocked = adstock_geometric(spend_array, self.params['theta'])
                else:
                    adstocked = adstock_weibull(spend_array, 
                                              self.params['shape'], 
                                              self.params['scale'])
                
                saturated = saturation_hill(adstocked, 
                                          self.params['saturation_alpha'],
                                          self.params['saturation_gamma'])
                X_list.append(saturated.reshape(-1, 1))
            
            X_media = np.hstack(X_list)
            if self.base_vars_data is not None:
                X = np.hstack([X_media, self.base_vars_data])
            else:
                X = X_media
            
            # Predict total response
            pred = self.model.predict(X)
            total_response = np.sum(pred)
            
            # Maximize response (minimize negative)
            return -total_response
        
        # Optimize
        initial_guess = [total_budget / n_channels] * n_channels
        result = minimize(objective_func, initial_guess, method='L-BFGS-B', 
                         bounds=bounds, options={'maxiter': 1000})
        
        # Format results
        optimal_budgets = result.x
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
        """Calculate ROAS for a channel."""
        # Simplified ROAS calculation
        # In practice, this would use the response curve
        return 2.0  # Placeholder

# %%
# End-to-End MMM Workflow Function
def run_robyn_mmm(data, date_col='wk_strt_dt', dep_var='sales',
                  media_spend_cols=None, base_vars=None,
                  adstock_type='geometric', trials=5, 
                  optimize_hyperparams=True):
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
        Media spend columns (auto-detected if None)
    base_vars : list, optional
        Baseline variables
    adstock_type : str
        'geometric' or 'weibull'
    trials : int
        Number of optimization trials
    optimize_hyperparams : bool
        Whether to optimize hyperparameters
    
    Returns:
    --------
    dict : Complete MMM results
    """
    print("="*80)
    print("ROBYN-STYLE MMM WORKFLOW")
    print("="*80)
    
    # 1. Prepare data
    print("\n[1/7] Preparing data...")
    if media_spend_cols is None:
        media_spend_cols = [col for col in data.columns if "mdsp_" in col]
    
    if base_vars is None:
        base_vars = [col for col in data.columns if any(prefix in col 
                   for prefix in ['me_', 'hldy_', 'seas_', 'st_', 'mkrdn_'])]
    
    # Extract media data
    media_data = {}
    for col in media_spend_cols:
        channel_name = col.replace('mdsp_', '')
        media_data[channel_name] = data[col].values
    
    # Extract target and baseline
    target = data[dep_var].values
    base_vars_array = data[base_vars].values if len(base_vars) > 0 else None
    
    print(f"  - Media channels: {list(media_data.keys())}")
    print(f"  - Baseline variables: {len(base_vars)}")
    print(f"  - Data points: {len(data)}")
    
    # 2. Optimize hyperparameters
    print("\n[2/7] Optimizing hyperparameters...")
    if optimize_hyperparams:
        optimizer = MMMOptimizer(adstock_type=adstock_type)
        best_params = optimizer.optimize(
            media_data, target, base_vars_array,
            maxiter=30, popsize=10
        )
        print(f"  - Best NRMSE: {best_params['nrmse']:.4f}")
        print(f"  - Parameters: {best_params}")
    else:
        # Use default parameters
        if adstock_type == 'geometric':
            best_params = {'theta': 0.5, 'ridge_alpha': 1.0, 
                          'saturation_alpha': 1.0, 'saturation_gamma': 1.0}
        else:
            best_params = {'shape': 1.0, 'scale': 1.0, 'ridge_alpha': 1.0,
                          'saturation_alpha': 1.0, 'saturation_gamma': 1.0}
    
    # 3. Transform media variables
    print("\n[3/7] Transforming media variables...")
    X_list = []
    for channel, channel_data in media_data.items():
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
        X_list.append(saturated.reshape(-1, 1))
    
    X_media = np.hstack(X_list)
    if base_vars_array is not None:
        X = np.hstack([X_media, base_vars_array])
    else:
        X = X_media
    
    # 4. Fit model
    print("\n[4/7] Fitting MMM model...")
    model = MMMModel(alpha=best_params['ridge_alpha'], normalize=True)
    model.fit(X, target)
    y_pred = model.predict(X)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(target, y_pred))
    nrmse = rmse / (np.max(target) - np.min(target))
    r2 = r2_score(target, y_pred)
    mae = mean_absolute_error(target, y_pred)
    
    print(f"  - R²: {r2:.4f}")
    print(f"  - NRMSE: {nrmse:.4f}")
    print(f"  - RMSE: {rmse:.2f}")
    print(f"  - MAE: {mae:.2f}")
    
    # 5. Model diagnostics
    print("\n[5/7] Generating model diagnostics...")
    
    # Plot actual vs predicted
    plot.figure(figsize=(14, 5))
    
    plot.subplot(1, 2, 1)
    plot.scatter(target, y_pred, alpha=0.6)
    plot.plot([target.min(), target.max()], [target.min(), target.max()], 
             'r--', lw=2, label='Perfect Prediction')
    plot.xlabel('Actual Sales', fontsize=11)
    plot.ylabel('Predicted Sales', fontsize=11)
    plot.title(f'Actual vs Predicted (R² = {r2:.3f})', fontsize=12, fontweight='bold')
    plot.legend()
    plot.grid(True, alpha=0.3)
    
    # Plot time series
    plot.subplot(1, 2, 2)
    plot.plot(data[date_col], target, label='Actual', linewidth=2, alpha=0.7)
    plot.plot(data[date_col], y_pred, label='Predicted', linewidth=2, alpha=0.7)
    plot.xlabel('Date', fontsize=11)
    plot.ylabel('Sales', fontsize=11)
    plot.title('Time Series: Actual vs Predicted', fontsize=12, fontweight='bold')
    plot.legend()
    plot.xticks(rotation=45)
    plot.grid(True, alpha=0.3)
    plot.tight_layout()
    plot.show()
    
    # 6. Response curves
    print("\n[6/7] Generating response curves...")
    response_curves = {}
    n_channels = len(media_data)
    rows = (n_channels + 1) // 2
    cols = 2 if n_channels > 1 else 1
    
    plot.figure(figsize=(14, 4 * rows))
    for idx, channel in enumerate(media_data.keys(), 1):
        spend_range, responses = generate_response_curves(
            model, media_data, base_vars_array, channel, best_params
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
    
    # 7. Budget allocation example
    print("\n[7/7] Budget allocation example...")
    allocator = BudgetAllocator(model, media_data, base_vars_array, best_params)
    total_budget = sum([data[col].sum() for col in media_spend_cols])
    allocation = allocator.allocate_budget(total_budget)
    
    print("\nOptimal Budget Allocation:")
    print("-" * 60)
    for channel, alloc in allocation.items():
        print(f"{channel:20s} | Budget: ${alloc['budget']:>12,.0f} | Share: {alloc['share']*100:>5.1f}%")
    
    # Compile results
    results = {
        'model': model,
        'params': best_params,
        'metrics': {
            'r2': r2,
            'nrmse': nrmse,
            'rmse': rmse,
            'mae': mae
        },
        'predictions': y_pred,
        'response_curves': response_curves,
        'budget_allocation': allocation,
        'feature_names': list(media_data.keys()) + base_vars,
        'coefficients': model.coefficients
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
    media_spend_cols=mdsp_col,
    base_vars=base_vars,
    adstock_type='geometric',
    trials=5,
    optimize_hyperparams=True
)
