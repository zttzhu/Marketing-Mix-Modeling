# Marketing Mix Modeling (MMM)

A comprehensive Marketing Mix Modeling implementation using Robyn-style methodology with adstock and saturation transformations, model validation, diagnostics, and budget optimization.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Data Format](#data-format)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Configuration](#configuration)
- [Outputs](#outputs)
- [Key Functions](#key-functions)
- [Metrics Explained](#metrics-explained)

## 🎯 Overview

This project implements a production-ready Marketing Mix Model (MMM) that:

- **Models media effectiveness** using adstock and saturation transformations
- **Validates models** with train/test splits and comprehensive diagnostics
- **Decomposes contributions** to understand baseline vs. media impact
- **Optimizes budgets** across media channels
- **Calculates business metrics** (ROAS, CPM, Effectiveness) by channel and year
- **Compares models** (Robyn MMM vs. Baseline OLS) for validation

## ✨ Features

### Core Functionality

- ✅ **Robyn-Style MMM Workflow**
  - Geometric and Weibull adstock transformations
  - Hill function saturation curves
  - Ridge regression with hyperparameter optimization
  - Differential evolution for parameter tuning

- ✅ **Model Validation**
  - Chronological train/test split (80/20 default)
  - Separate training and test metrics
  - Residual analysis (Q-Q plots, residual plots)
  - Model comparison tables

- ✅ **Contribution Decomposition**
  - Baseline contribution (intercept + control variables)
  - Media contribution by channel
  - Total predicted sales breakdown

- ✅ **Budget Optimization**
  - Optimal budget allocation across channels
  - ROAS calculation per channel
  - Budget constraints support

- ✅ **Channel Metrics Analysis**
  - ROAS (Return on Ad Spend)
  - CPM (Cost Per Mille)
  - Effectiveness (Revenue per Impression)
  - Due-to Contribution
  - Summary tables by channel and year

- ✅ **Visualizations**
  - Actual vs. Predicted plots
  - Response curves by channel
  - Residual diagnostics
  - Time series comparisons
  - Media spend analysis

## 📦 Requirements

```python
numpy
pandas
seaborn
matplotlib
scikit-learn
scipy
statsmodels
```

Install dependencies:
```bash
pip install numpy pandas seaborn matplotlib scikit-learn scipy statsmodels
```

## 📊 Data Format

The script expects a CSV file (`data.csv`) with the following column naming conventions:

### Required Columns

- **Date Column**: `wk_strt_dt` (weekly start date, format: YYYY-MM-DD)
- **Target Variable**: `sales` (dependent variable)

### Media Variables

- **Media Spend**: Columns prefixed with `mdsp_` (e.g., `mdsp_vidtr`, `mdsp_dm`, `mdsp_inst`)
- **Media Impressions**: Columns prefixed with `mdip_` (e.g., `mdip_vidtr`, `mdip_dm`, `mdip_inst`)

### Control Variables (Optional)

- **Macro Economics**: Columns prefixed with `me_`
- **Holidays**: Columns prefixed with `hldy_`
- **Seasonality**: Columns prefixed with `seas_`
- **Store Count**: Columns prefixed with `st_`
- **Discounts**: Columns prefixed with `mkrdn_`

### Example Data Structure

```
wk_strt_dt    | sales | mdsp_vidtr | mdip_vidtr | mdsp_dm | mdip_dm | me_ics_all | st_ct
2020-01-05    | 15000 | 50000      | 2000000    | 30000   | 1500000 | 0.95      | 100
2020-01-12    | 16500 | 55000      | 2200000    | 32000   | 1600000 | 0.96      | 100
...
```

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/zttzhu/Marketing-Mix-Modeling.git
cd Marketing-Mix-Modeling
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place your data file as `data.csv` in the project directory

## 💻 Usage

### Basic Usage

Simply run the script:
```bash
python mmm_script.py
```

Or use in Jupyter Notebook:
```python
# Run all cells in mmm_script.py
```

### Configuration

Edit the configuration section at the top of the script:

```python
TRAIN_TEST_SPLIT = 0.8  # Proportion of data for training
OPTIMIZATION_MAXITER = 30  # Maximum iterations for hyperparameter optimization
OPTIMIZATION_POPSIZE = 10  # Population size for differential evolution
ADSTOCK_TYPE = 'geometric'  # 'geometric' or 'weibull'
OPTIMIZE_HYPERPARAMS = True  # Whether to optimize hyperparameters
```

### Custom Usage

```python
# Run Robyn MMM workflow
results = run_robyn_mmm(
    data=mmm_data,
    date_col='wk_strt_dt',
    dep_var='sales',
    media_spend_cols=mdsp_col,
    base_vars=base_vars,
    adstock_type='geometric',
    optimize_hyperparams=True,
    train_test_split=0.8
)

# Calculate channel metrics by year
channel_metrics = calculate_channel_metrics_by_year(
    mmm_data,
    results,
    date_col='wk_strt_dt',
    dep_var='sales',
    media_spend_cols=mdsp_col,
    media_impression_cols=mdip_col
)
```

## 🔬 Methodology

### Robyn-Style MMM

The implementation follows the Robyn methodology:

1. **Adstock Transformation**
   - **Geometric**: `adstock[t] = x[t] + θ * adstock[t-1]`
   - **Weibull**: Uses Weibull distribution for decay weights

2. **Saturation Transformation**
   - **Hill Function**: `saturated = (x^γ) / (x^γ + α^γ)`
   - Captures diminishing returns

3. **Model Fitting**
   - Ridge regression with regularization
   - Hyperparameter optimization using differential evolution
   - Normalized features for stability

4. **Validation**
   - Chronological train/test split
   - Out-of-sample performance metrics
   - Residual diagnostics

### Baseline OLS Model

A simple OLS model is also provided for comparison:
- Log-transformed variables
- Linear regression on raw spend and control variables
- Same train/test split for fair comparison

## ⚙️ Configuration

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TRAIN_TEST_SPLIT` | 0.8 | Proportion of data for training |
| `OPTIMIZATION_MAXITER` | 30 | Max iterations for hyperparameter optimization |
| `OPTIMIZATION_POPSIZE` | 10 | Population size for differential evolution |
| `ADSTOCK_TYPE` | 'geometric' | Adstock type: 'geometric' or 'weibull' |
| `OPTIMIZE_HYPERPARAMS` | True | Whether to optimize hyperparameters |

### Hyperparameter Bounds

**Geometric Adstock:**
- θ (decay): [0.01, 0.99]
- Ridge α: [0.01, 100.0]
- Saturation α: [0.1, 10.0]
- Saturation γ: [0.1, 3.0]

**Weibull Adstock:**
- Shape: [0.1, 10.0]
- Scale: [0.1, 20.0]
- Ridge α: [0.01, 100.0]
- Saturation α: [0.1, 10.0]
- Saturation γ: [0.1, 3.0]

## 📈 Outputs

### 1. Model Metrics

**Training Metrics:**
- R² (Coefficient of Determination)
- NRMSE (Normalized Root Mean Squared Error)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)

**Test Metrics:**
- R², NRMSE, RMSE, MAE
- MAPE (Mean Absolute Percentage Error)

### 2. Visualizations

- **Actual vs. Predicted**: Scatter plots for train and test sets
- **Residual Analysis**: Residual plots and Q-Q plots
- **Time Series**: Actual vs. predicted over time
- **Response Curves**: Media response curves by channel
- **Media Spend Analysis**: Spend distribution and trends

### 3. Contribution Decomposition

- Baseline contribution (intercept + control variables)
- Media contribution by channel
- Total predicted sales

### 4. Budget Allocation

Optimal budget allocation table showing:
- Channel
- Allocated Budget
- Budget Share (%)
- ROAS

### 5. Channel Metrics by Year

Comprehensive summary table with:
- **Spend**: Total spend per channel per year
- **Impressions**: Total impressions per channel per year
- **ROAS**: Return on Ad Spend
- **CPM**: Cost Per Mille (cost per 1,000 impressions)
- **Effectiveness**: Incremental revenue per impression
- **Due-to Contribution**: Media contribution from model

### 6. Model Comparison

Side-by-side comparison of:
- Robyn MMM vs. Baseline OLS
- Performance metrics
- Improvement percentages

### 7. Exported Files

- `media_channel_metrics_by_year.csv`: Detailed metrics by channel and year

## 🔧 Key Functions

### `run_robyn_mmm()`

Main workflow function that:
1. Prepares and splits data
2. Optimizes hyperparameters
3. Transforms media variables (adstock + saturation)
4. Fits Ridge regression model
5. Calculates metrics and diagnostics
6. Generates response curves
7. Optimizes budget allocation

**Returns:** Dictionary with model, metrics, predictions, decomposition, and more

### `calculate_channel_metrics_by_year()`

Calculates comprehensive metrics for each media channel by year:
- ROAS, CPM, Effectiveness
- Due-to Contribution
- Summary tables and pivot tables

**Returns:** DataFrame with metrics by channel and year

### `show_media_spend()`

Analyzes and visualizes media spend across channels:
- Summary statistics
- Time series plots
- Distribution plots
- Comparison charts

### `BudgetAllocator`

Class for optimizing budget allocation:
- Maximizes response given budget constraints
- Calculates ROAS per channel
- Supports channel bounds

### `MMMModel`

Main model class:
- Ridge regression with normalization
- Adstock and saturation transformations
- Feature scaling

## 📊 Metrics Explained

### ROAS (Return on Ad Spend)
```
ROAS = Incremental Revenue / Spend
     = Due-to Contribution / Spend
```
- **Interpretation**: Revenue generated per dollar spent
- **Example**: ROAS of 2.5 means $2.50 revenue per $1.00 spent

### CPM (Cost Per Mille)
```
CPM = (Spend / Impressions) × 1000
```
- **Interpretation**: Cost to reach 1,000 impressions
- **Example**: CPM of $5.00 means $5 to reach 1,000 impressions

### Effectiveness
```
Effectiveness = Incremental Revenue / Impressions
              = Due-to Contribution / Impressions
```
- **Interpretation**: Revenue generated per impression
- **Example**: $0.001 per impression

### Due-to Contribution
- **Interpretation**: Incremental revenue attributed to a media channel
- **Calculation**: Media contribution from model decomposition
- **Use**: Understand which channels drive sales

## 📝 Example Workflow

```python
# 1. Load data
mmm_data = pd.read_csv("data.csv")
mmm_data['wk_strt_dt'] = pd.to_datetime(mmm_data['wk_strt_dt'])

# 2. Identify variables
mdsp_col = [col for col in mmm_data.columns if "mdsp_" in col]
mdip_col = [col for col in mmm_data.columns if "mdip_" in col]
base_vars = [col for col in mmm_data.columns if any(
    prefix in col for prefix in ['me_', 'hldy_', 'seas_', 'st_', 'mkrdn_']
)]

# 3. Run MMM workflow
results = run_robyn_mmm(
    mmm_data,
    date_col='wk_strt_dt',
    dep_var='sales',
    media_spend_cols=mdsp_col,
    base_vars=base_vars,
    adstock_type='geometric',
    optimize_hyperparams=True
)

# 4. Calculate channel metrics
channel_metrics = calculate_channel_metrics_by_year(
    mmm_data, results,
    media_spend_cols=mdsp_col,
    media_impression_cols=mdip_col
)

# 5. Access results
print(f"Test R²: {results['metrics']['test']['r2']:.4f}")
print(f"Budget Allocation: {results['budget_allocation']}")
```

## 🎓 Best Practices

1. **Data Quality**
   - Ensure no missing values in key columns
   - Check for outliers in spend and sales
   - Validate date formats

2. **Model Selection**
   - Compare geometric vs. Weibull adstock
   - Use train/test metrics to select best model
   - Consider business context when interpreting results

3. **Hyperparameter Tuning**
   - Start with default parameters
   - Increase `OPTIMIZATION_MAXITER` for better results (slower)
   - Use `OPTIMIZE_HYPERPARAMS=False` for quick iterations

4. **Interpretation**
   - Focus on test metrics (out-of-sample performance)
   - Consider contribution decomposition for business insights
   - Use ROAS and effectiveness for budget decisions

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

Marketing Data Science Team

## 🙏 Acknowledgments

- Inspired by Facebook's Robyn methodology
- Uses scikit-learn, scipy, and statsmodels libraries

---

For detailed code review and recommendations, see `MMM_SCRIPT_REVIEW.md`.
