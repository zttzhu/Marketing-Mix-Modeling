# %%
print("Hello Marketing Mix Modeling")
# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plot
import prophet as Prophet
import statsmodels.api as sm
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
# We will need to find a way to derive half saturation point here
# %%
# Baseline MMM using OLS on raw spend and control variables
media_cols_baseline = ['mdsp_vidtr', 'mdsp_dm', 'mdsp_inst']
control_cols_baseline = ['me_ics_all', 'st_ct']

# Log-transform sales and features to stabilize variance
model_data = mmm_data.copy()
model_data['sales'] = np.log1p(model_data['sales'])
for col in media_cols_baseline + control_cols_baseline:
    model_data[col] = np.log1p(model_data[col])

# Chronological train/test split
split_point = int(len(model_data) * 0.8)
train = model_data.iloc[:split_point]
test = model_data.iloc[split_point:]

# Fit simple linear regression
X_train = sm.add_constant(train[media_cols_baseline + control_cols_baseline])
y_train = train['sales']
baseline_model = sm.OLS(y_train, X_train).fit()

# Evaluate on holdout set
X_test = sm.add_constant(test[media_cols_baseline + control_cols_baseline])
y_test = test['sales']
y_pred = baseline_model.predict(X_test)
rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(baseline_model.summary())
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")

# Plot actual vs predicted sales
plot.figure(figsize=(12,6))
plot.plot(test['wk_strt_dt'], np.expm1(y_test), label='Actual')
plot.plot(test['wk_strt_dt'], np.expm1(y_pred), label='Predicted')
plot.legend()
