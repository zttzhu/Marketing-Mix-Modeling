# %%
print("Hello Marketing Mix Modeling")
# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plot
pd.set_option('display.float_format', '{:.2f}'.format)

# %%
mmm_data = pd.read_csv("data.csv")
# %%
mmm_data
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