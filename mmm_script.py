# %%
print("Hello Marketing Mix Modeling")
# %%
import numpy as np
import pandas as pd
# %%
mmm_data = pd.read_csv("data.csv")
# %%
mmm_data
# %%
# understand data
mmm_data.describe()
mmm_data.columns
# %%
# 1. Media Variables 
# media impression columns
mdip_col = [col for col in mmm_data.columns if "mdip_" in col]
# media spend columns
mdsp_col = [col for col in mmm_data.columns if "mdsp_" in col]

# 2. Base Variables (Control variables)
# macro economics variables 
me_col = [col for col in mmm_data.columns if "me_" in col]
# %%
# create adstock function
def adstock(support,half_life):
    decay = np.exp(np.log(0.5) / half_life)
    num_period = support.shape[0]
    adstock = np.zeros(num_period)
    for i in range(num_period):
        if i ==0 :
            adstock[i] = (1.0 - decay[i]) * support[i]
        else:
            adstock[i] = (1.0 - decay[i]) * support[i] + decay[i] * adstock[i-1]

