# Baseline OLS Model Improvements

## Current Issues
- **R² = -0.245** (model performing worse than predicting the mean)
- Only using 3 media channels + 2 control variables
- No adstock (lagged effects) transformation
- No saturation effects
- Missing critical seasonal/promotional variables

## Recommended Improvements

### 1. **Include ALL Relevant Variables**
Instead of using only first 3 media channels and 2 controls:
- Use ALL media spend variables (13 channels available)
- Use ALL control variables (holidays, seasonality, macroeconomics, store count, discounts)
- This captures the full picture of what drives sales

### 2. **Add Adstock Transformation (Lagged Effects)**
Marketing effects don't happen instantly - they decay over time:
```python
def apply_adstock(x, decay_rate=0.5):
    """
    Apply geometric adstock transformation
    decay_rate: between 0-1, how much effect carries to next period
    """
    adstocked = np.zeros_like(x)
    adstocked[0] = x[0]
    for i in range(1, len(x)):
        adstocked[i] = x[i] + decay_rate * adstocked[i-1]
    return adstocked
```

### 3. **Add Saturation Effects (Diminishing Returns)**
More spend doesn't mean proportional returns:
```python
def hill_saturation(x, alpha=1.0, gamma=0.5):
    """
    Hill saturation transformation
    alpha: half-saturation point
    gamma: shape parameter (controls curve steepness)
    """
    return x**gamma / (alpha**gamma + x**gamma)
```

### 4. **Better Feature Engineering**

#### Seasonal Features:
- Week of year
- Month indicators
- Quarter indicators  
- Year-end holiday period indicator

#### Promotional Features:
- Discount levels (mrkdn_* variables)
- Holiday indicators (already in data)
- Value-added promotions (va_pub_* variables)

#### Interaction Terms:
- Media spend × seasonal indicators
- Media spend × promotional periods

### 5. **Improved Model Architecture**

**Option A: Enhanced Linear Model**
```python
# Use Ridge regression with cross-validation for regularization
from sklearn.linear_model import RidgeCV

# Apply transformations
X_adstocked = apply_adstock_to_media(X_media)
X_saturated = apply_saturation(X_adstocked)

# Combine with all controls
X_full = np.hstack([X_saturated, X_controls])

# Fit with regularization
model = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
model.fit(X_train, y_train)
```

**Option B: Gradient Boosting (XGBoost/LightGBM)**
- Handles non-linear relationships automatically
- Captures complex interactions
- More robust to outliers

### 6. **Cross-Validation Strategy**
Instead of single train/test split:
```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
# Evaluate on multiple folds to get robust metrics
```

### 7. **Standardization/Scaling**
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

## Implementation Priority

### **Quick Wins (Implement First):**
1. ✅ Use ALL media channels and control variables
2. ✅ Add standardization/scaling
3. ✅ Use Ridge regression with regularization

### **Medium Effort (High Impact):**
4. ✅ Add simple adstock transformation (geometric decay)
5. ✅ Include more seasonal features
6. ✅ Add interaction terms for promotions × media

### **Advanced (For Best Performance):**
7. ✅ Optimize adstock and saturation parameters
8. ✅ Try gradient boosting models
9. ✅ Implement time series cross-validation

## Expected Results After Improvements

| Metric | Current | Expected After Improvements |
|--------|---------|----------------------------|
| R² | -0.245 | 0.60 - 0.85+ |
| RMSE | Very High | 60-70% reduction |
| MAPE | Very High | < 15-20% |

## Code Example: Improved Baseline Model

```python
# ============================================================================
# IMPROVED BASELINE OLS MODEL
# ============================================================================

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd

# 1. USE ALL VARIABLES (not just first 3)
media_cols_full = mdsp_col  # All 13 media channels
control_cols_full = base_vars  # All control variables

print(f"Using {len(media_cols_full)} media channels")
print(f"Using {len(control_cols_full)} control variables")

# 2. APPLY ADSTOCK TRANSFORMATION
def geometric_adstock(x, decay=0.5):
    adstocked = np.zeros_like(x, dtype=float)
    adstocked[0] = x[0]
    for i in range(1, len(x)):
        adstocked[i] = x[i] + decay * adstocked[i-1]
    return adstocked

# Apply adstock to media variables
model_data = mmm_data.copy()
for col in media_cols_full:
    model_data[f'{col}_adstock'] = geometric_adstock(model_data[col].values, decay=0.5)

# 3. PREPARE DATA
# Split chronologically
split_point = int(len(model_data) * TRAIN_TEST_SPLIT)
train_data = model_data.iloc[:split_point]
test_data = model_data.iloc[split_point:]

# Create feature matrix with adstocked media + all controls
feature_cols = [f'{col}_adstock' for col in media_cols_full] + control_cols_full

X_train = train_data[feature_cols].values
y_train = train_data['sales'].values
X_test = test_data[feature_cols].values  
y_test = test_data['sales'].values

# 4. SCALE FEATURES
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. FIT RIDGE REGRESSION (with regularization to prevent overfitting)
model = Ridge(alpha=100.0)  # Regularization strength
model.fit(X_train_scaled, y_train)

# 6. EVALUATE
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

# Metrics
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae_test = mean_absolute_error(y_test, y_pred_test)
mape_test = np.mean(np.abs((y_test - y_pred_test) / (y_test + 1e-10))) * 100

print(f"\nImproved Model Results:")
print(f"Train R²: {r2_train:.4f}")
print(f"Test R²: {r2_test:.4f}")
print(f"Test RMSE: {rmse_test:,.0f}")
print(f"Test MAE: {mae_test:,.0f}")
print(f"Test MAPE: {mape_test:.2f}%")

# 7. PLOT RESULTS
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 6))
plt.plot(test_data['wk_strt_dt'], y_test, label='Actual', linewidth=2, alpha=0.8)
plt.plot(test_data['wk_strt_dt'], y_pred_test, label='Predicted', linewidth=2, alpha=0.8)
plt.legend()
plt.xlabel('Date', fontsize=12)
plt.ylabel('Sales', fontsize=12)
plt.title(f'Improved Baseline Model: R² = {r2_test:.3f}', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

## Next Steps

1. Implement the improved baseline model above
2. Compare results with current Robyn MMM model
3. If still unsatisfactory, add saturation effects
4. Consider trying XGBoost/LightGBM for non-linear patterns
5. Perform hyperparameter optimization for adstock decay rates

## References
- [Meta Robyn](https://github.com/facebookexperimental/Robyn) - Industry standard MMM approach
- [PyMC Marketing](https://www.pymc-marketing.io/) - Bayesian MMM in Python
- [LightweightMMM](https://github.com/google/lightweight_mmm) - Google's MMM framework
