# Marketing Mix Model - Problem Analysis & Solutions

## 🔴 The Problem

Your baseline OLS model showed **R² = -0.245**, meaning:
- The model performed **worse than just predicting the average sales** every time
- Predictions were completely flat (~0.5-1.0e8) while actual sales had huge spikes (up to 3.5e8)
- The model failed to capture any meaningful patterns in the data

## 🔍 Root Causes

### 1. **Using Too Few Variables**
**Before:**
- Only 3 media channels (out of 13 available)
- Only 2 control variables (out of 30+ available)
- Missing critical seasonal, holiday, and promotional factors

**Why This Failed:**
Sales are driven by many factors. Using only 5 variables to predict a complex outcome like sales is like trying to explain a movie with just 5 words.

### 2. **No Adstock (Lagged Effects)**
**Before:**
- Treated all marketing as having immediate impact only
- Ignored that advertising effects carry over multiple weeks

**Why This Failed:**
If you see a TV ad today, you might buy the product next week. The old model assumed you'd only buy today. This misses 80% of marketing's actual impact.

### 3. **No Saturation/Diminishing Returns**
**Before:**
- Assumed doubling spend = doubling sales (linear relationship)

**Why This Failed:**
Real world: First $1M in ads is very effective, but the 10th $1M has less impact. The model couldn't capture this.

### 4. **No Feature Scaling**
**Before:**
- Variables had vastly different scales (sales in millions, some controls 0-1)
- Made model coefficients unstable

### 5. **No Regularization**
**Before:**
- OLS regression with no penalty for overfitting
- High risk with many correlated media channels

---

## ✅ Solutions Implemented

### 1. **Use ALL Available Features**
```python
# NOW using:
- All 13 media spend channels (mdsp_*)
- All 30+ control variables:
  - Macro economics (me_*)
  - Holidays (hldy_*)
  - Seasonality (seas_*)
  - Store count (st_*)
  - Discounts/promotions (mrkdn_*, va_pub_*)
```

**Expected Impact:** Captures full picture of sales drivers → **+30-40% R² improvement**

### 2. **Add Adstock Transformation**
```python
def geometric_adstock_simple(x, decay=0.5):
    """Marketing effects decay over time"""
    adstocked = np.zeros_like(x, dtype=float)
    adstocked[0] = x[0]
    for i in range(1, len(x)):
        adstocked[i] = x[i] + decay * adstocked[i-1]
    return adstocked
```

**How It Works:**
- If you spend $100k on ads in week 1:
  - Week 1 effect: $100k
  - Week 2 effect: $100k + 0.5 × (week 1 effect) = $150k total
  - Week 3 effect: new spend + 0.5 × (week 2 effect)
  - etc.

**Expected Impact:** Captures delayed marketing effects → **+20-30% R² improvement**

### 3. **Use Ridge Regression (Regularization)**
```python
baseline_ridge_model = Ridge(alpha=100.0)
```

**Why Ridge vs OLS:**
- OLS: Finds best fit, but can overfit with many features
- Ridge: Adds penalty for large coefficients → prevents overfitting
- Works better when features are correlated (media channels often are)

**Expected Impact:** More stable predictions → **+10-15% R² improvement**

### 4. **Feature Scaling/Standardization**
```python
scaler_baseline = StandardScaler()
X_train_scaled = scaler_baseline.fit_transform(X_train)
```

**Why This Helps:**
- Puts all features on same scale (mean=0, std=1)
- Ridge regression works much better with scaled features
- Coefficients become directly comparable

**Expected Impact:** Better model convergence → **+5-10% R² improvement**

---

## 📊 Expected Results

| Metric | Before (Old Model) | After (Improved Model) | Change |
|--------|-------------------|------------------------|--------|
| **R² (Test)** | -0.245 | **0.60 - 0.80** | +0.85 to +1.05 |
| **RMSE** | Very High (~150M+) | **30-50M** | 60-70% reduction |
| **MAPE** | >100% | **15-25%** | 75-85% reduction |
| **Predictions** | Flat line | **Captures peaks/valleys** | Realistic |

---

## 🚀 Next Steps to Run Improved Model

1. **Run the updated script:**
   ```bash
   python mmm_script.py
   ```

2. **Check the new metrics:**
   - Look for "IMPROVED BASELINE MODEL RESULTS"
   - R² should now be positive and ideally > 0.6

3. **Interpret the output:**
   - **R² > 0.6**: Good! Model is working
   - **R² 0.4-0.6**: Decent, but room for improvement
   - **R² < 0.4**: Still issues, may need further tuning

4. **Review feature importance:**
   - The script will show top 10 most important features
   - Helps understand what drives sales most

---

## 🔧 Further Improvements (If Needed)

If R² is still < 0.6 after running improved model:

### A. **Optimize Adstock Decay Rate**
Currently using decay=0.5 (50% carryover). Try:
```python
# Test different decay rates
for decay in [0.3, 0.5, 0.7, 0.9]:
    # Fit model with this decay
    # Compare R²
```

### B. **Add Saturation Effects**
```python
def hill_saturation(x, alpha=1.0, gamma=0.5):
    return x**gamma / (alpha**gamma + x**gamma)
```

### C. **Try Different Regularization Strengths**
```python
from sklearn.linear_model import RidgeCV
model = RidgeCV(alphas=[1, 10, 100, 1000])  # Auto-selects best alpha
```

### D. **Add Interaction Terms**
```python
# e.g., TV spend × Holiday periods
# Media often works better during promotions
```

### E. **Try Advanced Models**
```python
from xgboost import XGBRegressor
model = XGBRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1
)
# Handles non-linearities automatically
```

---

## 📚 Key Takeaways

1. **More variables ≠ overfitting** (with proper regularization)
2. **Marketing has delayed effects** (adstock is critical for MMM)
3. **Regularization is essential** when features are correlated
4. **Feature scaling matters** for Ridge/Lasso regression
5. **Domain knowledge > complex models** (use what makes business sense)

---

## 🎯 Success Criteria

Your model is working well if:
- ✅ **R² > 0.6** on test set
- ✅ **Predictions follow actual sales trends** (peaks and valleys aligned)
- ✅ **Feature importance makes business sense** (top drivers are logical)
- ✅ **Residuals are random** (no clear patterns in residual plot)
- ✅ **MAPE < 20%** (predictions within 20% of actuals on average)

---

## 🆘 Troubleshooting

**If R² is still negative:**
- Check for data leakage (future data in training set)
- Verify train/test split is chronological
- Check for missing values or infinite values
- Ensure scaling is fit on train only, transform on test

**If R² is very high in training (>0.95) but low in test (<0.5):**
- Overfitting! Increase Ridge alpha (regularization)
- Reduce number of features
- Use cross-validation

**If predictions are all very similar:**
- Check feature variance (maybe all scaled to 0?)
- Ensure adstock is working (print values to verify)
- Try different alpha values in Ridge

---

For any issues, check:
1. Console output for error messages
2. Feature importance (are coefficients reasonable?)
3. Residual plots (should be random scatter)
4. Data distribution (any extreme outliers?)
