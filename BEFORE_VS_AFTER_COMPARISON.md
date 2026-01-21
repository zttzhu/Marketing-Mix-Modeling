# Before vs After: Baseline Model Comparison

## Visual Comparison

### OLD MODEL (R² = -0.245) ❌
```
[Week 1] → TV Spend: $100k → Immediate Impact: $100k → Sales Effect
[Week 2] → TV Spend: $50k  → Immediate Impact: $50k  → Sales Effect
[Week 3] → TV Spend: $0    → Immediate Impact: $0    → Sales Effect
                              
Only 3 media channels used
Only 2 control variables used
No carryover effects
Linear relationships only
No regularization
```

**Result:** Model can't explain sales spikes → Predicts flat line → R² is NEGATIVE

---

### NEW MODEL (Expected R² = 0.60-0.80) ✅
```
[Week 1] → TV Spend: $100k → Adstocked: $100k        → Sales Effect
[Week 2] → TV Spend: $50k  → Adstocked: $100k+$50k   → Sales Effect (carryover!)
[Week 3] → TV Spend: $0    → Adstocked: $75k          → Sales Effect (decay)

All 13 media channels used
All 30+ control variables used (holidays, seasonality, etc.)
Adstock transformation (carryover effects)
Feature scaling + Ridge regularization
```

**Result:** Model captures complex patterns → Follows actual trends → R² is POSITIVE

---

## Side-by-Side Feature Comparison

| Aspect | OLD Model | NEW Model |
|--------|-----------|-----------|
| **Media Channels** | 3 channels (mdsp_dm, mdsp_inst, mdsp_nsp) | All 13 channels |
| **Control Variables** | 2 controls (me_ics_all, me_gas_dpg) | All ~30 controls |
| **Adstock** | ❌ None | ✅ Geometric decay (0.5) |
| **Saturation** | ❌ None | Not yet (can add) |
| **Scaling** | ❌ None | ✅ StandardScaler |
| **Regularization** | ❌ None (OLS) | ✅ Ridge (alpha=100) |
| **Log Transform** | ✅ Yes | ❌ No (using raw) |
| **Features Used** | 5 total | 43+ total |
| **Model Type** | statsmodels OLS | sklearn Ridge |

---

## What Changed in the Code

### 1. Variable Selection
```python
# OLD (BAD)
media_cols_baseline = mdsp_col[:3]  # Only first 3
control_cols_baseline = base_vars[:2]  # Only first 2

# NEW (GOOD)
media_cols_baseline = mdsp_col  # ALL media channels
control_cols_baseline = base_vars  # ALL control variables
```

### 2. Adstock Transformation
```python
# NEW: Add this function
def geometric_adstock_simple(x, decay=0.5):
    adstocked = np.zeros_like(x, dtype=float)
    adstocked[0] = x[0]
    for i in range(1, len(x)):
        adstocked[i] = x[i] + decay * adstocked[i-1]
    return adstocked

# Apply to all media
for col in media_cols_baseline:
    model_data[f'{col}_adstock'] = geometric_adstock_simple(
        model_data[col].values, decay=0.5
    )
```

### 3. Feature Scaling
```python
# NEW: Scale features before fitting
scaler_baseline = StandardScaler()
X_train_scaled = scaler_baseline.fit_transform(X_train)
X_test_scaled = scaler_baseline.transform(X_test)
```

### 4. Ridge Regression
```python
# OLD (BAD)
baseline_ols_model = sm.OLS(y_train, X_train).fit()

# NEW (GOOD)
baseline_ridge_model = Ridge(alpha=100.0)
baseline_ridge_model.fit(X_train_scaled, y_train)
```

---

## How Adstock Works (Example)

### Scenario: TV Ad Campaign
- **Week 1:** Spend $1M on TV ads
- **Week 2:** Spend $0 (no ads)
- **Week 3:** Spend $0 (no ads)

### Without Adstock (OLD MODEL):
```
Week 1 Sales Effect: $1M × coefficient = impact
Week 2 Sales Effect: $0 × coefficient = 0 impact ❌ WRONG!
Week 3 Sales Effect: $0 × coefficient = 0 impact ❌ WRONG!
```
**Problem:** Assumes ad impact ends immediately when spending stops.

### With Adstock (NEW MODEL, decay=0.5):
```
Week 1 Adstocked Value: $1M
Week 2 Adstocked Value: $1M × 0.5 = $500k (carryover effect!)
Week 3 Adstocked Value: $500k × 0.5 = $250k (still some effect!)

Week 1 Sales Effect: $1M × coefficient = impact
Week 2 Sales Effect: $500k × coefficient = continued impact ✅ CORRECT!
Week 3 Sales Effect: $250k × coefficient = decaying impact ✅ CORRECT!
```
**Benefit:** Captures realistic advertising decay over time.

---

## Why This Matters for Your Data

Looking at your chart, the actual sales show:
- **Huge spikes** (up to 3.5e8) → likely holiday periods or major promotions
- **Strong seasonality** → year-end peaks
- **High volatility** → week-to-week changes

### OLD Model Couldn't Capture This Because:
1. Missing holiday indicators (only used 2 controls)
2. Missing seasonal variables (seas_prd_*, seas_week_*)
3. No adstock (couldn't explain why sales stayed high for weeks after big media push)
4. Only 3 media channels (missing 10 other important drivers)

### NEW Model Can Capture This Because:
1. ✅ **Has ALL holiday indicators** (hldy_Black Friday, hldy_Christmas, etc.)
2. ✅ **Has ALL seasonal variables** (seas_prd_1-12, seas_week_40-48)
3. ✅ **Adstock explains sustained effects** after media spends
4. ✅ **All 13 media channels** included
5. ✅ **Discount/promotion variables** (mrkdn_*, va_pub_*)

---

## Expected Visual Improvement

### Before (R² = -0.245):
```
Actual:    /\    /\        /\
          /  \  /  \  /\  /  \___
         /    \/    \/  \/
        
Predicted: _________________ (flat line at mean)
```

### After (Expected R² = 0.60-0.80):
```
Actual:    /\    /\        /\
          /  \  /  \  /\  /  \___
         /    \/    \/  \/
        
Predicted: /\   /\       /\
          /  \_/  \  /\_/  \___
         /        \/
         
(Following the actual trends!)
```

---

## Diagnostic Metrics to Watch

When you run the improved model, you should see:

### ✅ **Good Signs:**
- R² test > 0.6
- R² train ≈ R² test (within 0.1-0.15)
- MAPE < 20%
- Residuals randomly scattered (no pattern)
- Top features make business sense

### ⚠️ **Warning Signs:**
- R² train >> R² test (e.g., 0.9 vs 0.5) → Overfitting
- R² still negative → Something fundamentally wrong
- All predictions very similar → Model not learning
- Top features are all noise variables → Bad model

### 🔴 **Red Flags:**
- R² test < 0 → Model worse than baseline
- MAPE > 50% → Predictions way off
- Feature coefficients are huge → Scaling issue
- Predictions outside data range → Extrapolation problem

---

## Testing the Improvements

Run this and compare:

```python
# After running mmm_script.py, check:

1. Console output for "IMPROVED BASELINE MODEL RESULTS"
   - Is R² positive?
   - Is R² > 0.6?

2. Chart: "Test Set: Improved Baseline Model"
   - Does predicted line follow actual trends?
   - Are peaks/valleys aligned?

3. Residual plot:
   - Points randomly scattered? ✅ Good
   - Clear pattern visible? ❌ Bad

4. Top 10 features:
   - Do they make business sense?
   - Are important media channels included?
```

---

## Quick Wins if R² is Still Low

If R² < 0.5 after improvements:

1. **Tune adstock decay:**
   ```python
   # Try decay values: 0.3, 0.5, 0.7, 0.9
   # Higher decay = longer carryover
   ```

2. **Tune Ridge alpha:**
   ```python
   # Try alpha values: 10, 50, 100, 500, 1000
   # Higher alpha = more regularization
   ```

3. **Check for outliers:**
   ```python
   # Remove extreme sales values?
   # Or use log transformation?
   ```

4. **Add polynomial features:**
   ```python
   from sklearn.preprocessing import PolynomialFeatures
   poly = PolynomialFeatures(degree=2, interaction_only=True)
   ```

5. **Try different model:**
   ```python
   from sklearn.ensemble import RandomForestRegressor
   # Or XGBoost, LightGBM
   ```

---

## Summary

**What We Fixed:**
1. ✅ Using ALL variables (not just 5)
2. ✅ Adding adstock transformation
3. ✅ Feature scaling
4. ✅ Ridge regularization

**Expected Outcome:**
- R² improvement from **-0.245** to **0.60-0.80** (+0.85 to +1.05 gain)
- Predictions that actually follow sales trends
- Meaningful feature importance insights

**Next Step:**
Run `python mmm_script.py` and check if R² is now positive and > 0.6! 🚀
