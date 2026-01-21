# Quick Start Checklist: Testing the Improved Model

## ✅ Pre-Flight Checklist

- [ ] **Data file exists:** `data.csv` is in the project directory
- [ ] **Python environment ready:** All packages installed (sklearn, pandas, numpy, matplotlib, statsmodels)
- [ ] **Script updated:** `mmm_script.py` has the improved baseline model code

---

## 🚀 Running the Improved Model

### Step 1: Execute the Script
```bash
python mmm_script.py
```

### Step 2: Find the Results
Look for this section in the console output:
```
================================================================================
IMPROVED BASELINE OLS MMM
================================================================================
Media channels (13): ['dm', 'inst', 'nsp', 'auddig', ...]
Control variables (43): 43 features
Using adstock transformation with decay=0.5
Using Ridge regression with regularization
```

### Step 3: Check the Metrics
Look for:
```
--------------------------------------------------------------------------------
IMPROVED BASELINE MODEL RESULTS
--------------------------------------------------------------------------------

Training Set Metrics:
  R²:   [SHOULD BE 0.70-0.90]
  RMSE: [Lower is better]
  MAE:  [Lower is better]
  MAPE: [Should be < 20%]

Test Set Metrics:
  R²:   [SHOULD BE 0.60-0.80] ← KEY METRIC!
  RMSE: [Lower is better]
  MAE:  [Lower is better]
  MAPE: [Should be < 25%]
```

---

## 📊 Interpreting Results

### ✅ SUCCESS - Model is Working Well
- [x] Test R² > 0.60
- [x] Train R² and Test R² are close (within 0.10-0.15)
- [x] MAPE < 25%
- [x] Chart shows predicted line following actual trends
- [x] Residual plot shows random scatter (no clear pattern)

**Action:** Proceed to use the model! Move on to budget optimization and ROI analysis.

---

### ⚠️ DECENT - Model is OK but Could Be Better
- [x] Test R² between 0.40-0.60
- [x] Some alignment in charts but not perfect
- [x] MAPE between 25-40%

**Action:** Try tuning hyperparameters (see Tuning Guide below)

---

### 🔴 FAILURE - Model Still Has Issues
- [ ] Test R² < 0.40 or negative
- [ ] Predictions are flat or don't follow trends
- [ ] MAPE > 50%
- [ ] Huge gap between Train R² and Test R² (e.g., 0.9 vs 0.3)

**Action:** See Troubleshooting Guide below

---

## 🔧 Quick Tuning Guide

### If R² = 0.40-0.60 (Decent but Can Improve)

#### Option 1: Tune Adstock Decay
```python
# In mmm_script.py, around line 1285, change:
adstock_decay = 0.5  # Try: 0.3, 0.5, 0.7, 0.9

# Test each value, pick the one with highest test R²
```

**How to choose:**
- **Lower decay (0.3):** Media effects fade quickly (e.g., social media, display ads)
- **Higher decay (0.7-0.9):** Media effects last long (e.g., TV, brand campaigns)

#### Option 2: Tune Ridge Alpha (Regularization)
```python
# In mmm_script.py, around line 1307, change:
baseline_ridge_model = Ridge(alpha=100.0)  # Try: 10, 50, 100, 500, 1000

# Test each value, pick the one with highest test R²
```

**How to choose:**
- **Lower alpha (10-50):** Less regularization, model can fit more complex patterns
- **Higher alpha (500-1000):** More regularization, prevents overfitting

#### Option 3: Use RidgeCV (Auto-Tune Alpha)
```python
# Replace the Ridge line with:
from sklearn.linear_model import RidgeCV
baseline_ridge_model = RidgeCV(alphas=[10, 50, 100, 200, 500, 1000])
baseline_ridge_model.fit(X_train_baseline_scaled, y_train_baseline)
print(f"Best alpha: {baseline_ridge_model.alpha_}")
```

---

## 🆘 Troubleshooting Guide

### Problem 1: R² is Still Negative
**Possible Causes:**
1. Train/test split issue (future data leaking into training)
2. Data mismatch (wrong columns, missing values)
3. Scaling applied incorrectly

**Solutions:**
```python
# Check 1: Verify split is chronological
print(f"Train end date: {train_baseline['wk_strt_dt'].max()}")
print(f"Test start date: {test_baseline['wk_strt_dt'].min()}")
# Test should start AFTER train ends

# Check 2: Look for missing values
print(f"Missing values in train: {X_train_baseline.isna().sum().sum()}")
print(f"Missing values in test: {X_test_baseline.isna().sum().sum()}")
# Should be 0

# Check 3: Check for infinite values
print(f"Infinite values in train: {np.isinf(X_train_baseline).sum().sum()}")
print(f"Infinite values in test: {np.isinf(X_test_baseline).sum().sum()}")
# Should be 0
```

### Problem 2: Train R² = 0.95, Test R² = 0.30 (Overfitting)
**Solution:**
```python
# Increase regularization
baseline_ridge_model = Ridge(alpha=1000.0)  # Increase from 100

# OR reduce features (use top N most important)
# Re-run with fewer features
```

### Problem 3: Predictions Are All Very Similar (No Variance)
**Possible Causes:**
1. Features have no variance after scaling
2. Adstock not working correctly

**Solutions:**
```python
# Check feature variance
print(f"Feature variance: {X_train_baseline_scaled.var(axis=0)}")
# Should see mix of values around 1.0

# Check adstock is working
print(f"Media before adstock: {mmm_data[mdsp_col[0]].head()}")
print(f"Media after adstock: {model_data[f'{mdsp_col[0]}_adstock'].head()}")
# Values should be different!
```

### Problem 4: Feature Importance Doesn't Make Sense
**Example:** Top feature is "seas_week_43" with huge coefficient

**Possible Causes:**
1. Not enough regularization
2. Multicollinearity between features

**Solutions:**
```python
# Increase regularization
baseline_ridge_model = Ridge(alpha=500.0)

# OR check correlation between top features
top_features = feature_importance.head(10)['Feature'].tolist()
corr_matrix = train_baseline[top_features].corr()
print(corr_matrix)
# If many correlations > 0.8, you have multicollinearity
```

---

## 📈 What to Do After Success

Once you have Test R² > 0.60:

### 1. **Analyze Feature Importance**
```
Top 10 Most Important Features:
  Feature                 Coefficient
  mdsp_vidtr_adstock      45000.23     ← Video TV is key driver!
  hldy_Black Friday       123000.50    ← Black Friday huge impact
  seas_week_48            89000.12     ← Week 48 (late Nov) matters
  ...
```

**Questions to ask:**
- Do top features make business sense?
- Are expensive media channels showing ROI?
- Which holidays drive most sales?

### 2. **Compare with Robyn MMM**
Look at the "MODEL COMPARISON" section:
```
Metric      Robyn MMM (Test)    Baseline OLS (Test)    Improvement
R²          0.75                0.65                    -13.3%
RMSE        32,145              38,234                  +18.9%
...
```

**If Baseline > Robyn:** Great! Simpler model works better.
**If Robyn > Baseline:** Robyn's adstock/saturation optimization adds value.

### 3. **Budget Optimization**
Use the model to find optimal budget allocation:
- Which channels have highest ROI?
- Where should you increase/decrease spend?

### 4. **Document Findings**
Create a summary:
- Model performance metrics
- Top sales drivers
- Recommended budget changes
- Expected sales impact

---

## 🎯 Success Metrics Cheat Sheet

| Metric | Excellent | Good | Fair | Poor |
|--------|-----------|------|------|------|
| **R²** | > 0.75 | 0.60-0.75 | 0.40-0.60 | < 0.40 |
| **MAPE** | < 15% | 15-25% | 25-40% | > 40% |
| **RMSE** | < 20M | 20-40M | 40-60M | > 60M |
| **Train-Test Gap** | < 0.10 | 0.10-0.15 | 0.15-0.25 | > 0.25 |

---

## 📝 Quick Reference Commands

```bash
# Run the model
python mmm_script.py

# If you get import errors:
pip install scikit-learn pandas numpy matplotlib statsmodels seaborn scipy

# Check Python version (should be 3.7+):
python --version

# Run in Jupyter (if available):
jupyter notebook mmm_script.py
```

---

## 📞 Need Help?

If still stuck after following this checklist:

1. **Check data quality:**
   - Look at `data.csv` for missing values, outliers
   - Verify date range makes sense

2. **Review console output:**
   - Any error messages?
   - Any warnings about convergence?

3. **Compare with documentation:**
   - Read `MODEL_IMPROVEMENTS_SUMMARY.md`
   - Review `BEFORE_VS_AFTER_COMPARISON.md`

4. **Test with simpler data:**
   - Try with just 1 media channel first
   - Build up complexity gradually

---

**Good luck! 🚀**

Remember: R² > 0.60 is the goal. Anything above that is great success!
