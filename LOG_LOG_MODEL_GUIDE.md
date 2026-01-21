# Log-Log Model & Impressions Implementation Guide

## Overview

The MMM script now supports:
1. **Log-Log Transformation**: `log(sales) = β₀ + β₁·log(transformed_media) + ...`
2. **Impressions Instead of Spend**: Uses `mdip_*` columns (media impressions) instead of `mdsp_*` (media spend)

---

## Configuration

### Enable/Disable Features

In the **CONFIGURATION** section (lines 16-40):

```python
USE_LOG_LOG_MODEL = True   # True = log-log, False = linear
USE_IMPRESSIONS = True     # True = impressions (mdip_*), False = spend (mdsp_*)
```

### Four Possible Combinations

| Configuration | Model Type | Media Variable | Use Case |
|--------------|------------|----------------|----------|
| `USE_LOG_LOG_MODEL=True`<br>`USE_IMPRESSIONS=True` | Log-Log | Impressions | **✅ Recommended**: Elasticity analysis with direct exposure metrics |
| `USE_LOG_LOG_MODEL=True`<br>`USE_IMPRESSIONS=False` | Log-Log | Spend | Elasticity analysis with budget data |
| `USE_LOG_LOG_MODEL=False`<br>`USE_IMPRESSIONS=True` | Linear | Impressions | Direct contribution analysis with impressions |
| `USE_LOG_LOG_MODEL=False`<br>`USE_IMPRESSIONS=False` | Linear | Spend | Traditional ROI/ROAS analysis |

---

## Log-Log Model: How It Works

### Transformation Pipeline

```
1. Raw Impressions (mdip_*) or Spend (mdsp_*)
   ↓
2. Adstock Transformation (carryover effects)
   ↓
3. Saturation Transformation (diminishing returns)
   ↓
4. Log Transformation: log(1 + transformed_value)
   ↓
5. Model Fitting: log(sales) ~ log(transformed_media)
   ↓
6. Predictions in Log Space
   ↓
7. Transform Back: exp(prediction) - 1 = predicted_sales
```

### Mathematical Form

#### Log-Log Model (USE_LOG_LOG_MODEL = True):
```
log(sales) = β₀ + β₁·log(f(impressions₁)) + β₂·log(f(impressions₂)) + ... + controls

where f(x) = saturation(adstock(x))
```

#### Linear Model (USE_LOG_LOG_MODEL = False):
```
sales = β₀ + β₁·f(impressions₁) + β₂·f(impressions₂) + ... + controls
```

---

## Interpretation Guide

### Log-Log Model Coefficients (Elasticities)

When `USE_LOG_LOG_MODEL = True`:

**Coefficient = 0.5 means:**
- **1% increase** in transformed impressions → **0.5% increase** in sales
- This is called **elasticity**

**Example:**
```python
# Coefficient for TV impressions: 0.35
# Current TV impressions: 10M (transformed to 5M after adstock/saturation)
# Current sales: $100M

# If TV impressions increase by 10%:
# Sales increase ≈ 10% × 0.35 = 3.5%
# New sales ≈ $100M × 1.035 = $103.5M
```

**Interpretation Table:**

| Coefficient (β) | Interpretation | Sales Response |
|----------------|----------------|----------------|
| β > 1.0 | **Elastic**: Sales very responsive | 1% media ↑ → >1% sales ↑ |
| β = 1.0 | **Unit elastic**: Proportional response | 1% media ↑ → 1% sales ↑ |
| 0 < β < 1.0 | **Inelastic**: Sales less responsive | 1% media ↑ → <1% sales ↑ |
| β ≈ 0 | **No effect**: Media not driving sales | No impact |

### Linear Model Coefficients (Absolute Contributions)

When `USE_LOG_LOG_MODEL = False`:

**Coefficient = 1000 means:**
- **1 unit increase** in transformed impressions → **$1000 increase** in sales
- This is **absolute contribution**

**Example:**
```python
# Coefficient for TV: 1000
# Current TV transformed value: 5M

# If transformed TV increases by 1 unit (after adstock/saturation):
# Sales increase by $1000
```

---

## Why Use Log-Log Model?

### ✅ Advantages

1. **Elasticity Interpretation**
   - Coefficients are scale-independent
   - Easy to compare across channels (e.g., "TV is 2x more elastic than social")

2. **Multiplicative Relationships**
   - Captures percentage changes naturally
   - Better for modeling real-world marketing effects

3. **Handles Wide Range of Values**
   - Compresses large values, expands small values
   - More stable with outliers

4. **Diminishing Returns Built-In**
   - Log transformation naturally models decreasing marginal returns
   - Complements saturation effects

5. **Better Model Fit (Often)**
   - Sales typically have multiplicative relationships with media
   - More homoscedastic residuals (constant variance)

### ⚠️ Disadvantages

1. **Cannot Handle Zeros Directly**
   - Uses `log1p(x) = log(1+x)` to handle zeros
   - Small bias when values are close to zero

2. **Less Intuitive for Budget Allocation**
   - Elasticities don't directly translate to "$1 spent = $X sales"
   - Need to convert elasticities to marginal ROI for budget optimization

3. **Interpretation Requires Care**
   - Non-linear transformation makes simple contribution decomposition harder
   - Need to think in percentages, not absolute values

---

## Why Use Impressions Instead of Spend?

### ✅ Reasons to Use Impressions (mdip_*)

1. **Direct Exposure Metric**
   - Impressions = actual number of times ad was seen
   - More direct measure of marketing "dose"

2. **Separates Volume from Price**
   - Spend = impressions × CPM (cost per thousand impressions)
   - Impressions isolates the volume effect
   - Useful when CPM varies significantly

3. **Better for Cross-Channel Comparison**
   - All channels measured in same unit (impressions)
   - Easier to compare reach across channels

4. **Media Mix Optimization**
   - Optimize based on reach efficiency
   - Budget allocation can consider CPM separately

### ✅ Reasons to Use Spend (mdsp_*)

1. **Direct ROI/ROAS Calculation**
   - Spend directly ties to budget and profitability
   - Easier to calculate return on ad spend

2. **Budget-Constrained Optimization**
   - Optimization naturally aligns with budget limits
   - CFO/Finance teams think in spend, not impressions

3. **Simpler Communication**
   - Executives understand "spend $1M → get $5M sales"
   - Less explanation needed for stakeholders

---

## Model Comparison Example

### Scenario
- Channel: TV
- Current impressions: 10M
- Current spend: $500K
- Current sales: $100M

### Log-Log with Impressions
```python
USE_LOG_LOG_MODEL = True
USE_IMPRESSIONS = True

# Model: log(sales) ~ log(transformed_impressions)
# Coefficient β_TV = 0.4

# Interpretation:
# 10% increase in TV impressions → 4% increase in sales
# New impressions: 11M
# Expected sales: $100M × 1.04 = $104M
# Incremental sales: $4M
```

### Linear with Spend
```python
USE_LOG_LOG_MODEL = False
USE_IMPRESSIONS = False

# Model: sales ~ transformed_spend
# Coefficient β_TV = 8.0

# Interpretation:
# 1 unit increase in transformed spend → $8 increase in sales
# If spend increases by $50K (10%), transformed spend might increase by X units
# Expected sales increase: X × $8
```

---

## Running the Model

### Step 1: Set Configuration

```python
# In mmm_script.py, lines 21-22
USE_LOG_LOG_MODEL = True   # Enable log-log transformation
USE_IMPRESSIONS = True     # Use impressions instead of spend
```

### Step 2: Run the Script

```bash
python mmm_script.py
```

### Step 3: Check Output

Look for this in the console:

```
================================================================================
ROBYN-STYLE MMM WORKFLOW
================================================================================
Model Type: Log-Log
Media Variable: Impressions (mdip_*)

[1/8] Preparing data...
  - Applying log transformation to sales and media variables...
  - Media channels: ['dm', 'inst', 'nsp', 'auddig', 'audtr', 'vidtr', ...]
  - Baseline variables: 43
  - Training data points: 160
  - Test data points: 41
```

### Step 4: Interpret Results

```
Test Set Metrics:
  - R²: 0.7234      # How well model fits
  - RMSE: 15234.23  # Prediction error in $ (original scale)
  - MAPE: 12.45%    # Average % error
```

**Model Coefficients (Top Channels):**
```
Channel: vidtr    Coefficient: 0.45   # 1% ↑ impressions → 0.45% ↑ sales
Channel: audtr    Coefficient: 0.32   # 1% ↑ impressions → 0.32% ↑ sales
Channel: on       Coefficient: 0.28   # 1% ↑ impressions → 0.28% ↑ sales
```

---

## Comparing Models

### Quick Test: Run Both Models

**Test 1: Log-Log with Impressions**
```python
USE_LOG_LOG_MODEL = True
USE_IMPRESSIONS = True
# Run and record: R², RMSE, MAPE
```

**Test 2: Linear with Spend**
```python
USE_LOG_LOG_MODEL = False
USE_IMPRESSIONS = False
# Run and record: R², RMSE, MAPE
```

**Compare:**
- Which has higher R²? (better fit)
- Which has lower MAPE? (better % accuracy)
- Which residuals look more random? (check residual plots)

### Decision Guide

Choose **Log-Log with Impressions** if:
- ✅ You want elasticity-based insights
- ✅ Your team thinks in terms of reach/frequency
- ✅ Sales have multiplicative relationships with media
- ✅ You need scale-independent metrics

Choose **Linear with Spend** if:
- ✅ You need direct ROI/ROAS calculations
- ✅ Budget allocation is the primary goal
- ✅ Stakeholders prefer "dollar in → dollar out" metrics
- ✅ Linear relationships fit your data well

---

## Advanced: Custom Combinations

### Use Both in Parallel

Run two models and compare:

```python
# Model 1: Log-Log Impressions
results_log_imp = run_robyn_mmm(
    data, use_log_transform=True, use_impressions=True
)

# Model 2: Linear Spend
results_lin_spend = run_robyn_mmm(
    data, use_log_transform=False, use_impressions=False
)

# Compare R²
print(f"Log-Log R²: {results_log_imp['metrics']['test']['r2']:.4f}")
print(f"Linear R²: {results_lin_spend['metrics']['test']['r2']:.4f}")
```

### Hybrid Approach

1. **Model with impressions** (elasticity insights)
2. **Convert elasticities to spend-based ROI** using CPM data
3. **Optimize budget** considering both efficiency and cost

---

## Troubleshooting

### Issue: R² is negative or very low

**Solution:**
- Try switching between log-log and linear
- Check if data has zeros/negative values (log can't handle negatives)
- Verify impressions/spend data quality

### Issue: Coefficients are huge or tiny

**For log-log:**
- Coefficients should be between -2 and 2 typically
- If much larger, check data scaling

**For linear:**
- Coefficients depend on scale of data
- Use `normalize=True` in model (already default)

### Issue: Predictions are all similar

**Solution:**
- Check feature variance after transformation
- Increase regularization (`ridge_alpha`)
- Try different adstock/saturation parameters

---

## References

### Theory
- Log-log models: [Econometrics textbooks, elasticity chapters]
- MMM with elasticity: Meta Robyn documentation
- Adstock transformations: Clarke (1976), "Econometric Measurement of Duration of Advertising Effect on Sales"

### Practical Guides
- `MODEL_IMPROVEMENTS_SUMMARY.md` - Why baseline model failed
- `BEFORE_VS_AFTER_COMPARISON.md` - Model comparison examples
- `README.md` - Full project documentation

---

## Summary: Quick Start

**Recommended Setup:**
```python
USE_LOG_LOG_MODEL = True   # Elasticity interpretation
USE_IMPRESSIONS = True     # Direct exposure metric
OPTIMIZE_HYPERPARAMS = True  # Find best adstock/saturation
```

**Expected Output:**
- Coefficients as elasticities (0-1 range typically)
- Predictions in original sales scale (auto-converted)
- R² typically 0.6-0.85 for good models
- Interpretable: "1% more TV impressions → 0.4% more sales"

**Next Steps:**
1. Run model and check R²/MAPE
2. Review coefficient magnitudes (elasticities)
3. Compare with linear model if needed
4. Use for budget optimization and scenario planning
