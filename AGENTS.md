# AGENTS.md

## Scope
These instructions apply to the whole repository.

## Project Summary
This repo contains a Python Marketing Mix Modeling (MMM) workflow centered on `mmm_script.py`.
The script includes data loading, EDA, Robyn-style transformations (adstock + saturation), model fitting, diagnostics, and budget allocation. The workflow now includes steps for Bayesian modeling with iterative prior refinement.

## Environment
Use Python 3.10+.

Suggested setup (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install numpy pandas seaborn matplotlib scikit-learn scipy statsmodels pymc
```

## Runbook
### Baseline Model
1. Ensure `data.csv` is in the repository root.
2. Run the baseline model:
   ```powershell
   python mmm_script.py
   ```
3. Analyze outputs:
   - Check metrics in the console.
   - Review plots and `media_channel_metrics_by_year.csv`.
   - Validate ROAS for each media channel (see Validation Checklist).

### Bayesian Model
1. Set up priors based on baseline results:
   - Use realistic ranges informed by baseline outputs.
   - Example: ROAS priors for linear TV ($1.5–$4), digital ($2–$5), streaming TV ($2–$6).
2. Implement Bayesian modeling:
   - Use PyMC or another Bayesian library.
   - Define priors for key parameters (e.g., adstock, saturation).
3. Refine priors iteratively:
   - Run the Bayesian model.
   - Adjust priors based on posterior distributions and diagnostics.
4. Validate results:
   - Ensure ROAS values are realistic.
   - Compare Bayesian outputs to baseline metrics.

## Data Contract
Required columns:
- `wk_strt_dt`: weekly date (parseable as datetime)
- `sales`: dependent variable

Media columns:
- Spend: prefix `mdsp_`
- Impressions: prefix `mdip_`

Optional control variables are identified by prefixes:
- `me_`, `hldy_`, `seas_`, `st_`, `mkrdn_`

## Coding Guidelines
- Preserve current column-prefix conventions; many selections depend on prefix matching.
- Keep model configuration constants near the top of `mmm_script.py`.
- Prefer adding/refining functions over duplicating logic in new cells/blocks.
- Keep changes backward-compatible with existing CSV inputs unless explicitly requested.
- If you change behavior, update `README.md` examples and parameter descriptions.
- For Bayesian modeling, ensure compatibility with the existing workflow.

## Validation Checklist
After code changes:
1. Run `python mmm_script.py` and verify it completes without exceptions.
2. Confirm key outputs are produced (metrics in console, plots, and `media_channel_metrics_by_year.csv` when relevant).
3. Re-check date parsing and media/control column auto-detection.
4. If optimization settings change, note runtime impact in documentation.
5. Check ROAS for each Media Channel for each year:
   - Linear TV: $1.5–$4
   - Digital: $2–$5
   - Streaming TV: $2–$6
   - Investigate anomalies (e.g., 0 or excessively high ROAS).
6. For the validation metrics:
   - Testing dataset R² should not deviate significantly from training dataset R².
   - R² cannot be negative.
   - Include additional metrics (e.g., MAPE, RMSE) for diagnostics.

## Known Practical Notes
- `mmm_script.py` is notebook-style (`# %%`) but can be run as a script.
- Plotting is interactive; changes that affect plotting should still work in non-notebook execution.
- Hyperparameter optimization (`differential_evolution`) can be slow; keep defaults practical.
- Bayesian modeling may require longer runtimes; document runtime impacts.

## File Map
- `mmm_script.py`: primary MMM workflow and analysis code
- `mmm_project.py`: small project/helper entry file
- `README.md`: usage and methodology documentation
- `MMM_SCRIPT_REVIEW.md`: review notes and recommendations
- `data.csv`, `mmm_data.csv`, `Sample Media Spend Data.csv`: local datasets

## Out of Scope
- Do not add external services or database dependencies unless explicitly requested.
- Do not rename core input columns (`wk_strt_dt`, `sales`, `mdsp_*`, `mdip_*`) without migration guidance.
