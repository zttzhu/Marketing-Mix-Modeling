# AGENTS.md

## Scope
These instructions apply to the whole repository.

## Project Summary
This repo contains a Python Marketing Mix Modeling (MMM) workflow centered on `mmm_script.py`.
The script includes data loading, EDA, Robyn-style transformations (adstock + saturation), model fitting, diagnostics, and budget allocation.

## Environment
Use Python 3.10+.

Suggested setup (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install numpy pandas seaborn matplotlib scikit-learn scipy statsmodels
```

## Runbook
Primary run command:
```powershell
python mmm_script.py
```

The script expects `data.csv` in the repository root.

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

## Validation Checklist
After code changes:
1. Run `python mmm_script.py` and verify it completes without exceptions.
2. Confirm key outputs are produced (metrics in console, plots, and `media_channel_metrics_by_year.csv` when relevant).
3. Re-check date parsing and media/control column auto-detection.
4. If optimization settings change, note runtime impact in documentation.
5. Check ROAS for each Media Channel for each year. Make sure these ROAS make sense and no crazy high ROAS (>$20) showing up. For example, linear tv has roas between $1.5 to $4, digital $2-$5, streaming tv $2-$6.
6. For the validation metrics, the testing dataset R square should be deviate too much from training dataset R square. 
## Known Practical Notes
- `mmm_script.py` is notebook-style (`# %%`) but can be run as a script.
- Plotting is interactive; changes that affect plotting should still work in non-notebook execution.
- Hyperparameter optimization (`differential_evolution`) can be slow; keep defaults practical.

## File Map
- `mmm_script.py`: primary MMM workflow and analysis code
- `mmm_project.py`: small project/helper entry file
- `README.md`: usage and methodology documentation
- `MMM_SCRIPT_REVIEW.md`: review notes and recommendations
- `data.csv`, `mmm_data.csv`, `Sample Media Spend Data.csv`: local datasets

## Out of Scope
- Do not add external services or database dependencies unless explicitly requested.
- Do not rename core input columns (`wk_strt_dt`, `sales`, `mdsp_*`, `mdip_*`) without migration guidance.
