# CLAUDE.md

## Project Overview

Python-based Marketing Mix Modeling (MMM) using Robyn-style methodology. The core workflow
lives in a single script (`mmm_script.py`) that handles data loading, media transformations
(adstock + saturation), Ridge regression fitting, diagnostics, and budget optimization.

## Environment

Python 3.10+ required.

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .\.venv\Scripts\Activate.ps1  # Windows PowerShell

pip install numpy pandas seaborn matplotlib scikit-learn scipy statsmodels pymc
```

No `requirements.txt` exists — install dependencies manually as above.

## Running the Project

```bash
python mmm_script.py
```

Plots are disabled by default (`ENABLE_PLOTS = False`). The script runs non-interactively and
prints metrics to the console. When complete, it produces `media_channel_metrics_by_year.csv`.

## Key Files

| File | Purpose |
|------|---------|
| `mmm_script.py` | Primary MMM workflow (~1,666 lines); all modeling logic lives here |
| `mmm_project.py` | Thin legacy entrypoint; directs users to `mmm_script.py` |
| `data.csv` | Primary dataset (required at repo root to run) |
| `AGENTS.md` | Repository-wide agent constraints and validation checklist |
| `README.md` | Full methodology, configuration reference, and usage examples |
| `skills/mmm-roas-audit/SKILL.md` | Custom `/mmm-roas-audit` skill for ROAS validation workflow |

## Data Contract

`data.csv` must exist in the repo root with these columns:

| Column | Description |
|--------|-------------|
| `wk_strt_dt` | Weekly start date (YYYY-MM-DD) |
| `sales` | Dependent variable (revenue) |
| `mdsp_*` | Media spend columns (one per channel) |
| `mdip_*` | Media impressions columns (one per channel) |

Optional control variable prefixes: `me_`, `hldy_`, `seas_`, `st_`, `mkrdn_`

**Never rename** `wk_strt_dt`, `sales`, `mdsp_*`, or `mdip_*` — column detection relies on
prefix matching throughout the script.

## Configuration Constants

All tunable parameters are at the top of `mmm_script.py`:

```python
TRAIN_TEST_SPLIT = 0.8           # 80/20 chronological split
OPTIMIZATION_MAXITER = 30        # Differential evolution iterations
OPTIMIZATION_POPSIZE = 10        # Population size for optimizer
ADSTOCK_TYPE = 'geometric'       # 'geometric' or 'weibull'
OPTIMIZE_HYPERPARAMS = True      # Set False for quick iterations
ENABLE_PLOTS = False             # Set True to render matplotlib charts
OPTIMIZER_VALIDATION_SPLIT = 0.2 # Inner validation split during optimization
COUNTERFACTUAL_REDUCTION = 1.0   # Attribution counterfactual level
MAX_REASONABLE_ROAS = 20.0       # ROAS cap in reported outputs
```

## Coding Guidelines

- Keep configuration constants at the top of `mmm_script.py`.
- Prefer adding/refining functions over duplicating logic.
- Preserve prefix-based column detection (`mdsp_`, `mdip_`, `me_`, etc.).
- Keep changes backward-compatible with existing CSV inputs unless explicitly asked.
- If you change behavior, update `README.md` examples and parameter descriptions.
- Do not add external services, databases, or new runtime dependencies without explicit approval.

## Validation Checklist (run after every code change)

1. `python mmm_script.py` completes without exceptions.
2. Model metrics print to console.
3. `media_channel_metrics_by_year.csv` is produced.
4. Date parsing and media/control column auto-detection still work.
5. ROAS values per channel per year fall within expected ranges:
   - Linear TV: $1.5–$4
   - Digital: $2–$5
   - Streaming TV: $2–$6
   - Flag any ROAS above 20 as a critical anomaly.
6. Test R² is not negative and does not deviate significantly from train R².

## Custom Skills

### `/mmm-roas-audit`

Triggers a structured ROAS audit workflow defined in `skills/mmm-roas-audit/SKILL.md`.
Use when asked to validate MMM outputs, review channel/year ROAS, or troubleshoot
suspicious optimization results.

The skill:
1. Verifies the data contract
2. Runs the pipeline and captures exceptions
3. Audits `media_channel_metrics_by_year.csv` for ROAS plausibility
4. Diagnoses implausible values (adstock/saturation extremes, multicollinearity, scaling issues)
5. Reports findings with ranked hypotheses and minimal recommended fixes

## Known Quirks

- `mmm_script.py` uses `# %%` cell markers (notebook-style) but runs fine as a plain script.
- `differential_evolution` optimization can be slow — increase `OPTIMIZATION_MAXITER` for
  quality at the cost of runtime; lower it for quick iteration.
- PyMC is listed as a dependency for Bayesian modeling (not yet fully implemented in main script).
- `media_channel_metrics_by_year.csv` is in `.gitignore` — it is a generated artifact.
- `run.log` is also gitignored.
