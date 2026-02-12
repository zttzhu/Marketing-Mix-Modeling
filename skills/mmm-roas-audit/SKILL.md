---
name: mmm-roas-audit
description: Audit and sanity-check Marketing Mix Modeling outputs with a focus on ROAS plausibility, channel-level/year-level diagnostics, and run completeness. Use when users ask to validate MMM results, review ROAS by channel/year, detect implausible values, or troubleshoot suspicious optimization outcomes in this repository.
---

# MMM ROAS Audit

Follow this workflow when asked to validate MMM output quality.

## Required Context

- Read `AGENTS.md` first for repository-wide constraints, data contract, and validation checklist.
- Do not restate project policies from `AGENTS.md`; apply them directly.

## Workflow

1. Verify input contract quickly.
- Confirm `data.csv` exists in repo root.
- Confirm required columns are present: `wk_strt_dt`, `sales`.
- Confirm media prefix conventions still hold (`mdsp_`, `mdip_`).

2. Run baseline pipeline.
- Execute `python mmm_script.py` from repository root.
- Capture exceptions and identify first failing stage.

3. Validate expected outputs.
- Confirm console model metrics are printed.
- Confirm plots render without runtime exceptions in script execution.
- Confirm `media_channel_metrics_by_year.csv` is produced when relevant.

4. Audit ROAS reasonableness by media channel and year.
- Read `media_channel_metrics_by_year.csv`.
- Flag any channel-year ROAS above 20 as critical.
- Use practical benchmark ranges for quick triage:
  - linear tv: 1.5 to 4
  - digital: 2 to 5
  - streaming tv: 2 to 6
- Treat benchmark misses as investigation signals, not automatic failures.

5. Diagnose implausible ROAS.
- Check transformed features for saturation/adstock parameter extremes.
- Check multicollinearity and control leakage.
- Check spend/impression scaling and date alignment issues.
- Check whether optimization bounds are too loose.

6. Report findings.
- Prioritize concrete issues, then assumptions, then next fixes.
- Include file references and commands used.
- If hyperparameter settings changed, note expected runtime impact.

## Response Template

Use this structure in findings:

- Status: pass/fail for run completion
- Output check: present/missing artifacts
- ROAS anomalies: channel-year rows and severity
- Likely causes: ranked hypotheses
- Recommended edits: minimal, testable code changes
- Verification: commands rerun and observed outcome

## Guardrails

- Keep backward compatibility with existing CSV inputs unless user asks otherwise.
- Preserve prefix-based column detection logic.
- Prefer small function-level edits over duplicated blocks.
