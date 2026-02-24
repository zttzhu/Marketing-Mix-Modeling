"""Report writers: summary.md, schema.json, missingness.csv."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def write_summary_md(summary: dict, outdir: Path) -> Path:
    """Write descriptive stats and shape info to summary.md."""
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / "summary.md"

    lines = [
        "# Dataset Summary\n",
        f"**Rows:** {summary['shape']['rows']}  ",
        f"**Columns:** {summary['shape']['columns']}\n",
        "## Column Data Types\n",
        "| Column | Type |",
        "|--------|------|",
    ]
    for col, dtype in summary["dtypes"].items():
        lines.append(f"| `{col}` | {dtype} |")

    numeric_summary = summary.get("numeric_summary", {})
    if numeric_summary:
        lines += [
            "\n## Numeric Column Statistics\n",
        ]
        # Collect all stats rows
        stats_keys = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
        columns = list(numeric_summary.keys())
        lines.append("| Stat | " + " | ".join(f"`{c}`" for c in columns) + " |")
        lines.append("|------|" + "|".join("---" for _ in columns) + "|")
        for stat in stats_keys:
            row_vals = []
            for col in columns:
                val = numeric_summary[col].get(stat, "")
                row_vals.append(f"{val:.4g}" if isinstance(val, float) else str(val))
            lines.append(f"| {stat} | " + " | ".join(row_vals) + " |")
    else:
        lines.append("\n_No numeric columns found._")

    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def write_schema_json(schema: list[dict], outdir: Path) -> Path:
    """Write column schema metadata to schema.json."""
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / "schema.json"
    out_path.write_text(json.dumps(schema, indent=2))
    return out_path


def write_missingness_csv(missingness: pd.DataFrame, outdir: Path) -> Path:
    """Write missingness audit to missingness.csv."""
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / "missingness.csv"
    missingness.to_csv(out_path, index=False)
    return out_path
