"""Core analysis functions: load, validate, and compute stats from a CSV."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV file and validate it is non-empty.

    Raises SystemExit with a clear message on invalid input.
    """
    path = Path(path)
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"Error: could not parse CSV: {exc}", file=sys.stderr)
        sys.exit(1)

    if df.empty or len(df.columns) == 0:
        print("Error: CSV is empty or has no data rows.", file=sys.stderr)
        sys.exit(1)

    return df


def compute_summary(df: pd.DataFrame) -> dict:
    """Return shape, dtypes, and descriptive stats for numeric columns."""
    numeric_df = df.select_dtypes(include="number")
    desc = numeric_df.describe().to_dict() if not numeric_df.empty else {}

    return {
        "shape": {"rows": df.shape[0], "columns": df.shape[1]},
        "dtypes": df.dtypes.astype(str).to_dict(),
        "numeric_summary": desc,
    }


def compute_schema(df: pd.DataFrame) -> list[dict]:
    """Return per-column metadata: name, dtype, unique count, sample values."""
    schema = []
    for col in df.columns:
        series = df[col]
        sample = series.dropna().head(3).tolist()
        schema.append(
            {
                "column": col,
                "dtype": str(series.dtype),
                "unique_count": int(series.nunique(dropna=True)),
                "missing_count": int(series.isna().sum()),
                "sample_values": sample,
            }
        )
    return schema


def compute_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with columns: column, missing_count, missing_pct."""
    missing_count = df.isna().sum()
    missing_pct = (missing_count / len(df) * 100).round(2)
    result = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": missing_count.values,
            "missing_pct": missing_pct.values,
        }
    )
    return result.sort_values("missing_pct", ascending=False).reset_index(drop=True)
