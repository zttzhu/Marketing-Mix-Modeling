"""Unit tests for analyze.py functions."""

from __future__ import annotations

import pandas as pd
import pytest

from dataset_insights.analyze import compute_missingness, compute_schema, compute_summary, load_csv


def test_load_csv_valid(sample_csv):
    df = load_csv(sample_csv)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (10, 5)


def test_load_csv_empty_exits(empty_csv):
    with pytest.raises(SystemExit) as exc_info:
        load_csv(empty_csv)
    assert exc_info.value.code != 0


def test_load_csv_missing_file(tmp_path):
    with pytest.raises(SystemExit) as exc_info:
        load_csv(tmp_path / "nonexistent.csv")
    assert exc_info.value.code != 0


def test_compute_summary_shape(sample_csv):
    df = load_csv(sample_csv)
    summary = compute_summary(df)
    assert summary["shape"]["rows"] == 10
    assert summary["shape"]["columns"] == 5


def test_compute_schema_columns(sample_csv):
    df = load_csv(sample_csv)
    schema = compute_schema(df)
    cols = [entry["column"] for entry in schema]
    assert set(cols) == {"id", "age", "salary", "department", "score"}


def test_compute_missingness_values(sample_csv):
    """Missingness counts must match the known values in the fixture CSV."""
    df = load_csv(sample_csv)
    miss = compute_missingness(df)
    miss_dict = dict(zip(miss["column"], miss["missing_count"]))

    # From the fixture:
    # age: row 3 is missing -> 1
    # score: rows 2, 6, 8 are missing -> 3
    # department: row 4 is missing -> 1
    # salary, id: fully populated -> 0
    assert miss_dict["age"] == 1
    assert miss_dict["score"] == 3
    assert miss_dict["department"] == 1
    assert miss_dict["salary"] == 0
    assert miss_dict["id"] == 0


def test_compute_missingness_pct(sample_csv):
    df = load_csv(sample_csv)
    miss = compute_missingness(df)
    miss_dict = dict(zip(miss["column"], miss["missing_pct"]))
    # score has 3/10 = 30%
    assert abs(miss_dict["score"] - 30.0) < 0.01


def test_compute_missingness_sorted(sample_csv):
    df = load_csv(sample_csv)
    miss = compute_missingness(df)
    pcts = miss["missing_pct"].tolist()
    assert pcts == sorted(pcts, reverse=True)
