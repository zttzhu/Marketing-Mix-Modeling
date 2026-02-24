"""CLI entrypoint for dataset-insights."""

from __future__ import annotations

from pathlib import Path

import click

from .analyze import compute_missingness, compute_schema, compute_summary, load_csv
from .plots import plot_correlation_heatmap, plot_distribution_histogram, plot_missingness_bar
from .reports import write_missingness_csv, write_schema_json, write_summary_md


@click.group()
@click.version_option(package_name="dataset-insights")
def main():
    """dataset-insights: instant orientation for any CSV dataset."""


@main.command()
@click.argument("csv_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--outdir",
    default="reports",
    show_default=True,
    help="Directory to write output files.",
    type=click.Path(),
)
def analyze(csv_path: str, outdir: str):
    """Analyze a CSV file and write reports + diagnostic plots to OUTDIR."""
    out = Path(outdir)

    click.echo(f"Loading {csv_path} ...")
    df = load_csv(csv_path)
    click.echo(f"  {df.shape[0]:,} rows x {df.shape[1]} columns")

    # --- Compute ---
    summary = compute_summary(df)
    schema = compute_schema(df)
    missingness = compute_missingness(df)

    # --- Write reports ---
    click.echo("\nWriting reports ...")
    p1 = write_summary_md(summary, out)
    click.echo(f"  {p1}")
    p2 = write_schema_json(schema, out)
    click.echo(f"  {p2}")
    p3 = write_missingness_csv(missingness, out)
    click.echo(f"  {p3}")

    # --- Generate plots ---
    click.echo("\nGenerating plots ...")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        click.echo("  Warning: no numeric columns found — skipping histogram and heatmap.")

    p4 = plot_distribution_histogram(df, out)
    if p4:
        click.echo(f"  {p4}")
    else:
        click.echo("  Skipped: distribution_histogram.png (no numeric columns)")

    p5 = plot_correlation_heatmap(df, out)
    if p5:
        click.echo(f"  {p5}")
    else:
        click.echo("  Skipped: correlation_heatmap.png (fewer than 2 numeric columns)")

    p6 = plot_missingness_bar(df, out)
    click.echo(f"  {p6}")

    # --- Console summary ---
    total_missing = int(missingness["missing_count"].sum())
    cols_with_missing = int((missingness["missing_count"] > 0).sum())
    click.echo(
        f"\nDone. {cols_with_missing}/{df.shape[1]} columns have missing data "
        f"({total_missing:,} total missing values)."
    )
    click.echo(f"Output written to: {out.resolve()}/")
