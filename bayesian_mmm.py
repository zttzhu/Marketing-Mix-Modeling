"""
Direct PyMC Bayesian Marketing Mix Model.

This script keeps the existing Robyn-style workflow intact and adds a
transparent Bayesian path focused on posterior diagnostics and ROAS auditing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import arviz as az
    import pymc as pm
    import pytensor.tensor as pt
except ImportError as exc:  # pragma: no cover - exercised only in missing envs
    raise SystemExit(
        "Missing Bayesian dependencies. Install them with:\n"
        "  pip install -r requirements.txt\n"
        f"Original import error: {exc}"
    ) from exc


DATE_COL = "wk_strt_dt"
TARGET_COL = "sales"
MEDIA_PREFIX = "mdsp_"
DEFAULT_TRAIN_TEST_SPLIT = 0.8
DEFAULT_MAX_LAG = 8

BENCHMARK_GROUPS = {
    "vidtr": ("Linear TV", 1.5, 4.0),
    "sem": ("Digital", 2.0, 5.0),
    "so": ("Digital", 2.0, 5.0),
    "on": ("Digital", 2.0, 5.0),
    "inst": ("Digital", 2.0, 5.0),
    "auddig": ("Digital", 2.0, 5.0),
    "viddig": ("Digital", 2.0, 5.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a direct PyMC Bayesian MMM with posterior ROAS audit."
    )
    parser.add_argument("--data", default="data.csv", help="Input CSV path.")
    parser.add_argument(
        "--output-dir",
        default="outputs/bayesian",
        help="Directory for generated Bayesian MMM outputs.",
    )
    parser.add_argument("--draws", type=int, default=1000, help="Posterior draws.")
    parser.add_argument("--tune", type=int, default=1000, help="Tuning draws.")
    parser.add_argument("--chains", type=int, default=4, help="MCMC chains.")
    parser.add_argument("--cores", type=int, default=1, help="Parallel sampler cores.")
    parser.add_argument(
        "--target-accept",
        type=float,
        default=0.9,
        help="NUTS target_accept.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--train-test-split",
        type=float,
        default=DEFAULT_TRAIN_TEST_SPLIT,
        help="Chronological train fraction.",
    )
    parser.add_argument(
        "--max-lag",
        type=int,
        default=DEFAULT_MAX_LAG,
        help="Maximum weekly adstock lag.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a short sampler run for smoke testing.",
    )
    parser.add_argument(
        "--save-trace",
        action="store_true",
        help="Save the ArviZ InferenceData NetCDF trace.",
    )
    return parser.parse_args()


def load_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = {DATE_COL, TARGET_COL}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = data.copy()
    data[DATE_COL] = pd.to_datetime(data[DATE_COL])
    data = data.sort_values(DATE_COL).reset_index(drop=True)
    if data[TARGET_COL].le(0).any():
        raise ValueError("Bayesian MMM requires positive sales values.")
    return data


def detect_media_columns(data: pd.DataFrame) -> List[str]:
    media_cols = [col for col in data.columns if col.startswith(MEDIA_PREFIX)]
    if not media_cols:
        raise ValueError(f"No media spend columns with prefix {MEDIA_PREFIX!r}.")
    return media_cols


def build_controls(data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    controls = pd.DataFrame(index=data.index)
    selected = [col for col in ["me_ics_all", "me_gas_dpg", "st_ct"] if col in data.columns]
    for col in selected:
        controls[col] = data[col].astype(float)

    holiday_cols = [col for col in data.columns if col.startswith("hldy_")]
    if holiday_cols:
        controls["holiday_any"] = data[holiday_cols].max(axis=1).astype(float)

    week_index = np.arange(len(data), dtype=float)
    for harmonic in (1, 2):
        angle = 2.0 * np.pi * harmonic * week_index / 52.18
        controls[f"season_sin_{harmonic}"] = np.sin(angle)
        controls[f"season_cos_{harmonic}"] = np.cos(angle)

    return controls, controls.columns.tolist()


def chronological_split(n_rows: int, train_test_split: float) -> int:
    if not 0.5 <= train_test_split < 1.0:
        raise ValueError("train_test_split must be in [0.5, 1.0).")
    split_idx = int(n_rows * train_test_split)
    if split_idx < 20 or n_rows - split_idx < 4:
        raise ValueError("Not enough rows for the requested train/test split.")
    return split_idx


def safe_scale(values: np.ndarray) -> np.ndarray:
    scale = np.percentile(values, 95, axis=0)
    fallback = np.maximum(np.max(values, axis=0), 1.0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-10), scale, fallback)
    return scale


def standardize_controls(
    controls: pd.DataFrame, split_idx: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if controls.empty:
        return (
            np.zeros((len(controls), 0), dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
        )

    values = controls.to_numpy(dtype=float)
    train = values[:split_idx]
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std = np.where(std > 1e-10, std, 1.0)
    return (values - mean) / std, mean, std


def build_lagged_media(media: np.ndarray, max_lag: int) -> np.ndarray:
    """Return lagged media with shape (lag, observation, channel)."""
    if max_lag < 0:
        raise ValueError("max_lag must be non-negative.")

    lagged = np.zeros((max_lag + 1, media.shape[0], media.shape[1]), dtype=float)
    for lag in range(max_lag + 1):
        if lag == 0:
            lagged[lag] = media
        else:
            lagged[lag, lag:, :] = media[:-lag, :]
    return lagged


def roas_prior_bounds(channel: str) -> Tuple[str, float, float]:
    group, target_low, target_high = channel_group(channel)
    if np.isnan(target_low) or np.isnan(target_high):
        # Broad but regularizing prior for channels without business benchmarks.
        return group, 0.2, 3.0
    return group, target_low, target_high


def build_media_effect_priors(
    media_spend: np.ndarray,
    split_idx: int,
    channel_names: List[str],
    y_scale: float,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build LogNormal priors for total train-period media contribution.

    The model normalizes each channel's transformed media shape over the training
    period. This makes the media effect parameter interpretable as total
    train-period contribution in scaled sales units, so ROAS priors can be
    translated into model priors directly.
    """
    rows = []
    mu = []
    sigma = []

    train_spend = media_spend[:split_idx].sum(axis=0)
    for idx, channel in enumerate(channel_names):
        group, roas_low, roas_high = roas_prior_bounds(channel)
        roas_median = float(np.sqrt(roas_low * roas_high))
        effect_median = max(train_spend[idx] * roas_median / y_scale, 1e-6)
        # Treat the low/high ROAS bounds as an approximate 90% prior interval.
        effect_sigma = max((np.log(roas_high) - np.log(roas_low)) / (2.0 * 1.645), 0.15)

        mu.append(np.log(effect_median))
        sigma.append(effect_sigma)
        rows.append(
            {
                "channel": channel,
                "benchmark_group": group,
                "train_spend": float(train_spend[idx]),
                "prior_roas_low": roas_low,
                "prior_roas_median": roas_median,
                "prior_roas_high": roas_high,
                "prior_effect_median_scaled": effect_median,
                "prior_lognormal_mu": np.log(effect_median),
                "prior_lognormal_sigma": effect_sigma,
            }
        )

    return np.array(mu, dtype=float), np.array(sigma, dtype=float), pd.DataFrame(rows)


def build_model(
    y_train_scaled: np.ndarray,
    lagged_media_train: np.ndarray,
    controls_train: np.ndarray,
    channel_names: List[str],
    control_names: List[str],
    max_lag: int,
    media_effect_prior_mu: np.ndarray,
    media_effect_prior_sigma: np.ndarray,
) -> pm.Model:
    coords = {
        "obs_id": np.arange(len(y_train_scaled)),
        "channel": channel_names,
        "control": control_names,
    }

    with pm.Model(coords=coords) as model:
        lagged_data = pt.as_tensor_variable(lagged_media_train.astype("float64"))
        control_data = pt.as_tensor_variable(controls_train.astype("float64"))

        intercept = pm.Normal("intercept", mu=1.0, sigma=0.5)
        theta = pm.Beta("theta", alpha=2.0, beta=2.0, dims="channel")
        saturation_lambda = pm.Gamma(
            "saturation_lambda", alpha=2.0, beta=1.0, dims="channel"
        )
        media_total_effect = pm.LogNormal(
            "media_total_effect",
            mu=media_effect_prior_mu,
            sigma=media_effect_prior_sigma,
            dims="channel",
        )

        if control_names:
            control_beta = pm.Normal("control_beta", mu=0.0, sigma=0.2, dims="control")
            control_mu = pt.dot(control_data, control_beta)
        else:
            control_mu = 0.0

        lag_idx = pt.arange(max_lag + 1)[:, None]
        weights = theta[None, :] ** lag_idx
        weights = weights / pt.sum(weights, axis=0, keepdims=True)
        adstocked = pt.sum(lagged_data * weights[:, None, :], axis=0)
        saturated = 1.0 - pt.exp(-saturation_lambda[None, :] * adstocked)
        saturated_share = saturated / (pt.sum(saturated, axis=0, keepdims=True) + 1e-8)
        media_contribution = saturated_share * media_total_effect[None, :]
        media_mu = pt.sum(media_contribution, axis=1)

        sigma = pm.HalfNormal("sigma", sigma=0.2)
        mu = intercept + control_mu + media_mu
        pm.Normal("sales_obs", mu=mu, sigma=sigma, observed=y_train_scaled, dims="obs_id")

    return model


def sample_model(model: pm.Model, args: argparse.Namespace) -> az.InferenceData:
    draws = min(args.draws, 100) if args.quick else args.draws
    tune = min(args.tune, 100) if args.quick else args.tune
    chains = min(args.chains, 2) if args.quick else args.chains

    with model:
        return pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=args.cores,
            target_accept=args.target_accept,
            random_seed=args.seed,
            return_inferencedata=True,
            progressbar=True,
        )


def posterior_array(idata: az.InferenceData, var_name: str) -> np.ndarray:
    arr = idata.posterior[var_name].stack(sample=("chain", "draw"))
    dims = [dim for dim in arr.dims if dim != "sample"] + ["sample"]
    arr = arr.transpose(*dims)
    values = np.asarray(arr.values)
    if values.ndim == 1:
        return values[:, None]
    return np.moveaxis(values, -1, 0)


def compute_posterior_outputs(
    idata: az.InferenceData,
    lagged_media_full: np.ndarray,
    controls_full: np.ndarray,
    y_scale: float,
    max_lag: int,
    split_idx: int,
) -> Dict[str, np.ndarray]:
    intercept = posterior_array(idata, "intercept")[:, 0]
    theta = posterior_array(idata, "theta")
    saturation_lambda = posterior_array(idata, "saturation_lambda")
    media_total_effect = posterior_array(idata, "media_total_effect")

    if "control_beta" in idata.posterior:
        control_beta = posterior_array(idata, "control_beta")
        control_scaled = np.einsum("oc,sc->so", controls_full, control_beta)
    else:
        control_scaled = np.zeros((len(intercept), lagged_media_full.shape[1]))

    lags = np.arange(max_lag + 1, dtype=float)
    weights = theta[:, None, :] ** lags[None, :, None]
    weights = weights / weights.sum(axis=1, keepdims=True)

    adstocked = np.einsum("slc,loc->soc", weights, lagged_media_full)
    saturated = 1.0 - np.exp(-saturation_lambda[:, None, :] * adstocked)
    train_saturated_sum = saturated[:, :split_idx, :].sum(axis=1)
    media_contrib_scaled = (
        saturated / (train_saturated_sum[:, None, :] + 1e-8)
    ) * media_total_effect[:, None, :]
    media_total_scaled = media_contrib_scaled.sum(axis=2)
    mu_scaled = intercept[:, None] + control_scaled + media_total_scaled

    return {
        "prediction": mu_scaled * y_scale,
        "media_contribution": media_contrib_scaled * y_scale,
        "baseline": (intercept[:, None] + control_scaled) * y_scale,
    }


def summarize_interval(values: np.ndarray) -> Tuple[float, float, float]:
    lower, median, upper = np.quantile(values, [0.05, 0.5, 0.95])
    return float(median), float(lower), float(upper)


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    residual = actual - predicted
    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))
    mape = float(np.mean(np.abs(residual / np.maximum(actual, 1e-10))) * 100.0)
    denom = np.sum((actual - actual.mean()) ** 2)
    r2 = float(1.0 - np.sum(residual**2) / denom) if denom > 0 else float("nan")
    return {"r2": r2, "rmse": rmse, "mae": mae, "mape": mape}


def channel_group(channel: str) -> Tuple[str, float, float]:
    return BENCHMARK_GROUPS.get(channel, ("Other", np.nan, np.nan))


def roas_status(roas_median: float, target_low: float, target_high: float) -> str:
    if np.isnan(target_low) or np.isnan(target_high):
        return "audit_only"
    if roas_median < target_low:
        return "below_target"
    if roas_median > target_high:
        return "above_target"
    return "within_target"


def build_roas_tables(
    media_contribution: np.ndarray,
    media_spend: np.ndarray,
    channel_names: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    contribution_rows = []
    roas_rows = []

    total_contrib_by_sample = media_contribution.sum(axis=(1, 2))
    total_contrib_median = np.median(total_contrib_by_sample)

    for idx, channel in enumerate(channel_names):
        spend = float(media_spend[:, idx].sum())
        contrib_samples = media_contribution[:, :, idx].sum(axis=1)
        contrib_median, contrib_lower, contrib_upper = summarize_interval(contrib_samples)
        roas_samples = contrib_samples / max(spend, 1e-10)
        roas_median, roas_lower, roas_upper = summarize_interval(roas_samples)
        group, target_low, target_high = channel_group(channel)
        share = contrib_median / total_contrib_median if total_contrib_median else np.nan

        contribution_rows.append(
            {
                "channel": channel,
                "benchmark_group": group,
                "spend": spend,
                "contribution_median": contrib_median,
                "contribution_lower": contrib_lower,
                "contribution_upper": contrib_upper,
                "contribution_share_median": share,
            }
        )
        roas_rows.append(
            {
                "channel": channel,
                "benchmark_group": group,
                "spend": spend,
                "contribution_median": contrib_median,
                "contribution_lower": contrib_lower,
                "contribution_upper": contrib_upper,
                "roas_median": roas_median,
                "roas_lower": roas_lower,
                "roas_upper": roas_upper,
                "target_lower": target_low,
                "target_upper": target_high,
                "status": roas_status(roas_median, target_low, target_high),
            }
        )

    return pd.DataFrame(contribution_rows), pd.DataFrame(roas_rows)


def diagnostics_summary(idata: az.InferenceData) -> Tuple[pd.DataFrame, Dict[str, float]]:
    var_names = [
        "intercept",
        "theta",
        "saturation_lambda",
        "media_total_effect",
        "sigma",
    ]
    if "control_beta" in idata.posterior:
        var_names.append("control_beta")

    summary = az.summary(idata, var_names=var_names, round_to=6)
    rhat = summary["r_hat"].dropna() if "r_hat" in summary else pd.Series(dtype=float)
    ess = summary["ess_bulk"].dropna() if "ess_bulk" in summary else pd.Series(dtype=float)
    divergences = int(idata.sample_stats["diverging"].sum().item())
    diag = {
        "divergences": divergences,
        "max_rhat": float(rhat.max()) if len(rhat) else float("nan"),
        "min_ess_bulk": float(ess.min()) if len(ess) else float("nan"),
    }
    return summary, diag


def write_outputs(
    output_dir: Path,
    idata: az.InferenceData,
    parameter_summary: pd.DataFrame,
    metrics: Dict[str, Dict[str, float]],
    diagnostics: Dict[str, float],
    contribution_df: pd.DataFrame,
    roas_df: pd.DataFrame,
    prior_df: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    parameter_summary.to_csv(output_dir / "bayesian_parameter_summary.csv")
    pd.DataFrame(
        [
            {"dataset": dataset, **values}
            for dataset, values in metrics.items()
        ]
    ).to_csv(output_dir / "bayesian_metrics.csv", index=False)
    contribution_df.to_csv(output_dir / "bayesian_channel_contributions.csv", index=False)
    roas_df.to_csv(output_dir / "bayesian_roas_audit.csv", index=False)
    prior_df.to_csv(output_dir / "bayesian_media_priors.csv", index=False)

    run_config = {
        "draws": min(args.draws, 100) if args.quick else args.draws,
        "tune": min(args.tune, 100) if args.quick else args.tune,
        "chains": min(args.chains, 2) if args.quick else args.chains,
        "target_accept": args.target_accept,
        "seed": args.seed,
        "quick": args.quick,
        "diagnostics": diagnostics,
    }
    (output_dir / "bayesian_run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    if args.save_trace:
        idata.to_netcdf(output_dir / "bayesian_trace.nc")


def print_summary(
    metrics: Dict[str, Dict[str, float]],
    diagnostics: Dict[str, float],
    roas_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    print("\n" + "=" * 80)
    print("BAYESIAN MMM SUMMARY")
    print("=" * 80)
    for dataset in ("train", "test"):
        values = metrics[dataset]
        print(
            f"{dataset.title():5s} | R2={values['r2']:.4f} "
            f"RMSE={values['rmse']:,.0f} MAE={values['mae']:,.0f} "
            f"MAPE={values['mape']:.2f}%"
        )

    print(
        "\nDiagnostics | "
        f"divergences={diagnostics['divergences']} "
        f"max_rhat={diagnostics['max_rhat']:.4f} "
        f"min_ess_bulk={diagnostics['min_ess_bulk']:.1f}"
    )

    display_cols = [
        "channel",
        "benchmark_group",
        "roas_median",
        "roas_lower",
        "roas_upper",
        "target_lower",
        "target_upper",
        "status",
    ]
    print("\nPosterior ROAS Audit:")
    print(
        roas_df[display_cols]
        .sort_values(["benchmark_group", "channel"])
        .to_string(index=False, float_format=lambda x: f"{x:.2f}")
    )
    print(f"\nOutputs written to: {output_dir}")


def run(args: argparse.Namespace) -> None:
    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    data = load_data(data_path)
    media_cols = detect_media_columns(data)
    channel_names = [col.replace(MEDIA_PREFIX, "") for col in media_cols]
    controls, control_names = build_controls(data)
    split_idx = chronological_split(len(data), args.train_test_split)

    media_spend = data[media_cols].to_numpy(dtype=float)
    media_scale = safe_scale(media_spend[:split_idx])
    media_norm = media_spend / media_scale
    lagged_media_full = build_lagged_media(media_norm, args.max_lag)

    controls_std, _, _ = standardize_controls(controls, split_idx)
    y = data[TARGET_COL].to_numpy(dtype=float)
    y_scale = float(y[:split_idx].mean())
    y_scaled = y / y_scale
    media_prior_mu, media_prior_sigma, prior_df = build_media_effect_priors(
        media_spend=media_spend,
        split_idx=split_idx,
        channel_names=channel_names,
        y_scale=y_scale,
    )

    model = build_model(
        y_train_scaled=y_scaled[:split_idx],
        lagged_media_train=lagged_media_full[:, :split_idx, :],
        controls_train=controls_std[:split_idx],
        channel_names=channel_names,
        control_names=control_names,
        max_lag=args.max_lag,
        media_effect_prior_mu=media_prior_mu,
        media_effect_prior_sigma=media_prior_sigma,
    )
    idata = sample_model(model, args)

    posterior = compute_posterior_outputs(
        idata=idata,
        lagged_media_full=lagged_media_full,
        controls_full=controls_std,
        y_scale=y_scale,
        max_lag=args.max_lag,
        split_idx=split_idx,
    )
    median_pred = np.median(posterior["prediction"], axis=0)
    metrics = {
        "train": regression_metrics(y[:split_idx], median_pred[:split_idx]),
        "test": regression_metrics(y[split_idx:], median_pred[split_idx:]),
    }

    contribution_df, roas_df = build_roas_tables(
        media_contribution=posterior["media_contribution"],
        media_spend=media_spend,
        channel_names=channel_names,
    )
    parameter_summary, diagnostics = diagnostics_summary(idata)
    write_outputs(
        output_dir=output_dir,
        idata=idata,
        parameter_summary=parameter_summary,
        metrics=metrics,
        diagnostics=diagnostics,
        contribution_df=contribution_df,
        roas_df=roas_df,
        prior_df=prior_df,
        args=args,
    )
    print_summary(metrics, diagnostics, roas_df, output_dir)


if __name__ == "__main__":
    run(parse_args())
