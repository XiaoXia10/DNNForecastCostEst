#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fit the scaling law to a user-specified set of already-trained cells.

Reads base_dir and dataset from the YAML config, finds each cell's
metrics.csv, extracts the requested metric, then fits and prints
the power-law surface.

Run:
    cd src/
    python framework/get_fit_surface.py \
        --config configs/experiment.yaml \
        --cells  4h:3 4h:12 24h:3 24h:12 \
        --metric RMSE \
        --resolutions 1h 4h 8h 12h 24h \
        --horizons 3 6 9 12 
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import argparse

import pandas as pd
import yaml

from framework.fit_surface import fit_surface, print_fit_summary
from framework.extrapolate import predict_full_surface, surface_to_dataframe
from framework.train_one_cell import data_dir_path


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def get_metric_values(config, cells, metric_name):
    """
    Read metric_name from metrics.csv for each (resolution, horizon) cell.

    Parameters
    ----------
    config      : dict — must contain base_dir and dataset keys
    cells       : list of (resolution_str, horizon_int)
    metric_name : str  — e.g. "RMSE", "MAE", "MSE"

    Returns
    -------
    list of float, one value per cell in the same order as cells
    """
    col = metric_name.upper()
    metric_values = []

    for resolution, horizon in cells:
        ddir         = data_dir_path(config, resolution, horizon)
        metrics_path = os.path.join(ddir, "metrics.csv")

        if not os.path.exists(metrics_path):
            raise FileNotFoundError(
                f"No metrics.csv for ({resolution}, h={horizon}) at:\n"
                f"  {metrics_path}\n"
                f"  Run train.py for this cell first."
            )

        df = pd.read_csv(metrics_path)
        if col not in df.columns:
            raise KeyError(
                f"Metric '{col}' not in {metrics_path}. "
                f"Available: {[c for c in df.columns if c not in ('resolution','horizon')]}"
            )

        metric_values.append(float(df[col].iloc[0]))

    return metric_values


def save_results(config, cells, metric_name, metrics, fit, surface_df=None):
    """Write fit_params.json, fit_cells.csv, and error_surface.csv into {base_dir}/{dataset}/results/{name}/."""
    out_dir = os.path.join(
        config["base_dir"], config["dataset"], "results", config["name"]
    )
    os.makedirs(out_dir, exist_ok=True)

    # ── fit_cells.csv — input cells and their metric values ───────────────────
    cells_df = pd.DataFrame({
        "resolution": [c[0] for c in cells],
        "horizon":    [c[1] for c in cells],
        metric_name.upper(): metrics,
    })
    cells_path = os.path.join(out_dir, "fit_cells.csv")
    cells_df.to_csv(cells_path, index=False)

    # ── fit_params.csv — one row per fitted parameter ─────────────────────────
    params_df = pd.DataFrame([{
        "form":         fit["form"],
        "metric":       metric_name.upper(),
        "n_samples":    fit["n_samples"],
        "C":            fit["C"],
        "log_C":        fit["log_C"],
        "alpha":        fit["alpha"],
        "beta":         fit["beta"],
        "se_log_C":     fit["se_log_C"],
        "se_alpha":     fit["se_alpha"],
        "se_beta":      fit["se_beta"],
        "ci_alpha_lo":  fit["ci_alpha"][0],
        "ci_alpha_hi":  fit["ci_alpha"][1],
        "ci_beta_lo":   fit["ci_beta"][0],
        "ci_beta_hi":   fit["ci_beta"][1],
        "t_stat_alpha": fit["t_stat_alpha"],
        "t_stat_beta":  fit["t_stat_beta"],
        "p_alpha":      fit["p_alpha"],
        "p_beta":       fit["p_beta"],
        "r2":           fit["r2"],
        "confidence":   fit["confidence"],
    }])
    params_path = os.path.join(out_dir, "fit_params.csv")
    params_df.to_csv(params_path, index=False)

    print(f"\n  Results saved to: {out_dir}/")
    print(f"    fit_cells.csv  — input cells and {metric_name.upper()} values")
    print(f"    fit_params.csv — fitted C, alpha, beta, SE, CI, R²")

    # ── error_surface.csv — extrapolated predictions across the full grid ─────
    if surface_df is not None:
        surface_path = os.path.join(out_dir, "error_surface.csv")
        surface_df.to_csv(surface_path, index=False)
        print(f"    error_surface.csv — extrapolated {metric_name.upper()} "
              f"with confidence intervals (alpha/beta/log_C uncertainty)")


def get_fit_surface(config, cells, metric_name, resolutions=None, horizons=None):
    metrics = get_metric_values(config, cells, metric_name)

    print(f"\n  Cells and {metric_name.upper()} values:")
    for (res, hor), val in zip(cells, metrics):
        print(f"    {res}  h={hor:>2d}  →  {val:.6f}")

    fit = fit_surface(cells, metric_values=metrics)
    print_fit_summary(fit)

    ######## Extrapolate the fitted surface across the requested grid 
    # Defaults to the resolutions/horizons already present in `cells` if the
    # caller doesn't ask for a wider grid.
    resolutions = resolutions or sorted({c[0] for c in cells})
    horizons    = horizons or sorted({c[1] for c in cells})
    confidence  = config.get("confidence", 0.95)

    surface    = predict_full_surface(fit, resolutions, horizons, confidence)
    surface_df = surface_to_dataframe(surface, sampled_cells=cells)

    observed = dict(zip(cells, metrics))
    surface_df["observed_error"] = surface_df.apply(
        lambda row: observed.get((row["resolution"], row["horizon"]), float("nan")),
        axis=1,
    )

    ci_pct = int(confidence * 100)

    print(f"\n  Extrapolated error surface ({len(resolutions)} resolutions × "
          f"{len(horizons)} horizons):")
    pivot = surface_df.pivot(
        index="resolution", columns="horizon", values="predicted_error"
    )
    pivot.index.name   = "resolution \\ horizon"
    pivot.columns.name = None
    print(pivot.to_string(float_format="{:.4f}".format))

    # ci_lower/ci_upper already propagate the joint uncertainty in log_C,
    # alpha, and beta (via the OLS covariance matrix + delta method in
    # extrapolate.predict_cell) into the metric's original units — this is
    # the alpha/beta CI, expressed as error rather than as exponents.
    print(f"\n  {ci_pct}% CI on predicted {metric_name.upper()} "
          f"(propagated from the alpha/beta/log_C covariance):")
    ci_cols = ["resolution", "horizon", "predicted_error", "ci_lower", "ci_upper"]
    print(surface_df[ci_cols].to_string(index=False, float_format="{:.4f}".format))

    save_results(config, cells, metric_name, metrics, fit, surface_df)

    return fit, surface_df


def _parse_cells(cell_strings):
    """Convert ['4h:3', '24h:12', ...] into [('4h', 3), ('24h', 12), ...]."""
    cells = []
    for s in cell_strings:
        parts = s.split(":")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(
                f"Invalid cell '{s}' — expected resolution:horizon (e.g. 4h:3)"
            )
        cells.append((parts[0].strip(), int(parts[1].strip())))
    return cells


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit scaling law to trained cells and print surface parameters."
    )
    parser.add_argument("--config", required=True,
                        help="Path to YAML experiment config")
    parser.add_argument("--cells", required=True, nargs="+",
                        metavar="RES:HOR",
                        help="Trained cells as resolution:horizon pairs "
                             "(e.g. 4h:3 4h:12 24h:3 24h:12)")
    parser.add_argument("--metric", default="RMSE",
                        help="Error metric to fit on (default: RMSE). "
                             "Choices: RMSE MAE MSE or user defined")
    parser.add_argument("--resolutions", nargs="+", default=None,
                        help="Resolutions to extrapolate the fitted surface over "
                             "(default: resolutions present in --cells)")
    parser.add_argument("--horizons", nargs="+", type=int, default=None,
                        help="Horizons to extrapolate the fitted surface over "
                             "(default: horizons present in --cells)")
    args = parser.parse_args()

    config = load_config(args.config)
    cells  = _parse_cells(args.cells)

    get_fit_surface(config, cells, args.metric, args.resolutions, args.horizons)
