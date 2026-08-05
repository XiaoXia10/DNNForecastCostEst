#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collect observed error metrics for every (resolution, horizon) cell already
trained in the requested grid, from the metrics.csv files written by train.py.

Unlike get_fit_surface.py, this doesn't fit or extrapolate anything — it's a
raw pull of already-trained results across the full grid, for plotting.

Run:
    cd src/
    python framework/collect_observed_grid.py \
        --config configs/experiment.yaml \
        --resolutions 1h 4h 8h 12h 24h \
        --horizons 3 6 9 12
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import argparse

import pandas as pd

from framework.train_one_cell import load_config, data_dir_path
from framework.cell_sampler import full_grid

METRIC_COLUMNS = ["RMSE", "MAE", "MSE", "MAPE", "NSE", "KGE"]


def collect_observed_grid(config, resolutions, horizons):
    """
    Read metrics.csv for every (resolution, horizon) cell in the grid.

    Returns
    -------
    DataFrame with one row per grid cell: resolution, horizon, trained (bool),
    and each metric in METRIC_COLUMNS (NaN for cells not yet trained).
    """
    rows = []
    for resolution, horizon in full_grid(resolutions, horizons):
        ddir         = data_dir_path(config, resolution, horizon)
        metrics_path = os.path.join(ddir, "metrics.csv")

        row = {"resolution": resolution, "horizon": horizon, "trained": False}
        row.update({m: float("nan") for m in METRIC_COLUMNS})

        if os.path.exists(metrics_path):
            df = pd.read_csv(metrics_path)
            row["trained"] = True
            for m in METRIC_COLUMNS:
                if m in df.columns:
                    row[m] = float(df[m].iloc[0])

        rows.append(row)

    return pd.DataFrame(rows)


def save_observed_grid(config, grid_df):
    """Write observed_grid.csv into {base_dir}/{dataset}/."""
    out_path = os.path.join(config["base_dir"], config["dataset"], "results", "observed_grid.csv")
    grid_df.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect observed error metrics across the full "
                     "(resolution x horizon) grid for plotting."
    )
    parser.add_argument("--config", required=True,
                        help="Path to YAML experiment config")
    parser.add_argument("--resolutions", nargs="+", required=True,
                        help="Resolutions in the grid, e.g. 1h 4h 8h 12h 24h")
    parser.add_argument("--horizons", nargs="+", type=int, required=True,
                        help="Horizons in the grid, e.g. 3 6 9 12")
    args = parser.parse_args()

    config  = load_config(args.config)
    grid_df = collect_observed_grid(config, args.resolutions, args.horizons)

    n_total   = len(grid_df)
    n_trained = int(grid_df["trained"].sum())
    print(f"\n  Observed grid: {n_trained}/{n_total} cells trained")
    print(grid_df.to_string(index=False, float_format="{:.4f}".format))

    missing = grid_df[~grid_df["trained"]]
    if not missing.empty:
        print(f"\n  Not yet trained ({len(missing)}):")
        for _, row in missing.iterrows():
            print(f"    resolution={row['resolution']}  horizon={int(row['horizon']):>2d}")

    out_path = save_observed_grid(config, grid_df)
    print(f"\n  Saved to: {out_path}")
