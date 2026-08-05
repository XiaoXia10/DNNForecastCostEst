#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot predicted mean +/- uncertainty band against measured values for one
(resolution, horizon) cell, using the mean_prediction.npy / uncertainty.npy
arrays saved by train.py.

uncertainty.npy holds the variance of the MC dropout realizations
(predictions.var(dim=0) in train.py), not a calibrated confidence interval,
so it is plotted directly as the band half-width.

Usage
-----
python framework/plot_uncertainty.py \
    --config configs/experiment_sw.yaml \
    --resolution 4h \
    --horizon 12
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from framework.train_one_cell import load_config, data_dir_path


def _load_outputs(ddir):
    """Load mean_prediction.npy and uncertainty.npy saved by train.py, plus the
    matching real values and time index. Returns (time, real, mean, uncertainty),
    each reshaped to (n*t, f) / (n*t,) in the row-major order written by train.py.
    """
    outputs_dir = os.path.join(ddir, "outputs")
    mean_pred   = np.load(os.path.join(outputs_dir, "mean_prediction.npy"))
    uncertainty = np.load(os.path.join(outputs_dir, "uncertainty.npy"))
    t_test      = np.load(os.path.join(ddir, "test_time.npy"))
    real_df     = pd.read_csv(os.path.join(ddir, "test_realy.csv"), index_col=0)

    n_features = mean_pred.shape[-1]
    mean_flat        = mean_pred.reshape(-1, n_features)
    uncertainty_flat = uncertainty.reshape(-1, n_features)
    time_flat        = pd.to_datetime(t_test.flatten())

    return time_flat, real_df.to_numpy(), mean_flat, uncertainty_flat


def plot_uncertainty(ddir, resolution, horizon):
    """Plot measured vs. predicted with an MC dropout uncertainty band for
    each output feature, saving one PNG per feature into ddir/outputs/.
    """
    time, real, mean, uncertainty = _load_outputs(ddir)

    n_features = mean.shape[1]
    for f in range(n_features):
        suffix = "" if n_features == 1 else f"_f{f}"

        fig = plt.figure(figsize=(20, 10))
        plt.plot(time, real[:, f], label="Measured", linewidth=3)
        plt.plot(time, mean[:, f], label="Predicted", linewidth=3)
        plt.fill_between(
            time, mean[:, f] - uncertainty[:, f], mean[:, f] + uncertainty[:, f],
            color="green", alpha=0.5, label="Uncertainty",
        )

        plt.title(f"{resolution} | horizon={horizon} | MC Dropout Uncertainty", fontsize=35)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.legend(fontsize=25)
        plt.xlabel("Date Time", fontsize=25)
        plt.ylabel("Value", fontsize=25)
        plt.savefig(
            os.path.join(ddir, "outputs", f"forecast_plot{suffix}.png"),
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot mean prediction with uncertainty band for one (resolution, horizon) cell.")
    parser.add_argument("--config",     required=True, help="Path to YAML experiment config")
    parser.add_argument("--resolution", required=True, help="e.g. 4h. This is the user data resolution")
    parser.add_argument("--horizon",    required=True, type=int, help="e.g. 12. User forecast horizon length")
    args = parser.parse_args()

    cfg  = load_config(args.config)
    ddir = data_dir_path(cfg, args.resolution, args.horizon)

    plot_uncertainty(ddir, args.resolution, args.horizon)
    print(f"Saved forecast plot to {os.path.join(ddir, 'outputs')}")
