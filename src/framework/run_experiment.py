#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment with full surface grid. All cells are trained here.

Usage
-----
python framework/run_experiment.py \
    --config configs/experiment_sw.yaml \
    --resolutions 1h 4h 8h 12h 24h \
    --horizons 3 6 9 12

The framework:
  1. Samples a sparse subset of the (resolution X horizon) grid
  2. Trains the user's model on each sampled cell
  3. Fits  log(E) = log(C) + α·log(r) + β·log(h)  to the sampled errors
  4. Extrapolates the full error surface with confidence intervals
  5. Saves results to  base_dir/dataset/results/<name>/
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import json
import argparse
import importlib

import yaml
import pandas as pd

from framework.cell_sampler import sample_cells, full_grid
from framework.fit_surface  import fit_surface, print_fit_summary
from framework.extrapolate  import predict_full_surface, surface_to_dataframe
from framework.train_one_cell import train_cell, load_config


# ── Helpers ────────────────────────────────────────────────────────────────────

def _out_dir(config):
    path = os.path.join(config["base_dir"], config["dataset"],
                        "results", config["name"])
    os.makedirs(path, exist_ok=True)
    return path


def _save_fit(fit, out_dir):
    """Persist fit parameters (excludes numpy arrays)."""
    scalar_keys = ("form", "C", "alpha", "beta", "log_C",
                   "se_log_C", "se_alpha", "se_beta", "r2", "n_samples")
    summary = {k: fit[k] for k in scalar_keys}
    with open(os.path.join(out_dir, "fit_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run(config_path, resolutions, horizons):
    config = load_config(config_path)

    # Import user-specified model modules to trigger @register
    for module in config.get("model_modules", []):
        importlib.import_module(module)

    metric  = config.get("metric", "RMSE")
    out_dir = _out_dir(config)
    n_total = len(resolutions) * len(horizons)

    print(f"\n{'═'*60}")
    print(f"  Experiment : {config['name']}")
    print(f"  Model      : {config['model']}")
    print(f"  Dataset    : {config['dataset']}")
    print(f"  Grid       : {len(resolutions)} resolutions X"
          f"{len(horizons)} horizons = {n_total} cells")
    print(f"  Sampling   : {config['sampling_budget']} cells "
          f"via {config['sampling_strategy']}")
    print(f"{'═'*60}\n")

    # ── Step 1: Sample cells ─────────────────────────────────────────────────
    sampled = sample_cells(
        resolutions, horizons,
        config["sampling_budget"],
        strategy=config["sampling_strategy"],
    )
    print(f"Cells selected for training ({len(sampled)}):")
    for r, h in sampled:
        print(f"  resolution={r}  horizon={h}")
    print()

    # ── Step 2: Train model on each sampled cell ──────────────────────────────
    all_results = []
    skipped     = []

    for resolution, horizon in sampled:
        print(f"▶ Training  resolution={resolution}  horizon={horizon}")
        try:
            metrics = train_cell(config, resolution, horizon)
            all_results.append(metrics)
            print(f"  {metric}: {metrics[metric]:.4f}\n")
        except FileNotFoundError as exc:
            print(f"  SKIPPED — {exc}\n")
            skipped.append((resolution, horizon))

    if not all_results:
        raise RuntimeError(
            "No cells trained successfully. "
            "Check that data directories exist for the requested resolutions and horizons."
        )

    if skipped:
        print(f"Skipped {len(skipped)} cell(s) — data not found: {skipped}\n")

    # ── Step 3: Fit scaling surface ───────────────────────────────────────────
    trained_cells   = [(r["resolution"], r["horizon"]) for r in all_results]
    metric_values   = [r[metric] for r in all_results]

    fit = fit_surface(trained_cells, metric_values,
                      form=config.get("scaling_form", "power_law"))
    print_fit_summary(fit)
    _save_fit(fit, out_dir)

    # ── Step 4: Extrapolate full surface ──────────────────────────────────────
    confidence = config.get("confidence", 0.95)
    surface    = predict_full_surface(fit, resolutions, horizons, confidence)
    surface_df = surface_to_dataframe(surface, sampled_cells=trained_cells)

    # Attach observed values for cells that were actually trained
    observed = {(r["resolution"], r["horizon"]): r[metric] for r in all_results}
    surface_df["observed_error"] = surface_df.apply(
        lambda row: observed.get((row["resolution"], row["horizon"]), float("nan")),
        axis=1,
    )

    surface_df.to_csv(os.path.join(out_dir, "error_surface.csv"), index=False)

    # ── Step 5: Print summary ─────────────────────────────────────────────────
    print(f"\nFull error surface — predicted {metric}:")
    pivot = surface_df.pivot(
        index="resolution", columns="horizon", values="predicted_error"
    )
    pivot.index.name   = "resolution \\ horizon"
    pivot.columns.name = None
    print(pivot.to_string(float_format="{:.4f}".format))

    print(f"\nSampled cells (observed):")
    obs_df = surface_df[surface_df["sampled"]][
        ["resolution", "horizon", "observed_error", "predicted_error",
         "ci_lower", "ci_upper"]
    ].reset_index(drop=True)
    print(obs_df.to_string(index=False, float_format="{:.4f}".format))

    print(f"\nResults saved → {out_dir}\n")
    return surface_df, fit


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the data-resolution error-surface experiment."
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to YAML experiment config (e.g. configs/experiment_sw.yaml)"
    )
    parser.add_argument(
        "--resolutions", nargs="+", required=True,
        help="Temporal resolutions to study  e.g.  --resolutions 1h 4h 8h 12h 24h"
    )
    parser.add_argument(
        "--horizons", nargs="+", required=True, type=int,
        help="Forecast horizons (timesteps)  e.g.  --horizons 3 6 9 12"
    )
    args = parser.parse_args()

    run(args.config, args.resolutions, args.horizons)
