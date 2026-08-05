#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predict the full (resolution × horizon) error surface from a fitted scaling law.
Uses the delta method in log-space for calibrated confidence intervals.
"""

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist
from framework.cell_sampler import RESOLUTION_HOURS


def predict_cell(fit, resolution, horizon, confidence=0.95):
    """
    Predict error at a single (resolution, horizon) cell.

    Returns
    -------
    mean  : float  — point estimate
    lower : float  — lower confidence bound
    upper : float  — upper confidence bound
    """
    log_r = np.log(float(RESOLUTION_HOURS[resolution]))
    log_h = np.log(float(horizon))
    x     = np.array([1.0, log_r, log_h])

    log_e_pred = float(x @ fit["coeffs"])
    mean = np.exp(log_e_pred)

    if fit.get("cov") is not None and fit["n_samples"] > 3:
        var_log_e = float(x @ fit["cov"] @ x)
        se_log_e  = np.sqrt(max(var_log_e, 0.0))
        df      = fit["n_samples"] - 3
        t_val   = t_dist.ppf((1 + confidence) / 2, df=df)
        lower   = np.exp(log_e_pred - t_val * se_log_e)
        upper   = np.exp(log_e_pred + t_val * se_log_e)
    else:
        lower = upper = np.nan

    return float(mean), float(lower), float(upper)


def predict_full_surface(fit, resolutions, horizons, confidence=0.95):
    """
    Predict error at every cell in the (resolutions × horizons) grid.

    Returns
    -------
    dict mapping (resolution, horizon) → (mean, lower_ci, upper_ci)
    """
    return {
        (r, h): predict_cell(fit, r, h, confidence)
        for r in resolutions
        for h in horizons
    }


def surface_to_dataframe(surface, sampled_cells=None):
    """
    Convert the surface dict to a tidy DataFrame suitable for saving and plotting.

    Columns: resolution, horizon, predicted_error, ci_lower, ci_upper, sampled
    """
    sampled_set = set(sampled_cells or [])
    rows = [
        {
            "resolution":      r,
            "horizon":         h,
            "predicted_error": mean,
            "ci_lower":        lower,
            "ci_upper":        upper,
            "sampled":         (r, h) in sampled_set,
        }
        for (r, h), (mean, lower, upper) in surface.items()
    ]
    return pd.DataFrame(rows)
