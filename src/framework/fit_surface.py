#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fit the scaling law  E = C * r^alpha * h^beta  to sampled (cell, error) pairs.

The power law is linearised in log-space:
    log(E) = log(C) + alpha*log(r) + beta*log(h)
and solved with OLS — giving closed-form parameter standard errors.
"""

import numpy as np
from scipy.stats import t as t_dist
from framework.cell_sampler import RESOLUTION_HOURS


def _encode(cells, metric_values):
    """Convert string resolution labels and horizon integers to numeric arrays."""
    r = np.array([RESOLUTION_HOURS[c[0]] for c in cells], dtype=float)
    h = np.array([c[1]               for c in cells], dtype=float)
    e = np.array(metric_values,                        dtype=float)
    return r, h, e


def fit_power_law(cells, metric_values, confidence=0.95):
    """
    Fit  E = C * r^alpha * h^beta  via OLS in log-space.

    Returns a dict containing:
        C, alpha, beta              — fitted parameters
        se_log_C, se_alpha,
        se_beta                     — standard errors
        ci_alpha, ci_beta           — (lower, upper) confidence intervals
        t_stat_alpha, t_stat_beta   — t-statistics  (H₀: param = 0)
        p_alpha, p_beta             — two-sided p-values
        r2                          — R² in log-space
        n_samples                   — number of sampled cells used
        coeffs, cov, X,
        log_e_observed              — internals needed by extrapolate.py
    """
    r, h, e = _encode(cells, metric_values)

    log_e = np.log(e)
    log_r = np.log(r)
    log_h = np.log(h)

    X = np.column_stack([np.ones(len(r)), log_r, log_h])
    coeffs, _, _, _ = np.linalg.lstsq(X, log_e, rcond=None)
    log_C, alpha, beta = coeffs

    y_hat = X @ coeffs
    n, p  = X.shape
    df    = n - p

    if n > p:
        sigma2 = np.sum((log_e - y_hat) ** 2) / df
        cov    = sigma2 * np.linalg.inv(X.T @ X)
        se     = np.sqrt(np.diag(cov))

        t_crit = float(t_dist.ppf((1 + confidence) / 2, df=df))

        # Confidence intervals
        ci_alpha = (float(alpha - t_crit * se[1]), float(alpha + t_crit * se[1]))
        ci_beta  = (float(beta  - t_crit * se[2]), float(beta  + t_crit * se[2]))

        # t-statistics and two-sided p-values
        t_stat_alpha = float(alpha / se[1]) if se[1] > 0 else np.nan
        t_stat_beta  = float(beta  / se[2]) if se[2] > 0 else np.nan
        p_alpha = float(2 * t_dist.sf(abs(t_stat_alpha), df=df))
        p_beta  = float(2 * t_dist.sf(abs(t_stat_beta),  df=df))
    else:
        cov = None
        se  = np.full(3, np.nan)
        ci_alpha = (np.nan, np.nan)
        ci_beta  = (np.nan, np.nan)
        t_stat_alpha = t_stat_beta = np.nan
        p_alpha = p_beta = np.nan

    ss_res = np.sum((log_e - y_hat) ** 2)
    ss_tot = np.sum((log_e - log_e.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "form":           "power_law",
        "C":              float(np.exp(log_C)),
        "alpha":          float(alpha),
        "beta":           float(beta),
        "log_C":          float(log_C),
        "se_log_C":       float(se[0]),
        "se_alpha":       float(se[1]),
        "se_beta":        float(se[2]),
        "ci_alpha":       ci_alpha,
        "ci_beta":        ci_beta,
        "t_stat_alpha":   t_stat_alpha,
        "t_stat_beta":    t_stat_beta,
        "p_alpha":        p_alpha,
        "p_beta":         p_beta,
        "confidence":     confidence,
        "r2":             float(r2),
        "n_samples":      n,
        "coeffs":         coeffs,
        "cov":            cov,
        "X":              X,
        "log_e_observed": log_e,
    }


def fit_surface(cells, metric_values, form="power_law", confidence=0.95):
    """
    Fit a scaling law to sampled (cell, error) pairs.

    Parameters
    ----------
    cells         : list of (resolution_str, horizon_int)
    metric_values : list of float  — one error value per cell
    form          : "power_law" | "auto"
    confidence    : float  — confidence level for alpha/beta intervals (default 0.95)
    """
    if form in ("power_law", "auto"):
        return fit_power_law(cells, metric_values, confidence=confidence)
    raise ValueError(f"Unknown scaling form '{form}'. Choose: power_law | auto")


def print_fit_summary(fit):
    ci  = int(fit.get("confidence", 0.95) * 100)
    nan = float("nan")

    def _fmt(v):
        return f"{v:.4f}" if v == v else "n/a"   # nan check

    print(f"\n{'─'*55}")
    print(f"  Scaling law:  log(E) = log(C) + α·log(r) + β·log(h)  [{fit['form']}]")
    print(f"  {'─'*51}")
    print(f"  {'Parameter':<10}  {'Estimate':>9}  {'SE':>9}  "
          f"  {f'{ci}% CI':<20}  {'t-stat':>8}  {'p-value':>8}")
    print(f"  {'─'*51}")

    rows = [
        ("log C",  fit["log_C"],  fit["se_log_C"],  (nan, nan),       nan,                   nan),
        ("alpha",  fit["alpha"],  fit["se_alpha"],   fit["ci_alpha"],  fit["t_stat_alpha"],   fit["p_alpha"]),
        ("beta",   fit["beta"],   fit["se_beta"],    fit["ci_beta"],   fit["t_stat_beta"],    fit["p_beta"]),
    ]
    for name, est, se, (lo, hi), t, pv in rows:
        ci_str = f"[{_fmt(lo)}, {_fmt(hi)}]" if lo == lo else "n/a"
        print(f"  {name:<10}  {_fmt(est):>9}  {_fmt(se):>9}  "
              f"  {ci_str:<20}  {_fmt(t):>8}  {_fmt(pv):>8}")

    print(f"  {'─'*51}")
    print(f"  R² (log-space): {fit['r2']:.4f}   |   "
          f"C = exp(log C) = {fit['C']:.4f}   |   n = {fit['n_samples']}")
    print(f"{'─'*55}\n")
