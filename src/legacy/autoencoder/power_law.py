#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Callable, Dict
import numpy as np
from os.path import join, dirname, abspath
import pandas as pd

def fit_power_law_surface(
    resolution: np.ndarray,
    horizon: np.ndarray,
    error: np.ndarray,
) -> Dict[str, object]:
    """
    Fit power-law surface:
        E[i,j] = C * resolution[i]^alpha * horizon[j]^beta

    Args:
        resolution: shape (N,)
        horizon: shape (M,)
        error: shape (N, M)

    Returns:
        dict with C, alpha, beta, predict()
    """

    resolution = np.asarray(resolution, dtype=float)
    horizon = np.asarray(horizon, dtype=float)
    error = np.asarray(error, dtype=float)

    N = resolution.shape[0]
    M = horizon.shape[0]

    if error.shape != (N, M):
        raise ValueError("error must have shape (len(resolution), len(horizon))")

    if np.any(resolution <= 0) or np.any(horizon <= 0) or np.any(error <= 0):
        raise ValueError("All inputs must be strictly positive")

    # build grid
    R, H = np.meshgrid(resolution, horizon, indexing="ij")

    # flatten
    r_flat = R.ravel()
    h_flat = H.ravel()
    e_flat = error.ravel()

    # log transform
    log_r = np.log(r_flat)
    log_h = np.log(h_flat)
    log_e = np.log(e_flat)

    # design matrix
    X = np.column_stack([np.ones_like(log_r), log_r, log_h])

    # solve
    coeffs, *_ = np.linalg.lstsq(X, log_e, rcond=None)

    log_C, alpha, beta = coeffs
    C = np.exp(log_C)

    def predict(
        resolution_new: np.ndarray,
        horizon_new: np.ndarray,
    ) -> np.ndarray:
        resolution_new = np.asarray(resolution_new, dtype=float)
        horizon_new = np.asarray(horizon_new, dtype=float)

        if np.any(resolution_new <= 0) or np.any(horizon_new <= 0):
            raise ValueError("Inputs must be positive")

        Rn, Hn = np.meshgrid(resolution_new, horizon_new, indexing="ij")
        return C * (Rn ** alpha) * (Hn ** beta)

    return {
        "C": C,
        "alpha": alpha,
        "beta": beta,
        "predict": predict,
    }


if __name__ == "__main__":
    resolution = np.array([1,4,8,12,24])
    horizon = np.array([3,6,12])

    # synthetic example
    # C_true, a_true, b_true = 2.0, -0.5, 0.8
    current_dir = dirname(abspath(__file__))
    base_dir = dirname(current_dir)

    error = pd.read_csv(join(base_dir, "data/LSTM_rmse_metrics.csv"), index_col = 0)    
    R, H = np.meshgrid(resolution, horizon, indexing="ij")
    # error = C_true * (R ** a_true) * (H ** b_true)

    model = fit_power_law_surface(resolution, horizon, error)

    print(model["C"], model["alpha"], model["beta"])