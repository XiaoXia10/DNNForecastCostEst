#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model-agnostic single-cell trainer.

Trains a registered model on one (resolution, horizon) cell, runs MC dropout
inference on the test set, and returns a metrics dict.

Dispatch: train_cell() reads the registered backend and calls either
  _train_pytorch()     — for torch.nn.Module models
  _train_tensorflow()  — for TensorFlow/Keras factory models
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import time
import importlib
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

from permetrics.regression import RegressionMetric
from framework.model_registry import get_backend, build_model, build_tf_model

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim


# ── Shared helpers ─────────────────────────────────────────────────────────────

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def data_dir_path(config, resolution, horizon):
    h = str(horizon)
    return os.path.join(
        config["base_dir"], config["dataset"],
        resolution, f"timestep{h}{h}{h}"
    )


def _load_data_stats(config, resolution):
    """Return (data_mean, data_std) from the resolution-level df.csv."""
    df_raw = pd.read_csv(
        os.path.join(config["base_dir"], config["dataset"], resolution, "df.csv"),
        index_col=0, parse_dates=True,
    )
    return df_raw.mean().values, df_raw.std().values


def _save_outputs_and_metrics(config, ddir, resolution, horizon,
                               mean_pred, uncertainty, y_test, t_test,
                               data_mean, data_std):
    """Reshape arrays, write CSVs and metrics.csv. Shared between both backends."""
    n, t, f = mean_pred.shape

    test_pred_arr   = mean_pred.reshape(n * t, f)
    test_uncert_arr = uncertainty.reshape(n * t, f)
    test_real_arr   = (y_test.reshape(y_test.shape[0] * y_test.shape[1], y_test.shape[2])
                       * data_std + data_mean)
    time_flat = t_test.flatten()

    pred_df   = pd.DataFrame(test_pred_arr,   index=time_flat)
    real_df   = pd.DataFrame(test_real_arr,   index=time_flat)
    uncert_df = pd.DataFrame(test_uncert_arr, index=time_flat)

    pred_df.to_csv(  os.path.join(ddir, "test_predy.csv"))
    real_df.to_csv(  os.path.join(ddir, "test_realy.csv"))
    uncert_df.to_csv(os.path.join(ddir, "test_uncert.csv"))

    metric_names = ["RMSE", "MAE", "MSE", "MAPE", "NSE", "KGE"]
    evaluator    = RegressionMetric(real_df.to_numpy(), pred_df.to_numpy())
    results      = evaluator.get_metrics_by_list_names(metric_names)

    with open(os.path.join(ddir, "metrics.csv"), "w") as fh:
        fh.write("resolution,horizon," + ",".join(metric_names) + "\n")
        fh.write(f"{resolution},{horizon}," +
                 ",".join(str(results[m]) for m in metric_names) + "\n")

    results["resolution"] = resolution
    results["horizon"]    = horizon
    return results


def _save_loss_plot(train_losses, val_losses, config, ddir, resolution, horizon):
    plt.figure()
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses,   label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MAE)")
    plt.title(f"{resolution} | horizon={horizon}")
    plt.legend()
    plt.savefig(os.path.join(ddir, f"loss_{config['model']}.png"),
                dpi=150, bbox_inches="tight")
    plt.close()


# ── PyTorch backend ────────────────────────────────────────────────────────────

def _load_split_pt(data_dir, split, batch_size, shuffle):
    """Load one data split as a PyTorch DataLoader with NaN filtering."""
    x = torch.tensor(np.load(os.path.join(data_dir, f"x_{split}.npy"))).float()
    y = torch.tensor(np.load(os.path.join(data_dir, f"y_{split}.npy"))).float()
    valid = ~(torch.isnan(x).any(dim=(1, 2)) | torch.isnan(y).any(dim=(1, 2)))
    x, y = x[valid], y[valid]
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)


def _mc_dropout_predict_pt(model, data_mean, data_std, input_data, num_real, mc_dropout_rate):
    """PyTorch MC dropout: run `num_real` forward passes in train() mode."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = mc_dropout_rate
    model.train()
    predictions = []
    for _ in range(num_real):
        with torch.no_grad():
            output = model(input_data)
            predictions.append((output * data_std) + data_mean)
    predictions = torch.stack(predictions)
    return predictions.mean(dim=0).numpy(), predictions.var(dim=0).numpy()


def _train_pytorch(config, ddir, resolution, horizon):

    model     = build_model(
        config["model"],
        input_size=config["input_dim"],
        hidden_size=config["hidden_dim"],
        output_size=config["output_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout_rate"],
    )
    bs            = config["batch_size"]
    train_loader  = _load_split_pt(ddir, "train", bs, shuffle=True)
    val_loader    = _load_split_pt(ddir, "val",   bs, shuffle=False)
    optimizer     = optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion     = nn.L1Loss()
    save_path     = os.path.join(ddir, f"best_{config['model']}.pth")

    best_val_loss = float("inf")
    counter       = 0
    train_losses, val_losses = [], []
    t0 = time.time()

    for epoch in range(config["epochs"]):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_losses.append(train_loss / len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xv, yv in val_loader:
                val_loss += criterion(model(xv), yv).item()
        avg_val = val_loss / len(val_loader)
        val_losses.append(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            counter += 1
            if counter >= config["patience"]:
                print(f"  [{resolution} h={horizon}] Early stop at epoch {epoch + 1}")
                break

    elapsed = (time.time() - t0) / 60
    print(f"  [{resolution} h={horizon}] Best val loss: {best_val_loss:.6f}  ({elapsed:.1f} min)")
    _save_loss_plot(train_losses, val_losses, config, ddir, resolution, horizon)

    # Inference — reload best weights
    model.load_state_dict(torch.load(save_path, map_location="cpu"))

    x_test = np.load(os.path.join(ddir, "x_test.npy"))
    y_test = np.load(os.path.join(ddir, "y_test.npy"))
    t_test = np.load(os.path.join(ddir, "test_time.npy"))
    data_mean, data_std = _load_data_stats(config, resolution)

    mean_pred, uncertainty = _mc_dropout_predict_pt(
        model, data_mean, data_std,
        torch.tensor(x_test).float(),
        config["num_real"], config["mc_dropout_rate"],
    )

    os.makedirs(os.path.join(ddir, "outputs"), exist_ok=True)

    np.save(os.path.join(ddir, "outputs", "mean_prediction.npy"), mean_pred)
    np.save(os.path.join(ddir, "outputs", "uncertainty.npy"), uncertainty)

    return _save_outputs_and_metrics(
        config, ddir, resolution, horizon,
        mean_pred, uncertainty, y_test, t_test, data_mean, data_std,
    )


# ── TensorFlow backend ─────────────────────────────────────────────────────────

def _nan_filter_np(x, y):
    """Remove samples with any NaN in x or y (numpy version of the PyTorch filter)."""
    valid = ~(np.isnan(x).any(axis=(1, 2)) | np.isnan(y).any(axis=(1, 2)))
    return x[valid], y[valid]


def _mc_dropout_predict_tf(model, data_mean, data_std, x_test, num_real):
    """TF MC dropout: call model with training=True to activate dropout layers.

    Note: dropout rate is fixed at model-build time; num_real controls the
    number of stochastic forward passes.
    """
    predictions = []
    for _ in range(num_real):
        output = model(x_test, training=True).numpy()
        predictions.append((output * data_std) + data_mean)
    predictions = np.stack(predictions)
    return predictions.mean(axis=0), predictions.var(axis=0)


def _train_tensorflow(config, ddir, resolution, horizon):
    import tensorflow as tf

    # ── Load data ─────────────────────────────────────────────────────────────
    x_train = np.load(os.path.join(ddir, "x_train.npy"))
    y_train = np.load(os.path.join(ddir, "y_train.npy"))
    x_val   = np.load(os.path.join(ddir, "x_val.npy"))
    y_val   = np.load(os.path.join(ddir, "y_val.npy"))
    x_test  = np.load(os.path.join(ddir, "x_test.npy"))
    y_test  = np.load(os.path.join(ddir, "y_test.npy"))
    t_test  = np.load(os.path.join(ddir, "test_time.npy"))

    x_train, y_train = _nan_filter_np(x_train, y_train)
    x_val,   y_val   = _nan_filter_np(x_val,   y_val)

    data_mean, data_std = _load_data_stats(config, resolution)

    # ── Build model (factory needs data for input/output shapes) ──────────────
    model     = build_tf_model(config["model"], x_train, y_train, config)
    save_path = os.path.join(ddir, f"best_{config['model']}.h5")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            save_path, save_best_only=True, monitor="val_loss", verbose=0,
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=config["patience"], restore_best_weights=True, verbose=0,
        ),
    ]

    # ── Train ─────────────────────────────────────────────────────────────────
    t0      = time.time()
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        callbacks=callbacks,
        verbose=0,
    )
    elapsed      = (time.time() - t0) / 60
    best_val     = min(history.history["val_loss"])
    print(f"  [{resolution} h={horizon}] Best val loss: {best_val:.6f}  ({elapsed:.1f} min)")

    _save_loss_plot(
        history.history["loss"], history.history["val_loss"],
        config, ddir, resolution, horizon,
    )

    ######## Inference — model already has best weights (restore_best_weights=True)
    mean_pred, uncertainty = _mc_dropout_predict_tf(
        model, data_mean, data_std, x_test, config["num_real"],
    )

    return _save_outputs_and_metrics(
        config, ddir, resolution, horizon,
        mean_pred, uncertainty, y_test, t_test, data_mean, data_std,
    )


# ── Public entry point ─────────────────────────────────────────────────────────

def train_cell(config, resolution, horizon):
    """
    Train the model registered in config["model"] on one (resolution, horizon)
    cell and return a metrics dict.

    Dispatches to _train_pytorch or _train_tensorflow based on the registered
    backend of the model.

    Returns
    -------
    dict with RMSE, MAE, MSE, MAPE, NSE, KGE plus resolution and horizon keys.
    """
    ddir = data_dir_path(config, resolution, horizon)
    if not os.path.exists(ddir):
        raise FileNotFoundError(
            f"No data for resolution={resolution}, horizon={horizon} at:\n  {ddir}"
        )

    backend = get_backend(config["model"])

    if backend == "pytorch":
        return _train_pytorch(config, ddir, resolution, horizon)
    elif backend == "tensorflow":
        return _train_tensorflow(config, ddir, resolution, horizon)
    else:
        raise ValueError(f"Unknown backend '{backend}' for model '{config['model']}'.")


# ── Standalone entry point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train one (resolution, horizon) cell.")
    parser.add_argument("--config",     required=True, help="Path to YAML experiment config")
    parser.add_argument("--resolution", required=True, help="e.g. 4h. This is the user data resolution")
    parser.add_argument("--horizon",    required=True, type=int, help="e.g. 12. User forecast horizon length")
    args = parser.parse_args()

    #config_dir = os.path.dirname(args.config)
    #if config_dir:
    #    os.makedirs(config_dir, exist_ok=True)

    cfg = load_config(args.config)
    for module in cfg.get("model_modules", []):
        importlib.import_module(module)

    metrics = train_cell(cfg, args.resolution, args.horizon)
    print("\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
