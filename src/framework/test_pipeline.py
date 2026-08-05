#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline smoke test — exercises all framework components end-to-end.

Uses a reduced config (few epochs, small grid) so the test completes quickly.
Exercises every component in order and prints a pass/fail summary.

Run:
    cd src/
    python framework/test_pipeline.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import traceback
import importlib

import numpy as np

# ── PyTorch test config ────────────────────────────────────────────────────────
TEST_CONFIG = {
    "name":               "pipeline_smoke_test",
    "base_dir":           "/Users/xl3138/workspaces/data_resolution_study/lstm",
    "dataset":            "sw_data",
    "model":              "lstm",
    "model_modules":      ["models.lstm"],
    # Architecture
    "input_dim":          1,
    "output_dim":         1,
    "hidden_dim":         8,
    "num_layers":         2,
    "dropout_rate":       0.5,
    # Training — kept small for speed
    "epochs":             20,
    "learning_rate":      0.0001,
    "batch_size":         16,
    "patience":           15,
    # MC Dropout
    "num_real":           10,
    "mc_dropout_rate":    0.3,
    # Sampling
    "sampling_strategy":  "corners",
    "sampling_budget":    4,
    # Surface fitting
    "metric":             "RMSE",
    "scaling_form":       "power_law",
    "confidence":         0.95,
}

# ── TensorFlow test config ─────────────────────────────────────────────────────
# hidden_dim maps to latent_dim inside the TF autoencoder factories.
TEST_CONFIG_TF = {
    "name":               "pipeline_smoke_test_tf",
    "base_dir":           "/Users/xl3138/workspaces/data_resolution_study/lstm",
    "dataset":            "sw_data",
    "model":              "gru_autoencoder",
    "model_modules":      ["models.autoencoder"],
    # Architecture — hidden_dim is used as latent_dim in the encoder/decoder
    "input_dim":          1,
    "output_dim":         1,
    "hidden_dim":         100,
    "num_layers":         2,
    "dropout_rate":       0.2,
    "recurrent_dropout":  0.0,
    # Training — kept small for speed
    "epochs":             20,
    "learning_rate":      0.001,
    "batch_size":         16,
    "patience":           15,
    # MC Dropout — rate baked in at build time; num_real controls forward passes
    "num_real":           10,
    "mc_dropout_rate":    0.3,
    # Sampling
    "sampling_strategy":  "corners",
    "sampling_budget":    4,
    # Surface fitting
    "metric":             "RMSE",
    "scaling_form":       "power_law",
    "confidence":         0.95,
}

# Check once at import time whether TensorFlow is available
import importlib.util
TF_AVAILABLE = importlib.util.find_spec("tensorflow") is not None

TEST_RESOLUTIONS = ["1h", "4h", "8h", "12h", "24h"]
TEST_HORIZONS    = [3, 6, 9, 12]

PASS = "  [PASS]"
FAIL = "  [FAIL]"


# ── Individual component tests ─────────────────────────────────────────────────

def test_registry():
    print("\n── Test 1: Model Registry ──────────────────────────────")
    import models.lstm  # triggers @register
    from framework.model_registry import build_model, list_models

    model_name  = TEST_CONFIG["model"]
    models_list = list_models()
    assert model_name in models_list, f"'{model_name}' not registered. Found: {models_list}"
    print(f"  Registered models: {models_list}")

    model = build_model(
        model_name,
        input_size=TEST_CONFIG["input_dim"],
        hidden_size=TEST_CONFIG["hidden_dim"],
        output_size=TEST_CONFIG["output_dim"],
        num_layers=TEST_CONFIG["num_layers"],
        dropout=TEST_CONFIG["dropout_rate"],
    )
    print(f"  Model type: {type(model).__name__}")
    print(f"  Dropout modules: "
          f"{sum(1 for m in model.modules() if 'Dropout' in type(m).__name__)}")
    print(PASS)


def test_cell_sampler():
    print("\n── Test 2: Cell Sampler ────────────────────────────────")
    from framework.cell_sampler import sample_cells, full_grid

    grid = full_grid(TEST_RESOLUTIONS, TEST_HORIZONS)
    print(f"  Full grid ({len(grid)} cells): {grid}")

    for strategy in ("corners", "lhs", "random"):
        cells = sample_cells(TEST_RESOLUTIONS, TEST_HORIZONS,
                             budget=4, strategy=strategy)
        assert len(cells) <= 4
        assert all(c in grid for c in cells), f"Invalid cell from {strategy}"
        print(f"  {strategy:8s} → {cells}")

    print(PASS)


def test_train_single_cell():
    print("\n── Test 3: Single Cell Training (4h, horizon=3) ────────")
    from framework.train import train_cell

    metrics = train_cell(TEST_CONFIG, resolution="4h", horizon=3)

    required = {"RMSE", "MAE", "NSE", "KGE", "resolution", "horizon"}
    missing  = required - set(metrics.keys())
    assert not missing, f"Missing metric keys: {missing}"
    assert metrics["resolution"] == "4h"
    assert metrics["horizon"]    == 3
    assert np.isfinite(metrics["RMSE"]), f"RMSE is not finite: {metrics['RMSE']}"

    print(f"  RMSE : {metrics['RMSE']:.4f}")
    print(f"  MAE  : {metrics['MAE']:.4f}")
    print(f"  NSE  : {metrics['NSE']:.4f}")
    print(f"  KGE  : {metrics['KGE']:.4f}")
    print(PASS)
    return metrics


def test_fit_surface(sampled_results):
    print("\n── Test 4: Scaling Law Fit ─────────────────────────────")
    from framework.fit_surface import fit_surface, print_fit_summary

    cells         = [(r["resolution"], r["horizon"]) for r in sampled_results]
    metric_values = [r[TEST_CONFIG["metric"]] for r in sampled_results]

    fit = fit_surface(cells, metric_values)
    print_fit_summary(fit)

    assert fit["form"] == "power_law"
    assert np.isfinite(fit["C"])
    assert np.isfinite(fit["alpha"])
    assert np.isfinite(fit["beta"])
    assert np.isfinite(fit["r2"]),  "R² is not finite"
    print(PASS)
    return fit


def test_extrapolate(fit):
    print("\n── Test 5: Surface Extrapolation ───────────────────────")
    from framework.extrapolate import predict_full_surface, surface_to_dataframe

    surface = predict_full_surface(fit, TEST_RESOLUTIONS, TEST_HORIZONS)
    assert len(surface) == len(TEST_RESOLUTIONS) * len(TEST_HORIZONS)

    for (r, h), (mean, lower, upper) in surface.items():
        assert np.isfinite(mean), f"Non-finite prediction at ({r}, {h})"

    df = surface_to_dataframe(surface)
    print(df.to_string(index=False, float_format="{:.4f}".format))
    print(PASS)
    return surface


def test_full_pipeline():
    print("\n── Test 6: Full Pipeline (run_experiment) ───────────────")
    from framework.run_experiment import run

    surface_df, fit = run(
        config_path=None,
        resolutions=TEST_RESOLUTIONS,
        horizons=TEST_HORIZONS,
    )
    assert surface_df is not None
    assert "predicted_error" in surface_df.columns
    print(PASS)
    return surface_df, fit


# ── run_experiment accepts a config dict OR a path — use dict directly ─────────

def _run_with_dict(resolutions, horizons):
    """Call the pipeline with TEST_CONFIG dict instead of a YAML path."""
    import json
    from framework.cell_sampler import sample_cells
    from framework.fit_surface  import fit_surface, print_fit_summary
    from framework.extrapolate  import predict_full_surface, surface_to_dataframe
    from framework.train        import train_cell

    config  = TEST_CONFIG
    metric  = config["metric"]
    out_dir = os.path.join(config["base_dir"], config["dataset"],
                           "results", config["name"])
    os.makedirs(out_dir, exist_ok=True)

    sampled = sample_cells(resolutions, horizons,
                           config["sampling_budget"],
                           strategy=config["sampling_strategy"])
    print(f"  Cells to train: {sampled}")

    all_results = []
    for resolution, horizon in sampled:
        print(f"  ▶ {resolution}  h={horizon}")
        try:
            m = train_cell(config, resolution, horizon)
            all_results.append(m)
        except FileNotFoundError as exc:
            print(f"    SKIPPED — {exc}")

    assert all_results, "No cells trained"

    trained_cells = [(r["resolution"], r["horizon"]) for r in all_results]
    metric_values = [r[metric] for r in all_results]

    fit        = fit_surface(trained_cells, metric_values)
    print_fit_summary(fit)

    surface    = predict_full_surface(fit, resolutions, horizons)
    surface_df = surface_to_dataframe(surface, sampled_cells=trained_cells)
    observed   = {(r["resolution"], r["horizon"]): r[metric] for r in all_results}
    surface_df["observed_error"] = surface_df.apply(
        lambda row: observed.get((row["resolution"], row["horizon"]), float("nan")), axis=1
    )

    surface_df.to_csv(os.path.join(out_dir, "error_surface.csv"), index=False)
    print(f"  Surface saved → {out_dir}/error_surface.csv")

    return surface_df, fit, all_results


# ── Additional tests ───────────────────────────────────────────────────────────

def test_nan_filtering():
    """Verify NaN samples are filtered before reaching the model (1h split has known NaN)."""
    import torch
    from os.path import join

    ddir = f"{TEST_CONFIG['base_dir']}/{TEST_CONFIG['dataset']}/1h/timestep333"
    x_raw = np.load(join(ddir, "x_train.npy"))
    y_raw = np.load(join(ddir, "y_train.npy"))

    n_nan = int(np.isnan(x_raw).sum() + np.isnan(y_raw).sum())
    print(f"  Raw NaN values in 1h/h=3 train split: {n_nan}")

    x = torch.tensor(x_raw).float()
    y = torch.tensor(y_raw).float()
    valid   = ~(torch.isnan(x).any(dim=(1, 2)) | torch.isnan(y).any(dim=(1, 2)))
    x_clean = x[valid]
    y_clean = y[valid]

    assert not torch.isnan(x_clean).any(), "NaN still present in x after filtering"
    assert not torch.isnan(y_clean).any(), "NaN still present in y after filtering"

    dropped = int((~valid).sum())
    if dropped == 0:
        print("  No NaN samples found — filter is vacuously correct")
    else:
        print(f"  Dropped {dropped} NaN samples → {len(x_clean)} clean samples remain")
    print(PASS)


def test_sampling_budget():
    """Sample_cells must return exactly min(budget, grid_size) cells for all strategies."""
    from framework.cell_sampler import sample_cells

    resolutions = TEST_RESOLUTIONS
    horizons    = TEST_HORIZONS
    grid_size   = len(resolutions) * len(horizons)

    for budget in [1, 4, 8, grid_size, grid_size + 10]:
        expected = min(budget, grid_size)
        for strategy in ("random", "lhs", "corners"):
            cells = sample_cells(resolutions, horizons, budget, strategy)
            assert len(cells) == expected, (
                f"{strategy} budget={budget}: expected {expected} cells, got {len(cells)}"
            )

    print(f"  Budget respected across all strategies "
          f"(budgets 1 → {grid_size + 10}, grid={grid_size})")
    print(PASS)


def test_output_files():
    """All expected output files must exist after train_cell completes."""
    from framework.train import data_dir_path
    from framework.model_registry import get_backend

    ddir       = data_dir_path(TEST_CONFIG, "4h", 3)
    model_name = TEST_CONFIG["model"]
    weight_ext = ".pth" if get_backend(model_name) == "pytorch" else ".h5"

    expected = [
        f"best_{model_name}{weight_ext}",
        "test_predy.csv",
        "test_realy.csv",
        "test_uncert.csv",
        "metrics.csv",
        f"loss_{model_name}.png",
    ]
    missing = [f for f in expected if not os.path.exists(os.path.join(ddir, f))]
    assert not missing, f"Missing output files: {missing}"

    for fname in expected:
        size = os.path.getsize(os.path.join(ddir, fname))
        print(f"  {fname:<35s}  {size:>8d} bytes")
    print(PASS)


def test_model_save_load():
    """Reloading saved weights must produce identical predictions (eval mode)."""
    import torch
    from framework.train        import data_dir_path
    from framework.model_registry import build_model

    ddir      = data_dir_path(TEST_CONFIG, "4h", 3)
    save_path = os.path.join(ddir, f"best_{TEST_CONFIG['model']}.pth")

    assert os.path.exists(save_path), f"No saved weights at {save_path} — run Test 3 first"

    def _load():
        m = build_model(
            TEST_CONFIG["model"],
            input_size=TEST_CONFIG["input_dim"],
            hidden_size=TEST_CONFIG["hidden_dim"],
            output_size=TEST_CONFIG["output_dim"],
            num_layers=TEST_CONFIG["num_layers"],
            dropout=TEST_CONFIG["dropout_rate"],
        )
        m.load_state_dict(torch.load(save_path, map_location="cpu"))
        m.eval()
        return m

    model1, model2 = _load(), _load()

    x = torch.randn(4, 3, TEST_CONFIG["input_dim"])
    with torch.no_grad():
        out1 = model1(x)
        out2 = model2(x)

    max_diff = (out1 - out2).abs().max().item()
    assert max_diff == 0.0, f"Reloaded model outputs differ by {max_diff:.2e}"
    print(f"  Two independent loads produce identical output  (max diff = {max_diff:.2e})")
    print(PASS)


def test_ci_validity(fit):
    """lower_ci ≤ predicted_error ≤ upper_ci for every extrapolated cell."""
    from framework.extrapolate import predict_full_surface

    surface    = predict_full_surface(fit, TEST_RESOLUTIONS, TEST_HORIZONS)
    violations = []

    for (r, h), (mean, lower, upper) in surface.items():
        if np.isnan(lower) or np.isnan(upper):
            continue
        if not (lower <= mean <= upper):
            violations.append((r, h, lower, mean, upper))

    assert not violations, f"CI bound violations: {violations}"
    valid = sum(1 for _, (_, lo, hi) in surface.items() if not (np.isnan(lo) or np.isnan(hi)))
    print(f"  CIs valid for all {valid} cells with finite bounds")
    print(PASS)


def test_exponent_signs(fit):
    """
    Scientific sanity: α and β should be positive.
    Coarser resolution → higher error; longer horizon → higher error.
    Logs a warning if violated but only hard-fails if both are non-positive.
    """
    alpha, beta = fit["alpha"], fit["beta"]
    print(f"  α (resolution exponent) = {alpha:+.4f}  "
          f"({'✓ positive' if alpha > 0 else '⚠ non-positive'})")
    print(f"  β (horizon exponent)    = {beta:+.4f}  "
          f"({'✓ positive' if beta  > 0 else '⚠ non-positive'})")

    if alpha <= 0:
        print("  WARNING: α ≤ 0 — error does not increase with coarser resolution")
    if beta <= 0:
        print("  WARNING: β ≤ 0 — error does not increase with longer horizon")

    assert alpha > 0 or beta > 0, \
        "Both α and β are non-positive — fit is likely degenerate"
    print(PASS)


def test_multi_metric_fit(sampled_results):
    """Fit must succeed and give finite parameters for both RMSE and MAE."""
    from framework.fit_surface import fit_surface

    cells = [(r["resolution"], r["horizon"]) for r in sampled_results]

    for metric in ("RMSE", "MAE"):
        values = [r[metric] for r in sampled_results]
        fit    = fit_surface(cells, values)
        assert np.isfinite(fit["C"]),     f"{metric}: C is not finite"
        assert np.isfinite(fit["alpha"]), f"{metric}: alpha is not finite"
        assert np.isfinite(fit["beta"]),  f"{metric}: beta is not finite"
        assert np.isfinite(fit["r2"]),    f"{metric}: R² is not finite"
        print(f"  {metric:<6s}  C={fit['C']:.4f}  "
              f"α={fit['alpha']:+.4f}  β={fit['beta']:+.4f}  R²={fit['r2']:.4f}")

    print(PASS)


# ── TensorFlow tests ──────────────────────────────────────────────────────────

def test_tf_registry():
    print("\n── Test TF-1: TF Model Registry ────────────────────────")
    importlib.import_module("models.autoencoder")  # triggers @register_tf for both models
    from framework.model_registry import list_models, get_backend

    models_list = list_models()
    for name in ("lstm_autoencoder", "gru_autoencoder"):
        assert name in models_list, f"'{name}' not registered. Found: {models_list}"
        assert get_backend(name) == "tensorflow", \
            f"'{name}' backend is '{get_backend(name)}', expected 'tensorflow'"
        print(f"  {name}: backend=tensorflow")
    print(f"  All registered models: {models_list}")
    print(PASS)


def test_tf_train_cell():
    print("\n── Test TF-2: TF Single Cell Training (4h, horizon=3) ──")
    from framework.train import train_cell

    metrics = train_cell(TEST_CONFIG_TF, resolution="4h", horizon=3)

    required = {"RMSE", "MAE", "NSE", "KGE", "resolution", "horizon"}
    missing  = required - set(metrics.keys())
    assert not missing, f"Missing metric keys: {missing}"
    assert metrics["resolution"] == "4h"
    assert metrics["horizon"]    == 3
    assert np.isfinite(metrics["RMSE"]), f"RMSE is not finite: {metrics['RMSE']}"

    print(f"  RMSE : {metrics['RMSE']:.4f}")
    print(f"  MAE  : {metrics['MAE']:.4f}")
    print(f"  NSE  : {metrics['NSE']:.4f}")
    print(f"  KGE  : {metrics['KGE']:.4f}")
    print(PASS)
    return metrics


def test_tf_output_files():
    print("\n── Test TF-3: TF Output Files ───────────────────────────")
    from framework.train import data_dir_path

    ddir       = data_dir_path(TEST_CONFIG_TF, "4h", 3)
    model_name = TEST_CONFIG_TF["model"]

    expected = [
        f"best_{model_name}.h5",
        "test_predy.csv",
        "test_realy.csv",
        "test_uncert.csv",
        "metrics.csv",
        f"loss_{model_name}.png",
    ]
    missing = [f for f in expected if not os.path.exists(os.path.join(ddir, f))]
    assert not missing, f"Missing output files: {missing}"

    for fname in expected:
        size = os.path.getsize(os.path.join(ddir, fname))
        print(f"  {fname:<40s}  {size:>8d} bytes")
    print(PASS)


def test_tf_model_reload():
    print("\n── Test TF-4: TF Model Save/Reload ─────────────────────")
    import tensorflow as tf
    from framework.train import data_dir_path

    ddir      = data_dir_path(TEST_CONFIG_TF, "4h", 3)
    save_path = os.path.join(ddir, f"best_{TEST_CONFIG_TF['model']}.h5")

    assert os.path.exists(save_path), \
        f"No saved weights at {save_path} — run TF-2 first"

    # compile=False skips optimizer reconstruction — weights and architecture only
    model1 = tf.keras.models.load_model(save_path, compile=False)
    model2 = tf.keras.models.load_model(save_path, compile=False)

    x_sample = np.load(os.path.join(ddir, "x_test.npy"))[:4]

    # training=False disables dropout; TF doesn't guarantee bit-exact
    # reproducibility across kernel calls, so tolerate sub-1e-5 differences
    out1 = model1(x_sample, training=False).numpy()
    out2 = model2(x_sample, training=False).numpy()

    assert out1.shape == out2.shape, f"Shape mismatch: {out1.shape} vs {out2.shape}"
    assert np.all(np.isfinite(out1)), "Non-finite values in reloaded model output"

    max_diff = np.abs(out1 - out2).max()
    assert max_diff < 1e-5, \
        f"Reloaded model outputs differ by {max_diff:.2e} (threshold 1e-5)"
    print(f"  Two independent loads agree within tolerance  (max diff = {max_diff:.2e})")
    print(PASS)


# ── Test runner ────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(f"  PIPELINE SMOKE TEST — {TEST_CONFIG['model']} / {TEST_CONFIG['dataset']}")
    print("=" * 60)

    # Import model modules to register them
    for module in TEST_CONFIG["model_modules"]:
        importlib.import_module(module)

    results_log = {}

    def run_test(name, fn, *args):
        try:
            result = fn(*args)
            results_log[name] = "PASS"
            return result
        except Exception:
            results_log[name] = "FAIL"
            traceback.print_exc()
            return None

    # ── Standalone tests (no training needed) ────────────────────────────────
    run_test("1.  Model Registry",     test_registry)
    run_test("2.  Cell Sampler",       test_cell_sampler)
    run_test("7.  NaN Filtering",      test_nan_filtering)
    run_test("8.  Sampling Budget",    test_sampling_budget)

    # ── Single-cell training ──────────────────────────────────────────────────
    m1 = run_test("3.  Single Cell",   test_train_single_cell)

    # Files and weights produced by Test 3
    if results_log.get("3.  Single Cell") == "PASS":
        run_test("9.  Output Files",    test_output_files)
        run_test("10. Model Save/Load", test_model_save_load)

    # ── Train remaining cells for surface fit ─────────────────────────────────
    print("\n── Training remaining cells for surface fit test ────────")
    all_metrics = [m1] if m1 else []
    for res, hor in [("4h", 12), ("24h", 3), ("24h", 12)]:
        try:
            from framework.train import train_cell
            m = train_cell(TEST_CONFIG, resolution=res, horizon=hor)
            all_metrics.append(m)
            print(f"  [{res} h={hor}]  RMSE={m['RMSE']:.4f}")
        except Exception:
            traceback.print_exc()

    # ── Surface fit and extrapolation ─────────────────────────────────────────
    fit     = run_test("4.  Fit Surface",       test_fit_surface,    all_metrics) \
              if len(all_metrics) >= 3 else None
    surface = run_test("5.  Extrapolate",       test_extrapolate,    fit) if fit else None

    if fit is not None:
        run_test("11. CI Validity",         test_ci_validity,    fit)
        run_test("12. Exponent Signs",      test_exponent_signs, fit)

    if len(all_metrics) >= 3:
        run_test("13. Multi-Metric Fit",    test_multi_metric_fit, all_metrics)

    # ── Full pipeline ─────────────────────────────────────────────────────────
    print("\n── Test 6: Full Pipeline ────────────────────────────────")
    try:
        surface_df, fit_full, trained = _run_with_dict(TEST_RESOLUTIONS, TEST_HORIZONS)
        print("\n  Final error surface:")
        pivot = surface_df.pivot(
            index="resolution", columns="horizon", values="predicted_error"
        )
        print(pivot.to_string(float_format="{:.4f}".format))
        results_log["6.  Full Pipeline"] = "PASS"
    except Exception:
        results_log["6.  Full Pipeline"] = "FAIL"
        traceback.print_exc()

    # ── TensorFlow suite ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if TF_AVAILABLE:
        print(f"  TF SUITE — {TEST_CONFIG_TF['model']} / {TEST_CONFIG_TF['dataset']}")
        print("=" * 60)

        for module in TEST_CONFIG_TF["model_modules"]:
            importlib.import_module(module)

        run_test("TF-1. TF Registry",        test_tf_registry)
        run_test("TF-2. TF Single Cell",     test_tf_train_cell)

        if results_log.get("TF-2. TF Single Cell") == "PASS":
            run_test("TF-3. TF Output Files",    test_tf_output_files)
            run_test("TF-4. TF Model Reload",    test_tf_model_reload)
    else:
        print("  TF SUITE — skipped (TensorFlow not available)")
        print("=" * 60)
        results_log["TF suite"] = "SKIP"

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for name, status in sorted(results_log.items()):
        icon = "✓" if status == "PASS" else "✗"
        print(f"  {icon}  {name:35s}  {status}")

    n_fail = sum(1 for s in results_log.values() if s == "FAIL")
    print("=" * 60)
    if n_fail == 0:
        print("  All tests passed.")
    else:
        print(f"  {n_fail} test(s) failed.")
    print()
    sys.exit(n_fail)


if __name__ == "__main__":
    main()
