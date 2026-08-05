# Cost Efficient Estimation of Deep Learning Forecast Error Surfaces from Sparse Resolution-Horizon Sampling: A Model Agnostic Framework

**Authors: Xiao Xia Liang, Dany Lauzon, Erwan Gloaguen, Reed Maxwell**

## Overview

This model-agnostic framework recovers a forecasting model's performance. In other words, its metric error across data resolution and forecast horizon from a minimal number of training runs. Characterizing how metric error scales normally requires a full grid sweep, with a model trained and evaluated at every combination of data resolution and forecast horizon. The framework replaces that sweep with a sparse-sampling and extrapolation procedure by fitting a low-parameter scaling model to the metric errors from a small subset of trained cells, predicts the error at the remaining resolution–horizon combinations, and reports calibrated uncertainty on those estimates. As an added feature, the framework also reports the forecasting model's own predictive uncertainty via Monte Carlo Bayesian dropout at inference. Together this gives users a quick tool to assess whether their data resolution and model are adequate, with model forecast uncertainty included.

1. Samples a sparse subset of the (resolution × horizon) grid (`random`,
   `lhs`, or `corners` strategies).
2. Trains a user-registered model on each sampled cell and runs MC
   dropout inference to get a predictive mean and uncertainty.
3. Fits a power-law scaling surface `log(E) = log(C) + α·log(r) + β·log(h)`
   to the sampled errors via OLS in log-space.
4. Extrapolates the fitted surface to the full grid, with confidence
   intervals computed via the delta method.
5. Framework Feature: Monte Carlo Bayesian dropout Uncertainty per model forecast

The framework is model-agnostic: any PyTorch `nn.Module` or TensorFlow/Keras
model can be plugged in through a registry, without touching the sampling,
fitting, or extrapolation code.

## Repository structure

```
src/
├── framework/
│   ├── cell_sampler.py          # (resolution, horizon) sampling strategies
│   ├── model_registry.py        # @register / @register_tf decorators
│   ├── train_one_cell.py        # trains one cell, runs MC dropout, saves outputs
│   ├── fit_surface.py           # power-law OLS fit in log-space
│   ├── extrapolate.py           # predicts the full surface + confidence intervals
│   ├── run_experiment.py        # orchestrates steps 1-4 end to end
│   ├── get_fit_surface.py       # fits + extrapolates from already-trained cells
│   ├── collect_observed_grid.py # pulls observed metrics for every trained cell in a grid
│   └── plot_uncertainty.py      # plots predicted mean +/- MC dropout uncertainty
├── configs/                # example experiment configs
├── models/                 # user-registered model definitions (e.g. lstm.py)
└── legacy/                 # earlier single-model scripts, kept for reference
```

## Installation

```
pip install numpy pandas matplotlib scipy pyyaml permetrics torch
```

Install `tensorflow` as well if you register a TensorFlow/Keras model.

## Usage

### 1. Define an experiment config

```yaml
# configs/experiment.yaml
base_dir: /path/to/base_dir
dataset: sw_data  # gw_unconfined_data | gw_confined_data | sw_data | karst_data
...
```

`dataset` selects which data subfolder under `base_dir` to use — swap it to
point the same config at a different dataset without duplicating the file.

```yaml
# configs/test_experiment_sw.yaml -> This is the test script
```

### 2. Register a model

```python
# models/lstm.py
from framework.model_registry import register

@register("lstm")
class BayesianLSTM(nn.Module):
    ...
```

### 3. Run the full pipeline

```
python framework/run_experiment.py \
    --config configs/experiment.yaml \
    --resolutions 1h 4h 8h 12h 24h \
    --horizons 3 6 9 12
```

This samples cells, trains the registered model on each, fits the scaling
surface, and writes `fit_summary.json` and `error_surface.csv` to
`base_dir/dataset/results/<name>/`.

### 4. Train or plot a single cell

```
python framework/train_one_cell.py --config configs/experiment.yaml --resolution 4h --horizon 3
python framework/plot_uncertainty.py --config configs/experiment.yaml --resolution 4h --horizon 3

Run this a few more times with different resolution and horizon (suggestion 4h:12, 12h:3, 12h:12)
This will create extrapolation points for the power law fitting
```
### 5. Fit and extrapolate from already-trained cells

If you've already trained some cells (step 4) and just want to fit the
scaling surface and extrapolate to a wider grid without rerunning training:

```
python framework/get_fit_surface.py \
        --config configs/experiment.yaml \
        --cells  4h:3 4h:12 12h:3 12h:12 \
        --metric RMSE \
        --resolutions 4h 8h 12h 24h \
        --horizons 3 6 9 12
```

`--metric` picks which column of `metrics.csv` to fit on (default `RMSE`).
`--resolutions`/`--horizons` define the grid to extrapolate over and default
to the resolutions/horizons already present in `--cells` if omitted. Results
are written to `base_dir/dataset/results/<name>/`:

- `fit_cells.csv` — the input cells and their metric values
- `fit_params.csv` — fitted `C`, `alpha`, `beta`, SE, CI, R²
- `error_surface.csv` — extrapolated metric with confidence intervals across
  the full requested grid

### 6. Collect observed metrics across the full grid

To pull the raw (non-extrapolated) observed error metrics for every cell
already trained within a grid — e.g. for plotting alongside the fitted
surface:

```
python framework/collect_observed_grid.py \
    --config configs/experiment.yaml \
    --resolutions 1h 4h 8h 12h 24h \
    --horizons 3 6 9 12
```

For each `(resolution, horizon)` in the grid, it reads that cell's
`metrics.csv` if the cell has been trained (`NaN` and `trained=False`
otherwise), prints a summary, and saves `observed_grid.csv` — one row per
grid cell — to `base_dir/dataset/`.

## License

MIT — see [LICENSE](LICENSE).
