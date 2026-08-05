import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from pathlib import Path
from os.path import join, dirname, abspath


# ── configuration ─────────────────────────────────────────────────────────────

# Graph WaveNet fixed sequence parameters — same for all resolutions
SEQ_LENGTH_X = 12    # input window length
SEQ_LENGTH_Y = 12    # forecast horizon (steps)
SHIFT        = 12    # shift step

# Resolutions and stations
RESOLUTIONS = ["1h", "4h", "8h", "12h", "24h"]
RES_HOURS   = {"1h": 1, "4h": 4, "8h": 8, "12h": 12, "24h": 24}
STATIONS    = [0, 1, 2, 3]

# Peak detection — tune to your discharge data
PEAK_PROMINENCE  = 0.15   # fraction of signal range
PEAK_DISTANCE    = 5      # minimum timesteps between peaks
MAX_MATCH_OFFSET = 15     # maximum timestep offset to match peaks
MIN_PEAKS        = 3      # skip if fewer peaks detected

# Output
current_dir = dirname(abspath(__file__))
base_dir    = dirname(current_dir)
RESULTS_DIR = Path(join(base_dir, "results", "phase_error"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── loading ───────────────────────────────────────────────────────────────────

def load_files(freq: str):
    """Load predicted and actual files for a given resolution."""
    folder   = join(base_dir, "data", freq,
                    f"timestep{SEQ_LENGTH_X}{SEQ_LENGTH_Y}{SHIFT}")
    pred_path = join(folder, "test_predy.csv")
    real_path = join(folder, "test_realy.csv")

    if not Path(pred_path).exists() or not Path(real_path).exists():
        print(f"  [skip] missing files for freq={freq} in {folder}")
        return None, None

    pred = pd.read_csv(pred_path, index_col=0, parse_dates=True)
    real = pd.read_csv(real_path, index_col=0, parse_dates=True)
    pred.columns = [int(c) for c in pred.columns]
    real.columns = [int(c) for c in real.columns]
    return pred, real


# ── peak detection ────────────────────────────────────────────────────────────

def detect_peaks_in_series(series: pd.Series) -> np.ndarray:
    """
    Detect peaks scaled to the signal range so it adapts across
    resolutions and stations with different discharge magnitudes.
    """
    values    = series.values
    sig_range = values.max() - values.min()
    if sig_range == 0:
        return np.array([], dtype=int)
    prom_abs = PEAK_PROMINENCE * sig_range
    peaks, _ = find_peaks(values, prominence=prom_abs, distance=PEAK_DISTANCE)
    return peaks


def match_peaks(actual_idx, pred_idx):
    """
    Greedy nearest-neighbour peak matching within MAX_MATCH_OFFSET steps.
    Returns list of (actual_pos, pred_pos, offset) where:
      offset > 0 → predicted peak delayed (rightward / lag)
      offset < 0 → predicted peak early   (leftward)
      offset = 0 → on time
    """
    matched = []
    used    = set()
    for a in actual_idx:
        candidates = [
            p for p in pred_idx
            if abs(int(p) - int(a)) <= MAX_MATCH_OFFSET and p not in used
        ]
        if not candidates:
            continue
        best   = min(candidates, key=lambda p: abs(int(p) - int(a)))
        offset = int(best) - int(a)
        matched.append((int(a), int(best), offset))
        used.add(best)
    return matched


# ── per station analysis ──────────────────────────────────────────────────────

def analyse_station(freq, station, real_series, pred_series):
    """Compute phase error statistics for one resolution x station."""
    actual_peaks = detect_peaks_in_series(real_series)
    pred_peaks   = detect_peaks_in_series(pred_series)

    if len(actual_peaks) < MIN_PEAKS:
        print(f"    [skip] freq={freq} station={station}: "
              f"only {len(actual_peaks)} peaks detected")
        return None

    matched = match_peaks(actual_peaks, pred_peaks)
    if not matched:
        print(f"    [skip] freq={freq} station={station}: no matched peaks")
        return None

    res_hours     = RES_HOURS[freq]
    offsets_steps = np.array([m[2] for m in matched])
    offsets_hours = offsets_steps * res_hours

    # real-time horizon covered by seq_length_y steps
    horizon_hours = SEQ_LENGTH_Y * res_hours

    return {
        "freq":              freq,
        "res_hours":         res_hours,
        "horizon_hours":     horizon_hours,
        "station":           station,
        "n_actual_peaks":    len(actual_peaks),
        "n_predicted_peaks": len(pred_peaks),
        "n_matched":         len(matched),
        "mean_offset_steps": float(np.mean(offsets_steps)),
        "mean_offset_hours": float(np.mean(offsets_hours)),
        "median_offset_hours": float(np.median(offsets_hours)),
        "std_offset_hours":  float(np.std(offsets_hours)),
        "max_lag_hours":     float(np.max(offsets_hours)),
        "pct_lagged":        float(np.mean(offsets_steps > 0) * 100),
        "pct_early":         float(np.mean(offsets_steps < 0) * 100),
        "pct_exact":         float(np.mean(offsets_steps == 0) * 100),
        # stored for plotting
        "_matched":      matched,
        "_actual_peaks": actual_peaks,
        "_pred_peaks":   pred_peaks,
        "_real":         real_series,
        "_pred":         pred_series,
    }


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_hydrograph_samples(result, n_samples=2):
    """Zoomed hydrograph windows around matched peaks showing timing offset."""
    real    = result["_real"]
    pred    = result["_pred"]
    matched = result["_matched"]
    freq    = result["freq"]
    station = result["station"]
    res_h   = result["res_hours"]
    window  = max(PEAK_DISTANCE * 4, 10)

    n_plots = min(n_samples, len(matched))
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4), sharey=False)
    if n_plots == 1:
        axes = [axes]

    for ax, (a_idx, p_idx, offset) in zip(axes, matched[:n_plots]):
        start = max(0, a_idx - window)
        end   = min(len(real), a_idx + window + 1)
        x     = np.arange(start, end)

        ax.plot(x, real.values[start:end], color="#185FA5",
                linewidth=1.8, label="Actual")
        ax.plot(x, pred.values[start:end], color="#D85A30",
                linewidth=1.8, linestyle="--", label="Predicted")
        ax.axvline(a_idx, color="#185FA5", alpha=0.35, linewidth=1.2)
        ax.axvline(p_idx, color="#D85A30", alpha=0.35,
                   linewidth=1.2, linestyle="--")

        if offset > 0:
            direction = f"+{offset} steps (lag)"
        elif offset < 0:
            direction = f"{offset} steps (early)"
        else:
            direction = "0 steps (on time)"

        ax.set_title(
            f"freq={freq}  station={station}\n"
            f"Phase offset: {direction} = {offset * res_h:+.0f}h",
            fontsize=10
        )
        ax.set_xlabel("Timestep index")
        ax.set_ylabel("Discharge")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    out = RESULTS_DIR / f"hydrograph_{freq}_station{station}.png"
    # fig.savefig(out, dpi=150, bbox_inches="tight")
    # plt.close()
    print(f"    Saved: {out.name}")


def plot_phase_error_surface(summary):
    """
    Heatmap of mean phase error (hours) across resolution x station.
    One subplot per station so spatial differences are visible.
    """
    resolutions = [r for r in RESOLUTIONS if r in summary["freq"].values]
    res_hours   = [RES_HOURS[r] for r in resolutions]

    fig, axes = plt.subplots(1, len(STATIONS),
                             figsize=(4.5 * len(STATIONS), 4), sharey=True)
    if len(STATIONS) == 1:
        axes = [axes]

    for ax, station in zip(axes, STATIONS):
        sub = summary[summary["station"] == station].copy()
        sub = sub.set_index("freq").reindex(resolutions)
        values = sub["mean_offset_hours"].values.reshape(-1, 1)

        vmax = summary["mean_offset_hours"].abs().max()
        im   = ax.imshow(values, aspect="auto", cmap="RdBu_r",
                         vmin=-vmax, vmax=vmax)

        ax.set_yticks(range(len(resolutions)))
        ax.set_yticklabels(resolutions)
        ax.set_xticks([])
        ax.set_title(f"Station {station}", fontsize=11)
        ax.set_ylabel("Resolution" if station == STATIONS[0] else "")

        for i, v in enumerate(values.flatten()):
            if not np.isnan(v):
                txt_col = "white" if abs(v) > 0.5 * vmax else "black"
                ax.text(0, i, f"{v:.1f}h", ha="center", va="center",
                        fontsize=10, color=txt_col, fontweight="bold")

    fig.colorbar(im, ax=axes, label="Mean phase error (hours)",
                 shrink=0.8, pad=0.02)
    fig.suptitle("Mean phase error per resolution and station\n"
                 "Red = lagged  |  Blue = early",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    out = RESULTS_DIR / "phase_error_surface.png"
    # fig.savefig(out, dpi=150, bbox_inches="tight")
    # plt.close()
    print(f"  Saved: {out.name}")


def plot_phase_vs_resolution(summary):
    """
    Line plot: mean phase error (hours) vs resolution, one line per station.
    Also marks the real-time horizon covered by seq_length_y steps at each
    resolution — if the lag exceeds this the peak is outside the forecast window.
    """
    colors  = ["#185FA5", "#1D9E75", "#D85A30", "#7F77DD"]
    markers = ["o", "s", "^", "D"]

    fig, ax = plt.subplots(figsize=(8, 5))

    # shade the real-time horizon per resolution as context
    for freq, res_h in RES_HOURS.items():
        if freq in summary["freq"].values:
            hor_h = SEQ_LENGTH_Y * res_h
            ax.scatter(res_h, hor_h, marker="|", s=200,
                       color="gray", zorder=2, linewidths=1.5)

    # horizon line
    res_h_vals = [RES_HOURS[f] for f in RESOLUTIONS if f in summary["freq"].values]
    hor_vals   = [SEQ_LENGTH_Y * h for h in res_h_vals]
    ax.plot(res_h_vals, hor_vals, color="gray", linewidth=1,
            linestyle=":", label=f"Forecast window ({SEQ_LENGTH_Y} steps)")

    for (station, grp), col, mk in zip(
        summary.groupby("station"), colors, markers
    ):
        grp = grp.sort_values("res_hours")
        ax.plot(grp["res_hours"], grp["mean_offset_hours"],
                color=col, marker=mk, linewidth=1.8, markersize=7,
                label=f"Station {station}")
        ax.fill_between(
            grp["res_hours"],
            grp["mean_offset_hours"] - grp["std_offset_hours"],
            grp["mean_offset_hours"] + grp["std_offset_hours"],
            alpha=0.10, color=col,
        )

    ax.axhline(0, color="#888780", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Resolution (hours)")
    ax.set_ylabel("Mean phase error (hours)")
    ax.set_title(
        f"Phase error vs resolution  |  seq_x={SEQ_LENGTH_X}, "
        f"seq_y={SEQ_LENGTH_Y}, shift={SHIFT}\n"
        "Dotted line = real-time forecast window  |  "
        "Positive = rightward shift (lag)"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    out = RESULTS_DIR / "phase_vs_resolution.png"
    # fig.savefig(out, dpi=150, bbox_inches="tight")
    # plt.close()
    print(f"  Saved: {out.name}")


def plot_pct_lagged(summary):
    """Grouped bar chart: % lagged peaks per resolution x station."""
    resolutions = [r for r in RESOLUTIONS if r in summary["freq"].values]
    colors      = ["#185FA5", "#1D9E75", "#D85A30", "#7F77DD"]
    x           = np.arange(len(resolutions))
    width       = 0.8 / len(STATIONS)

    fig, ax = plt.subplots(figsize=(9, 4))
    for k, (station, col) in enumerate(zip(STATIONS, colors)):
        grp    = summary[summary["station"] == station]
        grp    = grp.set_index("freq").reindex(resolutions)
        offset = (k - len(STATIONS) / 2 + 0.5) * width
        ax.bar(x + offset, grp["pct_lagged"].values,
               width=width * 0.9, color=col, alpha=0.85,
               label=f"Station {station}")

    ax.axhline(50, color="#888780", linewidth=0.8,
               linestyle="--", label="50% reference")
    ax.set_xticks(x)
    ax.set_xticklabels(resolutions)
    ax.set_xlabel("Resolution")
    ax.set_ylabel("% of peaks with rightward shift")
    ax.set_title("Fraction of lagged peak predictions per resolution and station")
    ax.set_ylim(0, 110)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    out = RESULTS_DIR / "pct_lagged.png"
    # fig.savefig(out, dpi=150, bbox_inches="tight")
    # plt.close()
    print(f"  Saved: {out.name}")


# ── summary table ─────────────────────────────────────────────────────────────

def print_summary(summary):
    print("\n" + "=" * 92)
    print(f"{'PHASE ERROR SUMMARY':^92}")
    print("=" * 92)
    print(f"  {'freq':>5}  {'hor(h)':>6}  {'stn':>3}  {'peaks':>6}  "
          f"{'matched':>7}  {'mean(h)':>8}  {'med(h)':>7}  "
          f"{'std(h)':>7}  {'% lag':>6}  {'% early':>7}")
    print("-" * 92)
    for _, r in summary.sort_values(["res_hours", "station"]).iterrows():
        print(f"  {r.freq:>5}  {int(r.horizon_hours):>6}  "
              f"{int(r.station):>3}  {int(r.n_actual_peaks):>6}  "
              f"{int(r.n_matched):>7}  {r.mean_offset_hours:>8.2f}  "
              f"{r.median_offset_hours:>7.2f}  {r.std_offset_hours:>7.2f}  "
              f"{r.pct_lagged:>5.1f}%  {r.pct_early:>6.1f}%")
    print("=" * 92)

    print("\n  Key findings:")
    worst = summary.loc[summary["mean_offset_hours"].idxmax()]
    print(f"  Largest mean lag        : freq={worst.freq}, "
          f"station={int(worst.station)}, "
          f"mean={worst.mean_offset_hours:.2f}h "
          f"(forecast window={int(worst.horizon_hours)}h)")

    most  = summary.loc[summary["pct_lagged"].idxmax()]
    print(f"  Most consistently lagged: freq={most.freq}, "
          f"station={int(most.station)}, "
          f"{most.pct_lagged:.1f}% of peaks")

    # check if lag exceeds forecast window at coarse resolutions
    exceeded = summary[summary["mean_offset_hours"] >= summary["horizon_hours"]]
    if not exceeded.empty:
        print(f"\n  WARNING — mean lag exceeds forecast window at:")
        for _, r in exceeded.iterrows():
            print(f"    freq={r.freq}, station={int(r.station)}: "
                  f"lag={r.mean_offset_hours:.1f}h >= window={int(r.horizon_hours)}h")
        print("  → Peaks are systematically predicted AFTER the window closes.")

    corr = summary["res_hours"].corr(summary["mean_offset_hours"])
    print(f"\n  Correlation (resolution vs mean phase error): r = {corr:.3f}")
    if corr > 0.5:
        print("  → Phase lag increases with coarser resolution, as expected.")
    print()


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("Phase error analysis — GRU Autoencoder multivariate forecasting")
    print(f"seq_length_x={SEQ_LENGTH_X}  seq_length_y={SEQ_LENGTH_Y}  shift={SHIFT}")
    print(f"Peak prominence : {PEAK_PROMINENCE} x signal range")
    print(f"Peak distance   : {PEAK_DISTANCE} steps minimum")
    print(f"Match window    : +/-{MAX_MATCH_OFFSET} steps")
    print("=" * 65)

    all_results = []

    for freq in RESOLUTIONS:
        print(f"\nResolution = {freq}  "
              f"(forecast window = {SEQ_LENGTH_Y * RES_HOURS[freq]}h real-time)")
        pred_df, real_df = load_files(freq)
        if pred_df is None:
            continue

        for station in STATIONS:
            if station not in real_df.columns:
                print(f"    [skip] station {station} not found in data")
                continue

            result = analyse_station(
                freq=freq,
                station=station,
                real_series=real_df[station],
                pred_series=pred_df[station],
            )
            if result is None:
                continue

            plot_hydrograph_samples(result, n_samples=2)
            all_results.append(result)

    if not all_results:
        print("\nNo results — check base_dir and folder structure.")
    else:
        summary = pd.DataFrame([
            {k: v for k, v in r.items() if not k.startswith("_")}
            for r in all_results
        ])
        summary.to_csv(RESULTS_DIR / "phase_error_summary.csv", index=False)

        print_summary(summary)
        plot_phase_error_surface(summary)
        plot_phase_vs_resolution(summary)
        plot_pct_lagged(summary)

        print(f"All outputs saved to: {RESULTS_DIR}/")
