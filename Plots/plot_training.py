"""
plot_training.py

Plot per-episode training diagnostics for a single model run.

"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SMOOTH = 20

#  directories ---------
ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_DIR = os.path.join(ROOT, "runs", "lt6", "lstm")
# RUN_DIR = os.path.join(ROOT, "runs", "lt10", "lstm") # Set to your experiment

CSV      = os.path.join(RUN_DIR, "monitor.monitor.csv")
EVAL_NPZ = os.path.join(RUN_DIR, "evaluations.npz")
OUT      = os.path.join(RUN_DIR, "training_diagnostics.png")

#  Load data ─

df        = pd.read_csv(CSV, skiprows=1)
timesteps = df["l"].cumsum().values

eval_data = None
if os.path.exists(EVAL_NPZ):
    d = np.load(EVAL_NPZ)
    eval_data = {
        "timesteps": d["timesteps"],
        "means":     d["results"].mean(axis=1),
        "stds":      d["results"].std(axis=1),
        "mins":      d["results"].min(axis=1),
        "maxs":      d["results"].max(axis=1),
    }

has_diag = "ep_service_level" in df.columns

#  Helpers ─

def smooth(arr, w):
    if len(arr) < w:
        return arr, np.arange(len(arr))
    return np.convolve(arr, np.ones(w) / w, mode="valid"), np.arange(w - 1, len(arr))

def plot_series(ax, ts, raw, label, color, ylabel, w=SMOOTH, hline=None):
    sm, sx = smooth(raw, w)
    ax.plot(ts, raw, alpha=0.15, color=color, linewidth=0.7)
    ax.plot(ts[sx], sm, color=color, linewidth=2, label=label)
    if hline is not None:
        ax.axhline(hline, color="grey", linewidth=0.6, linestyle="--")
    ax.set_xlabel("Timestep")
    ax.set_ylabel(ylabel)

#  Figure

nrows = 2 if has_diag else 1
fig, axes = plt.subplots(nrows, 3, figsize=(18, 5 * nrows))
model_name = os.path.basename(RUN_DIR).upper()
lt_name    = os.path.basename(os.path.dirname(RUN_DIR))
fig.suptitle(
    f"Training diagnostics — {lt_name.replace('lt', 'Lead time = ')}  |  {model_name}  "
    f"({len(df):,} episodes, {int(timesteps[-1]):,} timesteps)"
)
if nrows == 1:
    axes = axes.reshape(1, -1)

#  Row 0: reward | deterministic eval | service level ─
ax = axes[0, 0]
plot_series(ax, timesteps, df["r"].values, "stochastic (training)", "steelblue",
            "Episode reward", hline=0)
ax.plot(timesteps, df["r"].expanding().mean().values,
        color="steelblue", linewidth=1.2, linestyle="--", label="running mean", alpha=0.6)
if eval_data is not None:
    ax.plot(eval_data["timesteps"], eval_data["means"],
            color="crimson", linewidth=2.5, label="deterministic eval", zorder=5)
    ax.fill_between(eval_data["timesteps"], eval_data["mins"], eval_data["maxs"],
                    color="crimson", alpha=0.12, label="eval min/max")
ax.set_title("Episode reward: stochastic vs deterministic")
ax.legend(fontsize=8)

ax = axes[0, 1]
if eval_data is not None:
    ax.plot(eval_data["timesteps"], eval_data["means"],
            color="crimson", linewidth=2.5, label="mean")
    ax.fill_between(eval_data["timesteps"],
                    eval_data["means"] - eval_data["stds"],
                    eval_data["means"] + eval_data["stds"],
                    color="crimson", alpha=0.2, label="±1 std")
    ax.fill_between(eval_data["timesteps"], eval_data["mins"], eval_data["maxs"],
                    color="crimson", alpha=0.08, label="min/max")
    best_idx = np.argmax(eval_data["means"])
    ax.axvline(eval_data["timesteps"][best_idx], color="gold", linewidth=1.5,
               linestyle="--", label=f"best @ {eval_data['timesteps'][best_idx]:,}")
    ax.axhline(eval_data["means"][best_idx], color="gold", linewidth=1, linestyle=":")
    ax.axhline(0, color="grey", linewidth=0.6, linestyle="--")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Deterministic episode reward")
    ax.set_title(f"Deterministic eval  (best mean = {eval_data['means'][best_idx]:.1f})")
    ax.legend(fontsize=8)
else:
    ax.text(0.5, 0.5, "evaluations.npz not found", transform=ax.transAxes,
            ha="center", va="center", color="grey")
    ax.set_title("Deterministic eval")

if has_diag:
    plot_series(axes[0, 2], timesteps, df["ep_service_level"].values,
                "service level", "seagreen", "Service level (sold / demand)", hline=1.0)
    axes[0, 2].set_title("Service level")
    axes[0, 2].legend(fontsize=8)
else:
    axes[0, 2].set_visible(False)

#  Row 1: mean inventory | orders vs demand | stockout fraction
if has_diag:
    plot_series(axes[1, 0], timesteps, df["ep_mean_inventory"].values,
                "mean inventory", "darkorange", "Units")
    axes[1, 0].set_title("Mean on-hand inventory per episode")
    axes[1, 0].legend(fontsize=8)

    ax = axes[1, 1]
    sm_ord, sx_ord = smooth(df["ep_orders"].values, SMOOTH)
    sm_dem, sx_dem = smooth(df["ep_demand"].values, SMOOTH)
    ax.plot(timesteps, df["ep_orders"].values, alpha=0.12, color="royalblue", linewidth=0.7)
    ax.plot(timesteps, df["ep_demand"].values, alpha=0.12, color="tomato",    linewidth=0.7)
    ax.plot(timesteps[sx_ord], sm_ord, color="royalblue", linewidth=2, label="orders")
    ax.plot(timesteps[sx_dem], sm_dem, color="tomato",    linewidth=2, label="demand")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Units per episode")
    ax.set_title("Total orders vs total demand per episode")
    ax.legend(fontsize=8)

    stockout_frac = df["ep_stockout_steps"].values / df["l"].values
    plot_series(axes[1, 2], timesteps, stockout_frac,
                "stockout fraction", "mediumpurple", "Fraction of steps", hline=0)
    axes[1, 2].set_title("Stockout fraction (steps with lost sales / episode length)")
    axes[1, 2].legend(fontsize=8)

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved: {os.path.abspath(OUT)}")
os.startfile(os.path.abspath(OUT))
