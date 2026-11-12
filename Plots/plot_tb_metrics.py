"""
plot_tb_metrics.py

Read TensorBoard event files from all runs and produce a grid of PPO health
metrics (explained_variance, approx_kl, clip_fraction, entropy_loss,
value_loss, policy_gradient_loss).

"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tbparse import SummaryReader

RUNS_DIR = "../runs"
OUT      = os.path.join(RUNS_DIR, "tb_metrics.png")

METRICS = [
    ("train/explained_variance",    "Explained variance",    "seagreen"),
    ("train/approx_kl",             "Approx KL divergence",  "steelblue"),
    ("train/clip_fraction",         "Clip fraction",         "darkorange"),
    ("train/entropy_loss",          "Entropy loss",          "mediumpurple"),
    ("train/value_loss",            "Value loss",            "crimson"),
    ("train/policy_gradient_loss",  "Policy gradient loss",  "saddlebrown"),
]

MODEL_STYLES = {
    "lstm": dict(linestyle="-",  linewidth=2,   label="LSTM"),
    "mlp":  dict(linestyle="--", linewidth=1.8, label="MLP"),
}


TB_PREFIX = {"lstm": "RecurrentPPO", "mlp": "PPO"}

def find_tb_dir(lt_path, model_type):
    """Return the TensorBoard event directory for a given lt and model type."""
    prefix = TB_PREFIX[model_type]
    candidates = sorted(
        [e for e in os.listdir(lt_path)
         if e.startswith(prefix) and os.path.isdir(os.path.join(lt_path, e))]
    )
    if candidates:
        return os.path.join(lt_path, candidates[-1])  # take most recent if multiple
    return None


def load_scalar(tb_dir, tag):
    """Return (steps, values) arrays for a scalar tag from a TensorBoard dir."""
    try:
        reader = SummaryReader(tb_dir, pivot=False, extra_columns=set())
        df = reader.scalars
        df = df[df["tag"] == tag].sort_values("step")
        if df.empty:
            return None, None
        return df["step"].values, df["value"].values
    except Exception:
        return None, None


#  Collect data

lt_dirs = sorted(
    [d for d in os.listdir(RUNS_DIR) if d.startswith("lt") and
     os.path.isdir(os.path.join(RUNS_DIR, d))],
    key=lambda d: int(d.replace("lt", ""))
)

if not lt_dirs:
    print("No runs/lt* directories found.")
    raise SystemExit

#  Plot

n_metrics = len(METRICS)
n_lt      = len(lt_dirs)

fig, axes = plt.subplots(n_metrics, n_lt,
                          figsize=(6 * n_lt, 3.5 * n_metrics),
                          sharex=False, sharey="row")

if n_lt == 1:
    axes = axes.reshape(-1, 1)

fig.suptitle("PPO Training Health Metrics — LSTM vs MLP across lead times", fontsize=14)

for col, lt_name in enumerate(lt_dirs):
    lt_path = os.path.join(RUNS_DIR, lt_name)
    lt_val  = lt_name.replace("lt", "")

    for row, (tag, label, color) in enumerate(METRICS):
        ax = axes[row, col]

        plotted_any = False
        for model_type in ["lstm", "mlp"]:
            tb_dir = find_tb_dir(lt_path, model_type)
            if tb_dir is None:
                continue

            steps, vals = load_scalar(tb_dir, tag)
            if steps is None or len(steps) == 0:
                continue

            style    = MODEL_STYLES[model_type]
            line_col = color if model_type == "lstm" else "grey"
            ax.plot(steps / 1_000_000, vals,
                    color=line_col, alpha=0.2, linewidth=0.7)

            w = max(1, len(vals) // 40)
            if len(vals) >= w * 2:
                kernel   = np.ones(w) / w
                smoothed = np.convolve(vals, kernel, mode="valid")
                sx       = steps[w - 1:] / 1_000_000
                ax.plot(sx, smoothed, color=line_col,
                        linestyle=style["linestyle"],
                        linewidth=style["linewidth"],
                        label=style["label"])
            else:
                ax.plot(steps / 1_000_000, vals, color=line_col,
                        linestyle=style["linestyle"],
                        linewidth=style["linewidth"],
                        label=style["label"])
            plotted_any = True

        if not plotted_any:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center", color="grey", fontsize=9)

        if row == 0:
            ax.set_title(f"Lead time = {lt_val}", fontsize=11, fontweight="bold")
        if col == 0:
            ax.set_ylabel(label, fontsize=9)
        if row == n_metrics - 1:
            ax.set_xlabel("Timestep (millions)", fontsize=9)

        ax.tick_params(labelsize=8)
        if plotted_any and row == 0 and col == 0:
            ax.legend(fontsize=8)

handles = [
    plt.Line2D([0], [0], color="dimgrey", linestyle="-",  linewidth=2, label="LSTM"),
    plt.Line2D([0], [0], color="dimgrey", linestyle="--", linewidth=1.8, label="MLP"),
]
fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=10,
           bbox_to_anchor=(0.5, -0.01))

plt.tight_layout(rect=[0, 0.02, 1, 0.97])
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved: {os.path.abspath(OUT)}")
os.startfile(os.path.abspath(OUT))
