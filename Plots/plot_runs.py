"""
plot_runs.py
────────────────────────────────────────────────────────────────────────────────
Load RL rewards from eval_results.json files, re-evaluate the (s,S) baseline
live with the correct post-heatmap parameters, and produce comparison plots.

Run: python Plots/plot_runs.py
Saves:
    runs/comparison_boxplot.png
    runs/comparison_bar.png
    runs/comparison_vs_leadtime.png
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from environment.inventory import InventoryEnv
from environment.demand import RegimeSwitchingDemand
from environment.config import ENV_CONFIG
from models.sS_policy import sSPolicy


SS_PARAMS = {
    0:  (15,  30),
    3:  (85,  95),
    6:  (140, 150),
    10: (65,  75),
}

LAMBDAS        = [5, 25]
PERSISTENCE    = "medium"
EPISODE_LENGTH = ENV_CONFIG["episode_length"]
EVAL_SEED      = 42
N_EVAL         = 30

RUNS_DIR = os.path.join(ROOT, "runs")
OUT_BOX  = os.path.join(RUNS_DIR, "comparison_boxplot.png")
OUT_BAR  = os.path.join(RUNS_DIR, "comparison_bar.png")
OUT_LINE = os.path.join(RUNS_DIR, "comparison_vs_leadtime.png")

COLOURS = {"ss": "#4C72B0", "mlp": "#55A868", "lstm": "#DD8452"}
LABELS  = {"ss": "sS (optimal)", "mlp": "SB3 MLP", "lstm": "SB3 LSTM"}

# Load RL rewards from JSON

def load_rl_results():
    records = {}
    for lt_dir in sorted(os.listdir(RUNS_DIR),
                         key=lambda d: int(d.replace("lt", "")) if d.startswith("lt") else -1):
        if not lt_dir.startswith("lt"):
            continue
        lt = int(lt_dir.replace("lt", ""))
        records[lt] = {}
        for model in ["lstm", "mlp"]:
            path = os.path.join(RUNS_DIR, lt_dir, model, "eval_results.json")
            if os.path.exists(path):
                with open(path) as f:
                    d = json.load(f)
                records[lt][model] = d["model"]["rewards"]
    return records

# Re-evaluate sS live

def eval_ss(lt, s, S):
    env = InventoryEnv(
        demand_model=RegimeSwitchingDemand(PERSISTENCE, LAMBDAS),
        lead_time=lt,
        episode_length=EPISODE_LENGTH,
    )
    policy  = sSPolicy(s=s, S=S)
    rewards = []
    for ep in range(N_EVAL):
        np.random.seed(EVAL_SEED + ep)
        obs, _ = env.reset(seed=EVAL_SEED + ep)
        total, done = 0.0, False
        while not done:
            obs, r, term, trunc, _ = env.step(policy.act(obs))
            done = term or trunc
            total += r
        rewards.append(total)
    return rewards

# Main

rl_results = load_rl_results()
lead_times = sorted(rl_results.keys())

print("Re-evaluating sS baselines…")
ss_results = {}
for lt in lead_times:
    if lt in SS_PARAMS:
        s, S = SS_PARAMS[lt]
        print(f"  lt={lt}  sS(s={s}, S={S})…")
        ss_results[lt] = {"rewards": eval_ss(lt, s, S), "s": s, "S": S}
    else:
        print(f"  lt={lt}  no sS params defined — skipping")

# Plots

model_order = ["ss", "mlp", "lstm"]
n_lt = len(lead_times)

fig_box, ax_box = plt.subplots(1, n_lt, figsize=(5 * n_lt, 5), sharey=False)
fig_bar, ax_bar = plt.subplots(1, n_lt, figsize=(5 * n_lt, 5), sharey=False)
if n_lt == 1:
    ax_box, ax_bar = [ax_box], [ax_bar]

fig_box.suptitle("Reward distribution by lead time", fontsize=12)
fig_bar.suptitle("Mean ± std by lead time",          fontsize=12)

for i, lt in enumerate(lead_times):
    present, data, colours, xlabels = [], [], [], []

    for mt in model_order:
        if mt == "ss":
            if lt in ss_results:
                present.append(mt)
                data.append(ss_results[lt]["rewards"])
                colours.append(COLOURS["ss"])
                s, S = ss_results[lt]["s"], ss_results[lt]["S"]
                xlabels.append(f"sS({s},{S})")
        else:
            if mt in rl_results.get(lt, {}):
                present.append(mt)
                data.append(rl_results[lt][mt])
                colours.append(COLOURS[mt])
                xlabels.append(LABELS[mt])

    # Boxplot
    bp = ax_box[i].boxplot(data, patch_artist=True,
                           medianprops=dict(color="black", linewidth=2))
    for patch, c in zip(bp["boxes"], colours):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    ax_box[i].set_xticks(range(1, len(present) + 1))
    ax_box[i].set_xticklabels(xlabels, fontsize=9)
    ax_box[i].axhline(0, color="grey", linewidth=0.6, linestyle="--")
    ax_box[i].set_title(f"Lead time = {lt}")
    ax_box[i].set_ylabel("Episode reward")

    # Bar chart
    means = [np.mean(d) for d in data]
    stds  = [np.std(d)  for d in data]
    bars  = ax_bar[i].bar(xlabels, means, yerr=stds, color=colours, alpha=0.75,
                          capsize=6, edgecolor="white")
    ax_bar[i].axhline(0, color="grey", linewidth=0.6, linestyle="--")
    ax_bar[i].set_title(f"Lead time = {lt}")
    ax_bar[i].set_ylabel("Mean episode reward")
    for bar, mean, std in zip(bars, means, stds):
        ax_bar[i].text(
            bar.get_x() + bar.get_width() / 2,
            mean + (abs(std) * 0.1 if mean >= 0 else -abs(std) * 0.1),
            f"{mean:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

fig_box.tight_layout()
fig_bar.tight_layout()
fig_box.savefig(OUT_BOX, dpi=150, bbox_inches="tight")
fig_bar.savefig(OUT_BAR, dpi=150, bbox_inches="tight")
print(f"Saved: {os.path.abspath(OUT_BOX)}")
print(f"Saved: {os.path.abspath(OUT_BAR)}")

# Cross-lead-time line chart

fig_line, ax_line = plt.subplots(figsize=(8, 5))

for mt in model_order:
    lt_vals, mean_vals, std_vals = [], [], []
    for lt in lead_times:
        if mt == "ss":
            if lt in ss_results:
                r = ss_results[lt]["rewards"]
            else:
                continue
        else:
            r = rl_results.get(lt, {}).get(mt)
            if r is None:
                continue
        lt_vals.append(lt)
        mean_vals.append(np.mean(r))
        std_vals.append(np.std(r))

    if lt_vals:
        if mt == "ss" and lt_vals:
            s, S = SS_PARAMS[lt_vals[0]]
            label = f"sS (optimal per lt)"
        else:
            label = LABELS[mt]
        ax_line.plot(lt_vals, mean_vals, marker="o", linewidth=2,
                     color=COLOURS[mt], label=label)
        ax_line.fill_between(lt_vals,
                             np.array(mean_vals) - np.array(std_vals),
                             np.array(mean_vals) + np.array(std_vals),
                             color=COLOURS[mt], alpha=0.15)

ax_line.axhline(0, color="grey", linewidth=0.6, linestyle="--")
ax_line.set_xlabel("Lead time")
ax_line.set_ylabel("Mean episode reward")
ax_line.set_title("Mean reward vs lead time  (±1 std band)")
ax_line.set_xticks(lead_times)
ax_line.legend()
fig_line.tight_layout()
fig_line.savefig(OUT_LINE, dpi=150, bbox_inches="tight")
print(f"Saved: {os.path.abspath(OUT_LINE)}")

os.startfile(os.path.abspath(OUT_BOX))
os.startfile(os.path.abspath(OUT_BAR))
os.startfile(os.path.abspath(OUT_LINE))
