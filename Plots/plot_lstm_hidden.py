"""
plot_lstm_hidden.py

Extract and visualise the LSTM actor hidden state from a trained RecurrentPPO
model.  Run a single deterministic episode, record h at every step alongside
the true demand regime, then project down to 2D with PCA.

Usage
-----
    python plot_lstm_hidden.py [run_dir]

    run_dir  path to the lstm sub-directory, e.g. "runs/lt10/lstm"
             defaults to the lstm sub-dir of the most recently modified lt* folder

Output
------
    <run_dir>/lstm_hidden_state.png   (4-panel figure)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sb3_contrib import RecurrentPPO

from environment.inventory import InventoryEnv
from environment.demand import RegimeSwitchingDemand
from environment.config import ENV_CONFIG



ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_DIR    = os.path.join(ROOT, "runs", "lt0", "lstm")
MODEL_PATH = os.path.join(RUN_DIR, "best_model")
OUT        = os.path.join(RUN_DIR, "lstm_hidden_state.png")

LAMBDAS       = [5, 25]
PERSISTENCE   = "medium"
LEAD_TIME     = ENV_CONFIG["lead_time"]
EPISODE_LENGTH = ENV_CONFIG["episode_length"]

print(f"Loading model from {MODEL_PATH}.zip …")
model = RecurrentPPO.load(MODEL_PATH)

demand_model = RegimeSwitchingDemand(PERSISTENCE, LAMBDAS)
env = InventoryEnv(
    demand_model=demand_model,
    lead_time=LEAD_TIME,
    episode_length=EPISODE_LENGTH,
)

obs, _ = env.reset()
lstm_states    = None
episode_starts = np.ones((1,), dtype=bool)

hidden_states = []
regimes       = []
demands       = []
inventories   = []
actions       = []

done = False
while not done:
    if lstm_states is not None:
        h = lstm_states[0].squeeze()
    else:
        h = np.zeros(model.policy.lstm_actor.hidden_size)
    hidden_states.append(h)

    regimes.append(int(demand_model.current_state))

    action, lstm_states = model.predict(
        obs,
        state=lstm_states,
        episode_start=episode_starts,
        deterministic=True,
    )
    episode_starts = np.zeros((1,), dtype=bool)

    obs, _, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    demands.append(info["demand"])
    inventories.append(info["inventory"])
    actions.append(int(action))

H          = np.array(hidden_states)
regimes    = np.array(regimes)
demands    = np.array(demands)
inventories = np.array(inventories)
T          = len(regimes)
timesteps  = np.arange(T)

print(f"Episode length: {T} steps")
print(f"Regime 0 (low,  λ={LAMBDAS[0]}): {(regimes==0).sum()} steps  "
      f"({100*(regimes==0).mean():.1f}%)")
print(f"Regime 1 (high, λ={LAMBDAS[1]}): {(regimes==1).sum()} steps  "
      f"({100*(regimes==1).mean():.1f}%)")


pca    = PCA(n_components=2)
H_2d   = pca.fit_transform(H)
var_explained = pca.explained_variance_ratio_

print(f"PCA variance explained: PC1={var_explained[0]:.1%}  PC2={var_explained[1]:.1%}")

transitions = np.where(np.diff(regimes) != 0)[0] + 1


COLORS = {0: "#4C72B0", 1: "#DD8452"}
LABELS = {0: f"Low demand (λ={LAMBDAS[0]})", 1: f"High demand (λ={LAMBDAS[1]})"}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    f"LSTM Actor Hidden State Analysis  —  lt={LEAD_TIME}  "
    f"({T:,} steps,  PCA {var_explained[0]:.1%}+{var_explained[1]:.1%} var explained)",
    fontsize=13,
)

# Panel 1: PCA scatter coloured by true regime
ax = axes[0, 0]
for regime in [0, 1]:
    mask = regimes == regime
    ax.scatter(H_2d[mask, 0], H_2d[mask, 1],
               c=COLORS[regime], label=LABELS[regime],
               s=4, alpha=0.35, linewidths=0)
ax.set_xlabel(f"PC1 ({var_explained[0]:.1%})")
ax.set_ylabel(f"PC2 ({var_explained[1]:.1%})")
ax.set_title("Hidden state PCA — coloured by true demand regime")
ax.legend(markerscale=3, fontsize=9)

# Panel 2: PCA trajectory (first 500 steps) coloured by time
ax = axes[0, 1]
N_TRAJ = min(500, T)
sc = ax.scatter(H_2d[:N_TRAJ, 0], H_2d[:N_TRAJ, 1],
                c=timesteps[:N_TRAJ], cmap="viridis",
                s=8, alpha=0.7, linewidths=0)
for t in transitions[transitions < N_TRAJ]:
    ax.annotate("", xy=(H_2d[t, 0], H_2d[t, 1]),
                xytext=(H_2d[t-1, 0], H_2d[t-1, 1]),
                arrowprops=dict(arrowstyle="->", color="red", lw=1.2))
plt.colorbar(sc, ax=ax, label="Timestep")
ax.set_xlabel(f"PC1 ({var_explained[0]:.1%})")
ax.set_ylabel(f"PC2 ({var_explained[1]:.1%})")
ax.set_title(f"Hidden state trajectory (first {N_TRAJ} steps)\nred arrows = regime transitions")

# Panel 3: PC1 over time vs actual demand
ax  = axes[1, 0]
ax2 = ax.twinx()

regime_changes = np.concatenate([[0], transitions, [T]])
for i in range(len(regime_changes) - 1):
    start, end = regime_changes[i], regime_changes[i + 1]
    reg = regimes[start]
    ax.axvspan(start, end, alpha=0.08, color=COLORS[reg], linewidth=0)

ax.plot(timesteps, H_2d[:, 0], color="steelblue", linewidth=0.8, label="PC1")
ax2.plot(timesteps, demands, color="tomato", linewidth=0.6, alpha=0.6, label="demand")

ax.set_xlabel("Timestep")
ax.set_ylabel("PC1 of hidden state", color="steelblue")
ax2.set_ylabel("Demand (units)", color="tomato")
ax.set_title("PC1 of hidden state vs actual demand over time\n(shading = true regime)")

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")

# Panel 4: PCA scatter coloured by inventory level
ax = axes[1, 1]
sc = ax.scatter(H_2d[:, 0], H_2d[:, 1],
                c=inventories, cmap="RdYlGn",
                s=4, alpha=0.35, linewidths=0,
                vmin=0, vmax=np.percentile(inventories, 95))
plt.colorbar(sc, ax=ax, label="On-hand inventory (units)")
ax.set_xlabel(f"PC1 ({var_explained[0]:.1%})")
ax.set_ylabel(f"PC2 ({var_explained[1]:.1%})")
ax.set_title("Hidden state PCA — coloured by inventory level\n(reveals stock-tracking in hidden state)")

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"\nSaved: {os.path.abspath(OUT)}")
os.startfile(os.path.abspath(OUT))
