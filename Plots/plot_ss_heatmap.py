"""
plot_ss_heatmap.py
────────────────────────────────────────────────────────────────────────────────
1. Grid-search (s, S) parameters for the current lead time
2. Save a heatmap of mean reward across the search space
3. Evaluate the best (s, S) over 30 seeded episodes (same seeds as RL models)

Output
------
    runs/lt{N}/ss_heatmap.png
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from environment.inventory import InventoryEnv
from environment.demand import RegimeSwitchingDemand
from environment.config import ENV_CONFIG
from models.sS_policy import sSPolicy

# Config

LAMBDAS        = [5, 25]
PERSISTENCE    = "medium"
LEAD_TIME      = ENV_CONFIG["lead_time"]
EPISODE_LENGTH = ENV_CONFIG["episode_length"]

N_SEARCH   = 15
N_EVAL     = 30
EVAL_SEED  = 42
S_STEP     = 5

MAX_ORDER  = 40

mean_demand = np.mean(LAMBDAS)
base        = int(LEAD_TIME * mean_demand)

s_lo = max(5, base // 3)
s_hi = base + 80

S_lo = s_lo + S_STEP
S_hi = s_hi + MAX_ORDER

s_vals = np.arange(s_lo, s_hi, S_STEP)
S_vals = np.arange(S_lo, S_hi, S_STEP)

OUT = os.path.join(ROOT, "runs", f"lt{LEAD_TIME}", "ss_heatmap.png")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# Grid search

print(f"Lead time = {LEAD_TIME}  |  base reorder point = {base}")
print(f"s: {s_vals[0]} → {s_vals[-1]}  ({len(s_vals)} values)")
print(f"S: {S_vals[0]} → {S_vals[-1]}  ({len(S_vals)} values)")
print(f"{len(s_vals) * len(S_vals)} cells  ×  {N_SEARCH} episodes each\n")

rewards = np.full((len(S_vals), len(s_vals)), np.nan)

env = InventoryEnv(
    demand_model=RegimeSwitchingDemand(PERSISTENCE, LAMBDAS),
    lead_time=LEAD_TIME,
    episode_length=EPISODE_LENGTH,
)

for j, s in enumerate(s_vals):
    print(f"  s={int(s)}  ({j+1}/{len(s_vals)})")
    for i, S in enumerate(S_vals):
        if S <= s:
            continue

        policy     = sSPolicy(s=int(s), S=int(S))
        ep_rewards = []
        for ep in range(N_SEARCH):
            np.random.seed(EVAL_SEED + ep)
            obs, _ = env.reset(seed=EVAL_SEED + ep)
            total, done = 0.0, False
            while not done:
                obs, r, term, trunc, _ = env.step(policy.act(obs))
                done   = term or trunc
                total += r
            ep_rewards.append(total)

        rewards[i, j] = np.mean(ep_rewards)

#Find best cell

best_i, best_j = np.unravel_index(np.nanargmax(rewards), rewards.shape)
best_s  = int(s_vals[best_j])
best_S  = int(S_vals[best_i])
best_r  = rewards[best_i, best_j]

print(f"\nBest from grid search: s={best_s}, S={best_S}  (search mean={best_r:.2f})")

#Heatmap

df_heat = pd.DataFrame(rewards, index=S_vals.astype(int), columns=s_vals.astype(int))

fig, ax = plt.subplots(figsize=(13, 9))

sns.heatmap(
    df_heat,
    ax=ax,
    cmap="RdYlGn",
    mask=np.isnan(rewards),
    cbar_kws={"label": "Mean episode reward"},
    linewidths=0,
    xticklabels=5,
    yticklabels=5,
)

# Star on best cell
best_col = int(np.where(s_vals == best_s)[0][0])
best_row = int(np.where(S_vals == best_S)[0][0])
ax.scatter(best_col + 0.5, best_row + 0.5,
           marker="*", s=400, color="white", edgecolors="black",
           linewidths=0.8, zorder=5,
           label=f"Best: s={best_s}, S={best_S}  (search mean={best_r:.1f})")

ax.invert_yaxis()
ax.set_xlabel("Reorder point  s", fontsize=11)
ax.set_ylabel("Order-up-to level  S", fontsize=11)
ax.set_title(
    f"(s, S) policy grid search — lead time = {LEAD_TIME}\n"
    f"(λ={LAMBDAS}, {PERSISTENCE} persistence, {N_SEARCH} episodes per cell)",
    fontsize=12,
)
ax.legend(fontsize=9, loc="upper right")

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Heatmap saved: {os.path.abspath(OUT)}")
os.startfile(os.path.abspath(OUT))

# Full 30-episode evaluation of best (s, S)

print(f"\nEvaluating sS(s={best_s}, S={best_S}) over {N_EVAL} seeded episodes…")
policy     = sSPolicy(s=best_s, S=best_S)
ss_rewards = []

for ep in range(N_EVAL):
    np.random.seed(EVAL_SEED + ep)
    obs, _ = env.reset(seed=EVAL_SEED + ep)
    total, done = 0.0, False
    while not done:
        obs, r, term, trunc, _ = env.step(policy.act(obs))
        done   = term or trunc
        total += r
    ss_rewards.append(float(total))

print(f"\n── Results ──────────────────────────────────────────────")
print(f"  sS(s={best_s}, S={best_S})")
print(f"  mean = {np.mean(ss_rewards):.2f}  std = {np.std(ss_rewards):.2f}")
print(f"  min  = {np.min(ss_rewards):.2f}   max = {np.max(ss_rewards):.2f}")
