"""
plot_policy_comparison.py
────────────────────────────────────────────────────────────────────────────────
Evaluate LSTM, MLP, and (s,S) over 30 seeded episodes and compare operational
metrics: stockout fraction, service level, mean inventory, orders vs demand.

Usage
-----
    Set LEAD_TIME in env/config.py, set SS_s and SS_S below, then run:
        python Plots/plot_policy_comparison.py

Output
------
    runs/lt{N}/policy_comparison.png
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO

from environment.inventory import InventoryEnv
from environment.demand import RegimeSwitchingDemand
from environment.config import ENV_CONFIG
from models.sS_policy import sSPolicy

# Config

LAMBDAS        = [5, 25]
PERSISTENCE    = "medium"
LEAD_TIME      = ENV_CONFIG["lead_time"]
EPISODE_LENGTH = ENV_CONFIG["episode_length"]
EVAL_SEED      = 42
N_EVAL         = 300

# Set sS params
SS_s = 65
SS_S = 75


RUN_DIR   = os.path.join(ROOT, "runs", f"lt{LEAD_TIME}")
LSTM_PATH = os.path.join(RUN_DIR, "lstm", "best_model")
MLP_PATH  = os.path.join(RUN_DIR, "mlp",  "best_model")
OUT       = os.path.join(RUN_DIR, "policy_comparison.png")
os.makedirs(RUN_DIR, exist_ok=True)


def make_env():
    return InventoryEnv(
        demand_model=RegimeSwitchingDemand(PERSISTENCE, LAMBDAS),
        lead_time=LEAD_TIME,
        episode_length=EPISODE_LENGTH,
    )

def run_episode(env, policy_fn, seed):
    """Run one episode with a callable policy_fn(obs) -> action_idx. Returns metrics dict."""
    np.random.seed(seed)
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    total_demand = total_sold = total_lost = total_orders = stockout_steps = inv_sum = 0
    steps = 0

    done = False
    while not done:
        action = policy_fn(obs)
        obs, r, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward  += r
        total_demand  += info["demand"]
        total_sold    += info["sold"]
        total_lost    += info["lost_sales"]
        total_orders  += info["order"]
        stockout_steps += int(info["lost_sales"] > 0)
        inv_sum       += info["inventory"]
        steps         += 1

    return {
        "reward":           total_reward,
        "service_level":    total_sold / max(total_demand, 1),
        "stockout_frac":    stockout_steps / max(steps, 1),
        "mean_inventory":   inv_sum / max(steps, 1),
        "total_orders":     total_orders,
        "total_demand":     total_demand,
        "total_lost":       total_lost,
    }



policies = {}

# LSTM
if os.path.exists(LSTM_PATH + ".zip"):
    lstm_model = RecurrentPPO.load(LSTM_PATH)
    def lstm_policy_fn(obs, _state={"s": None, "starts": np.ones((1,), dtype=bool)}):
        action, _state["s"] = lstm_model.predict(
            obs, state=_state["s"],
            episode_start=_state["starts"], deterministic=True,
        )
        _state["starts"] = np.zeros((1,), dtype=bool)
        return int(action)
    policies["LSTM"] = lstm_model
    print(f"Loaded LSTM from {LSTM_PATH}.zip")
else:
    print(f"LSTM model not found at {LSTM_PATH}.zip — skipping")

# MLP
if os.path.exists(MLP_PATH + ".zip"):
    mlp_model = PPO.load(MLP_PATH)
    def mlp_policy_fn(obs):
        action, _ = mlp_model.predict(obs, deterministic=True)
        return int(action)
    policies["MLP"] = mlp_model
    print(f"Loaded MLP from {MLP_PATH}.zip")
else:
    print(f"MLP model not found at {MLP_PATH}.zip — skipping")

# sS
ss_policy = sSPolicy(s=SS_s, S=SS_S)
print(f"Using sS(s={SS_s}, S={SS_S})")

# Run evaluations

def eval_policy(policy_type, n=N_EVAL):
    env  = make_env()
    results = []
    for ep in range(n):
        seed = EVAL_SEED + ep
        if policy_type == "lstm":
            state   = {"s": None, "starts": np.ones((1,), dtype=bool)}
            np.random.seed(seed)
            obs, _  = env.reset(seed=seed)
            total_reward = total_demand = total_sold = total_lost = 0
            total_orders = stockout_steps = inv_sum = steps = 0
            done = False
            while not done:
                action, state["s"] = lstm_model.predict(
                    obs, state=state["s"],
                    episode_start=state["starts"], deterministic=True,
                )
                state["starts"] = np.zeros((1,), dtype=bool)
                obs, r, terminated, truncated, info = env.step(int(action))
                done = terminated or truncated
                total_reward  += r
                total_demand  += info["demand"]
                total_sold    += info["sold"]
                total_lost    += info["lost_sales"]
                total_orders  += info["order"]
                stockout_steps += int(info["lost_sales"] > 0)
                inv_sum       += info["inventory"]
                steps         += 1
            results.append({
                "reward":         total_reward,
                "service_level":  total_sold / max(total_demand, 1),
                "stockout_frac":  stockout_steps / max(steps, 1),
                "mean_inventory": inv_sum / max(steps, 1),
                "total_orders":   total_orders,
                "total_demand":   total_demand,
            })

        elif policy_type == "mlp":
            np.random.seed(seed)
            obs, _ = env.reset(seed=seed)
            total_reward = total_demand = total_sold = total_lost = 0
            total_orders = stockout_steps = inv_sum = steps = 0
            done = False
            while not done:
                action, _ = mlp_model.predict(obs, deterministic=True)
                obs, r, terminated, truncated, info = env.step(int(action))
                done = terminated or truncated
                total_reward  += r
                total_demand  += info["demand"]
                total_sold    += info["sold"]
                total_lost    += info["lost_sales"]
                total_orders  += info["order"]
                stockout_steps += int(info["lost_sales"] > 0)
                inv_sum       += info["inventory"]
                steps         += 1
            results.append({
                "reward":         total_reward,
                "service_level":  total_sold / max(total_demand, 1),
                "stockout_frac":  stockout_steps / max(steps, 1),
                "mean_inventory": inv_sum / max(steps, 1),
                "total_orders":   total_orders,
                "total_demand":   total_demand,
            })

        else:  # sS
            results.append(run_episode(env, ss_policy.act, seed))

    return results

print(f"\nEvaluating over {N_EVAL} seeded episodes each…")

all_results = {}
if "LSTM" in policies:
    print("  LSTM…")
    all_results["LSTM"] = eval_policy("lstm")
if "MLP" in policies:
    print("  MLP…")
    all_results["MLP"] = eval_policy("mlp")
print(f"  sS(s={SS_s}, S={SS_S})…")
all_results[f"sS({SS_s},{SS_S})"] = eval_policy("ss")

# Aggregate

def agg(results, key):
    return np.array([r[key] for r in results])

COLORS = {
    "LSTM":             "#DD8452",
    "MLP":              "#55A868",
    f"sS({SS_s},{SS_S})": "#4C72B0",
}

policy_names = list(all_results.keys())

print("\n── Summary ───────────────────────────────────────────────────────────")
for name, results in all_results.items():
    r = agg(results, "reward")
    sl = agg(results, "service_level")
    sf = agg(results, "stockout_frac")
    mi = agg(results, "mean_inventory")
    print(f"  {name:20s}  reward={r.mean():.1f}±{r.std():.1f}  "
          f"service={sl.mean():.3f}  stockout={sf.mean():.3f}  "
          f"inv={mi.mean():.1f}")

# Plot

METRICS = [
    ("reward",         "Episode reward",           None),
    ("service_level",  "Service level",            1.0),
    ("stockout_frac",  "Stockout fraction",        0.0),
    ("mean_inventory", "Mean on-hand inventory",   None),
    ("total_orders",   "Total units ordered",      None),
    ("total_demand",   "Total demand (reference)", None),
]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle(
    f"Policy comparison — lead time = {LEAD_TIME}  "
    f"(λ={LAMBDAS}, {PERSISTENCE} persistence, {N_EVAL} episodes)",
    fontsize=13,
)
axes = axes.flatten()

for ax, (key, label, hline) in zip(axes, METRICS):
    data   = [agg(all_results[n], key) for n in policy_names]
    colors = [COLORS.get(n, "grey") for n in policy_names]

    bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_xticks(range(1, len(policy_names) + 1))
    ax.set_xticklabels(policy_names, fontsize=9)
    ax.set_ylabel(label)
    ax.set_title(label)

    if hline is not None:
        ax.axhline(hline, color="grey", linewidth=0.8, linestyle="--", alpha=0.7)

    for i, d in enumerate(data):
        ax.text(i + 1, 0.01, f"{d.mean():.2f}",
                transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=8, color="black", fontweight="bold")

patches = [mpatches.Patch(color=COLORS.get(n, "grey"), label=n, alpha=0.75)
           for n in policy_names]
fig.legend(handles=patches, loc="lower center", ncol=len(policy_names),
           fontsize=10, bbox_to_anchor=(0.5, 0))

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"\nSaved: {os.path.abspath(OUT)}")
os.startfile(os.path.abspath(OUT))
