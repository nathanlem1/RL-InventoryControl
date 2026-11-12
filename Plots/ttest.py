import os, sys, json
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from environment.inventory import InventoryEnv
from environment.demand import RegimeSwitchingDemand
from environment.config import ENV_CONFIG
from models.sS_policy import sSPolicy

SS_PARAMS = {0: (15,30), 3: (85,95), 6: (140,150), 10: (65,75)}

def get_rewards(lt, model):
    if model == "ss":
        s, S = SS_PARAMS[lt]
        env = InventoryEnv(demand_model=RegimeSwitchingDemand("medium", [5,25]), lead_time=lt, episode_length=ENV_CONFIG["episode_length"])
        policy = sSPolicy(s, S)
        rewards = []
        for ep in range(30):
            obs, _ = env.reset(seed=42+ep)
            total, done = 0.0, False
            while not done:
                obs, r, term, trunc, _ = env.step(policy.act(obs))
                done = term or trunc
                total += r
            rewards.append(total)
        return np.array(rewards)
    else:
        path = os.path.join(ROOT, "runs", f"lt{lt}", model, "eval_results.json")
        with open(path) as f:
            return np.array(json.load(f)["model"]["rewards"])

for lt in [0, 3, 6, 10]:
    lstm = get_rewards(lt, "lstm")
    mlp  = get_rewards(lt, "mlp")
    ss   = get_rewards(lt, "ss")
    print(f"\nLead time {lt}")
    for a, b, label in [(lstm, ss, "LSTM vs sS"), (mlp, ss, "MLP  vs sS"), (lstm, mlp, "LSTM vs MLP")]:
        t, p = stats.ttest_ind(a, b, equal_var=False)
        print(f"  {label}:  t={t:.3f}  p={p:.4f}  {'*' if p<0.05 else 'ns'}")
