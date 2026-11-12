import os
import numpy as np

import gymnasium as gym
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from environment.inventory import InventoryEnv
from environment.demand import RegimeSwitchingDemand
from environment.config import ENV_CONFIG


class DiagnosticsWrapper(gym.Wrapper):
    """Accumulates per-step info into episode-level diagnostics injected at episode end."""

    def reset(self, **kwargs):
        self._demand = self._sold = self._lost = self._orders = 0.0
        self._stockout_steps = self._inv_sum = self._steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._demand         += info.get("demand", 0)
        self._sold           += info.get("sold", 0)
        self._lost           += info.get("lost_sales", 0)
        self._orders         += info.get("order", 0)
        self._stockout_steps += int(info.get("lost_sales", 0) > 0)
        self._inv_sum        += info.get("inventory", 0)
        self._steps          += 1

        if terminated or truncated:
            info["ep_demand"]         = self._demand
            info["ep_sold"]           = self._sold
            info["ep_lost_sales"]     = self._lost
            info["ep_orders"]         = self._orders
            info["ep_stockout_steps"] = self._stockout_steps
            info["ep_mean_inventory"] = self._inv_sum / max(self._steps, 1)
            info["ep_service_level"]  = self._sold / max(self._demand, 1)

        return obs, reward, terminated, truncated, info

#  Config

LAMBDAS         = [5, 25]
PERSISTENCE     = "medium"
LEAD_TIME       = ENV_CONFIG["lead_time"]
EPISODE_LENGTH  = ENV_CONFIG["episode_length"]
TOTAL_TIMESTEPS = 5_000_000

# Run directories
RUN_DIR  = os.path.join("runs", f"lt{LEAD_TIME}")
LSTM_DIR = os.path.join(RUN_DIR, "lstm")
MLP_DIR  = os.path.join(RUN_DIR, "mlp")

# Behavioural cloning warmstart
BC_SS_s     = 81
BC_SS_S     = 91
BC_STEPS    = 50_000
BC_EPOCHS   = 5
BC_LR       = 1e-3

# Evaluation seed
EVAL_SEED   = 42

#  Training

DIAG_KEYS = (
    "ep_demand", "ep_sold", "ep_lost_sales", "ep_orders",
    "ep_stockout_steps", "ep_mean_inventory", "ep_service_level",
)

def make_env(log_dir=None):
    env = InventoryEnv(
        demand_model=RegimeSwitchingDemand(PERSISTENCE, LAMBDAS),
        lead_time=LEAD_TIME,
        episode_length=EPISODE_LENGTH,
    )
    env = DiagnosticsWrapper(env)
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, filename=os.path.join(log_dir, "monitor"), info_keywords=DIAG_KEYS)
    return env


def make_env_raw():
    return InventoryEnv(
        demand_model=RegimeSwitchingDemand(PERSISTENCE, LAMBDAS),
        lead_time=LEAD_TIME,
        episode_length=EPISODE_LENGTH,
    )


#  Behavioural cloning warmstart

def bc_warmstart(model, ss_s, ss_S, n_steps=BC_STEPS, bc_epochs=BC_EPOCHS, bc_lr=BC_LR):
    import torch
    import torch.nn.functional as F
    from models.sS_policy import sSPolicy

    gamma = model.gamma
    env = make_env_raw()
    policy = sSPolicy(s=ss_s, S=ss_S)
    action_bins = np.array(env.action_bins)
    to_bin = lambda qty: int(np.argmin(np.abs(action_bins - qty)))

    #  Collect full episodes for MC returns
    print(f"BC: collecting {n_steps:,} sS(s={ss_s},S={ss_S}) transitions…")
    bc_obs, bc_acts, bc_returns = [], [], []

    while len(bc_obs) < n_steps:
        obs, _ = env.reset()
        ep_obs, ep_acts, ep_rews = [], [], []
        done = False
        while not done:
            raw = policy.act(obs)
            idx = to_bin(raw)
            ep_obs.append(obs.copy())
            ep_acts.append(idx)
            obs, rew, term, trunc, _ = env.step(idx)
            ep_rews.append(rew)
            done = term or trunc

        G, ep_returns = 0.0, []
        for r in reversed(ep_rews):
            G = r + gamma * G
            ep_returns.insert(0, G)

        bc_obs.extend(ep_obs)
        bc_acts.extend(ep_acts)
        bc_returns.extend(ep_returns)

    bc_obs     = bc_obs[:n_steps]
    bc_acts    = bc_acts[:n_steps]
    bc_returns = bc_returns[:n_steps]

    device    = model.device
    obs_t     = torch.tensor(np.array(bc_obs),     dtype=torch.float32, device=device)
    acts_t    = torch.tensor(np.array(bc_acts),     dtype=torch.long,    device=device)
    returns_t = torch.tensor(np.array(bc_returns),  dtype=torch.float32, device=device)

    #  Supervised training through policy internals
    lstm_pi  = model.policy.lstm_actor
    lstm_vf  = model.policy.lstm_critic if model.policy.lstm_critic is not None else lstm_pi
    mlp      = model.policy.mlp_extractor
    act_net  = model.policy.action_net
    val_net  = model.policy.value_net

    opt = torch.optim.Adam(model.policy.parameters(), lr=bc_lr)
    n   = len(bc_obs)

    print(f"BC: training for {bc_epochs} epochs…")
    for epoch in range(bc_epochs):
        perm = np.random.permutation(n)
        pi_loss_sum, vf_loss_sum, count = 0.0, 0.0, 0

        for start in range(0, n, 1024):
            mb        = perm[start:start + 1024]
            obs_b     = obs_t[mb]
            acts_b    = acts_t[mb]
            returns_b = returns_t[mb]
            B         = obs_b.shape[0]

            feats = model.policy.extract_features(obs_b, model.policy.features_extractor)

            # Actor path
            h0 = torch.zeros(lstm_pi.num_layers, B, lstm_pi.hidden_size, device=device)
            c0 = torch.zeros(lstm_pi.num_layers, B, lstm_pi.hidden_size, device=device)
            inp_pi = feats.unsqueeze(0) if not lstm_pi.batch_first else feats.unsqueeze(1)
            out_pi, _ = lstm_pi(inp_pi, (h0, c0))
            lat_pi = out_pi.squeeze(0) if not lstm_pi.batch_first else out_pi.squeeze(1)
            logits = act_net(mlp.forward_actor(lat_pi))

            # Critic path
            h0v = torch.zeros(lstm_vf.num_layers, B, lstm_vf.hidden_size, device=device)
            c0v = torch.zeros(lstm_vf.num_layers, B, lstm_vf.hidden_size, device=device)
            inp_vf = feats.unsqueeze(0) if not lstm_vf.batch_first else feats.unsqueeze(1)
            out_vf, _ = lstm_vf(inp_vf, (h0v, c0v))
            lat_vf = out_vf.squeeze(0) if not lstm_vf.batch_first else out_vf.squeeze(1)
            values = val_net(mlp.forward_critic(lat_vf)).squeeze(-1)

            pi_loss = F.cross_entropy(logits, acts_b)
            vf_loss = F.mse_loss(values, returns_b)
            loss    = pi_loss + vf_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5)
            opt.step()

            pi_loss_sum += pi_loss.item()
            vf_loss_sum += vf_loss.item()
            count       += 1

        print(f"  Epoch {epoch + 1}/{bc_epochs} | pi {pi_loss_sum/count:.4f}  vf {vf_loss_sum/count:.4f}")

    print("BC warmstart complete.")


def bc_warmstart_mlp(model, ss_s, ss_S, n_steps=BC_STEPS, bc_epochs=BC_EPOCHS, bc_lr=BC_LR):
    """Behavioural cloning warmstart for a non-recurrent MLP policy."""
    import torch
    import torch.nn.functional as F
    from models.sS_policy import sSPolicy

    gamma = model.gamma
    env = make_env_raw()
    policy = sSPolicy(s=ss_s, S=ss_S)
    action_bins = np.array(env.action_bins)
    to_bin = lambda qty: int(np.argmin(np.abs(action_bins - qty)))

    #  Collect full episodes for MC returns
    print(f"BC (MLP): collecting {n_steps:,} sS(s={ss_s},S={ss_S}) transitions…")
    bc_obs, bc_acts, bc_returns = [], [], []

    while len(bc_obs) < n_steps:
        obs, _ = env.reset()
        ep_obs, ep_acts, ep_rews = [], [], []
        done = False
        while not done:
            raw = policy.act(obs)
            idx = to_bin(raw)
            ep_obs.append(obs.copy())
            ep_acts.append(idx)
            obs, rew, term, trunc, _ = env.step(idx)
            ep_rews.append(rew)
            done = term or trunc

        G, ep_returns = 0.0, []
        for r in reversed(ep_rews):
            G = r + gamma * G
            ep_returns.insert(0, G)

        bc_obs.extend(ep_obs)
        bc_acts.extend(ep_acts)
        bc_returns.extend(ep_returns)

    bc_obs     = bc_obs[:n_steps]
    bc_acts    = bc_acts[:n_steps]
    bc_returns = bc_returns[:n_steps]

    device    = model.device
    obs_t     = torch.tensor(np.array(bc_obs),    dtype=torch.float32, device=device)
    acts_t    = torch.tensor(np.array(bc_acts),    dtype=torch.long,    device=device)
    returns_t = torch.tensor(np.array(bc_returns), dtype=torch.float32, device=device)

    #  Supervised training — direct MLP forward pass
    mlp     = model.policy.mlp_extractor
    act_net = model.policy.action_net
    val_net = model.policy.value_net

    opt = torch.optim.Adam(model.policy.parameters(), lr=bc_lr)
    n   = len(bc_obs)

    print(f"BC (MLP): training for {bc_epochs} epochs…")
    for epoch in range(bc_epochs):
        perm = np.random.permutation(n)
        pi_loss_sum, vf_loss_sum, count = 0.0, 0.0, 0

        for start in range(0, n, 1024):
            mb        = perm[start:start + 1024]
            obs_b     = obs_t[mb]
            acts_b    = acts_t[mb]
            returns_b = returns_t[mb]

            feats  = model.policy.extract_features(obs_b, model.policy.features_extractor)
            logits = act_net(mlp.forward_actor(feats))
            values = val_net(mlp.forward_critic(feats)).squeeze(-1)

            pi_loss = F.cross_entropy(logits, acts_b)
            vf_loss = F.mse_loss(values, returns_b)
            loss    = pi_loss + vf_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5)
            opt.step()

            pi_loss_sum += pi_loss.item()
            vf_loss_sum += vf_loss.item()
            count       += 1

        print(f"  Epoch {epoch + 1}/{bc_epochs} | pi {pi_loss_sum/count:.4f}  vf {vf_loss_sum/count:.4f}")

    print("BC (MLP) warmstart complete.")

#  SB3 LSTM

def train_lstm():
    os.makedirs(LSTM_DIR, exist_ok=True)
    env = DummyVecEnv([lambda: make_env(LSTM_DIR)])

    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        n_steps=10000,
        batch_size=2000,
        n_epochs=6,
        learning_rate=1e-4,
        gamma=0.99,
        gae_lambda=0.90,
        clip_range=0.1,
        ent_coef=0.05,
        vf_coef=1.0,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=RUN_DIR,
        policy_kwargs=dict(
            lstm_hidden_size=128,
            n_lstm_layers=1,
            net_arch=dict(pi=[128], vf=[128]),
        ),
    )

    bc_warmstart(model, ss_s=BC_SS_s, ss_S=BC_SS_S)

    eval_callback = EvalCallback(
        DummyVecEnv([make_env_raw]),
        best_model_save_path=LSTM_DIR,
        log_path=LSTM_DIR,
        eval_freq=200_000,
        n_eval_episodes=10,
        deterministic=True,
        verbose=1,
    )

    print(f"Training RecurrentPPO for {TOTAL_TIMESTEPS:,} timesteps…")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)
    model.save(os.path.join(LSTM_DIR, "final_model"))
    print(f"Final LSTM model saved to {LSTM_DIR}/final_model.zip")
    print(f"Best  LSTM model saved to {LSTM_DIR}/best_model.zip")
    return model


#  SB3 MLP

def train_mlp():
    os.makedirs(MLP_DIR, exist_ok=True)
    env = DummyVecEnv([lambda: make_env(MLP_DIR)])

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=10_000,
        batch_size=2000,
        n_epochs=6,
        learning_rate=1e-4,
        gamma=0.99,
        gae_lambda=0.90,
        clip_range=0.1,
        ent_coef=0.05,
        vf_coef=1.0,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=RUN_DIR,
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
    )

    bc_warmstart_mlp(model, ss_s=BC_SS_s, ss_S=BC_SS_S)

    eval_callback = EvalCallback(
        DummyVecEnv([make_env_raw]),
        best_model_save_path=MLP_DIR,
        log_path=MLP_DIR,
        eval_freq=200_000,
        n_eval_episodes=10,
        deterministic=True,
        verbose=1,
    )

    print(f"Training SB3 MLP for {TOTAL_TIMESTEPS:,} timesteps…")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)
    model.save(os.path.join(MLP_DIR, "final_model"))
    print(f"Final MLP model saved to {MLP_DIR}/final_model.zip")
    print(f"Best  MLP model saved to {MLP_DIR}/best_model.zip")
    return model


#  Evaluate best model and save results ─

def evaluate_best(model_type="mlp", n_episodes=30):
    """
    Load the best saved checkpoint for model_type, run n_episodes deterministic
    evaluation, find best sS policy, and save everything to
    runs/lt{N}/{model_type}/eval_results.json for later plotting.
    """
    import json
    from datetime import datetime
    from models.sS_policy import sSPolicy

    run_dir = LSTM_DIR if model_type == "lstm" else MLP_DIR
    model_path = os.path.join(run_dir, "best_model")

    if not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError(f"No best_model.zip found at {model_path}.zip — train first.")

    # Load model
    print(f"Loading {model_type.upper()} from {model_path}.zip …")
    if model_type == "lstm":
        model = RecurrentPPO.load(model_path)
    else:
        model = PPO.load(model_path)

    #  Evaluate model
    print(f"Evaluating {model_type.upper()} over {n_episodes} episodes…")
    env = make_env_raw()
    model_rewards = []

    for ep in range(n_episodes):
        np.random.seed(EVAL_SEED + ep)
        obs, _ = env.reset(seed=EVAL_SEED + ep)
        lstm_states    = None
        episode_starts = np.ones((1,), dtype=bool)
        total, done    = 0.0, False

        while not done:
            if model_type == "lstm":
                action, lstm_states = model.predict(
                    obs, state=lstm_states,
                    episode_start=episode_starts, deterministic=True,
                )
                episode_starts = np.array([done])
            else:
                action, _ = model.predict(obs, deterministic=True)

            obs, r, terminated, truncated, info = env.step(action)
            done   = terminated or truncated
            total += r
        model_rewards.append(float(total))

    print(f"  {model_type.upper()}: mean={np.mean(model_rewards):.2f}  "
          f"std={np.std(model_rewards):.2f}")

    #  Find best sS via grid search
    print("Grid-searching sS params…")
    mean_demand   = np.mean(LAMBDAS)
    base          = int(LEAD_TIME * mean_demand)
    s_lo, s_hi    = max(5, base - 30), base + 80
    best_r, best_s, best_S = -np.inf, None, None

    for s in range(s_lo, s_hi, 5):
        for S in range(s + 5, s + 300, 10):
            policy  = sSPolicy(s=s, S=S)
            rewards = []
            for i in range(5):
                np.random.seed(EVAL_SEED + i)
                obs, _ = env.reset(seed=EVAL_SEED + i)
                total, done = 0.0, False
                while not done:
                    obs, r, term, trunc, _ = env.step(policy.act(obs))
                    done   = term or trunc
                    total += r
                rewards.append(total)
            mean_r = float(np.mean(rewards))
            if mean_r > best_r:
                best_r, best_s, best_S = mean_r, s, S

    print(f"  Best sS: s={best_s}, S={best_S}  (search mean {best_r:.2f})")

    #  Full sS evaluation with best params
    print(f"Evaluating sS(s={best_s}, S={best_S}) over {n_episodes} episodes…")
    policy = sSPolicy(s=best_s, S=best_S)
    ss_rewards = []
    for ep in range(n_episodes):
        np.random.seed(EVAL_SEED + ep)
        obs, _ = env.reset(seed=EVAL_SEED + ep)
        total, done = 0.0, False
        while not done:
            obs, r, term, trunc, _ = env.step(policy.act(obs))
            done   = term or trunc
            total += r
        ss_rewards.append(float(total))

    print(f"  sS({best_s},{best_S}): mean={np.mean(ss_rewards):.2f}  "
          f"std={np.std(ss_rewards):.2f}")

    #  Save results
    results = {
        "lead_time":   LEAD_TIME,
        "model_type":  model_type,
        "lambdas":     LAMBDAS,
        "persistence": PERSISTENCE,
        "n_episodes":  n_episodes,
        "eval_seed":   EVAL_SEED,
        "timestamp":   datetime.now().isoformat(),
        "model": {
            "rewards": model_rewards,
            "mean":    float(np.mean(model_rewards)),
            "std":     float(np.std(model_rewards)),
            "min":     float(np.min(model_rewards)),
            "max":     float(np.max(model_rewards)),
        },
        "ss_policy": {
            "s":       best_s,
            "S":       best_S,
            "rewards": ss_rewards,
            "mean":    float(np.mean(ss_rewards)),
            "std":     float(np.std(ss_rewards)),
            "min":     float(np.min(ss_rewards)),
            "max":     float(np.max(ss_rewards)),
        },
    }

    out_path = os.path.join(run_dir, "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {os.path.abspath(out_path)}")
    return results



# Choose which one to train by commenting the other: MLP or LSTM
if __name__ == "__main__":
    train_lstm()
    # train_mlp()