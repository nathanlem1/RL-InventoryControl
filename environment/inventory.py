import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .config import ENV_CONFIG, COSTS
from collections import deque

class InventoryEnv(gym.Env):

    def __init__(self,
                 demand_model,
                 costs = "baseline",
                 lead_time=None,
                 num_episodes=None,
                 episode_length=None
                 ):

        super().__init__()
        self.cfg = ENV_CONFIG
        self.lead_time = lead_time if lead_time is not None else self.cfg['lead_time']
        self.costs = COSTS[costs]
        self.demand_model = demand_model
        self.num_episodes = num_episodes if num_episodes is not None else self.cfg['num_episodes']
        self.episode_length = episode_length if episode_length is not None else self.cfg['episode_length']
        self.inventory = self.cfg['initial_inventory']
        self.pipeline = []
        self.current_step = 0
        self.last_demand = 0
        self.action_bins = ENV_CONFIG["action_bins"]
        self.demand_history_len = self.cfg.get("demand_history_len", 5)
        self.demand_history = deque([0.0] * self.demand_history_len, maxlen=self.demand_history_len)

        obs_dim = 4 + self.lead_time + self.demand_history_len
        self.observation_space = spaces.Box(low=0.0, high=2.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.action_bins))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.inventory = np.random.randint(10, 80)
        self.pipeline = deque(
            [np.random.randint(0, 20) for _ in range(self.lead_time)],
            maxlen=self.lead_time
        )
        self.current_step = 0
        self.last_demand = 0
        self.demand_history = deque([0.0] * self.demand_history_len, maxlen=self.demand_history_len)

        self.demand_model.reset()

        return self._get_obs(), {}


    def _get_obs(self):
        return np.concatenate([
            [np.clip(self.inventory / 100, 0, 2)],
            [np.clip(self.last_demand / 50, 0, 2)],
            [np.clip((self.inventory + sum(self.pipeline)) / 200, 0, 2)],
            [np.clip(sum(self.pipeline) / (self.cfg["max_order"] * self.lead_time if self.lead_time > 0 else 1), 0, 2)],
            [np.clip(p / self.cfg["max_order"], 0, 2) for p in self.pipeline],
            list(self.demand_history),
        ]).astype(np.float32)


    def step(self, action_idx):
        action_idx = int(np.clip(action_idx, 0, len(self.action_bins) - 1))
        action = int(self.action_bins[action_idx])

        max_ip = self.cfg.get("max_inventory_position", 500)
        current_ip = self.inventory + sum(self.pipeline)
        action = max(0, min(action, max_ip - current_ip))

        if self.lead_time > 0:
            arrival = self.pipeline.popleft()
            self.inventory += arrival
            self.pipeline.append(action)
        else:
            self.inventory += action

        demand = self.demand_model.step()
        self.last_demand = demand
        self.demand_history.append(np.clip(demand / 50.0, 0, 2))

        sold = min(self.inventory, demand)
        lost_sales = max(0,(demand - sold))
        self.inventory -= sold

        revenue = sold * self.costs["selling_price"]

        pipeline = list(self.pipeline)
        raw_holding_cost = self.inventory * self.costs["holding_cost"]
        holding_cost = raw_holding_cost
        stockout_cost = lost_sales * self.costs["stockout_cost"]
        order_cost = action * self.costs["unit_order_cost"]
        fixed_cost = self.costs["fixed_order_cost"] if action > 0 else 0
        inventory = self.inventory
        total_cost = holding_cost + stockout_cost + order_cost + fixed_cost
        reward = float(np.tanh((revenue - total_cost) / 50))


        self.current_step += 1
        done = self.current_step >= self.episode_length

        info = {
            "demand": demand,
            "sold": sold,
            "lost_sales": lost_sales,
            "pipeline": pipeline,
            "inventory": inventory,
            "order":action,

        }

        return self._get_obs(), reward, False, done, info

