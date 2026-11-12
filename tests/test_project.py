"""
tests/test_project.py
Simple unit tests for the supply chain optimisation (inventory control) project.
Run with: python -m pytest tests/test_project.py -v
"""

import os
import sys
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from environment.inventory import InventoryEnv
from environment.demand import RegimeSwitchingDemand
from environment.config import ENV_CONFIG
from models.sS_policy import sSPolicy


def make_env(lead_time=0):
    return InventoryEnv(
        demand_model=RegimeSwitchingDemand("medium", [5, 25]),
        lead_time=lead_time,
        episode_length=100,
    )


class TestDemand:

    def test_demand_non_negative(self):
        d = RegimeSwitchingDemand("medium", [5, 25])
        d.reset(seed=0)
        for _ in range(200):
            assert d.step() >= 0

    def test_demand_is_integer(self):
        d = RegimeSwitchingDemand("medium", [5, 25])
        d.reset(seed=1)
        for _ in range(50):
            assert isinstance(d.step(), int)

    def test_demand_reset_is_reproducible(self):
        d = RegimeSwitchingDemand("medium", [5, 25])
        d.reset(seed=42)
        run1 = [d.step() for _ in range(20)]
        d.reset(seed=42)
        run2 = [d.step() for _ in range(20)]
        assert run1 == run2

    def test_low_persistence_switches_more(self):
        """Low persistence should switch regimes more often than high."""
        results = {}
        for level in ["low", "medium"]:
            d = RegimeSwitchingDemand(level, [5, 25])
            d.reset(seed=0)
            demands = [d.step() for _ in range(1000)]
            results[level] = np.std(demands)
        assert results["low"] >= results["medium"]


# ── Environment: reset ────────────────────────────────────────────────────────

class TestEnvReset:

    def test_reset_returns_obs_and_dict(self):
        env = make_env()
        result = env.reset(seed=0)
        assert isinstance(result, tuple) and len(result) == 2

    def test_obs_shape_lead_time_0(self):
        env = make_env(lead_time=0)
        obs, _ = env.reset(seed=0)
        expected = 4 + 0 + ENV_CONFIG["demand_history_len"]
        assert obs.shape == (expected,)

    def test_obs_shape_lead_time_3(self):
        env = make_env(lead_time=3)
        obs, _ = env.reset(seed=0)
        expected = 4 + 3 + ENV_CONFIG["demand_history_len"]
        assert obs.shape == (expected,)

    def test_obs_shape_lead_time_10(self):
        env = make_env(lead_time=10)
        obs, _ = env.reset(seed=0)
        expected = 4 + 10 + ENV_CONFIG["demand_history_len"]
        assert obs.shape == (expected,)

    def test_obs_in_valid_range(self):
        env = make_env(lead_time=3)
        obs, _ = env.reset(seed=0)
        assert np.all(obs >= 0.0) and np.all(obs <= 2.0)

    def test_step_resets_to_zero(self):
        env = make_env()
        env.reset(seed=0)
        assert env.current_step == 0


# ── Environment: step ─────────────────────────────────────────────────────────

class TestEnvStep:

    def test_step_returns_five_values(self):
        env = make_env()
        env.reset(seed=0)
        result = env.step(0)
        assert len(result) == 5

    def test_reward_is_float(self):
        env = make_env()
        env.reset(seed=0)
        _, reward, _, _, _ = env.step(0)
        assert isinstance(reward, float)

    def test_reward_bounded_by_tanh(self):
        """Reward uses tanh so must be in (-1, 1)."""
        env = make_env(lead_time=3)
        env.reset(seed=0)
        for _ in range(200):
            _, r, _, _, _ = env.step(np.random.randint(0, 8))
            assert -1.0 <= r <= 1.0

    def test_inventory_never_negative(self):
        env = make_env(lead_time=3)
        env.reset(seed=0)
        for _ in range(200):
            env.step(0)
            assert env.inventory >= 0

    def test_episode_terminates_at_length(self):
        env = make_env()
        env.reset(seed=0)
        done = False
        steps = 0
        while not done:
            _, _, term, trunc, _ = env.step(3)
            done = term or trunc
            steps += 1
        assert steps == 100  # episode_length

    def test_info_keys_present(self):
        env = make_env()
        env.reset(seed=0)
        _, _, _, _, info = env.step(2)
        for key in ["demand", "sold", "lost_sales", "inventory", "order"]:
            assert key in info

    def test_sold_leq_demand(self):
        env = make_env()
        env.reset(seed=0)
        for _ in range(100):
            _, _, _, _, info = env.step(0)
            assert info["sold"] <= info["demand"]

    def test_lost_sales_non_negative(self):
        env = make_env()
        env.reset(seed=0)
        for _ in range(100):
            _, _, _, _, info = env.step(0)
            assert info["lost_sales"] >= 0

    def test_order_zero_action(self):
        """Action index 0 should always place a zero-unit order."""
        env = make_env()
        env.reset(seed=0)
        _, _, _, _, info = env.step(0)
        assert info["order"] == 0

    def test_action_clipping(self):
        """Out-of-range action indices should be silently clipped."""
        env = make_env()
        env.reset(seed=0)
        env.step(999)
        env.step(-1)

    def test_pipeline_length_matches_lead_time(self):
        lt = 6
        env = make_env(lead_time=lt)
        env.reset(seed=0)
        for _ in range(10):
            env.step(3)
            assert len(env.pipeline) == lt

    def test_zero_lead_time_instant_delivery(self):
        """With lt=0 an order should arrive in the same step."""
        env = make_env(lead_time=0)
        env.reset(seed=0)
        env.inventory = 0
        before = env.inventory
        env.step(7)
        assert env.inventory >= 0


# ── (s,S) policy ──────────────────────────────────────────────────────────────

class TestSSPolicy:

    def test_assert_S_greater_than_s(self):
        with pytest.raises(AssertionError):
            sSPolicy(s=50, S=30)

    def test_returns_int(self):
        policy = sSPolicy(s=15, S=30)
        env = make_env()
        obs, _ = env.reset(seed=0)
        action = policy.act(obs)
        assert isinstance(action, int)

    def test_action_in_valid_range(self):
        policy = sSPolicy(s=15, S=30)
        env = make_env(lead_time=3)
        obs, _ = env.reset(seed=0)
        for _ in range(100):
            action = policy.act(obs)
            assert 0 <= action < len(ENV_CONFIG["action_bins"])
            obs, _, _, _, _ = env.step(action)

    def test_no_order_above_threshold(self):
        """When IP > s, policy should return action 0 (no order)."""
        policy = sSPolicy(s=10, S=50)
        env = make_env()
        obs, _ = env.reset(seed=0)
        obs = obs.copy()
        obs[2] = 200 / ENV_CONFIG["inv_position_scale"]
        assert policy.act(obs) == 0

    def test_orders_when_below_threshold(self):
        """When IP <= s, policy should order (action > 0)."""
        policy = sSPolicy(s=100, S=150)
        obs = np.zeros(4 + ENV_CONFIG["demand_history_len"], dtype=np.float32)
        obs[2] = 5 / ENV_CONFIG["inv_position_scale"]
        assert policy.act(obs) > 0

    def test_full_episode_runs(self):
        """Policy should complete a full episode without errors."""
        policy = sSPolicy(s=15, S=30)
        env = make_env(lead_time=0)
        obs, _ = env.reset(seed=42)
        done, total = False, 0.0
        while not done:
            obs, r, term, trunc, _ = env.step(policy.act(obs))
            done = term or trunc
            total += r
        assert isinstance(total, float)
