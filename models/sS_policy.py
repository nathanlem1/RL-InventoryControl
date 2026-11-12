import numpy as np
from environment.config import ENV_CONFIG

_ip_scale   = ENV_CONFIG['inv_position_scale']
_action_bins = np.array(ENV_CONFIG['action_bins'], dtype=int)


class sSPolicy:
    def __init__(self, s: float, S: float):
        assert S > s, "S must be greater than s"
        self.s = s
        self.S = S

    def act(self, state):
        inventory_position = state[2] * _ip_scale

        if inventory_position <= self.s:
            desired_qty = float(self.S - inventory_position)
            desired_qty = max(0.0, desired_qty)
            return int(np.argmin(np.abs(_action_bins - desired_qty)))
        else:
            return 0
