import numpy as np
from .config import *

class RegimeSwitchingDemand:
    def __init__(self, persistence_level, lambdas):
        self.P = np.array(PERSISTENCE[persistence_level]["transition_matrix"])
        self.lambdas = lambdas
        self.current_state = None

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.current_state = np.random.choice([0,1])
        return self.step()

    def step(self):
        self.current_state = np.random.choice(
            [0,1],
            p=self.P[self.current_state]
        )
        demand = np.random.poisson(self.lambdas[self.current_state])
        return max(0,int(demand))

