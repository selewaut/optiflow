from abc import ABC, abstractmethod

import numpy as np


# Abstract class for different demand distrubtions arrays
class Demand(ABC):
    def __init__(self, n_periods):
        self.n_periods = n_periods
        self.demand = None
    
    @abstractmethod
    def generate_demand(self):
        pass


# Deterministic demand
class DeterministicDemand(Demand):
    def __init__(self, n_periods: int, demand: int):
        super().__init__(n_periods)
        self.demand = demand

    def generate_demand(self):
        # return array of demand
        return np.full(self.n_periods, self.demand)

class NormalDemand(Demand):
    def __init__(self, n_periods: int, mean: float, std: float):
        super().__init__(n_periods)
        self.mean = mean
        self.std = std

    def generate_demand(self):
        # return array of demand
        return np.random.normal(self.mean, self.std, self.n_periods)


class PoissonDemand(Demand):
    def __init__(self, n_periods: int, mean: float):
        super().__init__(n_periods)
        self.mean = mean

    def generate_demand(self):
        # return array of demand
        return np.random.poisson(self.mean, self.n_periods)
    
class NegativeBinomialDemand(Demand):
    def __init__(self, n_periods: int, n:int, p:float):
        super().__init__(n_periods)
        self.n = n
        self.p = p

    def generate_demand(self):
        # return array of demand
        return np.random.negative_binomial(self.n, self.p, self.n_periods)