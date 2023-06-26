# Simulator environment for inventory management strategies experimenting and testing
# 1. Demand generating class
# 2. Inventory management strategy class
# 3. Simulator class
# 4. Simulator environment class
import scipy.stats as stats
import random
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from abc import ABC, abstractmethod

from optiflow.simulator.demand_generators import Demand
from optiflow.base.single_echelon_optimization.order_model import OrderQuantityModel
from typing import List, Tuple, Dict, Union, ArrayLike
from dataclasses import dataclass

@dataclass
class Item:
    id: int
    price: float
    unit_cost: float
    holding_cost: float
    ordering_cost: float

class Environment(ABC):
    def __init__(self, n_periods: int, demand: Demand, initial_inventory: ArrayLike, item: Item):
        self.n_periods = n_periods
        self.demand = demand
        self.demand_array = self.demand.generate_demand()
        self.inventory = np.zeros(self.n_periods)
        self.orders = np.zeros(self.n_periods)
        self.backorders = np.zeros(self.n_periods)
        self.lost_sales = np.zeros(self.n_periods)
        #
        self.rewards = np.zeros(self.n_periods)
        

    def reset(self):
        self.demand_array = self.demand.generate_demand()
        self.inventory = np.zeros(self.n_periods)

    @abstractmethod
    def step(self, action: int):
        pass

    @abstractmethod
    def get_state(self):
        pass

    @abstractmethod
    def get_reward(self):
        pass

    @abstractmethod
    def get_done(self):
        pass

    @abstractmethod
    def get_info(self):
        pass

class SingleEchelonCont(Environment):
    def __init__(self, n_periods: int, demand: Demand, initial_inventory: ArrayLike, strategy: OrderQuantityModel, review_strategy = "continous", reorder_point=0, backorders_allowed=False):
        super().__init__(n_periods, demand, initial_inventory)
        self.strategy = strategy
        self.backorders_allowed = backorders_allowed
        self.order_size,  = self.strategy.calculate_order_quantities(self.demand_array)
        

    def step(self, action: int, t: int):
        # Add incoming orders to inventory
        self.inventory[t] += self.orders[t]
        # Substract demand from inventory D_t + B_t
        if self.backorders_allowed:
            self.inventory[t] -= self.demand_array[t] + self.backorders[t]
        else:
            self.inventory[t] -= self.demand_array[t]
        # Cap inventory to zero:
        self.inventory[t] = max(0, self.inventory[t])
        # Lost Sales
        self.lost_sales[t] = max(0, self.demand_array[t]- self.inventory[t])
        # Sales
        self.sales_qty[t] = self.demand_array[t] - self.lost_sales[t]

        # If inventory reaches reorder point order
        if self.inventory[t] <= self.reorder_point:
            # Calculate order quantity for next period
            self.orders[t] = round(self.strategy.calculate_order_quantities(self.demand_array[t+1]))
        # if not below. Order is already initialized to zero.

        # Calculate reward
        # make order[t] boolean if grater than zero in count_orders
        count_orders = 1 if self.orders[t] > 0 else 0

        self.reward = self.inventory[t] * self.item.unit_cost + self.inventory[t] * self.holding_cost + self.item.unit_price * self.sales_qty[t] - count_orders * self.item.ordering_cost
    
    def get_state(self):
        return self.inventory, self.orders, self.backorders, self.lost_sales

    def get_reward(self):
        return self.reward
    
    def get_done(self):
        return self.done
    
    def get_info(self):
        return self.info
    
    # simulate
    def simulate(self):
        for t in range(self.n_periods):
            self.step(t)
            total_reward = self.get_reward()
        return self.inventory, self.orders, self.backorders, self.lost_sales

    def save_data(self):
        # save data arrays in CSV file with pandas
        # concatenate arrays
        data = np.concatenate((self.inventory, self.orders, self.backorders, self.lost_sales, self.reward, self.demand_array), axis=1)
        # create pandas dataframe
        df = pd.DataFrame(data, columns=['inventory', 'orders', 'backorders', 'lost_sales', 'reward', 'demand'])
        # save dataframe to csv
        df.to_csv('runs/experiment/result.csv', index=False)
    



        


        

