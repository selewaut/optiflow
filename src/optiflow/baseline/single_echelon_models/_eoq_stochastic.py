from optiflow.base.single_echelon_optimization.order_model import OrderQuantityModel
from optiflow.baseline.single_echelon_models._prob_factory import get_distribution
import numpy as np


# generate class dependant on OrderQuantityModel that considers stochastic demand
class StochasticDemandOrderQuantityModel(OrderQuantityModel):
    def __init__(self, ordering_cost, holding_cost, demand_distribution='norm',  n_periods=1, **kwargs):
        self.ordering_cost = ordering_cost
        self.holding_cost = holding_cost
        self.demand_distribution = get_distribution(demand_distribution)
        self.distribution = demand_distribution
        self.n_periods = n_periods


    def reorder_point(self, p, **kwargs):
        """
        Estimates the quantile function of the distribution at the specified probability level.
        
        Args:
        p (float): The probability level at which to estimate the quantile function.
        **kwargs: Additional keyword arguments to pass to the ppf method of the distribution object.
        """
        # Estimate quantile at which service level is met. 
        # This inventory value should be the same as CS + SS
        return self.distribution.ppf(p, **kwargs)
    

    def expected_service_level(self, **kwargs):
        """
        Estimates the expected service level. Calculates
        """
        # estimate cumulative distribution given distribution parameters and initial cycle inventory value.
        return self.distribution.cdf(**kwargs)
    
    def safety_stock(self, service_level: float, demand: np.array):
        if self.demand_distribution == 'norm':
            sigma_d = np.std(demand)
            # safety normal stock
            ss = self.distribution.ppf(service_level) * sigma_d * np.sqrt(self.n_periods)
            # cycle normal stock.
            cs = np.mean(demand)
            return ss, cs
        
    def calculate_order_quantities(self, demand):
        """Calculates the order quantities using the EOQ model.

        Parameters
        ----------
        demand : array-like of shape (n_periods,)
            The demand for the next n periods.
        Returns
        -------
        order_quantities : array-like of shape (n_periods,)
            The order quantities for each of the next n periods.
        """
        eoq = np.sqrt((2 * self.ordering_cost * np.sum(demand)) / self.holding_cost)
        self.eoq = eoq
        optimal_cost = self.total_cost(demand, Q=eoq)
        return eoq, optimal_cost    


    def total_cost(self, demand, Q):
        """Calculates the total cost of ordering and holding inventory.

        Parameters
        ----------
        demand : array-like of shape (n_periods,)
            The demand for the next n periods.
        Returns
        -------
        total_cost : float
            The total cost of ordering and holding inventory.
        """
        cost = self.holding_cost * Q / 2 + self.ordering_cost * np.sum(demand) / Q
        self.cost = cost
        return cost