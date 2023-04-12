from optiflow.base.single_echelon_optimization.order_model import OrderQuantityModel
import numpy as np


class EOQ(OrderQuantityModel):
    """Economic Order Quantity (EOQ) model for inventory optimization.

    This class implements the EOQ model, which calculates the order quantity
    that minimizes the total cost of ordering and holding inventory.

    Parameters
    ----------
    OrderQuantityModel : class
        The abstract base class for order quantity models.

    Attributes
    ----------
    eoq : float
        The economic order quantity that minimizes the total cost of ordering
        and holding inventory.

    Methods
    -------
    calculate_order_quantities(demand_forecasts, inventory_levels)
        Calculates the order quantities for the given demand forecasts and inventory levels
        using the EOQ model.

    """

    def __init__(self, ordering_cost, holding_cost):
        """
        Parameters
        ----------
        ordering_cost : float
            The cost of placing an order.
        holding_cost : float
            The cost of holding one unit of inventory for one period.
        """
        self.ordering_cost = ordering_cost
        self.holding_cost = holding_cost
        self.eoq = None
        self.cost = None

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


class EOQProduction(OrderQuantityModel):
    """Economic Order Quantity with production rate (EOQ) model for inventory optimization.

    This class implements the EOQ model, which calculates the order quantity
    that minimizes the total cost of ordering and holding inventory.
    """

    def __init__(self, ordering_cost, holding_cost, production_rate):
        """
        Parameters
        ----------
        ordering_cost : float
            The cost of placing an order.
        holding_cost : float
            The cost of holding one unit of inventory for one period.
        production_rate : float
            The production rate for the next n periods.
        """
        self.ordering_cost = ordering_cost
        self.holding_cost = holding_cost
        self.production_rate = production_rate
        self.eoq = None
        self.cost = None

    def calculate_order_quantities(self, demand):
        """Calculates the order quantities using the EOQ model with production rate additional parameter

        Parameters
        ----------
        demand : array-like of shape (n_periods,)
            The demand for the next n periods.
        Returns
        -------
        order_quantities : array-like of shape (n_periods,)
            The order quantities for each of the next n periods.
        """
        # we define 'rho'
        production_throughput = 1 - demand / self.production_rate

        eoq = np.sqrt(
            (2 * self.ordering_cost * np.sum(demand))
            / (self.holding_cost * production_throughput)
        )
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
        # we define 'rho'
        production_throughput = 1 - demand / self.production_rate
        cost = (
            self.holding_cost * production_throughput * Q / 2
            + self.ordering_cost * np.sum(demand) / Q
        )
        self.cost = cost
        return cost


class EOQBackorders(OrderQuantityModel):
    """Economic Order Quantity with backorders (EOQ) model for inventory optimization.

    This class implements the EOQ model with backorders, which calculates the order quantity
    that minimizes the total cost of ordering and holding inventory.
    """

    def __init__(self, ordering_cost, holding_cost, stockout_cost):
        self.ordering_cost = ordering_cost
        self.holding_cost = holding_cost
        self.stockout_cost = stockout_cost
        self.critical_ratio = stockout_cost / (holding_cost + stockout_cost)
        self.eoq = None
        self.cost = None
        self.backorders = None

    def calculate_order_quantities(self, demand):
        eoq = np.sqrt(
            2
            * self.ordering_cost
            * demand
            * (self.holding_cost + self.stockout_cost)
            / (self.holding_cost * self.stockout_cost)
        )
        self.eoq = eoq

        b = (1 - self.critical_ratio) * eoq
        optimal_cost = self.total_cost(demand, Q=eoq)
        return eoq, optimal_cost, b

    def total_cost(self, demand, Q, B):
        cost = demand / Q * self.ordering_cost + ((Q - B) ** 2) / (2 * Q) *self.holding_cost + (B**2)/2(Q) * self.stockout_cost
        self.cost = cost
        return cost
