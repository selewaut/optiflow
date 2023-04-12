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

    def calculate_order_quantities(self, demand_forecasts, inventory_levels):
        """Calculates the order quantities using the EOQ model.

        Parameters
        ----------
        demand_forecasts : array-like of shape (n_periods,)
            The demand forecasts for the next n periods.
        inventory_levels : array-like of shape (n_periods,)
            The inventory levels at the beginning of each of the next n periods.

        Returns
        -------
        order_quantities : array-like of shape (n_periods,)
            The order quantities for each of the next n periods.
        """
        eoq = np.sqrt((2 * self.ordering_cost * np.sum(demand_forecasts)) / self.holding_cost)
        self.eoq = eoq

        return self.eoq
