from abc import ABC, abstractmethod

class OrderQuantityModel(ABC):
    """Abstract base class for order quantity models.
    Methods
    -------
    calculate_order_quantities(demand_forecasts, inventory_levels)
        Calculates the order quantities for the given demand forecasts and inventory levels.

    """

    @abstractmethod
    def calculate_order_quantities(self, demand_forecasts, inventory_levels):
        """Calculates the order quantities for the given demand forecasts and inventory levels.

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
        pass
