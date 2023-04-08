import unittest
from optiflow.single_echelon_models._eoq import EOQ
import numpy as np

class EOQTest(unittest.TestCase):
    def setUp(self):
        self.demand_forecasts = np.array([100, 200, 300])
        self.inventory_levels = np.array([50, 100, 150])
        self.model = EOQ(holding_cost=1, ordering_cost=1)

    def test_calculate_order_quantities(self):
        order_quantity = self.model.calculate_order_quantities(self.demand_forecasts, self.inventory_levels)
        expected_order_quantity = 34.64101615137755

        34.64101615137755
        print(expected_order_quantity)
        self.assertEqual(order_quantity, expected_order_quantity)
