import unittest
from optiflow.baseline.single_echelon_models._eoq import EOQ, EOQProduction
import numpy as np


class EOQTest(unittest.TestCase):
    # TODO migrate away from unittest
    def setUp(self):
        self.demand_forecasts = 600
        holding_cost = 1.75
        ordering_cost = 50
        self.expected_order_quantity = 185.16401995451028
        self.expected_cost = 324.037034920393
        self.model = EOQ(holding_cost=holding_cost, ordering_cost=ordering_cost)

    def test_calculate_order_quantities(self):
        order_quantity, total_cost = self.model.calculate_order_quantities(
            self.demand_forecasts
        )
        self.assertEqual(order_quantity, self.expected_order_quantity)

    def test_calculate_total_cost(self):
        print(self.demand_forecasts)
        total_cost = self.model.total_cost(
            self.demand_forecasts, Q=self.expected_order_quantity
        )
        self.assertEqual(total_cost, self.expected_cost)


class EOQTestProduction(unittest.TestCase):
    # TODO migrate away from unittest
    def setUp(self):
        self.demand_forecasts = 1300
        self.expected_order_quantity = 626.8084945889684
        self.expected_cost = 33.183979125298336
        holding_cost = 0.225
        ordering_cost = 8
        production_rate = 1700
        self.model = EOQProduction(
            holding_cost=holding_cost,
            ordering_cost=ordering_cost,
            production_rate=production_rate,
        )

    def test_calculate_order_quantities(self):
        order_quantity, total_cost = self.model.calculate_order_quantities(
            self.demand_forecasts
        )
        self.assertEqual(order_quantity, self.expected_order_quantity)

    def test_calculate_total_cost(self):
        print(self.demand_forecasts)
        total_cost = self.model.total_cost(
            self.demand_forecasts, Q=self.expected_order_quantity
        )
        self.assertEqual(total_cost, self.expected_cost)