import pytest
from optiflow.baseline.single_echelon_models._eoq import EOQ, EOQProduction
import numpy as np



# EOQ Tests
# Generate test using pytest.mark.parametrize and using as inputs: "demand_forecasts, holding_cost, ordering_cost, expected_order_quantity, expected_cost"
# TODO: migrate away from unittest




@pytest.mark.parametrize("demand_forecasts, holding_cost, ordering_cost, expected_order_quantity, expected_cost", [(600, 1.75, 50, 185.16401995451028, 324.037034920393)])
def test_calculate_order_quantities(demand_forecasts, holding_cost, ordering_cost, expected_order_quantity, expected_cost):
    model = EOQ(holding_cost=holding_cost, ordering_cost=ordering_cost)
    order_quantity, total_cost = model.calculate_order_quantities(demand_forecasts)
    assert order_quantity == expected_order_quantity
    assert total_cost == expected_cost

@pytest.mark.parametrize("demand_forecasts, holding_cost, ordering_cost,production_rate, expected_order_quantity, expected_cost", [(1300, 0.225, 8,1700,  626.8084945889684, 33.183979125298336)])
def test_calculate_order_quantities_production(demand_forecasts, holding_cost, ordering_cost,production_rate, expected_order_quantity, expected_cost):
    model = EOQProduction(holding_cost=holding_cost, ordering_cost=ordering_cost, production_rate=production_rate)
    order_quantity, total_cost = model.calculate_order_quantities(demand_forecasts)
    assert order_quantity == expected_order_quantity
    assert total_cost == expected_cost

# Generate tests into a class to share inputs across different method tests, dont use unit test for this
