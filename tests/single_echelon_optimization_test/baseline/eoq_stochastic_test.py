import pytest
from optiflow.baseline.single_echelon_models._eoq_stochastic import EOQStochasticDemand



# EOQ stochastic tests.
# Generate test for EOQStochastic method reorder_point, assuming a normal distribution of demand. 
@pytest.mark.parametrize("ordering_cost, holding_cost, demand_distribution, n_periods,service_level ,mean, std, expected_reorder_point, ", [(50, 1.75, "norm", 1, 0.8413, 100, 25, 124.99537734036862)])
def test_reorder_point(ordering_cost, holding_cost, demand_distribution, n_periods,service_level ,mean, std, expected_reorder_point):
    model = EOQStochasticDemand(ordering_cost=ordering_cost, holding_cost=holding_cost, demand_distribution=demand_distribution, n_periods=n_periods, service_level=service_level, loc=mean, scale=std)
    reorder_point = model.reorder_point(service_level=service_level, loc=mean, scale=std)
    assert reorder_point == expected_reorder_point


@pytest.mark.parametrize("ordering_cost, holding_cost, demand_distribution, n_periods,initial_inventory ,mean, std, expected_service_level", [(50, 1.75, "norm", 1, 125, 100, 25, 0.8413447460685429)])
def test_expected_service_level(ordering_cost, holding_cost, demand_distribution, n_periods, initial_inventory,mean, std, expected_service_level):
    model = EOQStochasticDemand(ordering_cost=ordering_cost, holding_cost=holding_cost, demand_distribution=demand_distribution, n_periods=n_periods, initial_inventory=initial_inventory, loc=mean, scale=std)
    cycle_ss = model.expected_service_level(initial_inventory=initial_inventory, loc=mean, scale=std)
    assert cycle_ss == expected_service_level

