"""Known fixtures for lead-time demand, the loss function, fill rate, and EOQ."""

from __future__ import annotations

import math

import pytest
from scipy.integrate import quad
from scipy.stats import gamma, norm

from service_parts.inventory.math import (
    choose_distribution,
    cycle_service_level,
    demand_during_lead_time,
    expected_shortage,
    normal_unit_loss,
    order_quantity,
    raw_economic_order_quantity,
    reorder_point_from_z,
    unit_fill_rate,
    z_for_fill_rate,
)


def test_demand_during_lead_time_splits_demand_and_lead_time_noise():
    mean, std = demand_during_lead_time(
        weekly_mean=10, weekly_std=2, lead_time_mean=5, lead_time_std=1
    )
    # Mean is 10 units/week * 5 weeks. Variance is 5 * 2^2 + 10^2 * 1^2.
    assert mean == pytest.approx(50)
    assert std**2 == pytest.approx(5 * 4 + 100 * 1)


def test_fixed_lead_time_uses_the_square_root_of_lead_time_rule():
    mean, std = demand_during_lead_time(
        weekly_mean=20, weekly_std=5, lead_time_mean=4, lead_time_std=0
    )
    assert mean == pytest.approx(80)
    assert std == pytest.approx(10)


def test_distribution_choice_follows_the_published_rule():
    assert choose_distribution(20, 4) == "normal"
    assert choose_distribution(20, 20) == "gamma"
    assert choose_distribution(3, 0.3) == "gamma"
    assert choose_distribution(0, 0) == "degenerate"


def test_normal_unit_loss_at_zero_is_the_standard_normal_density():
    assert normal_unit_loss(0) == pytest.approx(1 / math.sqrt(2 * math.pi))


def test_normal_unit_loss_matches_numerical_integral():
    z = float(norm.ppf(0.95))

    def integrand(x: float) -> float:
        return (x - z) * norm.pdf(x)

    numeric, _error = quad(integrand, z, math.inf, epsabs=1e-10)
    assert normal_unit_loss(z) == pytest.approx(numeric, rel=1e-6)


def test_gamma_shortage_matches_exponential_closed_form():
    # Shape 1 and scale 10 is an exponential with mean 10. E[(Y-r)+] = 10 * exp(-r/10).
    for reorder in (0, 5, 10, 25):
        assert expected_shortage(10, 10, "gamma", reorder) == pytest.approx(
            10 * math.exp(-reorder / 10), rel=1e-9
        )


def test_gamma_shortage_matches_numerical_integral():
    mean, std, reorder = 20.0, 15.0, 25.0
    variance = std**2
    shape = mean**2 / variance
    scale = variance / mean

    def integrand(y: float) -> float:
        return (y - reorder) * gamma.pdf(y, shape, scale=scale)

    numeric, _error = quad(integrand, reorder, math.inf, epsabs=1e-8)
    assert expected_shortage(mean, std, "gamma", reorder) == pytest.approx(numeric, rel=1e-6)


def test_fill_rate_identity_and_clipping():
    assert unit_fill_rate(10, 200) == pytest.approx(0.95)
    assert unit_fill_rate(0, 200) == 1.0
    assert unit_fill_rate(250, 200) == 0.0


def test_worked_example_normal_safety_stock_and_fill_rate():
    mean, std = demand_during_lead_time(20, 5, 4, 0)
    z = float(norm.ppf(0.95))
    reorder = reorder_point_from_z(mean, std, "normal", z)
    shortage = expected_shortage(mean, std, "normal", reorder)
    fill = unit_fill_rate(shortage, 40)
    assert reorder - mean == pytest.approx(std * z)
    assert shortage == pytest.approx(std * normal_unit_loss(z))
    assert fill == pytest.approx(1 - shortage / 40)
    assert cycle_service_level(mean, std, "normal", reorder) == pytest.approx(0.95)


def test_fill_rate_inversion_round_trips_for_a_normal_part():
    mean, std, quantity = 80.0, 10.0, 40.0
    z = 1.2
    reorder = reorder_point_from_z(mean, std, "normal", z)
    fill = unit_fill_rate(expected_shortage(mean, std, "normal", reorder), quantity)
    recovered, feasible = z_for_fill_rate(mean, std, "normal", quantity, fill, z_max=3.5)
    assert feasible
    assert recovered == pytest.approx(z, abs=1e-3)


def test_economic_order_quantity_known_value_and_rounding():
    # Holding cost per unit is 0.25 * 40 = 10.
    # EOQ = sqrt(2 * 5200 * 80 / 10) = sqrt(83200).
    assert raw_economic_order_quantity(5200, 80, 40, 0.25) == pytest.approx(math.sqrt(83200))
    assert order_quantity(5200, 80, 40, 0.25, moq=1) == 289
    assert order_quantity(5200, 80, 40, 0.25, moq=300) == 300
    assert order_quantity(0, 80, 40, 0.25, moq=5) == 0
    # sqrt(2 * 100 * 5 / 10) = sqrt(100) = 10 exactly, so the quantity stays 10.
    assert order_quantity(100, 5, 40, 0.25, moq=1) == 10
