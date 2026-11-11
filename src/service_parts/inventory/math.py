"""Inventory math for a continuous-review (reorder point, order quantity) policy.

The planning bucket is one week. Annual quantities use 52 weeks. Demand during
lead time and lead time itself are treated as independent.

Demand during lead time
    mean = weekly_mean * lead_time_mean
    variance = lead_time_mean * weekly_variance
               + weekly_mean^2 * lead_time_variance

Safety stock for a normal lead-time demand at cycle service level alpha
    z = inverse of the standard normal cdf at alpha
    safety stock = z * std of lead-time demand
    reorder point = mean lead-time demand + safety stock

Expected shortage per replenishment cycle, normal lead-time demand
    std * G(z)
    G(z) = phi(z) - z * (1 - Phi(z))

Unit fill rate
    1 - expected shortage per cycle / order quantity
    clipped to [0, 1]. The identity is the standard continuous-review
    approximation. It overstates the shortage once that shortage exceeds one
    order quantity, so the result is clipped rather than allowed to go negative.

Gamma lead-time demand uses the same fill-rate identity with the gamma loss
    E[(Y - reorder point)+]
    = mean * survival(reorder point; shape + 1)
      - reorder point * survival(reorder point; shape)
in the shape-scale parameterization matched to the mean and variance.
"""

from __future__ import annotations

import math

from scipy.stats import gamma, norm


def demand_during_lead_time(
    weekly_mean: float,
    weekly_std: float,
    lead_time_mean: float,
    lead_time_std: float,
) -> tuple[float, float]:
    """Return the mean and standard deviation of demand during lead time."""
    if weekly_mean < 0 or weekly_std < 0 or lead_time_mean < 0 or lead_time_std < 0:
        raise ValueError("demand and lead-time moments cannot be negative")
    mean = weekly_mean * lead_time_mean
    variance = (
        lead_time_mean * weekly_std**2 + weekly_mean**2 * lead_time_std**2
    )
    return mean, math.sqrt(max(variance, 0.0))


def choose_distribution(
    mean: float,
    std: float,
    *,
    cv_threshold: float = 0.80,
    min_mean: float = 8.0,
) -> str:
    """Pick a lead-time demand curve.

    Normal is used when lead-time demand is large enough, and not so variable,
    that a symmetric curve is a fair planning approximation. Gamma, matched to
    the same mean and variance, is used for small or skewed lead-time demand.
    A part with no uncertainty is degenerate.
    """
    if mean <= 1e-9 or std <= 1e-8:
        return "degenerate"
    coefficient = std / mean
    if coefficient <= cv_threshold and mean >= min_mean:
        return "normal"
    return "gamma"


def normal_unit_loss(z: float) -> float:
    """Standard normal loss G(z) = phi(z) - z * (1 - Phi(z))."""
    z = float(z)
    return float(norm.pdf(z) - z * norm.sf(z))


def _gamma_moments(mean: float, std: float) -> tuple[float, float]:
    variance = std * std
    shape = mean * mean / variance
    scale = variance / mean
    return shape, scale


def expected_shortage(
    mean: float,
    std: float,
    distribution: str,
    reorder_point: float,
) -> float:
    """Expected units short per replenishment cycle, E[(demand - reorder point)+]."""
    reorder_point = max(float(reorder_point), 0.0)
    if distribution == "degenerate" or std <= 1e-12:
        return max(mean - reorder_point, 0.0)
    if distribution == "normal":
        z = (reorder_point - mean) / std
        return max(std * normal_unit_loss(z), 0.0)
    if distribution == "gamma":
        if mean <= 1e-12:
            return 0.0
        shape, scale = _gamma_moments(mean, std)
        value = mean * gamma.sf(reorder_point, shape + 1.0, scale=scale) - reorder_point * gamma.sf(
            reorder_point, shape, scale=scale
        )
        return float(max(value, 0.0))
    raise ValueError(f"Unknown distribution: {distribution}")


def cycle_service_level(
    mean: float,
    std: float,
    distribution: str,
    reorder_point: float,
) -> float:
    """Probability that lead-time demand does not exceed the reorder point."""
    if distribution == "degenerate" or std <= 1e-12:
        return 1.0 if reorder_point >= mean - 1e-9 else 0.0
    if distribution == "normal":
        return float(norm.cdf((reorder_point - mean) / std))
    if distribution == "gamma":
        if mean <= 1e-12:
            return 1.0
        shape, scale = _gamma_moments(mean, std)
        return float(gamma.cdf(max(reorder_point, 0.0), shape, scale=scale))
    raise ValueError(f"Unknown distribution: {distribution}")


def reorder_point_from_z(
    mean: float,
    std: float,
    distribution: str,
    z: float,
) -> float:
    """Map a normal score to a reorder point.

    For a normal curve, z is the safety factor: reorder point = mean + z * std.
    For a gamma curve, z is converted through the standard normal cdf so that
    the reorder point is the matching cycle-service quantile. z = 0 is the
    median of either curve.
    """
    if distribution == "degenerate" or std <= 1e-12:
        return max(mean, 0.0)
    if distribution == "normal":
        return max(mean + float(z) * std, 0.0)
    if distribution == "gamma":
        if mean <= 1e-12:
            return 0.0
        probability = float(norm.cdf(z))
        probability = min(max(probability, 1e-12), 1.0 - 1e-12)
        shape, scale = _gamma_moments(mean, std)
        return float(gamma.ppf(probability, shape, scale=scale))
    raise ValueError(f"Unknown distribution: {distribution}")


def reorder_point_for_cycle_service(
    mean: float,
    std: float,
    distribution: str,
    cycle_service: float,
) -> float:
    """Reorder point that hits a cycle service level on the chosen curve."""
    if not 0.0 < cycle_service < 1.0:
        raise ValueError("cycle service must be between 0 and 1")
    if distribution == "degenerate" or std <= 1e-12:
        return max(mean, 0.0)
    if distribution == "normal":
        return reorder_point_from_z(mean, std, distribution, float(norm.ppf(cycle_service)))
    if distribution == "gamma":
        if mean <= 1e-12:
            return 0.0
        shape, scale = _gamma_moments(mean, std)
        probability = min(max(cycle_service, 1e-12), 1.0 - 1e-12)
        return float(gamma.ppf(probability, shape, scale=scale))
    raise ValueError(f"Unknown distribution: {distribution}")


def unit_fill_rate(shortage_per_cycle: float, order_quantity: float) -> float:
    """Share of one replenishment cycle's demand filled from stock."""
    if order_quantity <= 0:
        return 1.0 if shortage_per_cycle <= 1e-12 else 0.0
    ratio = shortage_per_cycle / order_quantity
    if ratio <= 0:
        return 1.0
    if ratio >= 1:
        return 0.0
    return 1.0 - ratio


def raw_economic_order_quantity(
    annual_demand: float,
    order_cost: float,
    unit_cost: float,
    holding_rate: float,
) -> float:
    """Classic EOQ, sqrt(2 * D * S / (h * c)), before the minimum order quantity."""
    if annual_demand <= 0 or order_cost <= 0 or unit_cost <= 0 or holding_rate <= 0:
        raise ValueError("EOQ inputs must be positive")
    holding_cost = holding_rate * unit_cost
    return math.sqrt(2.0 * annual_demand * order_cost / holding_cost)


def order_quantity(
    annual_demand: float,
    order_cost: float,
    unit_cost: float,
    holding_rate: float,
    moq: int,
) -> int:
    """EOQ rounded up to a whole unit, and at least the minimum order quantity.

    A part with no annual demand returns 0. The caller decides whether a
    line-down part with no forecast still keeps an insurance unit.
    """
    if annual_demand <= 1e-12:
        return 0
    raw = raw_economic_order_quantity(annual_demand, order_cost, unit_cost, holding_rate)
    quantity = max(raw, float(moq), 1.0)
    return int(math.ceil(quantity - 1e-9))


def z_for_fill_rate(
    mean: float,
    std: float,
    distribution: str,
    order_quantity_units: float,
    fill_rate: float,
    z_max: float,
) -> tuple[float, bool]:
    """Smallest z in [0, z_max] whose unit fill rate meets the target.

    The second value is False when even z_max misses the target.
    """
    if not 0.0 < fill_rate < 1.0:
        raise ValueError("fill rate must be between 0 and 1")
    if z_max <= 0:
        raise ValueError("z_max must be positive")

    def fill_at(z: float) -> float:
        reorder = reorder_point_from_z(mean, std, distribution, z)
        shortage = expected_shortage(mean, std, distribution, reorder)
        return unit_fill_rate(shortage, order_quantity_units)

    if fill_at(0.0) >= fill_rate:
        return 0.0, True
    if fill_at(z_max) < fill_rate - 1e-12:
        return float(z_max), False
    low = 0.0
    high = float(z_max)
    for _ in range(60):
        mid = 0.5 * (low + high)
        if fill_at(mid) >= fill_rate:
            high = mid
        else:
            low = mid
    return high, True
