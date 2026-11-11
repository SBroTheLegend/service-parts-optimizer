"""Blanket rules the optimized policy is compared with.

Weeks of cover puts the same number of weeks of average demand on every part
and ignores variability, unit cost, and criticality. Uniform cycle service
uses one probability of no stockout and still ignores unit cost.
"""

from __future__ import annotations

from service_parts.config import PolicyConfig
from service_parts.inventory.math import reorder_point_for_cycle_service
from service_parts.model import GridPoint, PreparedPart
from service_parts.policy.optimize import point_from_reorder, zero_point


def baseline_point(part: PreparedPart, config: PolicyConfig) -> GridPoint:
    if part.kind != "stocked":
        return zero_point()
    if config.baseline_mode == "weeks_of_cover":
        reorder = part.ddlt_mean + config.baseline_cover_weeks * part.weekly_demand_mean
    elif config.baseline_mode == "cycle_service":
        reorder = reorder_point_for_cycle_service(
            part.ddlt_mean,
            part.ddlt_std,
            part.distribution,
            config.baseline_cycle_service,
        )
    else:
        raise ValueError(f"Unknown baseline mode: {config.baseline_mode}")
    score = 0.0
    if part.ddlt_std > 1e-12:
        score = (reorder - part.ddlt_mean) / part.ddlt_std
    return point_from_reorder(part, reorder, score)
