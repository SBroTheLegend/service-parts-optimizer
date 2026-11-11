"""Shared records for a prepared part, a policy grid, and a finished pack."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import pandas as pd

from service_parts.config import PolicyConfig


class GridPoint(NamedTuple):
    z: float
    reorder_point: float
    safety_stock: float
    cycle_service: float
    shortage_per_cycle: float
    fill_rate: float
    annual_shortages: float
    safety_investment: float
    on_hand_units: float
    on_hand_investment: float


class PartGrid(NamedTuple):
    points: tuple[GridPoint, ...]
    floor_feasible: bool
    kind: str


@dataclass(frozen=True)
class PreparedPart:
    part_id: str
    part_number: str
    description: str
    commodity: str
    criticality: str
    primary_region: str
    demand_pattern: str
    abc_class: str
    xyz_class: str
    unit_cost: float
    order_cost: float
    moq: int
    order_quantity: int
    weekly_demand_mean: float
    weekly_demand_std: float
    annual_demand: float
    lead_time_weeks_mean: float
    lead_time_weeks_std: float
    ddlt_mean: float
    ddlt_std: float
    distribution: str
    kind: str
    min_fill: float


@dataclass
class PlanningModel:
    """Order quantities and lead-time demand are fixed before a target is chosen.

    Moving the fill-rate target or the baseline rule reuses this model. Changing
    the holding rate or the lead time builds a new one, because both change the
    order quantity or the demand during lead time.
    """

    parts: list[PreparedPart]
    grids: list[PartGrid]
    applications: list[tuple[int, int]]
    frontier: pd.DataFrame
    config: PolicyConfig


@dataclass
class PolicyPack:
    recommendations: pd.DataFrame
    frontier: pd.DataFrame
    segment_summary: pd.DataFrame
    policy_totals: pd.DataFrame
    summary: dict
