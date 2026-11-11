"""The greedy frontier matches an exhaustive search on a known grid."""

from __future__ import annotations

import itertools

import pytest

from service_parts.config import PolicyConfig
from service_parts.model import GridPoint, PartGrid, PreparedPart
from service_parts.policy.optimize import (
    build_frontier,
    build_part_grid,
    choose_indices,
    state_totals,
    step_for_target,
)


def _point(on_hand: float, annual_short: float) -> GridPoint:
    return GridPoint(
        z=0.0,
        reorder_point=0.0,
        safety_stock=0.0,
        cycle_service=0.0,
        shortage_per_cycle=annual_short,
        fill_rate=0.0,
        annual_shortages=annual_short,
        safety_investment=on_hand,
        on_hand_units=0.0,
        on_hand_investment=on_hand,
    )


class _Stub:
    def __init__(self, annual_demand: float):
        self.annual_demand = annual_demand
        self.kind = "stocked"


def test_greedy_matches_enumeration_on_a_known_two_part_grid():
    parts = [_Stub(100), _Stub(80)]
    grids = [
        PartGrid(
            points=(
                _point(100, 30),
                _point(110, 12),
                _point(130, 4),
            ),
            floor_feasible=True,
            kind="stocked",
        ),
        PartGrid(
            points=(
                _point(50, 20),
                _point(80, 5),
                _point(120, 1),
            ),
            floor_feasible=True,
            kind="stocked",
        ),
    ]
    frontier, _applications = build_frontier(parts, grids)
    assert frontier["system_fill_rate"].is_monotonic_increasing
    assert frontier["on_hand_investment"].is_monotonic_increasing

    demand = 180
    for target, expected_on_hand, expected_met in (
        (0.90, 190.0, True),
        (0.95, 210.0, True),
        (0.99, 250.0, False),
    ):
        step, met = step_for_target(frontier, target)
        row = frontier.loc[frontier["step"] == step].iloc[0]
        assert met is expected_met
        assert row["on_hand_investment"] == pytest.approx(expected_on_hand)
        best = None
        for indexes in itertools.product((0, 1, 2), repeat=2):
            totals = state_totals(parts, grids, indexes)
            if totals["system_fill_rate"] + 1e-12 >= target:
                if best is None or totals["on_hand_investment"] < best:
                    best = totals["on_hand_investment"]
        if expected_met:
            assert best == pytest.approx(expected_on_hand)
        else:
            assert best is None
            assert row["system_fill_rate"] == pytest.approx(1 - 5 / demand)


def _prepared(**overrides) -> PreparedPart:
    values = dict(
        part_id="P0001",
        part_number="HFS-HYD-1000",
        description="Pump cartridge 40, flanged",
        commodity="hydraulics",
        criticality="degraded",
        primary_region="North Basin",
        demand_pattern="smooth",
        abc_class="B",
        xyz_class="X",
        unit_cost=40.0,
        order_cost=70.0,
        moq=1,
        order_quantity=30,
        weekly_demand_mean=10.0,
        weekly_demand_std=2.0,
        annual_demand=520.0,
        lead_time_weeks_mean=4.0,
        lead_time_weeks_std=0.2,
        ddlt_mean=40.0,
        ddlt_std=6.0,
        distribution="normal",
        kind="stocked",
        min_fill=0.50,
    )
    values.update(overrides)
    return PreparedPart(**values)


def test_real_grids_agree_with_exhaustive_search():
    config = PolicyConfig(
        z_step=1.0,
        z_max=2.0,
        min_fill_line_down=0.50,
        min_fill_degraded=0.50,
        min_fill_convenience=0.50,
    )
    parts = [
        _prepared(part_id="P0001", unit_cost=15, order_quantity=25, annual_demand=800, ddlt_mean=60, ddlt_std=18),
        _prepared(part_id="P0002", unit_cost=400, order_quantity=12, annual_demand=200, ddlt_mean=30, ddlt_std=4),
        _prepared(
            part_id="P0003",
            unit_cost=80,
            order_quantity=8,
            annual_demand=120,
            ddlt_mean=12,
            ddlt_std=9,
            distribution="gamma",
            min_fill=0.50,
        ),
    ]
    grids = [build_part_grid(part, config) for part in parts]
    assert all(len(grid.points) >= 2 for grid in grids)
    frontier, applications = build_frontier(parts, grids)
    target = 0.97
    indices, met, _step = choose_indices(parts, grids, frontier, applications, target)
    assert met
    chosen = state_totals(parts, grids, indices)
    lengths = [range(len(grid.points)) for grid in grids]
    best_cost = None
    for indexes in itertools.product(*lengths):
        totals = state_totals(parts, grids, indexes)
        if totals["system_fill_rate"] + 1e-9 >= target:
            cost = totals["on_hand_investment"]
            best_cost = cost if best_cost is None else min(best_cost, cost)
    assert chosen["on_hand_investment"] == pytest.approx(best_cost)
    assert chosen["system_fill_rate"] + 1e-12 >= target
