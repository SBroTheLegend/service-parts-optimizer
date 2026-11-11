"""Minimum on-hand inventory for a system fill rate, after criticality floors.

Each part starts at the cheapest reorder point that meets its floor. Further
protection is taken in small steps. Expected shortage is convex in the reorder
point, so each step prevents fewer shortages per dollar than the step before
it. Spending in that order traces the convex efficient frontier.

A coarse step can overshoot the fill-rate target. After the frontier first
clears the target, any step that can be removed without falling back through
the target is removed, worst dollar first. On the production grid those steps
are a small slice of one part's safety stock. The trimmed policy is the one
that is recommended.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Sequence

import pandas as pd

from service_parts.config import PolicyConfig
from service_parts.inventory.math import (
    cycle_service_level,
    expected_shortage,
    reorder_point_from_z,
    unit_fill_rate,
    z_for_fill_rate,
)
from service_parts.model import GridPoint, PartGrid, PreparedPart


def zero_point() -> GridPoint:
    return GridPoint(
        z=0.0,
        reorder_point=0.0,
        safety_stock=0.0,
        cycle_service=1.0,
        shortage_per_cycle=0.0,
        fill_rate=1.0,
        annual_shortages=0.0,
        safety_investment=0.0,
        on_hand_units=0.0,
        on_hand_investment=0.0,
    )


def insurance_point(part: PreparedPart) -> GridPoint:
    """One unit on the shelf. It is not a replenishment cycle, so there is no cycle stock."""
    investment = part.unit_cost
    return GridPoint(
        z=0.0,
        reorder_point=1.0,
        safety_stock=1.0,
        cycle_service=1.0,
        shortage_per_cycle=0.0,
        fill_rate=1.0,
        annual_shortages=0.0,
        safety_investment=investment,
        on_hand_units=1.0,
        on_hand_investment=investment,
    )


def point_from_reorder(part: PreparedPart, reorder_point: float, z: float) -> GridPoint:
    reorder_point = max(float(reorder_point), 0.0)
    raw_shortage = expected_shortage(
        part.ddlt_mean, part.ddlt_std, part.distribution, reorder_point
    )
    fill = unit_fill_rate(raw_shortage, part.order_quantity)
    used_shortage = raw_shortage
    if part.order_quantity > 0 and raw_shortage > part.order_quantity:
        used_shortage = float(part.order_quantity)
    annual_short = part.annual_demand * (1.0 - fill) if part.kind == "stocked" else 0.0
    safety = max(reorder_point - part.ddlt_mean, 0.0)
    cycle_units = part.order_quantity / 2.0 if part.kind == "stocked" else 0.0
    on_hand_units = cycle_units + safety
    service = cycle_service_level(
        part.ddlt_mean, part.ddlt_std, part.distribution, reorder_point
    )
    return GridPoint(
        z=float(z),
        reorder_point=reorder_point,
        safety_stock=safety,
        cycle_service=service,
        shortage_per_cycle=used_shortage,
        fill_rate=fill,
        annual_shortages=annual_short,
        safety_investment=safety * part.unit_cost,
        on_hand_units=on_hand_units,
        on_hand_investment=on_hand_units * part.unit_cost,
    )


def _z_values(step: float, z_max: float) -> list[float]:
    count = int(math.floor(z_max / step + 1e-9))
    values = [round(index * step, 10) for index in range(count + 1)]
    if not values or values[-1] < z_max - 1e-9:
        values.append(round(z_max, 10))
    return values


def _keep_improving(points: list[GridPoint]) -> tuple[GridPoint, ...]:
    """Drop steps that do not cut expected shortages. Keep investment non-decreasing."""
    kept: list[GridPoint] = [points[0]]
    for point in points[1:]:
        previous = kept[-1]
        if point.annual_shortages > previous.annual_shortages + 1e-7:
            continue
        if point.annual_shortages > previous.annual_shortages - 1e-8:
            continue
        if point.on_hand_investment + 1e-4 < previous.on_hand_investment:
            continue
        kept.append(point)
    return tuple(kept)


def build_part_grid(part: PreparedPart, config: PolicyConfig) -> PartGrid:
    if part.kind == "insurance":
        return PartGrid(points=(insurance_point(part),), floor_feasible=True, kind="insurance")
    if part.kind == "not_stocked":
        return PartGrid(points=(zero_point(),), floor_feasible=True, kind="not_stocked")

    z_floor, feasible = z_for_fill_rate(
        part.ddlt_mean,
        part.ddlt_std,
        part.distribution,
        part.order_quantity,
        part.min_fill,
        config.z_max,
    )
    scores = [z for z in _z_values(config.z_step, config.z_max) if z > z_floor + 1e-8]
    scores.insert(0, round(z_floor, 10))
    raw = [
        point_from_reorder(
            part,
            reorder_point_from_z(part.ddlt_mean, part.ddlt_std, part.distribution, z),
            z,
        )
        for z in scores
    ]
    return PartGrid(points=_keep_improving(raw), floor_feasible=feasible, kind="stocked")


def state_totals(parts: Sequence, grids: Sequence[PartGrid], indices: Sequence[int]) -> dict:
    on_hand = 0.0
    safety = 0.0
    shorts = 0.0
    demand = 0.0
    for part, grid, index in zip(parts, grids, indices):
        point = grid.points[index]
        on_hand += point.on_hand_investment
        safety += point.safety_investment
        if part.kind == "stocked":
            shorts += point.annual_shortages
            demand += part.annual_demand
    fill = 1.0 if demand <= 0 else 1.0 - shorts / demand
    return {
        "on_hand_investment": on_hand,
        "safety_investment": safety,
        "annual_shortages": shorts,
        "annual_demand": demand,
        "system_fill_rate": fill,
    }


def _next_ratio(points: tuple[GridPoint, ...], index: int) -> tuple[float, float] | None:
    if index >= len(points) - 1:
        return None
    current = points[index]
    nxt = points[index + 1]
    shortage_cut = current.annual_shortages - nxt.annual_shortages
    investment_add = nxt.on_hand_investment - current.on_hand_investment
    if shortage_cut <= 1e-8:
        return None
    if investment_add <= 1e-4:
        return math.inf, shortage_cut
    return shortage_cut / investment_add, shortage_cut


def build_frontier(
    parts: Sequence,
    grids: Sequence[PartGrid],
) -> tuple[pd.DataFrame, list[tuple[int, int]]]:
    """Full greedy path from the floors to the safety-stock cap."""
    count = len(parts)
    current = [0] * count
    applications: list[tuple[int, int]] = []
    records = [dict(state_totals(parts, grids, current), step=0)]
    heap: list[tuple[float, float, int, int]] = []

    def push(part_index: int) -> None:
        move = _next_ratio(grids[part_index].points, current[part_index])
        if move is None:
            return
        ratio, shortage_cut = move
        heapq.heappush(
            heap,
            (-ratio, -shortage_cut, part_index, current[part_index]),
        )

    for part_index in range(count):
        push(part_index)

    while heap:
        _neg_ratio, _neg_cut, part_index, from_index = heapq.heappop(heap)
        if current[part_index] != from_index:
            continue
        current[part_index] = from_index + 1
        applications.append((part_index, from_index + 1))
        totals = state_totals(parts, grids, current)
        totals["step"] = len(applications)
        records.append(totals)
        push(part_index)

    frontier = pd.DataFrame.from_records(records)
    return frontier, applications


def indices_at_step(
    n_parts: int,
    applications: Sequence[tuple[int, int]],
    step: int,
) -> list[int]:
    indices = [0] * n_parts
    if step < 0:
        raise ValueError("step cannot be negative")
    for part_index, to_index in applications[:step]:
        indices[part_index] = to_index
    return indices


def step_for_target(frontier: pd.DataFrame, target: float) -> tuple[int, bool]:
    fills = frontier["system_fill_rate"].to_numpy()
    hits = [index for index, fill in enumerate(fills) if fill + 1e-12 >= target]
    if not hits:
        return int(frontier["step"].iloc[-1]), False
    return int(frontier["step"].iloc[hits[0]]), True


def step_for_fill_at_least(frontier: pd.DataFrame, fill_rate: float) -> int | None:
    fills = frontier["system_fill_rate"].to_numpy()
    hits = [index for index, fill in enumerate(fills) if fill + 1e-12 >= fill_rate]
    if not hits:
        return None
    return int(frontier["step"].iloc[hits[0]])


def step_within_budget(frontier: pd.DataFrame, budget: float) -> int | None:
    affordable = frontier.index[frontier["on_hand_investment"] <= budget + 1.0]
    if len(affordable) == 0:
        return None
    return int(frontier.loc[affordable[-1], "step"])


def trim_to_target(parts: Sequence, grids: Sequence[PartGrid], indices: Sequence[int], target: float) -> list[int]:
    """Drop grid steps that are not needed to hold the fill-rate target.

    Floors stay in place because a part is never stepped below its first point.
    """
    current = list(indices)
    part_list = list(parts)
    shorts = 0.0
    demand = 0.0
    for part, grid, index in zip(part_list, grids, current):
        if part.kind == "stocked":
            shorts += grid.points[index].annual_shortages
            demand += part.annual_demand
    if demand <= 0:
        return current
    allowed = demand * (1.0 - target)

    while True:
        if shorts > allowed + 1e-8:
            return current
        worst: tuple[float, float, int, float] | None = None
        for part_index, grid in enumerate(grids):
            index = current[part_index]
            if index <= 0:
                continue
            previous = grid.points[index - 1]
            here = grid.points[index]
            extra_short = 0.0
            if part_list[part_index].kind == "stocked":
                extra_short = previous.annual_shortages - here.annual_shortages
            if shorts + extra_short > allowed + 1e-8:
                continue
            move = _next_ratio(grid.points, index - 1)
            ratio = move[0] if move is not None else 0.0
            released = here.on_hand_investment - previous.on_hand_investment
            candidate = (ratio, -released, part_index, extra_short)
            if worst is None or candidate[:3] < worst[:3]:
                worst = candidate
        if worst is None:
            return current
        _ratio, _released, part_index, extra_short = worst
        current[part_index] -= 1
        shorts += extra_short


def choose_indices(
    parts: Sequence,
    grids: Sequence[PartGrid],
    frontier: pd.DataFrame,
    applications: Sequence[tuple[int, int]],
    target: float,
) -> tuple[list[int], bool, int]:
    """Return trimmed positions, whether the target was reached, and the frontier step."""
    step, met = step_for_target(frontier, target)
    indices = indices_at_step(len(list(parts)), applications, step)
    if met:
        indices = trim_to_target(parts, grids, indices, target)
    return indices, met, step
