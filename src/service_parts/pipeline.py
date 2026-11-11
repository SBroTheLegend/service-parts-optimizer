"""Turn a parts catalog into a baseline, a frontier, and one recommended policy."""

from __future__ import annotations

import math

import pandas as pd

from service_parts.config import (
    ABC_ORDER,
    CRITICALITY_ORDER,
    PATTERN_ORDER,
    XYZ_ORDER,
    PolicyConfig,
    validate_config,
)
from service_parts.data.schema import validate_catalog
from service_parts.inventory.math import (
    choose_distribution,
    demand_during_lead_time,
    order_quantity,
)
from service_parts.model import PartGrid, PlanningModel, PolicyPack, PreparedPart
from service_parts.policy.baseline import baseline_point
from service_parts.policy.classify import assign_abc, assign_xyz
from service_parts.policy.optimize import (
    build_frontier,
    build_part_grid,
    choose_indices,
    indices_at_step,
    state_totals,
    step_for_fill_at_least,
    step_within_budget,
)


def prepare_parts(catalog: pd.DataFrame, config: PolicyConfig) -> list[PreparedPart]:
    validate_config(config)
    validate_catalog(catalog)
    frame = catalog.copy()
    frame["part_id"] = frame["part_id"].astype(str)
    annual = frame["weekly_demand_mean"].astype(float) * config.weeks_per_year
    usage = annual * frame["unit_cost"].astype(float)
    frame["abc_class"] = assign_abc(usage)
    frame["xyz_class"] = assign_xyz(frame["weekly_demand_mean"], frame["weekly_demand_std"])
    parts: list[PreparedPart] = []
    for row in frame.to_dict(orient="records"):
        weekly_mean = float(row["weekly_demand_mean"])
        weekly_std = float(row["weekly_demand_std"])
        lead_mean = float(row["lead_time_weeks_mean"]) * config.lead_time_scale
        lead_std = float(row["lead_time_weeks_std"]) * config.lead_time_scale
        ddlt_mean, ddlt_std = demand_during_lead_time(
            weekly_mean, weekly_std, lead_mean, lead_std
        )
        distribution = choose_distribution(
            ddlt_mean,
            ddlt_std,
            cv_threshold=config.normal_cv_threshold,
            min_mean=config.normal_min_mean,
        )
        annual_demand = weekly_mean * config.weeks_per_year
        criticality = str(row["criticality"])
        quantity = 0
        if weekly_mean > 1e-9:
            quantity = order_quantity(
                annual_demand,
                float(row["order_cost"]),
                float(row["unit_cost"]),
                config.holding_rate,
                int(row["moq"]),
            )
        if quantity <= 0:
            kind = "insurance" if criticality == "line_down" else "not_stocked"
            quantity = 0
        else:
            kind = "stocked"
        parts.append(
            PreparedPart(
                part_id=str(row["part_id"]),
                part_number=str(row["part_number"]),
                description=str(row["description"]),
                commodity=str(row["commodity"]),
                criticality=criticality,
                primary_region=str(row["primary_region"]),
                demand_pattern=str(row["demand_pattern"]),
                abc_class=str(row["abc_class"]),
                xyz_class=str(row["xyz_class"]),
                unit_cost=float(row["unit_cost"]),
                order_cost=float(row["order_cost"]),
                moq=int(row["moq"]),
                order_quantity=quantity,
                weekly_demand_mean=weekly_mean,
                weekly_demand_std=weekly_std,
                annual_demand=annual_demand,
                lead_time_weeks_mean=lead_mean,
                lead_time_weeks_std=lead_std,
                ddlt_mean=ddlt_mean,
                ddlt_std=ddlt_std,
                distribution=distribution,
                kind=kind,
                min_fill=config.min_fill(criticality),
            )
        )
    parts.sort(key=lambda part: part.part_id)
    return parts


def build_model(catalog: pd.DataFrame, config: PolicyConfig | None = None) -> PlanningModel:
    config = config or PolicyConfig()
    validate_config(config)
    parts = prepare_parts(catalog, config)
    grids = [build_part_grid(part, config) for part in parts]
    frontier, applications = build_frontier(parts, grids)
    return PlanningModel(
        parts=parts,
        grids=grids,
        applications=applications,
        frontier=frontier,
        config=config,
    )


def run_pipeline(catalog: pd.DataFrame, config: PolicyConfig | None = None) -> PolicyPack:
    config = config or PolicyConfig()
    model = build_model(catalog, config)
    return select_policy(
        model,
        target_fill_rate=config.target_fill_rate,
        baseline_mode=config.baseline_mode,
        baseline_cover_weeks=config.baseline_cover_weeks,
        baseline_cycle_service=config.baseline_cycle_service,
    )


def select_policy(
    model: PlanningModel,
    *,
    target_fill_rate: float,
    baseline_mode: str,
    baseline_cover_weeks: float,
    baseline_cycle_service: float,
) -> PolicyPack:
    if not 0.0 < target_fill_rate < 1.0:
        raise ValueError("target fill rate must be between 0 and 1")
    selection = PolicyConfig(
        target_fill_rate=target_fill_rate,
        holding_rate=model.config.holding_rate,
        baseline_mode=baseline_mode,
        baseline_cover_weeks=baseline_cover_weeks,
        baseline_cycle_service=baseline_cycle_service,
        lead_time_scale=model.config.lead_time_scale,
        z_step=model.config.z_step,
        z_max=model.config.z_max,
        weeks_per_year=model.config.weeks_per_year,
        normal_cv_threshold=model.config.normal_cv_threshold,
        normal_min_mean=model.config.normal_min_mean,
        min_fill_line_down=model.config.min_fill_line_down,
        min_fill_degraded=model.config.min_fill_degraded,
        min_fill_convenience=model.config.min_fill_convenience,
    )
    validate_config(selection)
    indices, target_met, step = choose_indices(
        model.parts,
        model.grids,
        model.frontier,
        model.applications,
        target_fill_rate,
    )
    recommendations = _recommendations(model, indices, selection)
    baseline_frame_totals = _policy_totals_from_frame(recommendations, "baseline")
    recommended_totals = _policy_totals_from_frame(recommendations, "optimized")

    matched_step = step_for_fill_at_least(
        model.frontier, baseline_frame_totals["system_fill_rate"]
    )
    same_budget_step = step_within_budget(
        model.frontier, baseline_frame_totals["on_hand_investment"]
    )
    matched_totals = _totals_at_step(model, matched_step) if matched_step is not None else None
    budget_totals = (
        _totals_at_step(model, same_budget_step) if same_budget_step is not None else None
    )
    frontier = _flag_frontier(
        model.frontier,
        recommended_step=step,
        matched_step=matched_step,
        same_budget_step=same_budget_step,
    )
    segments = summarize_segments(recommendations)
    above_floor = any(
        index > 0 and grid.kind == "stocked"
        for index, grid in zip(indices, model.grids)
    )
    summary = _summary(
        recommendations=recommendations,
        baseline=baseline_frame_totals,
        recommended=recommended_totals,
        matched=matched_totals,
        same_budget=budget_totals,
        selection=selection,
        above_floor=above_floor,
        frontier_step=step,
        target_met=target_met,
        grids=model.grids,
    )
    policy_totals = _policy_table(summary)
    return PolicyPack(
        recommendations=recommendations,
        frontier=frontier,
        segment_summary=segments,
        policy_totals=policy_totals,
        summary=summary,
    )


def _recommendations(
    model: PlanningModel,
    indices: list[int],
    config: PolicyConfig,
) -> pd.DataFrame:
    rows = []
    for part, grid, index in zip(model.parts, model.grids, indices):
        optimized = grid.points[index]
        baseline = baseline_point(part, config)
        cycle_units = part.order_quantity / 2.0 if part.kind == "stocked" else 0.0
        displayed_quantity = part.order_quantity
        if part.kind == "insurance":
            displayed_quantity = max(part.moq, 1)
        coefficient = (
            part.weekly_demand_std / part.weekly_demand_mean
            if part.weekly_demand_mean > 1e-12
            else math.nan
        )
        rows.append(
            {
                "part_id": part.part_id,
                "part_number": part.part_number,
                "description": part.description,
                "commodity": part.commodity,
                "criticality": part.criticality,
                "primary_region": part.primary_region,
                "demand_pattern": part.demand_pattern,
                "abc_class": part.abc_class,
                "xyz_class": part.xyz_class,
                "unit_cost": part.unit_cost,
                "order_cost": part.order_cost,
                "moq": part.moq,
                "order_quantity": displayed_quantity,
                "weekly_demand_mean": part.weekly_demand_mean,
                "weekly_demand_std": part.weekly_demand_std,
                "demand_cv": coefficient,
                "annual_demand": part.annual_demand,
                "lead_time_weeks_mean": part.lead_time_weeks_mean,
                "lead_time_weeks_std": part.lead_time_weeks_std,
                "ddlt_mean": part.ddlt_mean,
                "ddlt_std": part.ddlt_std,
                "distribution": part.distribution,
                "cycle_stock_units": cycle_units,
                "cycle_stock_investment": cycle_units * part.unit_cost,
                "pipeline_units": part.ddlt_mean,
                "pipeline_investment": part.ddlt_mean * part.unit_cost,
                "min_fill_rate": part.min_fill,
                "baseline_reorder_point": baseline.reorder_point,
                "baseline_safety_stock": baseline.safety_stock,
                "baseline_cycle_service": baseline.cycle_service if part.kind == "stocked" else math.nan,
                "baseline_fill_rate": baseline.fill_rate if part.kind == "stocked" else math.nan,
                "baseline_annual_shortages": baseline.annual_shortages,
                "baseline_safety_investment": baseline.safety_investment,
                "baseline_on_hand_units": baseline.on_hand_units,
                "baseline_on_hand_investment": baseline.on_hand_investment,
                "optimized_reorder_point": optimized.reorder_point,
                "optimized_safety_stock": optimized.safety_stock,
                "optimized_cycle_service": optimized.cycle_service if part.kind == "stocked" else math.nan,
                "optimized_fill_rate": optimized.fill_rate if part.kind == "stocked" else math.nan,
                "optimized_annual_shortages": optimized.annual_shortages,
                "optimized_safety_investment": optimized.safety_investment,
                "optimized_on_hand_units": optimized.on_hand_units,
                "optimized_on_hand_investment": optimized.on_hand_investment,
                "floor_met_baseline": _floor_met(part, baseline.fill_rate, baseline.on_hand_units),
                "floor_met_optimized": _floor_met(
                    part, optimized.fill_rate, optimized.on_hand_units
                ),
                "planning_treatment": part.kind,
                "in_service_metric": part.kind == "stocked",
                "position_note": _position_note(part, grid, index),
            }
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values("part_id", kind="mergesort").reset_index(drop=True)


def _floor_met(part: PreparedPart, fill_rate: float, on_hand_units: float) -> bool:
    if part.kind == "not_stocked":
        return True
    if part.kind == "insurance":
        return on_hand_units + 1e-9 >= 1.0
    return fill_rate + 1e-9 >= part.min_fill


def _position_note(part: PreparedPart, grid: PartGrid, index: int) -> str:
    if part.kind == "not_stocked":
        return "No forward demand. Not stocked."
    if part.kind == "insurance":
        return (
            "No forward demand. One unit stays on the shelf because a line-down "
            "part with no forecast still needs a spare."
        )
    if not grid.floor_feasible:
        return (
            "Criticality floor is not reachable inside the safety-stock cap. "
            "Held at the cap."
        )
    if index <= 0:
        return (
            "Held at the criticality floor. Another dollar of safety stock "
            "prevents more shortages on a different part."
        )
    if index >= len(grid.points) - 1:
        return "Held at the safety-stock cap."
    return (
        "Raised above the criticality floor. A dollar of safety stock here "
        "prevents more expected shortages than a dollar left at the floor."
    )


def _policy_totals_from_frame(frame: pd.DataFrame, prefix: str) -> dict:
    service = frame[frame["in_service_metric"]]
    demand = float(service["annual_demand"].sum())
    shorts = float(service[f"{prefix}_annual_shortages"].sum())
    fill = 1.0 if demand <= 0 else 1.0 - shorts / demand
    line_down = frame["criticality"].eq("line_down") & ~frame[f"floor_met_{prefix}"]
    any_miss = ~frame[f"floor_met_{prefix}"]
    return {
        "system_fill_rate": fill,
        "on_hand_investment": float(frame[f"{prefix}_on_hand_investment"].sum()),
        "safety_investment": float(frame[f"{prefix}_safety_investment"].sum()),
        "annual_shortages": shorts,
        "annual_demand": demand,
        "line_down_below_floor": int(line_down.sum()),
        "parts_below_floor": int(any_miss.sum()),
    }


def _totals_at_step(model: PlanningModel, step: int) -> dict:
    indices = indices_at_step(len(model.parts), model.applications, step)
    totals = state_totals(model.parts, model.grids, indices)
    line_down_miss = 0
    parts_miss = 0
    for part, grid, index in zip(model.parts, model.grids, indices):
        point = grid.points[index]
        met = _floor_met(part, point.fill_rate, point.on_hand_units)
        if not met:
            parts_miss += 1
            if part.criticality == "line_down":
                line_down_miss += 1
    totals["line_down_below_floor"] = line_down_miss
    totals["parts_below_floor"] = parts_miss
    totals["step"] = step
    return totals


def _flag_frontier(
    frontier: pd.DataFrame,
    *,
    recommended_step: int,
    matched_step: int | None,
    same_budget_step: int | None,
) -> pd.DataFrame:
    flagged = frontier.copy()
    flagged["is_recommended"] = flagged["step"].eq(recommended_step)
    flagged["is_matched_fill"] = (
        flagged["step"].eq(matched_step) if matched_step is not None else False
    )
    flagged["is_same_budget"] = (
        flagged["step"].eq(same_budget_step) if same_budget_step is not None else False
    )
    return flagged


def summarize_segments(recommendations: pd.DataFrame) -> pd.DataFrame:
    sections = [
        ("criticality", "criticality", CRITICALITY_ORDER),
        ("abc", "abc_class", ABC_ORDER),
        ("xyz", "xyz_class", XYZ_ORDER),
        ("demand_pattern", "demand_pattern", PATTERN_ORDER),
        ("commodity", "commodity", tuple(sorted(recommendations["commodity"].unique()))),
    ]
    rows = []
    for segment_type, column, order in sections:
        present = [name for name in order if (recommendations[column] == name).any()]
        extras = [
            name
            for name in recommendations[column].unique()
            if name not in present
        ]
        for name in present + extras:
            group = recommendations[recommendations[column] == name]
            rows.append(_segment_row(group, segment_type, str(name)))
    return pd.DataFrame(rows)


def _segment_row(group: pd.DataFrame, segment_type: str, segment: str) -> dict:
    service = group[group["in_service_metric"]]
    demand = float(service["annual_demand"].sum())

    def fill(column: str) -> float:
        if demand <= 0:
            return math.nan
        return 1.0 - float(service[column].sum()) / demand

    line_down = group["criticality"].eq("line_down")
    return {
        "segment_type": segment_type,
        "segment": segment,
        "parts": int(len(group)),
        "stocked_parts": int(group["in_service_metric"].sum()),
        "annual_demand": demand,
        "baseline_fill_rate": fill("baseline_annual_shortages"),
        "optimized_fill_rate": fill("optimized_annual_shortages"),
        "baseline_on_hand_investment": float(group["baseline_on_hand_investment"].sum()),
        "optimized_on_hand_investment": float(group["optimized_on_hand_investment"].sum()),
        "baseline_safety_investment": float(group["baseline_safety_investment"].sum()),
        "optimized_safety_investment": float(group["optimized_safety_investment"].sum()),
        "baseline_line_down_below_floor": int((line_down & ~group["floor_met_baseline"]).sum()),
        "optimized_line_down_below_floor": int(
            (line_down & ~group["floor_met_optimized"]).sum()
        ),
    }


def _summary(
    *,
    recommendations: pd.DataFrame,
    baseline: dict,
    recommended: dict,
    matched: dict | None,
    same_budget: dict | None,
    selection: PolicyConfig,
    above_floor: bool,
    frontier_step: int,
    target_met: bool,
    grids: list,
) -> dict:
    if not target_met:
        reason = (
            "The safety-stock cap is reached before the system target. "
            "Every part that can still take stock is at the cap."
        )
    elif not above_floor:
        reason = (
            "Criticality floors already meet the system fill-rate target, "
            "so no part is raised above its floor."
        )
    else:
        reason = (
            "Criticality floors are met first. Further safety stock is added "
            "where each dollar prevents the most expected shortages."
        )
    cycle = float(recommendations["cycle_stock_investment"].sum())
    pipeline = float(recommendations["pipeline_investment"].sum())
    holding = selection.holding_rate
    infeasible = sum(1 for grid in grids if grid.kind == "stocked" and not grid.floor_feasible)
    if selection.baseline_mode == "weeks_of_cover":
        baseline_label = f"{selection.baseline_cover_weeks:g} weeks of cover"
    else:
        baseline_label = f"uniform {selection.baseline_cycle_service:.0%} cycle service"
    return {
        "part_count": int(len(recommendations)),
        "stocked_part_count": int(recommendations["in_service_metric"].sum()),
        "not_stocked_part_count": int(
            (recommendations["planning_treatment"] == "not_stocked").sum()
        ),
        "insurance_part_count": int((recommendations["planning_treatment"] == "insurance").sum()),
        "baseline_label": baseline_label,
        "recommended_label": (
            f"{selection.target_fill_rate:.1%} system fill rate, criticality floors"
        ),
        "recommendation_reason": reason,
        "target_fill_rate": selection.target_fill_rate,
        "target_met": target_met,
        "holding_rate": holding,
        "lead_time_scale": selection.lead_time_scale,
        "baseline_fill_rate": baseline["system_fill_rate"],
        "optimized_fill_rate": recommended["system_fill_rate"],
        "baseline_on_hand_investment": baseline["on_hand_investment"],
        "optimized_on_hand_investment": recommended["on_hand_investment"],
        "on_hand_delta": recommended["on_hand_investment"] - baseline["on_hand_investment"],
        "baseline_safety_investment": baseline["safety_investment"],
        "optimized_safety_investment": recommended["safety_investment"],
        "cycle_stock_investment": cycle,
        "pipeline_investment": pipeline,
        "baseline_annual_holding_cost": holding * baseline["on_hand_investment"],
        "optimized_annual_holding_cost": holding * recommended["on_hand_investment"],
        "baseline_annual_shortages": baseline["annual_shortages"],
        "optimized_annual_shortages": recommended["annual_shortages"],
        "baseline_line_down_below_floor": baseline["line_down_below_floor"],
        "optimized_line_down_below_floor": recommended["line_down_below_floor"],
        "baseline_parts_below_floor": baseline["parts_below_floor"],
        "optimized_parts_below_floor": recommended["parts_below_floor"],
        "floors_infeasible": infeasible,
        "matched_fill_on_hand_investment": None
        if matched is None
        else matched["on_hand_investment"],
        "matched_fill_rate": None if matched is None else matched["system_fill_rate"],
        "same_budget_fill_rate": None if same_budget is None else same_budget["system_fill_rate"],
        "same_budget_on_hand_investment": None
        if same_budget is None
        else same_budget["on_hand_investment"],
        "frontier_step_at_target": frontier_step,
    }


def _policy_table(summary: dict) -> pd.DataFrame:
    def row(name: str, fill, on_hand, safety, shortages, line_down) -> dict:
        return {
            "policy": name,
            "system_fill_rate": fill,
            "on_hand_investment": on_hand,
            "safety_investment": safety,
            "cycle_stock_investment": summary["cycle_stock_investment"],
            "pipeline_investment": summary["pipeline_investment"],
            "annual_holding_cost": None
            if on_hand is None
            else summary["holding_rate"] * on_hand,
            "annual_shortages": shortages,
            "line_down_below_floor": line_down,
        }

    rows = [
        row(
            "baseline",
            summary["baseline_fill_rate"],
            summary["baseline_on_hand_investment"],
            summary["baseline_safety_investment"],
            summary["baseline_annual_shortages"],
            summary["baseline_line_down_below_floor"],
        ),
        row(
            "recommended",
            summary["optimized_fill_rate"],
            summary["optimized_on_hand_investment"],
            summary["optimized_safety_investment"],
            summary["optimized_annual_shortages"],
            summary["optimized_line_down_below_floor"],
        ),
    ]
    if summary["matched_fill_on_hand_investment"] is not None:
        rows.append(
            row(
                "matched_baseline_fill",
                summary["matched_fill_rate"],
                summary["matched_fill_on_hand_investment"],
                None,
                None,
                None,
            )
        )
    if summary["same_budget_fill_rate"] is not None:
        rows.append(
            row(
                "same_on_hand_budget",
                summary["same_budget_fill_rate"],
                summary["same_budget_on_hand_investment"],
                None,
                None,
                None,
            )
        )
    return pd.DataFrame(rows)
