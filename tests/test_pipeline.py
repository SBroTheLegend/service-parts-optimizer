"""Policy totals, criticality floors, and a smoke test through the file pack."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from service_parts.cli import main
from service_parts.config import PolicyConfig
from service_parts.data.generate import generate_dataset
from service_parts.data.load import load_catalog
from service_parts.pipeline import run_pipeline
from service_parts.policy.baseline import baseline_point
from service_parts.pipeline import prepare_parts

ROOT = Path(__file__).resolve().parents[1]


def _catalog() -> pd.DataFrame:
    rows = [
        _row("P0001", "HFS-HYD-1000", "degraded", "smooth", 25, 4, 4, 0.3, 80, 60),
        _row("P0002", "HFS-CTL-1001", "line_down", "intermittent", 1.2, 2.4, 8, 1.2, 90, 70),
        _row("P0003", "HFS-PWR-1002", "convenience", "smooth", 8, 1.2, 6, 0.5, 2400, 110),
        _row("P0004", "HFS-SEN-1003", "line_down", "inactive", 0, 0, 5, 0.4, 600, 80),
        _row("P0005", "HFS-FLT-1004", "convenience", "inactive", 0, 0, 2, 0.2, 18, 40),
    ]
    return pd.DataFrame(rows)


def _row(part_id, part_number, criticality, pattern, mean, std, lead, lead_std, cost, order_cost):
    return {
        "part_id": part_id,
        "part_number": part_number,
        "description": f"Synthetic part {part_number}",
        "commodity": "hydraulics",
        "criticality": criticality,
        "primary_region": "North Basin",
        "unit_cost": cost,
        "order_cost": order_cost,
        "moq": 1,
        "weekly_demand_mean": mean,
        "weekly_demand_std": std,
        "lead_time_weeks_mean": lead,
        "lead_time_weeks_std": lead_std,
        "demand_pattern": pattern,
    }


def test_policy_comparison_totals_match_the_part_file():
    pack = run_pipeline(_catalog(), PolicyConfig(target_fill_rate=0.97, baseline_cover_weeks=2))
    rec = pack.recommendations
    summary = pack.summary

    assert len(rec) == 5
    assert summary["part_count"] == 5
    assert rec["optimized_on_hand_investment"].sum() == pytest.approx(
        summary["optimized_on_hand_investment"]
    )
    assert rec["baseline_on_hand_investment"].sum() == pytest.approx(
        summary["baseline_on_hand_investment"]
    )
    assert rec["optimized_safety_investment"].sum() == pytest.approx(
        summary["optimized_safety_investment"]
    )
    assert rec["baseline_safety_investment"].sum() == pytest.approx(
        summary["baseline_safety_investment"]
    )
    assert (
        rec["cycle_stock_investment"] + rec["optimized_safety_investment"]
    ).sum() == pytest.approx(rec["optimized_on_hand_investment"].sum())
    assert (
        rec["cycle_stock_investment"] + rec["baseline_safety_investment"]
    ).sum() == pytest.approx(rec["baseline_on_hand_investment"].sum())

    service = rec[rec["in_service_metric"]]
    demand = service["annual_demand"].sum()
    optimized_fill = 1 - service["optimized_annual_shortages"].sum() / demand
    baseline_fill = 1 - service["baseline_annual_shortages"].sum() / demand
    assert optimized_fill == pytest.approx(summary["optimized_fill_rate"])
    assert baseline_fill == pytest.approx(summary["baseline_fill_rate"])
    assert summary["target_met"]
    assert summary["optimized_fill_rate"] + 1e-9 >= 0.97

    stocked = rec[rec["planning_treatment"] == "stocked"]
    assert (stocked["optimized_fill_rate"] + 1e-6 >= stocked["min_fill_rate"]).all()
    assert summary["optimized_line_down_below_floor"] == 0

    insurance = rec[rec["planning_treatment"] == "insurance"].iloc[0]
    assert insurance["part_id"] == "P0004"
    assert insurance["optimized_on_hand_units"] == pytest.approx(1)
    assert insurance["optimized_on_hand_investment"] == pytest.approx(600)
    assert insurance["baseline_on_hand_units"] == pytest.approx(0)
    assert insurance["floor_met_optimized"]
    assert not insurance["floor_met_baseline"]
    assert not insurance["in_service_metric"]

    idle = rec[rec["planning_treatment"] == "not_stocked"].iloc[0]
    assert idle["baseline_on_hand_investment"] == pytest.approx(0)
    assert idle["optimized_on_hand_investment"] == pytest.approx(0)

    frontier = pack.frontier
    assert (frontier["system_fill_rate"].diff().dropna() >= -1e-9).all()
    assert (frontier["on_hand_investment"].diff().dropna() >= -0.05).all()

    criticality = pack.segment_summary[pack.segment_summary["segment_type"] == "criticality"]
    assert set(criticality["segment"]) == {"line_down", "degraded", "convenience"}
    line_down = criticality[criticality["segment"] == "line_down"].iloc[0]
    assert line_down["optimized_line_down_below_floor"] == 0
    assert line_down["baseline_line_down_below_floor"] >= 1


def test_weeks_of_cover_safety_stock_ignores_variability():
    catalog = _catalog()
    config = PolicyConfig(baseline_cover_weeks=2)
    parts = {part.part_id: part for part in prepare_parts(catalog, config)}
    stocked = parts["P0001"]
    point = baseline_point(stocked, config)
    assert point.safety_stock == pytest.approx(2 * stocked.weekly_demand_mean)
    assert point.reorder_point == pytest.approx(stocked.ddlt_mean + point.safety_stock)


def test_pipeline_smoke_writes_the_policy_pack(tmp_path):
    data = tmp_path / "data"
    output = tmp_path / "out"
    generate_dataset(data, n_parts=24, n_weeks=30, seed=11)
    assert main(
        [
            "run",
            "--data",
            str(data),
            "--output",
            str(output),
            "--target-fill",
            "0.96",
            "--cover-weeks",
            "2",
        ]
    ) == 0
    recommendations = pd.read_csv(output / "policy_recommendations.csv")
    assert len(recommendations) == 24
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["part_count"] == 24
    assert summary["target_fill_rate"] == pytest.approx(0.96)
    assert (output / "policy_totals.csv").exists()
    assert (output / "segment_summary.csv").exists()
    assert (output / "tradeoff_curve.csv").exists()
    for name in ("tradeoff_curve.png", "before_after_demand_pattern.png"):
        payload = (output / name).read_bytes()
        assert payload.startswith(b"\x89PNG")
        assert len(payload) > 5_000
    catalog = load_catalog(data)
    pack = run_pipeline(catalog, PolicyConfig(target_fill_rate=0.96, baseline_cover_weeks=2))
    assert pack.summary["optimized_on_hand_investment"] == pytest.approx(
        summary["optimized_on_hand_investment"]
    )


def test_committed_halden_sample_is_fictional_and_runnable():
    data = ROOT / "data" / "halden_sample"
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["fictional"] is True
    assert manifest["network"] == "Halden Field Service"
    catalog = load_catalog(data)
    assert len(catalog) == 200
    assert catalog["part_id"].is_unique
    assert set(catalog["criticality"]) <= {"line_down", "degraded", "convenience"}
    history = pd.read_csv(data / "weekly_demand.csv")
    assert (history["demand_units"] >= 0).all()
    assert set(history["part_id"]) == set(catalog["part_id"])
    inactive = set(catalog.loc[catalog["demand_pattern"] == "inactive", "part_id"])
    recent = history[history["week_index"] > history["week_index"].max() - 52]
    recent_inactive = recent[recent["part_id"].isin(inactive)]
    assert recent_inactive["demand_units"].sum() == 0
    pack = run_pipeline(catalog, PolicyConfig())
    assert pack.summary["part_count"] == 200
    assert pack.summary["target_met"]
    assert pack.summary["optimized_line_down_below_floor"] == 0
    committed = json.loads((ROOT / "outputs" / "summary.json").read_text(encoding="utf-8"))
    assert pack.summary["optimized_on_hand_investment"] == pytest.approx(
        committed["optimized_on_hand_investment"]
    )
    assert pack.summary["baseline_fill_rate"] == pytest.approx(committed["baseline_fill_rate"])
    feasible = pack.recommendations["planning_treatment"].eq("stocked")
    stocked = pack.recommendations.loc[feasible]
    assert (stocked["optimized_fill_rate"] + 1e-6 >= stocked["min_fill_rate"]).all()
