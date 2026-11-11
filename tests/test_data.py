"""Catalog validation, history moments, and the synthetic generator."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from service_parts.data.demand import estimate_weekly_demand
from service_parts.data.generate import generate_dataset
from service_parts.data.schema import CatalogError, validate_catalog


def _valid_row(**overrides) -> dict:
    row = {
        "part_id": "P0001",
        "part_number": "HFS-HYD-1000",
        "description": "Pump cartridge 40, flanged",
        "commodity": "hydraulics",
        "criticality": "degraded",
        "primary_region": "North Basin",
        "unit_cost": 80.0,
        "order_cost": 60.0,
        "moq": 1,
        "weekly_demand_mean": 10.0,
        "weekly_demand_std": 2.0,
        "lead_time_weeks_mean": 4.0,
        "lead_time_weeks_std": 0.4,
        "demand_pattern": "smooth",
    }
    row.update(overrides)
    return row


def test_catalog_rejects_a_negative_unit_cost():
    frame = pd.DataFrame([_valid_row(unit_cost=-5)])
    with pytest.raises(CatalogError, match="unit_cost"):
        validate_catalog(frame)


def test_catalog_rejects_duplicate_part_ids():
    frame = pd.DataFrame([_valid_row(), _valid_row(part_number="HFS-HYD-1001")])
    with pytest.raises(CatalogError, match="duplicate part_id"):
        validate_catalog(frame)


def test_estimate_weekly_demand_uses_the_trailing_window_only():
    rows = []
    for week in range(1, 11):
        rows.append({"part_id": "A", "week_index": week, "demand_units": week})
    rows.append({"part_id": "B", "week_index": 10, "demand_units": 0})
    rows.append({"part_id": "B", "week_index": 9, "demand_units": 0})
    history = pd.DataFrame(rows)
    summary = estimate_weekly_demand(history, window=4).set_index("part_id")
    window = [7, 8, 9, 10]
    mean = sum(window) / 4
    variance = sum((value - mean) ** 2 for value in window) / 3
    assert summary.loc["A", "weekly_demand_mean"] == pytest.approx(mean)
    assert summary.loc["A", "weekly_demand_std"] == pytest.approx(math.sqrt(variance))
    assert summary.loc["B", "weekly_demand_mean"] == pytest.approx(0)
    assert summary.loc["B", "weekly_demand_std"] == pytest.approx(0)


def test_generator_is_deterministic_and_marks_the_sample_as_fictional(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    generate_dataset(first, n_parts=16, n_weeks=20, seed=3)
    generate_dataset(second, n_parts=16, n_weeks=20, seed=3)
    assert (first / "parts.csv").read_bytes() == (second / "parts.csv").read_bytes()
    assert (first / "weekly_demand.csv").read_bytes() == (second / "weekly_demand.csv").read_bytes()
    manifest = (first / "manifest.json").read_text(encoding="utf-8")
    assert '"fictional": true' in manifest
    other = tmp_path / "other"
    generate_dataset(other, n_parts=16, n_weeks=20, seed=4)
    assert (first / "parts.csv").read_bytes() != (other / "parts.csv").read_bytes()
