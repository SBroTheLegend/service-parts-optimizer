"""Read a catalog directory. Planning parameters stay in parts.csv."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from service_parts.data.demand import estimate_weekly_demand
from service_parts.data.schema import CatalogError, validate_catalog, validate_history


def load_catalog(data_dir: Path | str) -> pd.DataFrame:
    path = Path(data_dir) / "parts.csv"
    if not path.exists():
        raise CatalogError(f"No parts.csv in {data_dir}")
    frame = pd.read_csv(
        path,
        dtype={
            "part_id": str,
            "part_number": str,
            "description": str,
            "commodity": str,
            "criticality": str,
            "primary_region": str,
            "demand_pattern": str,
        },
    )
    validate_catalog(frame)
    frame["part_id"] = frame["part_id"].astype(str)
    frame["moq"] = frame["moq"].astype(int)
    return frame.reset_index(drop=True)


def load_history(data_dir: Path | str) -> pd.DataFrame:
    path = Path(data_dir) / "weekly_demand.csv"
    if not path.exists():
        raise CatalogError(f"No weekly_demand.csv in {data_dir}")
    frame = pd.read_csv(path, dtype={"part_id": str})
    validate_history(frame)
    return frame


def load_manifest(data_dir: Path | str) -> dict | None:
    path = Path(data_dir) / "manifest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def apply_demand_source(
    catalog: pd.DataFrame,
    history: pd.DataFrame | None,
    source: str,
    window: int,
) -> pd.DataFrame:
    """Keep catalog moments, or overwrite them from the trailing demand window."""
    if source == "catalog":
        return catalog
    if source != "history":
        raise ValueError("demand source must be 'catalog' or 'history'")
    if history is None:
        raise CatalogError("History demand was requested, but no history was loaded.")
    moments = estimate_weekly_demand(history, window=window)
    base = catalog.drop(columns=["weekly_demand_mean", "weekly_demand_std"])
    merged = base.merge(
        moments[["part_id", "weekly_demand_mean", "weekly_demand_std"]],
        on="part_id",
        how="left",
    )
    if merged["weekly_demand_mean"].isna().any():
        raise CatalogError("Demand history is missing part ids that are in the catalog.")
    inactive = merged["weekly_demand_mean"] <= 1e-12
    merged.loc[inactive, "weekly_demand_std"] = 0.0
    validate_catalog(merged)
    return merged
