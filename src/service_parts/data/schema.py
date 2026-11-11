"""Catalog checks. Bad rows stop the run instead of becoming quiet zeros."""

from __future__ import annotations

import pandas as pd

from service_parts.config import CRITICALITY_ORDER


REQUIRED_COLUMNS = (
    "part_id",
    "part_number",
    "description",
    "commodity",
    "criticality",
    "primary_region",
    "unit_cost",
    "order_cost",
    "moq",
    "weekly_demand_mean",
    "weekly_demand_std",
    "lead_time_weeks_mean",
    "lead_time_weeks_std",
    "demand_pattern",
)

HISTORY_COLUMNS = ("part_id", "week_index", "demand_units")


class CatalogError(ValueError):
    """The parts file or the demand history cannot be planned as written."""


def validate_catalog(frame: pd.DataFrame) -> None:
    problems: list[str] = []
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        problems.append("missing columns: " + ", ".join(missing))
        _raise(problems)
    if frame.empty:
        problems.append("catalog has no parts")
        _raise(problems)

    identifiers = frame["part_id"].astype(str)
    if identifiers.str.strip().eq("").any():
        problems.append("blank part_id")
    if identifiers.duplicated().any():
        problems.append("duplicate part_id")
    if frame["part_number"].astype(str).duplicated().any():
        problems.append("duplicate part_number")

    unknown = sorted(set(frame["criticality"]) - set(CRITICALITY_ORDER))
    if unknown:
        problems.append("unknown criticality: " + ", ".join(unknown))

    numeric = [
        "unit_cost",
        "order_cost",
        "moq",
        "weekly_demand_mean",
        "weekly_demand_std",
        "lead_time_weeks_mean",
        "lead_time_weeks_std",
    ]
    for column in numeric:
        coerced = pd.to_numeric(frame[column], errors="coerce")
        if coerced.isna().any():
            problems.append(f"{column} has non-numeric values")

    if problems:
        _raise(problems)

    costs = pd.to_numeric(frame["unit_cost"])
    orders = pd.to_numeric(frame["order_cost"])
    moq = pd.to_numeric(frame["moq"])
    mean = pd.to_numeric(frame["weekly_demand_mean"])
    std = pd.to_numeric(frame["weekly_demand_std"])
    lead = pd.to_numeric(frame["lead_time_weeks_mean"])
    lead_std = pd.to_numeric(frame["lead_time_weeks_std"])

    if (costs <= 0).any():
        problems.append("unit_cost must be positive")
    if (orders <= 0).any():
        problems.append("order_cost must be positive")
    if (moq < 1).any() or (moq != moq.round()).any():
        problems.append("moq must be a whole number of at least 1")
    if (mean < 0).any() or (std < 0).any():
        problems.append("weekly demand moments cannot be negative")
    if ((mean <= 1e-12) & (std > 1e-9)).any():
        problems.append("weekly demand std must be 0 when the mean is 0")
    if (lead <= 0).any() or (lead_std < 0).any():
        problems.append("lead time mean must be positive and its std cannot be negative")
    if frame["description"].astype(str).str.strip().eq("").any():
        problems.append("description cannot be blank")
    _raise(problems)


def validate_history(frame: pd.DataFrame) -> None:
    problems: list[str] = []
    missing = [column for column in HISTORY_COLUMNS if column not in frame.columns]
    if missing:
        problems.append("history missing columns: " + ", ".join(missing))
        _raise(problems)
    if frame.empty:
        problems.append("demand history is empty")
    demand = pd.to_numeric(frame["demand_units"], errors="coerce")
    weeks = pd.to_numeric(frame["week_index"], errors="coerce")
    if demand.isna().any() or weeks.isna().any():
        problems.append("history has non-numeric week_index or demand_units")
    elif (demand < 0).any():
        problems.append("demand history cannot be negative")
    _raise(problems)


def _raise(problems: list[str]) -> None:
    if not problems:
        return
    preview = problems[:8]
    extra = len(problems) - len(preview)
    suffix = f" (+{extra} more)" if extra else ""
    raise CatalogError("; ".join(preview) + suffix)
