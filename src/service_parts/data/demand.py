"""Replace catalog planning moments with sample moments from weekly history.

This is not a forecast. It is the trailing window a planner would use when the
catalog's weekly mean and standard deviation should follow recent demand.
"""

from __future__ import annotations

import pandas as pd

from service_parts.data.schema import validate_history


def estimate_weekly_demand(history: pd.DataFrame, window: int = 52) -> pd.DataFrame:
    """Mean and sample standard deviation over the last `window` weeks of each part."""
    if window < 2:
        raise ValueError("window must cover at least two weeks")
    validate_history(history)
    frame = history.copy()
    frame["part_id"] = frame["part_id"].astype(str)
    frame["week_index"] = pd.to_numeric(frame["week_index"])
    frame["demand_units"] = pd.to_numeric(frame["demand_units"])
    latest = frame.groupby("part_id")["week_index"].transform("max")
    recent = frame[frame["week_index"] > latest - window]
    grouped = recent.groupby("part_id")["demand_units"]
    summary = grouped.agg(
        weekly_demand_mean="mean",
        weekly_demand_std=lambda values: float(values.std(ddof=1)),
        observed_weeks="count",
    )
    summary["weekly_demand_std"] = summary["weekly_demand_std"].fillna(0.0)
    summary = summary.reset_index()
    return summary
