"""ABC by annual usage value, XYZ by the coefficient of variation of weekly demand."""

from __future__ import annotations

import pandas as pd


def assign_abc(annual_usage_value: pd.Series) -> pd.Series:
    """A until 80% of usage value, B until 95%, then C.

    The part that crosses a cutoff stays in the higher class. Ties keep the
    original row order.
    """
    classes = pd.Series("C", index=annual_usage_value.index, dtype="object")
    total = float(annual_usage_value.sum())
    if total <= 0:
        return classes
    ordered = annual_usage_value.sort_values(ascending=False, kind="mergesort")
    running = 0.0
    for index, value in ordered.items():
        previous = running / total
        running += float(value)
        if previous < 0.80:
            classes.loc[index] = "A"
        elif previous < 0.95:
            classes.loc[index] = "B"
        else:
            classes.loc[index] = "C"
    return classes


def assign_xyz(weekly_mean: pd.Series, weekly_std: pd.Series) -> pd.Series:
    """X if CV < 0.5, Y if CV < 1, otherwise Z. No demand is Z."""
    mean = weekly_mean.astype(float)
    std = weekly_std.astype(float)
    coefficient = std / mean.where(mean > 1e-12)
    classes = pd.Series("Z", index=mean.index, dtype="object")
    classes[coefficient < 1.0] = "Y"
    classes[coefficient < 0.5] = "X"
    return classes
