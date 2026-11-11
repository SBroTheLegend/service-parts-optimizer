"""Write the policy pack: part file, totals, segments, frontier, charts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from service_parts.model import PolicyPack
from service_parts.report.charts import write_charts


def write_policy_pack(pack: PolicyPack, output_dir: Path | str) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pack.recommendations.to_csv(output / "policy_recommendations.csv", index=False)
    pack.policy_totals.to_csv(output / "policy_totals.csv", index=False)
    pack.segment_summary.to_csv(output / "segment_summary.csv", index=False)
    _downsample_frontier(pack.frontier).to_csv(output / "tradeoff_curve.csv", index=False)
    (output / "summary.json").write_text(
        json.dumps(_jsonable(pack.summary), indent=2) + "\n",
        encoding="utf-8",
    )
    write_charts(pack, output)
    return output


def _downsample_frontier(frontier: pd.DataFrame, max_points: int = 240) -> pd.DataFrame:
    if len(frontier) <= max_points:
        return frontier.reset_index(drop=True)
    must = set()
    for column in ("is_recommended", "is_matched_fill", "is_same_budget"):
        if column in frontier.columns:
            must.update(frontier.index[frontier[column]].tolist())
    must.add(frontier.index[0])
    must.add(frontier.index[-1])
    stride = max(len(frontier) // max_points, 1)
    chosen = set(range(0, len(frontier), stride))
    # `frontier.index` may be a RangeIndex matching position.
    positions = set(chosen)
    for label in must:
        location = frontier.index.get_loc(label)
        if isinstance(location, int):
            positions.add(location)
    ordered = sorted(positions)
    return frontier.iloc[ordered].reset_index(drop=True)


def _jsonable(summary: dict) -> dict:
    rendered = {}
    for key, value in summary.items():
        if isinstance(value, bool):
            rendered[key] = value
        elif isinstance(value, int) and not isinstance(value, bool):
            rendered[key] = value
        elif isinstance(value, float):
            rendered[key] = None if value != value else value
        else:
            rendered[key] = value
    return rendered
