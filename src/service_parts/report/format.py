"""Plain-language figures for the CLI and the review screen."""

from __future__ import annotations

import math


def format_money(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    sign = "-" if value < 0 else ""
    amount = abs(float(value))
    if amount >= 1_000_000:
        return f"{sign}${amount / 1_000_000:.2f}M"
    if amount >= 1_000:
        return f"{sign}${amount:,.0f}"
    return f"{sign}${amount:,.2f}"


def format_pct(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value * 100:.1f}%"


def format_number(value: float | None, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value:,.{digits}f}"


def render_summary(summary: dict) -> str:
    """Short operating summary. Paths are added by the caller."""
    lines = [
        (
            f"{summary['part_count']} parts"
            f"  ·  {summary['stocked_part_count']} with forward demand"
            f"  ·  holding rate {summary['holding_rate']:.0%}"
        ),
        summary["recommendation_reason"],
        "",
        f"Baseline ({summary['baseline_label']})",
        f"  Expected system fill rate     {format_pct(summary['baseline_fill_rate'])}",
        f"  On-hand inventory             {format_money(summary['baseline_on_hand_investment'])}",
        f"  of which safety stock         {format_money(summary['baseline_safety_investment'])}",
        f"  Expected shortages per year   {summary['baseline_annual_shortages']:,.1f}",
        f"  Line-down parts below floor   {summary['baseline_line_down_below_floor']}",
        "",
        f"Recommended ({summary['recommended_label']})",
        f"  Expected system fill rate     {format_pct(summary['optimized_fill_rate'])}",
        f"  On-hand inventory             {format_money(summary['optimized_on_hand_investment'])}",
        f"  of which safety stock         {format_money(summary['optimized_safety_investment'])}",
        f"  Expected shortages per year   {summary['optimized_annual_shortages']:,.1f}",
        f"  Line-down parts below floor   {summary['optimized_line_down_below_floor']}",
    ]
    delta = summary["on_hand_delta"]
    if delta < -1:
        lines.append(f"  Working capital released      {format_money(-delta)}")
    elif delta > 1:
        lines.append(f"  Additional working capital    {format_money(delta)}")
    else:
        lines.append("  On-hand inventory is unchanged at this target.")
    lines.append("")
    lines.append(
        f"Cycle stock, both policies    {format_money(summary['cycle_stock_investment'])}"
    )
    lines.append(
        f"Pipeline inventory            {format_money(summary['pipeline_investment'])}"
    )
    lines.append(
        "Holding cost is applied to on-hand inventory only. "
        "Pipeline capital is reported separately because in-transit ownership "
        "depends on shipping terms."
    )
    matched = summary.get("matched_fill_on_hand_investment")
    same = summary.get("same_budget_fill_rate")
    if matched is not None or same is not None:
        lines.append("")
    if matched is not None:
        released = summary["baseline_on_hand_investment"] - matched
        if released > 1:
            movement = f"which releases {format_money(released)}"
        elif released < -1:
            movement = f"which takes {format_money(-released)} more"
        else:
            movement = "at about the same on-hand investment"
        lines.append(
            "Matching the baseline fill rate of "
            f"{format_pct(summary['baseline_fill_rate'])} on the efficient frontier "
            f"holds {format_money(matched)} on hand, {movement}."
        )
    if same is not None:
        lines.append(
            "Holding the baseline on-hand budget, the efficient frontier reaches "
            f"{format_pct(same)} fill rate."
        )
    if summary.get("floors_infeasible"):
        lines.append(
            f"{summary['floors_infeasible']} stocked parts cannot reach their floor "
            "inside the safety-stock cap."
        )
    return "\n".join(lines)
