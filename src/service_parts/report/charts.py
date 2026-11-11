"""Static charts for the policy pack. The review screen draws its own."""

from __future__ import annotations

from pathlib import Path

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from service_parts.model import PolicyPack


INK = "#1c2430"
TEAL = "#1f6f6a"
COPPER = "#9a4f2d"
PAPER = "#f7f4ef"
GRID = "#e2dacd"
MUTED = "#5c6670"


def write_charts(pack: PolicyPack, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _tradeoff(pack, output_dir / "tradeoff_curve.png")
    _before_after(pack, output_dir / "before_after_demand_pattern.png")


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.facecolor": PAPER,
            "figure.facecolor": PAPER,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
        }
    )


def _money_tick(value: float, _position: float) -> str:
    amount = abs(value)
    sign = "-" if value < 0 else ""
    if amount >= 1_000_000:
        return f"{sign}${amount / 1_000_000:.1f}M"
    if amount >= 1_000:
        return f"{sign}${amount / 1_000:.0f}k"
    return f"{sign}${amount:.0f}"


def _tradeoff(pack: PolicyPack, path: Path) -> None:
    _apply_style()
    frontier = pack.frontier
    summary = pack.summary
    figure, axis = plt.subplots(figsize=(10.2, 6.2))
    axis.plot(
        frontier["on_hand_investment"],
        frontier["system_fill_rate"],
        color=INK,
        linewidth=2.2,
        label="Efficient frontier",
        zorder=2,
    )
    axis.scatter(
        [summary["optimized_on_hand_investment"]],
        [summary["optimized_fill_rate"]],
        s=64,
        color=TEAL,
        zorder=3,
        label="Recommended",
    )
    axis.scatter(
        [summary["baseline_on_hand_investment"]],
        [summary["baseline_fill_rate"]],
        s=70,
        marker="D",
        color=COPPER,
        zorder=3,
        label="Baseline",
    )
    axis.axhline(
        summary["target_fill_rate"],
        color=TEAL,
        linewidth=0.8,
        linestyle="--",
        alpha=0.8,
        label=f"Target {summary['target_fill_rate']:.1%}",
    )
    axis.set_title("On-hand inventory versus expected system fill rate")
    axis.set_xlabel("On-hand inventory (cycle stock + safety stock)")
    axis.set_ylabel("Expected system fill rate")
    observed = [
        float(frontier["system_fill_rate"].min()),
        float(frontier["system_fill_rate"].max()),
        float(summary["baseline_fill_rate"]),
        float(summary["optimized_fill_rate"]),
    ]
    low = max(0.0, min(observed) - 0.006)
    axis.set_ylim(low, 1.004)
    span = 1.004 - low
    step = 0.01 if span <= 0.06 else 0.02 if span <= 0.12 else 0.05
    first = math.ceil((low + 1e-9) / step) * step
    ticks = []
    value = first
    while value <= 1.0001:
        ticks.append(round(value, 4))
        value += step
    if ticks:
        axis.set_yticks(ticks)
    axis.yaxis.set_major_formatter(plt.FuncFormatter(lambda tick, _pos: f"{tick:.0%}"))
    axis.xaxis.set_major_formatter(plt.FuncFormatter(_money_tick))
    axis.grid(True, axis="both")
    axis.legend(frameon=False, loc="lower right")
    figure.text(
        0.01,
        0.01,
        "Synthetic Halden Field Service sample. Cycle stock is the same on every point; the curve is safety stock.",
        color=MUTED,
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _before_after(pack: PolicyPack, path: Path) -> None:
    _apply_style()
    segments = pack.segment_summary
    order = ["smooth", "erratic", "intermittent"]
    pattern = segments[segments["segment_type"] == "demand_pattern"].copy()
    pattern = pattern[pattern["segment"].isin(order)]
    pattern["segment"] = pd.Categorical(pattern["segment"], categories=order, ordered=True)
    pattern = pattern.sort_values("segment")
    labels = [str(name).title() for name in pattern["segment"]]
    positions = list(range(len(pattern)))
    width = 0.36
    figure, axes = plt.subplots(1, 2, figsize=(10.6, 5.6))

    investment = axes[0]
    investment.bar(
        [position - width / 2 for position in positions],
        pattern["baseline_safety_investment"],
        width=width,
        color=COPPER,
        label="Baseline",
    )
    investment.bar(
        [position + width / 2 for position in positions],
        pattern["optimized_safety_investment"],
        width=width,
        color=TEAL,
        label="Recommended",
    )
    investment.set_xticks(positions, labels)
    investment.set_title("Safety-stock investment")
    investment.set_ylabel("Safety stock at unit cost")
    investment.yaxis.set_major_formatter(plt.FuncFormatter(_money_tick))
    investment.grid(True, axis="y")
    investment.legend(frameon=False)

    service = axes[1]
    service.bar(
        [position - width / 2 for position in positions],
        pattern["baseline_fill_rate"],
        width=width,
        color=COPPER,
        label="Baseline",
    )
    service.bar(
        [position + width / 2 for position in positions],
        pattern["optimized_fill_rate"],
        width=width,
        color=TEAL,
        label="Recommended",
    )
    fills = pd.concat(
        [pattern["baseline_fill_rate"], pattern["optimized_fill_rate"]]
    ).dropna()
    low = 0.90 if len(fills) and float(fills.min()) > 0.92 else 0.0
    if low == 0.0 and len(fills):
        low = max(0.0, float(fills.min()) - 0.05)
    service.set_ylim(low, 1.005)
    service.set_xticks(positions, labels)
    service.set_title("Demand-weighted fill rate")
    service.set_ylabel("Expected fill rate")
    service.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _pos: f"{value:.0%}"))
    service.grid(True, axis="y")

    figure.suptitle("Safety stock moves toward intermittent demand", fontsize=14)
    figure.text(
        0.01,
        0.01,
        "Fill rate is weighted by annual demand. The weeks-of-cover rule overstocks smooth parts and understocks intermittent ones.",
        color=MUTED,
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.96))
    figure.savefig(path, dpi=140)
    plt.close(figure)
