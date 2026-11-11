"""Review screen for the same policy model the command line writes to disk.

Run from the repository root:

    streamlit run app/streamlit_app.py --server.port 47321
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from service_parts.config import CRITICALITY_LABELS, PolicyConfig
from service_parts.data.load import load_catalog, load_manifest
from service_parts.data.schema import CatalogError
from service_parts.pipeline import build_model, select_policy
from service_parts.report.format import format_money, format_pct

DATA_DIR = ROOT / "data" / "halden_sample"

INK = "#1c2430"
TEAL = "#1f6f6a"
COPPER = "#9a4f2d"
PAPER = "#f7f4ef"
GRID = "#e2dacd"


def main() -> None:
    st.set_page_config(
        page_title="Service Parts Optimizer",
        page_icon="▣",
        layout="wide",
    )
    _css()
    st.title("Service Parts Optimizer")
    st.markdown(
        "Buy the fill rate with as little warehouse inventory as possible. "
        "The baseline is a weeks-of-cover rule. The recommendation meets a "
        "criticality floor on every part, then puts the next dollar of safety "
        "stock where it prevents the most expected shortages."
    )

    try:
        catalog = load_catalog(DATA_DIR)
    except CatalogError as exc:
        st.error(str(exc))
        st.code("python -m service_parts.cli generate && python -m service_parts.cli run")
        return

    manifest = load_manifest(DATA_DIR) or {}
    network = manifest.get("network", "Service-parts catalog")
    if manifest.get("fictional", True):
        st.caption(f"{network} is a synthetic sample. No real operator is represented.")

    defaults = PolicyConfig()
    with st.sidebar:
        st.header("Policy")
        target = st.slider(
            "System fill-rate target",
            min_value=0.90,
            max_value=0.995,
            value=float(defaults.target_fill_rate),
            step=0.005,
            help="Demand-weighted share of units filled from stock, across every stocked part.",
        )
        baseline_mode = st.radio(
            "Baseline",
            options=("weeks_of_cover", "cycle_service"),
            format_func=lambda mode: (
                "Weeks of cover" if mode == "weeks_of_cover" else "Uniform cycle service"
            ),
        )
        cover_weeks = st.slider(
            "Baseline cover (weeks)",
            min_value=0.0,
            max_value=8.0,
            value=float(defaults.baseline_cover_weeks),
            step=0.5,
            disabled=baseline_mode != "weeks_of_cover",
        )
        cycle_service = st.slider(
            "Baseline cycle service",
            min_value=0.80,
            max_value=0.99,
            value=float(defaults.baseline_cycle_service),
            step=0.01,
            disabled=baseline_mode != "cycle_service",
            help="Probability of no stockout in a replenishment cycle. Not the fill rate.",
        )
        holding_rate = st.slider(
            "Holding rate",
            min_value=0.10,
            max_value=0.40,
            value=float(defaults.holding_rate),
            step=0.01,
            help="Cost of capital plus warehousing, applied to on-hand inventory. It changes the order quantity.",
        )
        reduction = st.slider(
            "Lead-time reduction",
            min_value=0.0,
            max_value=0.40,
            value=0.0,
            step=0.05,
            help="Shrinks supplier lead time. Pipeline inventory and safety stock both move. Cycle stock does not.",
        )
        st.divider()
        st.caption(
            "Floors — line-down 98.5%, degraded 96%, convenience 90%. "
            "A line-down part with no forward demand keeps one unit."
        )

    try:
        with st.spinner("Computing the inventory–service frontier…"):
            model = _cached_model(float(holding_rate), 1.0 - float(reduction))
            pack = select_policy(
                model,
                target_fill_rate=float(target),
                baseline_mode=baseline_mode,
                baseline_cover_weeks=float(cover_weeks),
                baseline_cycle_service=float(cycle_service),
            )
    except (CatalogError, ValueError) as exc:
        st.error(str(exc))
        return

    summary = pack.summary
    _metrics(summary)
    st.markdown(summary["recommendation_reason"])

    chart_col, bar_col = st.columns((1.25, 1), gap="large")
    with chart_col:
        st.plotly_chart(_tradeoff_figure(pack), use_container_width=True)
        st.caption(
            "Cycle stock is identical along the curve. Movement is safety stock. "
            "The baseline is not required to sit on the curve."
        )
    with bar_col:
        st.plotly_chart(_pattern_figure(pack), use_container_width=True)
        st.caption(
            "Smooth parts give up safety stock. Intermittent parts receive it. "
            "A high network fill rate can still hide line-down parts; the segment table counts those misses."
        )

    st.subheader("Segments")
    segments = _segment_display(pack.segment_summary)
    st.dataframe(
        segments,
        use_container_width=True,
        hide_index=True,
        height=360,
        column_config=_money_and_fill_config(segments.columns),
    )

    st.subheader("Parts")
    _parts_section(pack.recommendations)

    with st.expander("Assumptions the numbers rest on"):
        st.markdown(
            """
- The planning bucket is one week. Annual demand is 52 times the weekly mean.
- Lead-time demand averages weekly demand times lead time. Its variance adds demand noise over that lead time and lead-time noise scaled by the demand rate. The two are treated as independent.
- On-hand inventory is half an order quantity plus safety stock. That is the warehouse working capital this screen moves.
- Pipeline inventory is expected demand during lead time, valued at unit cost. Safety-stock policy does not change it. A shorter lead time does.
- Fill rate is the share of units filled from stock. Cycle service is the chance a replenishment cycle has no stockout. The target is a fill rate.
- A normal curve is used when lead-time demand is reasonably large and not too skewed. Otherwise a gamma curve with the same mean and variance is used.
- Order quantity is an economic order quantity, rounded up to a whole unit and at least the minimum order quantity. Both policies use it.
- Holding cost in the summary is the holding rate times on-hand inventory. It is not applied to pipeline inventory.
            """
        )

    st.caption("Author: SBroTheLegend. Synthetic data only.")


@st.cache_data(show_spinner=False)
def _cached_model(holding_rate: float, lead_time_scale: float):
    catalog = load_catalog(DATA_DIR)
    config = PolicyConfig(holding_rate=holding_rate, lead_time_scale=lead_time_scale)
    return build_model(catalog, config)


def _metrics(summary: dict) -> None:
    fill_delta = summary["optimized_fill_rate"] - summary["baseline_fill_rate"]
    capital_delta = summary["on_hand_delta"]
    baseline_misses = summary["baseline_line_down_below_floor"]
    recommended_misses = summary["optimized_line_down_below_floor"]
    if recommended_misses < baseline_misses:
        floor_delta = f"{baseline_misses - recommended_misses} fewer than the baseline"
        floor_arrow = "down"
    elif recommended_misses > baseline_misses:
        floor_delta = f"{recommended_misses - baseline_misses} more than the baseline"
        floor_arrow = "up"
    else:
        floor_delta = "Same as the baseline"
        floor_arrow = "off"
    first = st.columns(4)
    first[0].metric("Baseline on-hand", format_money(summary["baseline_on_hand_investment"]))
    first[1].metric(
        "Recommended on-hand",
        format_money(summary["optimized_on_hand_investment"]),
        delta=format_money(capital_delta),
        delta_color="inverse",
    )
    first[2].metric(
        "System fill rate",
        format_pct(summary["optimized_fill_rate"]),
        delta=f"{fill_delta * 100:+.1f} pts vs baseline",
    )
    first[3].metric(
        "Line-down below floor",
        f"{recommended_misses}",
        delta=floor_delta,
        delta_color="off",
        delta_arrow=floor_arrow,
    )
    second = st.columns(4)
    second[0].metric(
        "Safety stock",
        format_money(summary["optimized_safety_investment"]),
        delta=format_money(
            summary["optimized_safety_investment"] - summary["baseline_safety_investment"]
        ),
        delta_color="inverse",
    )
    second[1].metric("Cycle stock", format_money(summary["cycle_stock_investment"]))
    second[2].metric("Pipeline inventory", format_money(summary["pipeline_investment"]))
    second[3].metric(
        "Shortages per year",
        f"{summary['optimized_annual_shortages']:,.0f}",
        delta=f"{summary['optimized_annual_shortages'] - summary['baseline_annual_shortages']:+,.0f}",
        delta_color="inverse",
    )


def _tradeoff_figure(pack) -> go.Figure:
    frontier = pack.frontier
    summary = pack.summary
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=frontier["on_hand_investment"],
            y=frontier["system_fill_rate"],
            mode="lines",
            name="Efficient frontier",
            line={"color": INK, "width": 2.4},
            hovertemplate="On-hand %{x:$,.0f}<br>Fill %{y:.2%}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[summary["optimized_on_hand_investment"]],
            y=[summary["optimized_fill_rate"]],
            mode="markers",
            name="Recommended",
            marker={"color": TEAL, "size": 12},
            hovertemplate="Recommended<br>On-hand %{x:$,.0f}<br>Fill %{y:.2%}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[summary["baseline_on_hand_investment"]],
            y=[summary["baseline_fill_rate"]],
            mode="markers",
            name="Baseline",
            marker={"color": COPPER, "size": 12, "symbol": "diamond"},
            hovertemplate="Baseline<br>On-hand %{x:$,.0f}<br>Fill %{y:.2%}<extra></extra>",
        )
    )
    figure.add_hline(
        y=summary["target_fill_rate"],
        line_dash="dot",
        line_color=TEAL,
        opacity=0.7,
        annotation_text="Target",
        annotation_position="top left",
    )
    figure.update_layout(**_layout("On-hand inventory versus expected fill rate"))
    figure.update_xaxes(title_text="On-hand inventory", tickprefix="$", separatethousands=True)
    figure.update_yaxes(title_text="Expected system fill rate", tickformat=".1%")
    return figure


def _pattern_figure(pack) -> go.Figure:
    segments = pack.segment_summary
    order = ["smooth", "erratic", "intermittent"]
    rows = segments[segments["segment_type"] == "demand_pattern"].copy()
    rows = rows[rows["segment"].isin(order)]
    rows["segment"] = pd.Categorical(rows["segment"], categories=order, ordered=True)
    rows = rows.sort_values("segment")
    labels = [str(name).title() for name in rows["segment"]]
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            name="Baseline safety stock",
            x=labels,
            y=rows["baseline_safety_investment"],
            marker_color=COPPER,
            hovertemplate="%{x}<br>%{y:$,.0f}<extra>Baseline</extra>",
        )
    )
    figure.add_trace(
        go.Bar(
            name="Recommended safety stock",
            x=labels,
            y=rows["optimized_safety_investment"],
            marker_color=TEAL,
            hovertemplate="%{x}<br>%{y:$,.0f}<extra>Recommended</extra>",
        )
    )
    figure.update_layout(**_layout("Safety stock by demand pattern"), barmode="group")
    figure.update_yaxes(title_text="Safety stock", tickprefix="$", separatethousands=True)
    return figure


def _layout(title: str) -> dict:
    return {
        "title": {"text": title, "font": {"size": 16, "color": INK}},
        "paper_bgcolor": PAPER,
        "plot_bgcolor": PAPER,
        "font": {"family": "Segoe UI, sans-serif", "color": INK, "size": 12},
        "margin": {"l": 64, "r": 16, "t": 48, "b": 48},
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        "height": 440,
        "xaxis": {"gridcolor": GRID},
        "yaxis": {"gridcolor": GRID},
    }


def _segment_display(segments: pd.DataFrame) -> pd.DataFrame:
    frame = segments.copy()
    frame.insert(0, "Cut", frame["segment_type"].str.replace("_", " ").str.title())
    frame.insert(1, "Segment", frame["segment"].map(lambda value: CRITICALITY_LABELS.get(value, value)))
    rename = {
        "parts": "Parts",
        "annual_demand": "Annual demand",
        "baseline_fill_rate": "Baseline fill",
        "optimized_fill_rate": "Recommended fill",
        "baseline_safety_investment": "Baseline safety-stock $",
        "optimized_safety_investment": "Recommended safety-stock $",
        "baseline_on_hand_investment": "Baseline on-hand",
        "optimized_on_hand_investment": "Recommended on-hand",
        "baseline_line_down_below_floor": "Line-down below floor, baseline",
        "optimized_line_down_below_floor": "Line-down below floor, recommended",
    }
    display = frame[["Cut", "Segment", *rename.keys()]].rename(columns=rename)
    return display


def _parts_section(recommendations: pd.DataFrame) -> None:
    filters = st.columns((1, 1, 1.4))
    criticality = filters[0].multiselect(
        "Criticality",
        options=list(CRITICALITY_LABELS),
        format_func=lambda value: CRITICALITY_LABELS[value],
        default=list(CRITICALITY_LABELS),
    )
    commodities = sorted(recommendations["commodity"].unique())
    commodity = filters[1].multiselect("Commodity", options=commodities, default=commodities)
    query = filters[2].text_input("Search part number or description", "")
    detail = st.checkbox("Show reorder points, lead-time demand, and the position note", value=False)

    mask = recommendations["criticality"].isin(criticality) & recommendations["commodity"].isin(
        commodity
    )
    if query.strip():
        needle = query.strip().lower()
        text = (
            recommendations["part_number"].str.lower()
            + " "
            + recommendations["description"].str.lower()
        )
        mask = mask & text.str.contains(needle, regex=False)
    subset = recommendations.loc[mask].copy()
    if subset.empty:
        st.info("No parts match this filter.")
        return

    _subset_caption(subset)
    display = _part_display(subset, detail)
    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        height=480,
        column_config=_money_and_fill_config(display.columns),
    )
    downloads = st.columns(2)
    downloads[0].download_button(
        "Download part recommendations",
        data=recommendations.to_csv(index=False).encode("utf-8"),
        file_name="policy_recommendations.csv",
        mime="text/csv",
    )
    downloads[1].download_button(
        "Download this filtered view",
        data=subset.to_csv(index=False).encode("utf-8"),
        file_name="policy_recommendations_filtered.csv",
        mime="text/csv",
    )


def _subset_caption(subset: pd.DataFrame) -> None:
    service = subset[subset["in_service_metric"]]
    demand = float(service["annual_demand"].sum())
    if demand <= 0:
        st.caption(f"{len(subset)} parts. None of them are in the fill-rate calculation.")
        return
    baseline = 1.0 - float(service["baseline_annual_shortages"].sum()) / demand
    optimized = 1.0 - float(service["optimized_annual_shortages"].sum()) / demand
    base_on_hand = float(subset["baseline_on_hand_investment"].sum())
    opt_on_hand = float(subset["optimized_on_hand_investment"].sum())
    st.caption(
        f"Filtered subset: {len(subset)} parts. "
        f"On-hand {format_money(base_on_hand)} → {format_money(opt_on_hand)}. "
        f"Fill rate {format_pct(baseline)} → {format_pct(optimized)}. "
        "The metrics at the top stay on the full network."
    )


def _part_display(frame: pd.DataFrame, detail: bool) -> pd.DataFrame:
    working = frame.copy()
    working["Criticality"] = working["criticality"].map(CRITICALITY_LABELS)
    columns = {
        "part_number": "Part",
        "description": "Description",
        "Criticality": "Criticality",
        "abc_class": "ABC",
        "xyz_class": "XYZ",
        "distribution": "Lead-time demand",
        "annual_demand": "Annual demand",
        "unit_cost": "Unit cost",
        "order_quantity": "Order quantity",
        "baseline_safety_stock": "Baseline safety stock",
        "optimized_safety_stock": "Recommended safety stock",
        "baseline_fill_rate": "Baseline fill",
        "optimized_fill_rate": "Recommended fill",
        "baseline_on_hand_investment": "Baseline on-hand",
        "optimized_on_hand_investment": "Recommended on-hand",
    }
    if detail:
        columns.update(
            {
                "ddlt_mean": "Lead-time demand, mean",
                "ddlt_std": "Lead-time demand, std",
                "lead_time_weeks_mean": "Lead time (weeks)",
                "baseline_reorder_point": "Baseline reorder point",
                "optimized_reorder_point": "Recommended reorder point",
                "baseline_cycle_service": "Baseline cycle service",
                "optimized_cycle_service": "Recommended cycle service",
                "pipeline_investment": "Pipeline inventory",
                "position_note": "Note",
            }
        )
    display = working[list(columns)].rename(columns=columns)
    return display.sort_values("Recommended on-hand", ascending=False)


def _money_and_fill_config(columns) -> dict:
    config = {}
    for name in columns:
        label = str(name)
        lowered = label.lower()
        if "fill" in lowered or "cycle service" in lowered:
            config[label] = st.column_config.NumberColumn(label, format="percent")
        elif "$" in label or "on-hand" in lowered or "unit cost" in lowered or "pipeline" in lowered:
            config[label] = st.column_config.NumberColumn(label, format="dollar")
        elif "annual demand" in lowered or "order quantity" in lowered or "safety stock" in lowered:
            config[label] = st.column_config.NumberColumn(label, format="%.1f")
        elif label.startswith("Lead-time demand,") or label.startswith("Lead time"):
            config[label] = st.column_config.NumberColumn(label, format="%.2f")
    return config


def _css() -> None:
    st.markdown(
        """
        <style>
            .stApp { background: #f4f1ea; }
            h1, h2, h3 {
                font-family: Palatino, "Iowan Old Style", Georgia, serif;
                letter-spacing: -0.02em;
            }
            [data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
            .block-container { padding-top: 1.4rem; max-width: 1180px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


main()
