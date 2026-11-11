"""Command line for the sample catalog and the policy pack."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from service_parts import __version__
from service_parts.config import PolicyConfig
from service_parts.data.generate import generate_dataset
from service_parts.data.load import apply_demand_source, load_catalog, load_history, load_manifest
from service_parts.data.schema import CatalogError
from service_parts.pipeline import run_pipeline
from service_parts.report.export import write_policy_pack
from service_parts.report.format import render_summary


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return Path.cwd()


def build_parser() -> argparse.ArgumentParser:
    root = repo_root()
    parser = argparse.ArgumentParser(
        prog="service-parts",
        description=(
            "Set safety stock for a service-parts network and compare a blanket "
            "baseline with a fill-rate policy."
        ),
    )
    parser.add_argument("--version", action="version", version=f"service-parts {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser(
        "generate",
        help="Write a synthetic parts catalog and weekly demand history.",
    )
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--parts", type=int, default=200)
    generate.add_argument("--weeks", type=int, default=156)
    generate.add_argument(
        "--output",
        type=Path,
        default=root / "data" / "halden_sample",
        help="Directory for parts.csv, weekly_demand.csv, and manifest.json.",
    )

    run = commands.add_parser(
        "run",
        help="Compute the baseline, the frontier, and the recommended policy.",
    )
    run.add_argument("--data", type=Path, default=root / "data" / "halden_sample")
    run.add_argument("--output", type=Path, default=root / "outputs")
    run.add_argument("--target-fill", type=float, default=PolicyConfig().target_fill_rate)
    run.add_argument("--holding-rate", type=float, default=PolicyConfig().holding_rate)
    run.add_argument(
        "--baseline",
        choices=("weeks_of_cover", "cycle_service"),
        default="weeks_of_cover",
    )
    run.add_argument("--cover-weeks", type=float, default=PolicyConfig().baseline_cover_weeks)
    run.add_argument(
        "--cycle-service",
        type=float,
        default=PolicyConfig().baseline_cycle_service,
    )
    run.add_argument(
        "--lead-time-reduction",
        type=float,
        default=0.0,
        help="Scale lead times down by this fraction, from 0 to 0.8.",
    )
    run.add_argument(
        "--demand-source",
        choices=("catalog", "history"),
        default="catalog",
        help="Use the catalog planning moments, or the trailing demand window.",
    )
    run.add_argument("--history-window", type=int, default=52)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            path = generate_dataset(
                args.output,
                n_parts=args.parts,
                n_weeks=args.weeks,
                seed=args.seed,
            )
            print(f"Wrote synthetic catalog to {path}")
            return 0
        if args.command == "run":
            return _run(args)
    except (CatalogError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    parser.error(f"Unknown command {args.command}")
    return 2


def _run(args: argparse.Namespace) -> int:
    if not 0.0 <= args.lead_time_reduction < 0.8:
        raise ValueError("lead-time reduction must be in [0, 0.8)")
    catalog = load_catalog(args.data)
    history = None
    if args.demand_source == "history":
        history = load_history(args.data)
    catalog = apply_demand_source(catalog, history, args.demand_source, args.history_window)
    config = PolicyConfig(
        target_fill_rate=args.target_fill,
        holding_rate=args.holding_rate,
        baseline_mode=args.baseline,
        baseline_cover_weeks=args.cover_weeks,
        baseline_cycle_service=args.cycle_service,
        lead_time_scale=1.0 - args.lead_time_reduction,
    )
    pack = run_pipeline(catalog, config)
    write_policy_pack(pack, args.output)
    manifest = load_manifest(args.data)
    header = "Service-parts policy pack"
    if manifest and manifest.get("network"):
        header = str(manifest["network"])
        if manifest.get("fictional"):
            header += " (synthetic)"
    print(header)
    print(render_summary(pack.summary))
    print(f"Policy pack written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
