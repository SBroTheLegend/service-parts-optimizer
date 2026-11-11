"""Synthetic Halden Field Service catalog.

Halden Field Service is fictional. Part numbers, regions, costs, lead times,
and demand are generated from a fixed seed. Nothing here is a real operator,
supplier, customer, or installed base.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from service_parts import __version__


COMMODITIES = (
    ("hydraulics", 0.20),
    ("controls", 0.18),
    ("filtration", 0.16),
    ("rotating", 0.16),
    ("sensors", 0.15),
    ("power", 0.15),
)

COMMODITY_CODE = {
    "hydraulics": "HYD",
    "controls": "CTL",
    "filtration": "FLT",
    "rotating": "ROT",
    "sensors": "SEN",
    "power": "PWR",
}

NOUNS = {
    "hydraulics": [
        "Pump cartridge",
        "Valve spool",
        "Cylinder seal kit",
        "Hose assembly",
        "Charge-pump rotor",
    ],
    "controls": [
        "I/O module",
        "Encoder",
        "Relay board",
        "Display unit",
        "Wire harness",
    ],
    "filtration": [
        "Pressure filter",
        "Return filter",
        "Breather",
        "Strainer basket",
        "Separator element",
    ],
    "rotating": [
        "Bearing set",
        "Coupling insert",
        "Impeller",
        "Shaft sleeve",
        "Wear ring",
    ],
    "sensors": [
        "Pressure transducer",
        "Temperature probe",
        "Proximity sensor",
        "Flow sensor",
        "Level switch",
    ],
    "power": [
        "Contactor",
        "Drive cooling fan",
        "Capacitor module",
        "Braking resistor",
        "Power supply",
    ],
}

FORMS = (
    "short body",
    "long body",
    "flanged",
    "threaded",
    "high flow",
    "low pressure",
    "field kit",
    "sealed",
)

SIZES = ("20", "32", "40", "50", "63", "80")

REGIONS = ("North Basin", "Coast Range", "Interior Plain")

CRITICALITY_WEIGHTS = {
    "hydraulics": (0.22, 0.48, 0.30),
    "controls": (0.42, 0.40, 0.18),
    "filtration": (0.06, 0.30, 0.64),
    "rotating": (0.28, 0.47, 0.25),
    "sensors": (0.38, 0.42, 0.20),
    "power": (0.34, 0.44, 0.22),
}

COST_LOGNORMAL = {
    "hydraulics": (6.1, 0.65),
    "controls": (6.4, 0.80),
    "filtration": (3.3, 0.50),
    "rotating": (5.3, 0.65),
    "sensors": (5.1, 0.55),
    "power": (5.7, 0.70),
}

LEAD_TIME = {
    "hydraulics": (5.0, 1.3),
    "controls": (10.0, 2.0),
    "filtration": (2.2, 0.45),
    "rotating": (6.5, 1.5),
    "sensors": (7.5, 1.7),
    "power": (8.5, 1.9),
}

CRITICALITY_NAMES = ("line_down", "degraded", "convenience")


def generate_dataset(
    output_dir: Path | str,
    *,
    n_parts: int = 200,
    n_weeks: int = 156,
    seed: int = 42,
) -> Path:
    """Write parts.csv, weekly_demand.csv, and manifest.json. Returns the directory."""
    if n_parts < 8:
        raise ValueError("generate at least 8 parts so every demand pattern appears")
    if n_weeks < 8:
        raise ValueError("generate at least 8 weeks of history")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    catalog_rows, history_rows = _build(rng, n_parts, n_weeks)
    catalog = pd.DataFrame(catalog_rows)
    history = pd.DataFrame(history_rows)
    catalog.to_csv(output / "parts.csv", index=False, float_format="%.6f")
    history.to_csv(output / "weekly_demand.csv", index=False)
    manifest = {
        "name": "halden_sample",
        "fictional": True,
        "network": "Halden Field Service",
        "description": (
            "Synthetic service-parts catalog for a fictional industrial "
            "field-service network. Part numbers, regions, costs, lead times, "
            "and demand are generated. No real operator, supplier, or customer "
            "is represented."
        ),
        "seed": seed,
        "parts": n_parts,
        "weeks": n_weeks,
        "weeks_per_year": 52,
        "generator": "service_parts.data.generate",
        "version": __version__,
        "author": "SBroTheLegend",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return output


def _counts(n: int, weights: list[float]) -> list[int]:
    raw = [weight * n for weight in weights]
    counts = [int(math.floor(value)) for value in raw]
    remainder = n - sum(counts)
    order = sorted(range(len(raw)), key=lambda index: raw[index] - counts[index], reverse=True)
    for index in order[:remainder]:
        counts[index] += 1
    return counts


def _build(rng: np.random.Generator, n_parts: int, n_weeks: int):
    pattern_weights = [0.45, 0.30, 0.20, 0.05]
    pattern_names = ["smooth", "erratic", "intermittent", "inactive"]
    patterns = []
    for name, count in zip(pattern_names, _counts(n_parts, pattern_weights)):
        patterns.extend([name] * count)
    pattern_array = np.array(patterns)
    rng.shuffle(pattern_array)

    commodity_names = [name for name, _weight in COMMODITIES]
    commodity_weights = [weight for _name, weight in COMMODITIES]
    commodities = []
    for name, count in zip(commodity_names, _counts(n_parts, commodity_weights)):
        commodities.extend([name] * count)
    commodity_array = np.array(commodities)
    rng.shuffle(commodity_array)

    catalog_rows = []
    history_rows = []
    for index in range(n_parts):
        commodity = str(commodity_array[index])
        pattern = str(pattern_array[index])
        criticality = str(
            rng.choice(CRITICALITY_NAMES, p=CRITICALITY_WEIGHTS[commodity])
        )
        part_id = f"P{index + 1:04d}"
        part_number = f"HFS-{COMMODITY_CODE[commodity]}-{1000 + index}"
        noun = str(rng.choice(NOUNS[commodity]))
        form = str(rng.choice(FORMS))
        size = str(rng.choice(SIZES))
        description = f"{noun} {size}, {form}"
        region = str(rng.choice(REGIONS))
        mu, sigma = COST_LOGNORMAL[commodity]
        unit_cost = float(np.clip(round(float(rng.lognormal(mu, sigma)), 2), 8.0, 12000.0))
        order_cost = round(float(rng.uniform(45.0, 160.0)), 2)
        if commodity == "filtration" and rng.random() < 0.55:
            moq = int(rng.choice([5, 10, 12]))
        elif rng.random() < 0.08:
            moq = int(rng.choice([2, 4, 5]))
        else:
            moq = 1
        center, spread = LEAD_TIME[commodity]
        lead_mean = float(np.clip(rng.normal(center, spread), 1.0, 18.0))
        lead_cv = float(rng.uniform(0.08, 0.22))
        lead_std = lead_mean * lead_cv
        weekly_mean, weekly_std, process = _demand_process(rng, pattern)
        demand = _simulate_weeks(rng, n_weeks, pattern, weekly_mean, weekly_std, process)
        catalog_rows.append(
            {
                "part_id": part_id,
                "part_number": part_number,
                "description": description,
                "commodity": commodity,
                "criticality": criticality,
                "primary_region": region,
                "unit_cost": unit_cost,
                "order_cost": order_cost,
                "moq": moq,
                "weekly_demand_mean": weekly_mean,
                "weekly_demand_std": weekly_std,
                "lead_time_weeks_mean": round(lead_mean, 4),
                "lead_time_weeks_std": round(lead_std, 4),
                "demand_pattern": pattern,
            }
        )
        for week, units in enumerate(demand, start=1):
            history_rows.append(
                {"part_id": part_id, "week_index": week, "demand_units": int(units)}
            )
    return catalog_rows, history_rows


def _demand_process(rng: np.random.Generator, pattern: str) -> tuple[float, float, dict]:
    if pattern == "inactive":
        return 0.0, 0.0, {}
    if pattern == "smooth":
        mean = float(np.clip(rng.lognormal(1.35, 0.55), 0.5, 40.0))
        cv = float(rng.uniform(0.12, 0.32))
        return mean, mean * cv, {"family": "gamma"}
    if pattern == "erratic":
        mean = float(np.clip(rng.lognormal(0.35, 0.60), 0.2, 22.0))
        cv = float(rng.uniform(0.55, 0.95))
        return mean, mean * cv, {"family": "gamma"}
    mean = float(np.clip(rng.lognormal(-0.35, 0.65), 0.08, 8.0))
    p_positive = float(rng.uniform(0.20, 0.42))
    cv_size = float(rng.uniform(0.40, 0.80))
    mean_size = mean / p_positive
    std_size = cv_size * mean_size
    variance = p_positive * (std_size**2 + mean_size**2) - mean**2
    std = math.sqrt(max(variance, 0.0))
    return mean, std, {
        "family": "intermittent",
        "p_positive": p_positive,
        "mean_size": mean_size,
        "std_size": std_size,
    }


def _simulate_weeks(
    rng: np.random.Generator,
    n_weeks: int,
    pattern: str,
    weekly_mean: float,
    weekly_std: float,
    process: dict,
) -> np.ndarray:
    """Integer weekly history. Planning moments stay on the catalog, not this sample."""
    if pattern == "inactive":
        demand = np.zeros(n_weeks, dtype=int)
        # Older weeks can show usage. The trailing year, which a history-based
        # run would read, stays at zero so the part plans as inactive.
        early = max(n_weeks - 52, 0)
        for week in range(early):
            if rng.random() < 0.22:
                demand[week] = int(rng.integers(1, 4))
        return demand
    if process.get("family") == "intermittent":
        std_size = process["std_size"]
        mean_size = process["mean_size"]
        if std_size <= 1e-8:
            sizes = np.full(n_weeks, mean_size)
        else:
            shape = (mean_size / std_size) ** 2
            scale = std_size**2 / mean_size
            sizes = rng.gamma(shape, scale=scale, size=n_weeks)
        positive = rng.random(n_weeks) < process["p_positive"]
        draws = np.where(positive, sizes, 0.0)
        return _stochastic_round(rng, draws)
    if weekly_std <= 1e-8:
        return _stochastic_round(rng, np.full(n_weeks, weekly_mean))
    shape = (weekly_mean / weekly_std) ** 2
    scale = weekly_std**2 / weekly_mean
    draws = rng.gamma(shape, scale=scale, size=n_weeks)
    return _stochastic_round(rng, draws)


def _stochastic_round(rng: np.random.Generator, draws: np.ndarray) -> np.ndarray:
    base = np.floor(draws)
    fraction = draws - base
    bumped = rng.random(draws.shape[0]) < fraction
    return (base + bumped).astype(int)
