"""Planning assumptions a user can change without touching the math."""

from __future__ import annotations

from dataclasses import dataclass


CRITICALITY_ORDER = ("line_down", "degraded", "convenience")
ABC_ORDER = ("A", "B", "C")
XYZ_ORDER = ("X", "Y", "Z")
PATTERN_ORDER = ("smooth", "erratic", "intermittent", "inactive")

CRITICALITY_LABELS = {
    "line_down": "Line-down",
    "degraded": "Degraded",
    "convenience": "Convenience",
}


@dataclass(frozen=True)
class PolicyConfig:
    """Defaults are planning assumptions, not estimated costs.

    The holding rate is a single cost of capital plus warehousing. Because it
    is the same for every part, it changes order quantities but it does not
    by itself change which part the safety-stock allocation prefers.
    """

    target_fill_rate: float = 0.99
    holding_rate: float = 0.20
    baseline_mode: str = "weeks_of_cover"
    baseline_cover_weeks: float = 4.0
    baseline_cycle_service: float = 0.95
    lead_time_scale: float = 1.0
    z_step: float = 0.05
    z_max: float = 3.5
    weeks_per_year: int = 52
    normal_cv_threshold: float = 0.80
    normal_min_mean: float = 8.0
    min_fill_line_down: float = 0.985
    min_fill_degraded: float = 0.960
    min_fill_convenience: float = 0.900

    def min_fill(self, criticality: str) -> float:
        floors = {
            "line_down": self.min_fill_line_down,
            "degraded": self.min_fill_degraded,
            "convenience": self.min_fill_convenience,
        }
        try:
            return floors[criticality]
        except KeyError as exc:
            raise ValueError(f"Unknown criticality: {criticality}") from exc


def validate_config(config: PolicyConfig) -> None:
    if not 0.0 < config.target_fill_rate < 1.0:
        raise ValueError("target fill rate must be between 0 and 1")
    if not 0.0 < config.holding_rate <= 1.0:
        raise ValueError("holding rate must be in (0, 1]")
    if config.baseline_mode not in {"weeks_of_cover", "cycle_service"}:
        raise ValueError("baseline mode must be 'weeks_of_cover' or 'cycle_service'")
    if config.baseline_cover_weeks < 0:
        raise ValueError("cover weeks cannot be negative")
    if not 0.0 < config.baseline_cycle_service < 1.0:
        raise ValueError("baseline cycle service must be between 0 and 1")
    if not 0.0 < config.lead_time_scale <= 1.0:
        raise ValueError("lead-time scale must be in (0, 1]")
    if config.z_step <= 0 or config.z_max <= 0:
        raise ValueError("safety-stock grid must be positive")
    if config.weeks_per_year < 1:
        raise ValueError("weeks_per_year must be positive")
    for name in ("min_fill_line_down", "min_fill_degraded", "min_fill_convenience"):
        value = getattr(config, name)
        if not 0.0 < value < 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    if not (
        config.min_fill_line_down >= config.min_fill_degraded >= config.min_fill_convenience
    ):
        raise ValueError("criticality floors must not rise as criticality falls")
