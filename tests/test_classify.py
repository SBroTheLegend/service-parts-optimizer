"""ABC and XYZ cutoffs."""

from __future__ import annotations

import pandas as pd

from service_parts.policy.classify import assign_abc, assign_xyz


def test_abc_keeps_the_crossing_part_in_the_higher_class():
    usage = pd.Series({"a": 60.0, "b": 25.0, "c": 10.0, "d": 5.0})
    classes = assign_abc(usage)
    assert classes.to_dict() == {"a": "A", "b": "A", "c": "B", "d": "C"}


def test_xyz_uses_the_coefficient_of_variation():
    mean = pd.Series({"x": 10.0, "y": 10.0, "z": 10.0, "none": 0.0})
    std = pd.Series({"x": 2.0, "y": 7.0, "z": 14.0, "none": 0.0})
    assert assign_xyz(mean, std).to_dict() == {"x": "X", "y": "Y", "z": "Z", "none": "Z"}
