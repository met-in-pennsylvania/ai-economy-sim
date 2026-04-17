"""Test that same seed + same scenario -> bitwise identical outputs."""

import pytest
from pathlib import Path

from ai_econ_sim.scenarios.loader import load_scenario
from ai_econ_sim.model import Model
from ai_econ_sim.analysis.outputs import build_time_series

REFERENCE_YAML = Path(__file__).parent.parent / "scenarios" / "fragmented.yaml"


def _run(quarters=4, scale=0.05):
    scenario = load_scenario(REFERENCE_YAML)
    scenario.horizon_quarters = quarters
    model = Model(scenario, population_scale=scale, dev_assertions=False)
    history = model.run()
    return build_time_series(history)


def test_identical_outputs_same_seed():
    df1 = _run()
    df2 = _run()
    assert df1.equals(df2), "Same seed produced different outputs"


def test_different_seeds_produce_different_outputs():
    scenario1 = load_scenario(REFERENCE_YAML)
    scenario1.horizon_quarters = 4
    scenario1.seed = 42

    scenario2 = load_scenario(REFERENCE_YAML)
    scenario2.horizon_quarters = 4
    scenario2.seed = 999

    m1 = Model(scenario1, population_scale=0.05, dev_assertions=False)
    m2 = Model(scenario2, population_scale=0.05, dev_assertions=False)

    h1 = m1.run()
    h2 = m2.run()

    df1 = build_time_series(h1)
    df2 = build_time_series(h2)

    # Different seeds should produce at least some different values
    assert not df1.equals(df2), "Different seeds produced identical outputs"
