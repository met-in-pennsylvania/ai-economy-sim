"""Tests for Monte Carlo runner and aggregation."""

import pytest
import numpy as np
from pathlib import Path

from ai_econ_sim.scenarios.loader import load_scenario
from ai_econ_sim.monte_carlo import run_monte_carlo, run_parameter_sweep, _compute_bands, QUANTILES

REFERENCE_YAML = Path(__file__).parent.parent / "scenarios" / "fragmented.yaml"


@pytest.fixture(scope="module")
def mc_results():
    scenario = load_scenario(REFERENCE_YAML)
    scenario.horizon_quarters = 4
    return run_monte_carlo(scenario, n_runs=5, population_scale=0.05)


def test_mc_run_count(mc_results):
    assert mc_results.n_runs == 5
    assert len(mc_results.runs) == 5


def test_mc_all_runs_same_length(mc_results):
    lengths = {len(df) for df in mc_results.runs}
    assert len(lengths) == 1
    assert lengths.pop() == 4


def test_mc_different_seeds_produce_variation(mc_results):
    gdp_finals = [df["nominal_gdp"].iloc[-1] for df in mc_results.runs]
    assert len(set(gdp_finals)) > 1, "All seeds produced identical GDP — no variation"


def test_bands_contain_all_quantiles(mc_results):
    quantiles_in_index = mc_results.bands.index.get_level_values("quantile").unique().tolist()
    for q in QUANTILES:
        assert q in quantiles_in_index, f"Missing quantile {q}"


def test_bands_p10_le_p50_le_p90(mc_results):
    p10 = mc_results.bands.xs(0.10, level="quantile")["nominal_gdp"]
    p50 = mc_results.bands.xs(0.50, level="quantile")["nominal_gdp"]
    p90 = mc_results.bands.xs(0.90, level="quantile")["nominal_gdp"]
    assert (p10 <= p50 + 1e-6).all(), "p10 > p50 for GDP"
    assert (p50 <= p90 + 1e-6).all(), "p50 > p90 for GDP"


def test_median_method(mc_results):
    med = mc_results.median()
    assert "nominal_gdp" in med.columns
    assert len(med) == 4


def test_band_method(mc_results):
    lo, hi = mc_results.band(0.10, 0.90)
    assert (hi["nominal_gdp"] >= lo["nominal_gdp"] - 1e-6).all()


def test_summary_stats(mc_results):
    stats = mc_results.summary_stats()
    assert "nominal_gdp" in stats.index
    assert "mean" in stats.columns
    assert "std" in stats.columns


def test_parameter_sweep():
    scenario = load_scenario(REFERENCE_YAML)
    scenario.horizon_quarters = 4

    sweep = run_parameter_sweep(
        scenario,
        param_name="oss_frontier_gap",
        param_values=[0.5, 2.0],
        setter=lambda s, v: setattr(s, "oss_frontier_gap", v),
        n_runs_per_value=3,
        population_scale=0.05,
    )
    assert len(sweep.mc_results) == 2
    assert sweep.param_name == "oss_frontier_gap"


def test_sweep_median_series(mc_results):
    scenario = load_scenario(REFERENCE_YAML)
    scenario.horizon_quarters = 4
    from ai_econ_sim.monte_carlo import run_parameter_sweep
    sweep = run_parameter_sweep(
        scenario, "oss_frontier_gap", [0.5, 1.0],
        setter=lambda s, v: setattr(s, "oss_frontier_gap", v),
        n_runs_per_value=3, population_scale=0.05,
    )
    series = sweep.median_series("nominal_gdp")
    assert list(series.columns) == [0.5, 1.0]
    assert len(series) == 4
