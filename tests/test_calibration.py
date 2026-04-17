"""Tests for BEA/BLS calibration data and model initialization."""

import pytest
from pathlib import Path

from ai_econ_sim.calibration.bea_bls import (
    GDP_SHARES, EMPLOYMENT_SHARES, MEDIAN_WAGES,
    BASE_GROWTH_RATES, NOMINAL_GDP, BENCHMARKS_2023,
    calibrated_initial_conditions,
)
from ai_econ_sim.scenarios.loader import load_scenario
from ai_econ_sim.config import SECTORS

CALIBRATED_YAML = Path(__file__).parent.parent / "scenarios" / "calibrated_baseline.yaml"


# ------------------------------------------------------------------
# Data integrity
# ------------------------------------------------------------------

def test_gdp_shares_sum_to_one():
    total = sum(GDP_SHARES.values())
    assert abs(total - 1.0) < 0.005, f"GDP shares sum to {total}, expected ~1.0"


def test_employment_shares_sum_to_one():
    total = sum(EMPLOYMENT_SHARES.values())
    assert abs(total - 1.0) < 0.005, f"Employment shares sum to {total}"


def test_all_sectors_covered():
    for s in SECTORS:
        assert s in GDP_SHARES, f"Missing GDP share for {s}"
        assert s in EMPLOYMENT_SHARES, f"Missing employment share for {s}"
        assert s in MEDIAN_WAGES, f"Missing wage for {s}"
        assert s in BASE_GROWTH_RATES, f"Missing growth rate for {s}"


def test_wages_positive_and_ordered():
    """ai_compute should have highest wages, services lowest."""
    assert MEDIAN_WAGES["ai_compute"] > MEDIAN_WAGES["knowledge_work"]
    assert MEDIAN_WAGES["knowledge_work"] > MEDIAN_WAGES["services"]
    for s in SECTORS:
        assert MEDIAN_WAGES[s] > 0


def test_nominal_gdp_reasonable():
    assert 15e12 < NOMINAL_GDP < 30e12, "Nominal GDP should be in $15–30T range"


def test_benchmarks_object_consistent():
    assert BENCHMARKS_2023.source_year == 2023
    assert set(BENCHMARKS_2023.gdp_shares.keys()) == set(SECTORS)


def test_calibrated_initial_conditions():
    ic = calibrated_initial_conditions()
    assert ic.nominal_gdp == NOMINAL_GDP
    assert set(ic.gdp_shares.keys()) == set(SECTORS)
    assert set(ic.employment_shares.keys()) == set(SECTORS)
    assert set(ic.initial_wages.keys()) == set(SECTORS)


# ------------------------------------------------------------------
# Calibrated baseline scenario loading
# ------------------------------------------------------------------

def test_calibrated_baseline_loads():
    scenario = load_scenario(CALIBRATED_YAML)
    assert scenario.name == "calibrated_baseline_2023"
    assert scenario.horizon_quarters == 40


def test_calibrated_baseline_initial_conditions():
    scenario = load_scenario(CALIBRATED_YAML)
    ic = scenario.initial_conditions
    assert abs(sum(ic.gdp_shares.values()) - 1.0) < 0.005
    assert abs(sum(ic.employment_shares.values()) - 1.0) < 0.005
    assert ic.initial_wages["ai_compute"] > ic.initial_wages["services"]


# ------------------------------------------------------------------
# Model initialization uses employment shares
# ------------------------------------------------------------------

@pytest.fixture(scope="module")
def calibrated_model():
    from ai_econ_sim.model import Model
    scenario = load_scenario(CALIBRATED_YAML)
    scenario.horizon_quarters = 2
    # 0.05 → ~501 workers total; ai_compute gets 13 workers (2.6%) vs 50 min floor
    return Model(scenario, population_scale=0.05)


def test_model_worker_shares_match_calibration(calibrated_model):
    """Workers per sector should be proportional to BLS employment shares."""
    m = calibrated_model
    total = sum(len(m.workers[s]) for s in SECTORS)
    for s in SECTORS:
        model_share = len(m.workers[s]) / total
        bench_share = EMPLOYMENT_SHARES[s]
        # Allow ±5pp absolute tolerance (small population rounding)
        assert abs(model_share - bench_share) < 0.05, (
            f"{s}: model share={model_share:.3f}, BLS={bench_share:.3f}"
        )


def test_model_services_largest_sector(calibrated_model):
    """Services must be the largest employment sector (BLS: 56%)."""
    m = calibrated_model
    counts = {s: len(m.workers[s]) for s in SECTORS}
    largest = max(counts, key=counts.get)
    assert largest == "services", f"Expected services to be largest, got {largest}"


def test_calibrated_model_runs(calibrated_model):
    """Calibrated model should run without errors."""
    scenario = load_scenario(CALIBRATED_YAML)
    scenario.horizon_quarters = 4
    from ai_econ_sim.model import Model
    m = Model(scenario, population_scale=0.01, dev_assertions=False)
    history = m.run()
    assert len(history) == 4
