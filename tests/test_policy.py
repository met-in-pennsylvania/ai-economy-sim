"""Tests for PolicyParams: UBI, retraining subsidy, AI windfall tax."""

from __future__ import annotations
import math
import pytest
from pathlib import Path

from ai_econ_sim.scenarios.loader import load_scenario, PolicyParams
from ai_econ_sim.model import Model
from ai_econ_sim.analysis.outputs import build_time_series

SCENARIO_DIR = Path(__file__).parent.parent / "scenarios"
YAML = SCENARIO_DIR / "calibrated_baseline.yaml"


@pytest.fixture(scope="module")
def baseline_run():
    scen = load_scenario(YAML)
    scen.horizon_quarters = 4
    model = Model(scen, population_scale=0.1, dev_assertions=True)
    history = model.run()
    return history


@pytest.fixture(scope="module")
def policy_run():
    scen = load_scenario(YAML)
    scen.horizon_quarters = 4
    scen.policy.ubi_annual = 12_000.0
    scen.policy.retraining_subsidy_rate = 0.80
    scen.policy.ai_windfall_tax_rate = 0.30
    model = Model(scen, population_scale=0.1, dev_assertions=True)
    history = model.run()
    return history


# ------------------------------------------------------------------
# Loader tests
# ------------------------------------------------------------------

def test_policy_params_default_values():
    """PolicyParams defaults are all zero (no-op)."""
    p = PolicyParams()
    assert p.ubi_annual == 0.0
    assert p.retraining_subsidy_rate == 0.0
    assert p.ai_windfall_tax_rate == 0.0


def test_scenario_has_policy_attr():
    """Loaded scenario exposes .policy attribute."""
    scen = load_scenario(YAML)
    assert hasattr(scen, "policy")
    assert isinstance(scen.policy, PolicyParams)


def test_policy_yaml_block_parsed():
    """A YAML with explicit policy block parses correctly."""
    import tempfile, pathlib, yaml
    yaml_data = {
        "scenario_name": "test_policy",
        "horizon_quarters": 4,
        "policy": {
            "ubi_annual": 6000.0,
            "retraining_subsidy_rate": 0.25,
            "ai_windfall_tax_rate": 0.15,
        },
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(yaml_data, f)
        fname = f.name
    scen = load_scenario(fname)
    assert scen.policy.ubi_annual == 6000.0
    assert scen.policy.retraining_subsidy_rate == 0.25
    assert scen.policy.ai_windfall_tax_rate == 0.15


# ------------------------------------------------------------------
# MacroAccounts fields
# ------------------------------------------------------------------

def test_macroaccts_policy_fields_exist(baseline_run):
    """MacroAccounts has all policy fields and they default to 0."""
    accts = baseline_run[0]
    assert hasattr(accts, "policy_ubi_disbursed")
    assert hasattr(accts, "policy_retraining_subsidized")
    assert hasattr(accts, "policy_windfall_tax_collected")
    assert accts.policy_ubi_disbursed == 0.0
    assert accts.policy_retraining_subsidized == 0
    assert accts.policy_windfall_tax_collected == 0.0


def test_no_policy_zero_ubi(baseline_run):
    """With no policy active, UBI disbursement is always 0."""
    for accts in baseline_run:
        assert accts.policy_ubi_disbursed == 0.0


def test_no_policy_zero_windfall(baseline_run):
    """With no policy active, windfall tax is always 0."""
    for accts in baseline_run:
        assert accts.policy_windfall_tax_collected == 0.0


# ------------------------------------------------------------------
# Windfall tax
# ------------------------------------------------------------------

def test_windfall_tax_positive_when_active(policy_run):
    """Windfall tax is > 0 in runs where ai_windfall_tax_rate > 0."""
    total = sum(a.policy_windfall_tax_collected for a in policy_run)
    assert total > 0.0, "Expected positive windfall tax with 30% rate"


def test_windfall_tax_added_to_revenue(policy_run):
    """tax_revenue includes windfall tax each quarter."""
    for accts in policy_run:
        # tax_revenue >= policy_windfall_tax_collected (it's also added)
        assert accts.tax_revenue >= accts.policy_windfall_tax_collected - 1e-6


def test_windfall_tax_nonnegativity(policy_run):
    """Windfall tax values are always >= 0."""
    for accts in policy_run:
        assert accts.policy_windfall_tax_collected >= 0.0


# ------------------------------------------------------------------
# Time series output
# ------------------------------------------------------------------

def test_timeseries_has_policy_columns(baseline_run):
    """build_time_series includes policy columns."""
    df = build_time_series(baseline_run)
    for col in ("policy_ubi_disbursed", "policy_retraining_subsidized", "policy_windfall_tax_collected"):
        assert col in df.columns, f"Missing column: {col}"


def test_timeseries_policy_columns_nonnegative(policy_run):
    """Policy columns in the time series are non-negative."""
    df = build_time_series(policy_run)
    assert (df["policy_ubi_disbursed"] >= 0).all()
    assert (df["policy_retraining_subsidized"] >= 0).all()
    assert (df["policy_windfall_tax_collected"] >= 0).all()


# ------------------------------------------------------------------
# Model still runs clean with policy active (invariants pass)
# ------------------------------------------------------------------

def test_policy_run_no_nan(policy_run):
    """Policy-active run produces no NaN/Inf in macro accounts."""
    for accts in policy_run:
        assert math.isfinite(accts.nominal_gdp)
        assert math.isfinite(accts.gini)
        assert math.isfinite(accts.real_gdp)
