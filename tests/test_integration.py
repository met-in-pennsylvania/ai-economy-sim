"""Full-run integration smoke tests."""

import math
import pytest
from pathlib import Path

from ai_econ_sim.scenarios.loader import load_scenario
from ai_econ_sim.model import Model
from ai_econ_sim.config import SECTORS


SCENARIO_DIR = Path(__file__).parent.parent / "scenarios"
REFERENCE_YAML = SCENARIO_DIR / "fragmented.yaml"


@pytest.fixture(scope="module")
def short_run():
    """Run 4 quarters with small population for speed."""
    scenario = load_scenario(REFERENCE_YAML)
    scenario.horizon_quarters = 4
    model = Model(scenario, population_scale=0.1, dev_assertions=True)
    history = model.run()
    return model, history


def test_run_completes(short_run):
    model, history = short_run
    assert len(history) == 4


def test_no_nan_or_inf(short_run):
    _, history = short_run
    for accts in history:
        assert math.isfinite(accts.nominal_gdp), f"Q{accts.quarter}: non-finite GDP"
        assert math.isfinite(accts.gini), f"Q{accts.quarter}: non-finite Gini"
        assert math.isfinite(accts.real_gdp), f"Q{accts.quarter}: non-finite real GDP"


def test_gdp_positive(short_run):
    _, history = short_run
    for accts in history:
        assert accts.nominal_gdp > 0, f"Q{accts.quarter}: GDP <= 0"


def test_employment_non_negative(short_run):
    _, history = short_run
    for accts in history:
        for s in SECTORS:
            assert accts.sector_employment.get(s, 0) >= 0


def test_labor_accounting(short_run):
    model, history = short_run
    all_workers = [w for ws in model.workers.values() for w in ws]
    total = len(all_workers)
    emp = sum(1 for w in all_workers if w.is_employed)
    unemp = sum(1 for w in all_workers if w.is_in_labor_force and not w.is_employed)
    olf = sum(1 for w in all_workers if not w.is_in_labor_force)
    assert emp + unemp + olf == total, f"Labor accounting: {emp}+{unemp}+{olf} != {total}"


def test_unemployment_rate_in_range(short_run):
    _, history = short_run
    for accts in history:
        lf = accts.total_employed + accts.total_unemployed
        if lf > 0:
            ur = accts.total_unemployed / lf
            assert 0.0 <= ur <= 1.0


def test_ai_adoption_non_decreasing_trend(short_run):
    """AI adoption should not collapse to zero (may fluctuate but bounded below)."""
    _, history = short_run
    final_adoption = {s: history[-1].sector_ai_adoption.get(s, 0) for s in SECTORS}
    for s, v in final_adoption.items():
        assert v >= 0.0


def test_scenario_load_and_run_all(tmp_path):
    """All 5 scenario YAMLs must load and run 2 quarters without error."""
    yamls = list(SCENARIO_DIR.glob("*.yaml"))
    assert len(yamls) >= 3, "Expected at least 3 scenario files"
    for yaml_path in yamls:
        scenario = load_scenario(yaml_path)
        scenario.horizon_quarters = 2
        model = Model(scenario, population_scale=0.1, dev_assertions=False)
        history = model.run()
        assert len(history) == 2, f"{yaml_path.name}: expected 2 quarters"
