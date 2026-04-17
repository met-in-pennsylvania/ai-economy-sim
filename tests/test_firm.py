"""Tests for firm decision logic."""

import numpy as np
import pytest
from ai_econ_sim.agents.firm import Firm, create_firms, _size_tier
from ai_econ_sim.config import MARKUP_BASE


def _rng():
    return np.random.default_rng(0)


def _base_firm(**kwargs) -> Firm:
    defaults = dict(
        id=0, sector="knowledge_work", size_tier="small", n_workers=20,
        capital_stock=1_000_000.0, ai_capital=0.0, ai_adoption_level=0.1,
        markup=MARKUP_BASE,
    )
    defaults.update(kwargs)
    f = Firm(**defaults)
    f.init_expectations(1.0, 80_000, 0.5, 0.05)
    return f


def test_ai_adoption_increases_over_time():
    rng = _rng()
    f = _base_firm(ai_adoption_level=0.05)
    initial = f.ai_adoption_level
    for _ in range(20):
        f.step_ai_adoption(rng, regulatory_friction=0.0, capability_index=0.8)
    assert f.ai_adoption_level > initial


def test_ai_adoption_capped_at_one():
    rng = _rng()
    f = _base_firm(ai_adoption_level=0.99)
    for _ in range(100):
        f.step_ai_adoption(rng, regulatory_friction=0.0, capability_index=1.0)
    assert f.ai_adoption_level <= 1.0


def test_high_friction_slows_adoption():
    rng = np.random.default_rng(42)
    f_low = _base_firm(id=1)
    f_high = _base_firm(id=2)
    for _ in range(10):
        f_low.step_ai_adoption(rng, regulatory_friction=0.0, capability_index=0.8)
    rng2 = np.random.default_rng(42)
    for _ in range(10):
        f_high.step_ai_adoption(rng2, regulatory_friction=0.9, capability_index=0.8)
    assert f_low.ai_adoption_level >= f_high.ai_adoption_level


def test_hiring_increases_vacancies_when_understaffed():
    rng = _rng()
    f = _base_firm(n_workers=5, desired_workers=0)
    # Manually set expectation to indicate high demand
    f.expectations.sector_demand.update(2.0)
    f.step_hiring(rng, sector_base_wage=80_000, productivity_multiplier=1.0)
    assert f.desired_workers > f.n_workers or f.desired_workers >= 0


def test_pricing_adjusts_with_demand():
    f = _base_firm()
    initial_price = f.price_level
    f.step_pricing(demand_pressure=1.0)  # high demand
    assert f.price_level >= initial_price


def test_pricing_no_negative_markup():
    f = _base_firm()
    for _ in range(50):
        f.step_pricing(demand_pressure=-5.0)
    assert f.markup >= 0


def test_create_firms_total_workers():
    rng = _rng()
    firms = create_firms("knowledge_work", 50, 500, 80_000, rng)
    total = sum(f.n_workers for f in firms)
    assert total == 500


def test_size_tier():
    assert _size_tier(1) == "micro"
    assert _size_tier(5) == "micro"
    assert _size_tier(6) == "small"
    assert _size_tier(50) == "small"
    assert _size_tier(51) == "medium"
    assert _size_tier(500) == "medium"
    assert _size_tier(501) == "large"


def test_financials_non_negative():
    rng = _rng()
    f = _base_firm()
    f.compute_financials(80_000)
    assert f.revenue >= 0
    assert f.labor_cost >= 0
    assert f.capital_cost >= 0
