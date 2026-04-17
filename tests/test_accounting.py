"""Tests for macro accounting identities."""

import numpy as np
import pytest
from ai_econ_sim.macro.accounting import MacroAccounting, MacroAccounts, _gini
from ai_econ_sim.config import SECTORS


def _make_accounting():
    gdp_shares = {"ai_compute": 0.02, "knowledge_work": 0.25, "services": 0.35, "goods": 0.25, "infrastructure": 0.13}
    return MacroAccounting(initial_nominal_gdp=25e12, gdp_shares=gdp_shares)


def _sector_dict(val):
    return {s: val for s in SECTORS}


def test_gdp_equals_labor_plus_capital():
    acc = _make_accounting()
    labor = {"ai_compute": 1e9, "knowledge_work": 10e9, "services": 12e9, "goods": 8e9, "infrastructure": 4e9}
    capital = {"ai_compute": 2e9, "knowledge_work": 4e9, "services": 6e9, "goods": 5e9, "infrastructure": 3e9}
    wages = {s: [50_000.0] * 100 for s in SECTORS}
    prices = _sector_dict(1.0)
    adoption = _sector_dict(0.05)

    accts = acc.compute(
        quarter=0,
        sector_labor_income=labor,
        sector_capital_income=capital,
        sector_employment={s: 100 for s in SECTORS},
        sector_wages=wages,
        sector_price_indices=prices,
        sector_ai_adoption=adoption,
        total_employed=500,
        total_unemployed=50,
        total_out_of_lf=10,
        all_worker_incomes=[50_000.0] * 500,
    )
    expected_gdp = sum(labor.values()) + sum(capital.values())
    assert abs(accts.nominal_gdp - expected_gdp) < 1.0


def test_labor_capital_shares_sum_to_one():
    acc = _make_accounting()
    labor = {s: 1e9 for s in SECTORS}
    capital = {s: 5e8 for s in SECTORS}
    wages = {s: [60_000.0] * 10 for s in SECTORS}
    prices = _sector_dict(1.0)
    adoption = _sector_dict(0.0)

    accts = acc.compute(
        quarter=1, sector_labor_income=labor, sector_capital_income=capital,
        sector_employment=_sector_dict(10), sector_wages=wages,
        sector_price_indices=prices, sector_ai_adoption=adoption,
        total_employed=50, total_unemployed=5, total_out_of_lf=0,
        all_worker_incomes=[60_000.0] * 50,
    )
    ls = acc.labor_share(accts)
    cs = acc.capital_share(accts)
    assert abs(ls + cs - 1.0) < 1e-9


def test_gini_zero_for_equal_incomes():
    incomes = np.array([50_000.0] * 100)
    g = _gini(incomes)
    assert abs(g) < 1e-9


def test_gini_positive_for_unequal_incomes():
    incomes = np.array([10_000.0] * 80 + [1_000_000.0] * 20)
    g = _gini(incomes)
    assert 0.0 < g <= 1.0


def test_gini_max_inequality():
    # One person has everything
    incomes = np.array([0.0] * 99 + [1_000_000.0])
    g = _gini(incomes)
    assert g > 0.9


def test_tax_revenue_positive():
    acc = _make_accounting()
    labor = {s: 1e9 for s in SECTORS}
    capital = {s: 5e8 for s in SECTORS}
    wages = {s: [60_000.0] * 10 for s in SECTORS}
    accts = acc.compute(
        quarter=0, sector_labor_income=labor, sector_capital_income=capital,
        sector_employment=_sector_dict(10), sector_wages=wages,
        sector_price_indices=_sector_dict(1.0), sector_ai_adoption=_sector_dict(0.0),
        total_employed=50, total_unemployed=5, total_out_of_lf=0,
        all_worker_incomes=[60_000.0] * 50,
    )
    assert accts.tax_revenue > 0


def test_non_negative_outputs():
    acc = _make_accounting()
    labor = {s: max(0, 1e9) for s in SECTORS}
    capital = {s: max(0, 5e8) for s in SECTORS}
    wages = {s: [60_000.0] * 10 for s in SECTORS}
    accts = acc.compute(
        quarter=0, sector_labor_income=labor, sector_capital_income=capital,
        sector_employment=_sector_dict(10), sector_wages=wages,
        sector_price_indices=_sector_dict(1.0), sector_ai_adoption=_sector_dict(0.0),
        total_employed=50, total_unemployed=5, total_out_of_lf=0,
        all_worker_incomes=[60_000.0] * 50,
    )
    assert accts.nominal_gdp >= 0
    assert accts.real_gdp >= 0
    assert accts.gini >= 0
