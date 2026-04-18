"""Macro accounting layer: GDP, labor/capital shares, price indices."""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from ai_econ_sim.config import SECTORS


@dataclass
class MacroAccounts:
    """Macro aggregate state for one quarter."""
    quarter: int

    # Sectoral value-added (nominal)
    sector_value_added: dict[str, float] = field(default_factory=dict)

    # Sectoral gross output
    sector_gross_output: dict[str, float] = field(default_factory=dict)

    # Sectoral employment (worker-count)
    sector_employment: dict[str, int] = field(default_factory=dict)

    # Sectoral median wages (annual)
    sector_median_wage: dict[str, float] = field(default_factory=dict)

    # Sectoral price indices (normalized to 1.0 at t=0)
    sector_price_index: dict[str, float] = field(default_factory=dict)

    # Sectoral AI adoption level [0,1]
    sector_ai_adoption: dict[str, float] = field(default_factory=dict)

    # Robotics adoption level per sector [0, max_labor_displacement]
    sector_robotics_adoption: dict[str, float] = field(default_factory=dict)

    # Interface realization fraction this quarter [0,1]
    interface_realization: float = 0.0

    # Factor income
    total_labor_income: float = 0.0
    total_capital_income: float = 0.0
    ai_sector_capital_income: float = 0.0

    # Aggregate
    nominal_gdp: float = 0.0
    real_gdp: float = 0.0    # deflated by aggregate price index
    aggregate_price_index: float = 1.0

    # Labor market
    total_employed: int = 0
    total_unemployed: int = 0
    total_out_of_lf: int = 0

    # Income distribution
    gini: float = 0.0

    # Tax revenue
    tax_revenue: float = 0.0

    # Composite AI capability index this quarter (mean over task capabilities)
    capability_index: float = 0.0

    # Demographic flows (set by Model._step_demographics / _step_firm_dynamics)
    retirements: int = 0
    new_entrants: int = 0
    firm_exits: int = 0
    firm_entries: int = 0

    # Generational employment shares
    gen_employment: dict[str, int] = field(default_factory=dict)

    # Knowledge-work median wages by occupation bucket
    kw_occupation_wages: dict[str, float] = field(default_factory=dict)

    # Retraining flows this quarter
    retraining_initiations: int = 0
    retraining_successes: int = 0
    retraining_failures: int = 0

    # Policy outcomes (all zero when policy is disabled)
    policy_ubi_disbursed: float = 0.0            # total UBI paid this quarter
    policy_retraining_subsidized: int = 0        # additional retraining initiations due to subsidy
    policy_windfall_tax_collected: float = 0.0   # AI-sector windfall tax revenue this quarter


class MacroAccounting:
    """
    Computes macro aggregates from sector-level and agent-level state.
    Enforces accounting identities.
    """

    def __init__(self, initial_nominal_gdp: float, gdp_shares: dict[str, float]):
        self._initial_gdp = initial_nominal_gdp
        self._gdp_shares = gdp_shares
        self._base_price_indices: dict[str, float] = {s: 1.0 for s in SECTORS}

    def compute(
        self,
        quarter: int,
        sector_labor_income: dict[str, float],
        sector_capital_income: dict[str, float],
        sector_employment: dict[str, int],
        sector_wages: dict[str, list[float]],
        sector_price_indices: dict[str, float],
        sector_ai_adoption: dict[str, float],
        total_employed: int,
        total_unemployed: int,
        total_out_of_lf: int,
        all_worker_incomes: list[float],
        sector_robotics_adoption: dict[str, float] | None = None,
        interface_realization: float = 0.0,
        corporate_tax_rate: float = 0.21,
        payroll_tax_rate: float = 0.15,
        income_tax_rate: float = 0.22,
    ) -> MacroAccounts:

        accts = MacroAccounts(quarter=quarter)

        for s in SECTORS:
            li = sector_labor_income.get(s, 0.0)
            ki = sector_capital_income.get(s, 0.0)
            accts.sector_value_added[s] = li + ki
            accts.sector_employment[s] = sector_employment.get(s, 0)
            wages = sector_wages.get(s, [])
            accts.sector_median_wage[s] = float(np.median(wages)) if wages else 0.0
            accts.sector_price_index[s] = sector_price_indices.get(s, 1.0)
            accts.sector_ai_adoption[s] = sector_ai_adoption.get(s, 0.0)
            rob = sector_robotics_adoption or {}
            accts.sector_robotics_adoption[s] = rob.get(s, 0.0)

        accts.interface_realization = interface_realization

        accts.total_labor_income = sum(sector_labor_income.values())
        accts.total_capital_income = sum(sector_capital_income.values())
        accts.ai_sector_capital_income = sector_capital_income.get("ai_compute", 0.0)

        # GDP = sum of value-added (expenditure = income in simplified model)
        accts.nominal_gdp = accts.total_labor_income + accts.total_capital_income

        # Aggregate price index (GDP-share weighted)
        agg_price = 0.0
        gdp = accts.nominal_gdp or 1.0
        for s in SECTORS:
            weight = accts.sector_value_added.get(s, 0.0) / gdp
            agg_price += weight * sector_price_indices.get(s, 1.0)
        accts.aggregate_price_index = max(agg_price, 1e-6)
        accts.real_gdp = accts.nominal_gdp / accts.aggregate_price_index

        accts.total_employed = total_employed
        accts.total_unemployed = total_unemployed
        accts.total_out_of_lf = total_out_of_lf

        # Gini from worker income distribution
        if all_worker_incomes:
            accts.gini = _gini(np.array(all_worker_incomes, dtype=float))

        # Tax revenue
        payroll_tax = accts.total_labor_income * payroll_tax_rate
        income_tax = accts.total_labor_income * income_tax_rate
        corporate_tax = accts.total_capital_income * corporate_tax_rate
        accts.tax_revenue = payroll_tax + income_tax + corporate_tax

        return accts

    def labor_share(self, accts: MacroAccounts) -> float:
        total = accts.total_labor_income + accts.total_capital_income
        return accts.total_labor_income / total if total > 0 else 0.0

    def capital_share(self, accts: MacroAccounts) -> float:
        return 1.0 - self.labor_share(accts)


def _gini(incomes: np.ndarray) -> float:
    """Compute Gini coefficient from income array."""
    if len(incomes) == 0:
        return 0.0
    incomes = incomes[incomes >= 0]
    if incomes.sum() == 0:
        return 0.0
    incomes = np.sort(incomes)
    n = len(incomes)
    index = np.arange(1, n + 1)
    return float((2 * index - n - 1).dot(incomes) / (n * incomes.sum()))
