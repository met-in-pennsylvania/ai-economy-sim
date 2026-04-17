"""Firm agent: state and quarterly decision logic."""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field

from ai_econ_sim.agents.expectations import FirmExpectations
from ai_econ_sim.config import (
    ADOPTION_BASE_RATE, ADOPTION_PEER_WEIGHT, ADOPTION_FIRM_SIZE_FRICTION,
    HIRING_SPEED, MARKUP_BASE, MARKUP_ADJUSTMENT_SPEED,
    WAGE_SKILL_PREMIUM, WAGE_IDIOSYNCRATIC_STD,
    KW_OCCUPATIONS,
)


@dataclass
class Firm:
    id: int
    sector: str
    size_tier: str                       # micro/small/medium/large
    n_workers: int                       # current headcount

    # Capital
    capital_stock: float                 # generic capital (stylized units)
    ai_capital: float                    # AI-specific capital

    # Workforce composition: occupation -> count (only meaningful for knowledge_work)
    workforce: dict[str, int] = field(default_factory=dict)

    # AI adoption
    ai_adoption_level: float = 0.0       # [0, 1]

    # Pricing
    price_level: float = 1.0            # relative to sector average
    markup: float = MARKUP_BASE

    # Financials (quarterly flow)
    revenue: float = 0.0
    labor_cost: float = 0.0
    capital_cost: float = 0.0
    profit: float = 0.0

    # Expectations (initialized post-construction via init_expectations)
    expectations: FirmExpectations | None = field(default=None, repr=False)

    # Productivity multiplier (updated by capability layer each quarter)
    productivity_multiplier: float = 1.0

    # Desired workforce (set during hiring step)
    desired_workers: int = 0

    # Vacancies posted this quarter
    vacancies: int = 0

    # Current wage offer (annual, for hiring)
    wage_offer: float = 0.0

    def init_expectations(
        self, sector_demand: float, own_wage: float,
        ai_capability: float, peer_adoption: float,
    ) -> None:
        self.expectations = FirmExpectations(sector_demand, own_wage, ai_capability, peer_adoption)

    # ------------------------------------------------------------------
    # Quarterly decision steps
    # ------------------------------------------------------------------

    def step_update_expectations(
        self,
        sector_demand: float,
        own_wage: float,
        ai_capability: float,
        peer_adoption: float,
    ) -> None:
        if self.expectations is None:
            self.init_expectations(sector_demand, own_wage, ai_capability, peer_adoption)
        else:
            self.expectations.update(sector_demand, own_wage, ai_capability, peer_adoption)

    def step_ai_adoption(
        self,
        rng: np.random.Generator,
        regulatory_friction: float,
        capability_index: float,
    ) -> None:
        """
        Probabilistic AI adoption update.
        DESIGN CHOICE: logistic-shaped adoption probability based on cost-benefit,
        peer pressure, and regulatory friction. Delta capped at ADOPTION_BASE_RATE.
        """
        if self.expectations is None:
            return

        # Cost-benefit: higher capability -> higher benefit
        cb_signal = capability_index * (1.0 - self.ai_adoption_level)

        # Peer pressure: if peers are ahead, nudge toward adoption
        peer_gap = max(0.0, self.expectations.peer_adoption.level - self.ai_adoption_level)
        peer_signal = ADOPTION_PEER_WEIGHT * peer_gap

        raw_signal = cb_signal + peer_signal

        # Apply frictions
        size_friction = ADOPTION_FIRM_SIZE_FRICTION.get(self.size_tier, 1.0)
        effective_signal = raw_signal * (1.0 - regulatory_friction) * size_friction

        # Logistic mapping to adoption delta
        prob = 1.0 / (1.0 + np.exp(-5.0 * (effective_signal - 0.3)))
        delta = prob * ADOPTION_BASE_RATE * rng.uniform(0.5, 1.5)
        self.ai_adoption_level = min(1.0, self.ai_adoption_level + delta)

    def step_hiring(
        self,
        rng: np.random.Generator,
        sector_base_wage: float,
        productivity_multiplier: float,
    ) -> int:
        """
        Compute desired workforce and post vacancies (or fire workers).
        Returns net change in desired headcount.
        """
        self.productivity_multiplier = productivity_multiplier

        if self.expectations is None:
            return 0

        # Expected demand drives desired output
        demand_level = self.expectations.sector_demand.level
        demand_trend = self.expectations.sector_demand.trend

        # Desired output proportional to expected demand
        desired_output_factor = max(0.1, demand_level + 0.5 * demand_trend)

        # Workers needed = desired_output / productivity_per_worker
        # Productivity per worker rises with AI adoption
        effective_productivity = max(0.5, productivity_multiplier)
        base_desired = max(1, round(self.n_workers * desired_output_factor / effective_productivity))
        self.desired_workers = base_desired

        gap = self.desired_workers - self.n_workers
        adjustment = round(gap * HIRING_SPEED)

        if adjustment > 0:
            self.vacancies = adjustment
        else:
            self.vacancies = 0
            # Fire immediately (with friction): reduction = |adjustment|
            self.n_workers = max(0, self.n_workers + adjustment)

        # Wage offer: sector base + skill noise
        skill_premium = rng.normal(1.0, WAGE_IDIOSYNCRATIC_STD)
        self.wage_offer = sector_base_wage * max(0.8, skill_premium)

        return adjustment

    def step_pricing(self, demand_pressure: float) -> None:
        """
        Adjust markup toward desired based on demand pressure.
        demand_pressure > 0 means more demand than supply -> raise price.
        """
        target_markup = MARKUP_BASE * (1.0 + 0.5 * demand_pressure)
        target_markup = max(0.05, min(0.40, target_markup))
        self.markup += MARKUP_ADJUSTMENT_SPEED * (target_markup - self.markup)
        # Price level reflects markup relative to baseline
        self.price_level = 1.0 + self.markup

    def compute_financials(self, sector_base_wage: float) -> None:
        """Update revenue, costs, profit for this quarter."""
        # Labor cost: workers * wage (quarterly = annual / 4)
        avg_wage = sector_base_wage * (1.0 + WAGE_SKILL_PREMIUM)
        self.labor_cost = self.n_workers * avg_wage / 4.0

        # Capital cost: 5% quarterly return on capital (stylized)
        self.capital_cost = (self.capital_stock + self.ai_capital) * 0.05

        # Revenue: value added proxy = labor + capital + markup profit
        self.revenue = (self.labor_cost + self.capital_cost) * self.price_level * self.productivity_multiplier
        self.profit = self.revenue - self.labor_cost - self.capital_cost

    def hire_worker(self) -> bool:
        """Accept one worker from the matching pool. Returns True if firm wanted a worker."""
        if self.vacancies > 0:
            self.n_workers += 1
            self.vacancies -= 1
            return True
        return False

    def fire_worker(self) -> bool:
        """Lay off one worker. Returns True if firing occurred."""
        if self.n_workers > 0 and self.n_workers > self.desired_workers:
            self.n_workers -= 1
            return True
        return False


def create_firms(
    sector: str,
    n_firms: int,
    n_workers_total: int,
    sector_base_wage: float,
    rng: np.random.Generator,
    initial_ai_adoption: float = 0.0,
) -> list[Firm]:
    """
    Initialize a population of firms for a sector with power-law size distribution.
    DESIGN CHOICE: power-law via Pareto sampling, then normalize to hit target total workers.
    """
    from ai_econ_sim.config import FIRM_SIZE_PARETO_ALPHA

    # Sample sizes from Pareto distribution
    raw_sizes = rng.pareto(FIRM_SIZE_PARETO_ALPHA, size=n_firms) + 1.0
    # Normalize to total workers using largest-remainder rounding
    proportions = raw_sizes / raw_sizes.sum() * n_workers_total
    sizes = np.floor(proportions).astype(int)
    sizes = np.maximum(sizes, 1)
    remainder = n_workers_total - sizes.sum()
    if remainder != 0:
        # Distribute remainder to firms with largest fractional parts
        fractions = proportions - np.floor(proportions)
        top_indices = np.argsort(fractions)[::-1][:abs(remainder)]
        sizes[top_indices] += int(np.sign(remainder))
    sizes = np.maximum(sizes, 1)

    firms = []
    for i, n_w in enumerate(sizes):
        tier = _size_tier(n_w)
        # Capital proportional to size
        cap = float(n_w) * 50_000 * rng.uniform(0.8, 1.2)
        ai_cap = float(n_w) * 5_000 * rng.uniform(0.5, 1.5) if sector == "ai_compute" else 0.0

        firm = Firm(
            id=i,
            sector=sector,
            size_tier=tier,
            n_workers=int(n_w),
            capital_stock=cap,
            ai_capital=ai_cap,
            ai_adoption_level=initial_ai_adoption * rng.uniform(0.8, 1.2),
            markup=MARKUP_BASE * rng.uniform(0.8, 1.2),
        )
        firms.append(firm)

    return firms


def _size_tier(n_workers: int) -> str:
    if n_workers <= 5:
        return "micro"
    if n_workers <= 50:
        return "small"
    if n_workers <= 500:
        return "medium"
    return "large"
