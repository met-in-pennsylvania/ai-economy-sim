"""Main Model class — orchestrates the quarterly simulation step."""

from __future__ import annotations
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from ai_econ_sim.config import (
    SECTORS, FIRM_COUNTS, WORKER_COUNTS, WAGE_BASE_BY_SECTOR,
    BASE_GROWTH_RATES, QUARTERS_PER_YEAR, AI_SPILLOVER,
    CORPORATE_TAX_RATE, PAYROLL_TAX_RATE, INCOME_TAX_RATE,
    RETRAINING_MIN_QUARTERS, RETRAINING_MAX_QUARTERS,
)
from ai_econ_sim.sectors import SECTOR_DEFINITIONS
from ai_econ_sim.scenarios.loader import Scenario
from ai_econ_sim.capability.tasks import compute_occupation_exposure
from ai_econ_sim.agents.firm import Firm, create_firms
from ai_econ_sim.agents.worker import (
    Worker, create_workers, _assign_generation, _sector_default_occupation,
    _sample_occupation, _retraining_success_prob, RetrainingState,
)
from ai_econ_sim.macro.accounting import MacroAccounting, MacroAccounts
from ai_econ_sim.capability.trajectory import CapabilityTrajectory

log = logging.getLogger(__name__)


class Model:
    """
    AI-Economy Macro Simulation v0.
    Orchestrates firms, workers, capability, and macro accounting.
    """

    def __init__(self, scenario: Scenario, population_scale: float = 1.0, dev_assertions: bool = True):
        self.scenario = scenario
        self.dev_assertions = dev_assertions
        self.quarter = 0

        self.rng = np.random.default_rng(scenario.seed)

        # Scale populations — use employment_shares from initial_conditions when provided,
        # otherwise fall back to config defaults.
        ic_emp = scenario.initial_conditions.employment_shares
        if ic_emp:
            total_firms_base = sum(FIRM_COUNTS.values())
            total_workers_base = sum(WORKER_COUNTS.values())
            self._firm_counts = {
                s: max(5, round(ic_emp.get(s, FIRM_COUNTS[s] / total_firms_base) * total_firms_base * population_scale))
                for s in SECTORS
            }
            self._worker_counts = {
                s: max(10, round(ic_emp[s] * total_workers_base * population_scale))
                for s in SECTORS
            }
        else:
            self._firm_counts = {s: max(5, round(n * population_scale)) for s, n in FIRM_COUNTS.items()}
            self._worker_counts = {s: max(50, round(n * population_scale)) for s, n in WORKER_COUNTS.items()}
        total_workers = sum(self._worker_counts.values())

        ic = scenario.initial_conditions

        # Initialize firms
        self.firms: dict[str, list[Firm]] = {}
        for s in SECTORS:
            n_firms = self._firm_counts[s]
            n_workers = self._worker_counts[s]
            base_wage = ic.initial_wages.get(s, WAGE_BASE_BY_SECTOR[s])
            initial_adoption = 0.05 if s == "ai_compute" else 0.02
            self.firms[s] = create_firms(s, n_firms, n_workers, base_wage, self.rng, initial_adoption)

        # Initialize workers matched to firms
        self.workers: dict[str, list[Worker]] = {}
        self._worker_id_offset = 0
        for s in SECTORS:
            base_wage = ic.initial_wages.get(s, WAGE_BASE_BY_SECTOR[s])
            # Build employer_id list matching workers to firms
            emp_ids = _assign_employer_ids(self.firms[s], self._worker_counts[s])
            workers = create_workers(s, self._worker_counts[s], base_wage, self.rng, emp_ids)
            # Apply global unique IDs
            for w in workers:
                w.id += self._worker_id_offset
            self._worker_id_offset += len(workers)
            self.workers[s] = workers

        # Initialize worker expectations
        for s in SECTORS:
            emp = len([w for w in self.workers[s] if w.is_employed])
            wages = [w.current_wage for w in self.workers[s] if w.is_employed]
            median_wage = float(np.median(wages)) if wages else WAGE_BASE_BY_SECTOR[s]
            for w in self.workers[s]:
                w.init_expectations(float(emp), median_wage)

        # Initialize firm expectations
        for s in SECTORS:
            sector_demand = 1.0  # normalized
            sector_wage = WAGE_BASE_BY_SECTOR.get(s, 50_000)
            cap_idx = self.scenario.capability.capability_index(0)
            peer_adoption = np.mean([f.ai_adoption_level for f in self.firms[s]])
            for firm in self.firms[s]:
                firm.init_expectations(sector_demand, sector_wage, cap_idx, peer_adoption)

        # Macro accounting
        self.accounting = MacroAccounting(
            initial_nominal_gdp=ic.nominal_gdp,
            gdp_shares=ic.gdp_shares,
        )

        # Time series storage
        self.history: list[MacroAccounts] = []

        # Sector demand indices (start at 1.0)
        self._sector_demand: dict[str, float] = {s: 1.0 for s in SECTORS}

        # Sector productivity multipliers
        self._productivity: dict[str, float] = {s: 1.0 for s in SECTORS}

        # Robotics adoption level per sector [0, max_labor_displacement]
        self._robotics_adoption: dict[str, float] = {s: 0.0 for s in SECTORS}

        # Track occupation exposures for knowledge_work
        self._kw_occupation_exposure: np.ndarray = np.zeros(4)

        # Retraining flow counters (reset each quarter)
        self._retraining_initiations: int = 0
        self._retraining_successes: int = 0
        self._retraining_failures: int = 0

        # Policy outcome counters (reset each quarter)
        self._policy_retraining_subsidized: int = 0
        self._policy_ubi_disbursed: float = 0.0

        # Demographic flow counters (reset each quarter)
        self._retirements_this_quarter: int = 0
        self._new_entrants_this_quarter: int = 0
        self._firm_exits_this_quarter: int = 0
        self._firm_entries_this_quarter: int = 0
        self._next_worker_id: int = self._worker_id_offset
        self._next_firm_id: dict[str, int] = {s: max((f.id for f in self.firms[s]), default=-1) + 1 for s in SECTORS}

        log.info("Model initialized: %d firms, %d workers, scenario=%s",
                 sum(len(v) for v in self.firms.values()),
                 sum(len(v) for v in self.workers.values()),
                 scenario.name)

    # ------------------------------------------------------------------
    # Main simulation step
    # ------------------------------------------------------------------

    def step(self) -> MacroAccounts:
        q = self.quarter
        log.debug("Quarter %d begin", q)

        capability = self.scenario.capability.capability_at(q)
        reliability_floors = self.scenario.capability.reliability_floors()
        cap_idx = float(capability.mean())

        # 1. Update KW occupation exposures (task-level model)
        kw_exposure = compute_occupation_exposure(capability, reliability_floors)
        self._kw_occupation_exposure = kw_exposure

        # 2. Update sector demand (driven by GDP growth + AI productivity)
        self._update_sector_demand(q, cap_idx)

        # 3. Update robotics adoption and compute sector productivity multipliers
        self._update_robotics_adoption(q)
        self._update_productivity(q, cap_idx, kw_exposure)

        # 4. Firm decisions
        self._step_firms(cap_idx)

        # 4b. Firm entry/exit dynamics (bankrupt firms exit; new firms enter growing sectors)
        self._step_firm_dynamics()

        # 4c. Reconcile layoffs: fire worker objects when firm headcount dropped
        self._reconcile_layoffs()

        # 5. Worker decisions (retraining, LFP)
        self._step_workers()

        # 5b. Demographics: aging, retirement, new entrants
        self._step_demographics()

        # 5c. Re-reconcile after workers may have left sectors via retraining
        self._reconcile_layoffs()

        # 6. Labor market matching
        self._step_labor_matching()

        # 7. Compute macro accounts
        accts = self._compute_macro()

        # 8. Assertions
        if self.dev_assertions:
            self._assert_invariants(accts)

        self.history.append(accts)
        self.quarter += 1
        log.debug("Quarter %d end: GDP=%.2fT, LF share=%.3f", q,
                  accts.nominal_gdp / 1e12, self.accounting.labor_share(accts))
        return accts

    def run(self, n_quarters: Optional[int] = None) -> list[MacroAccounts]:
        """Run full simulation."""
        horizon = n_quarters or self.scenario.horizon_quarters
        for _ in range(horizon):
            self.step()
        log.info("Simulation complete: %d quarters", horizon)
        return self.history

    # ------------------------------------------------------------------
    # Internal step helpers
    # ------------------------------------------------------------------

    def _update_robotics_adoption(self, q: int) -> None:
        """
        Ramp robotics adoption in affected sectors after deployment_start_quarter.
        Robotics adoption is sector-wide (not firm-level) and diffuses at a fixed rate
        up to max_labor_displacement. It represents the fraction of serviceable physical
        tasks that can now be handed to robot hardware.
        """
        rob = self.scenario.robotics
        if q < rob.deployment_start_quarter:
            return
        for s in rob.affected_sectors:
            current = self._robotics_adoption.get(s, 0.0)
            if current < rob.max_labor_displacement:
                self._robotics_adoption[s] = min(
                    rob.max_labor_displacement,
                    current + rob.diffusion_rate,
                )

    def _update_sector_demand(self, q: int, cap_idx: float) -> None:
        for s in SECTORS:
            base_growth = BASE_GROWTH_RATES.get(s, 0.02) / QUARTERS_PER_YEAR
            avg_adoption = np.mean([f.ai_adoption_level for f in self.firms[s]]) if self.firms[s] else 0.0

            if s == "ai_compute":
                extra = avg_adoption * 0.02
            else:
                extra = 0.0

            # Consumer AI discount: customers preferring human-delivered services
            # damp demand growth proportionally to how AI-saturated the sector is.
            discount = self.scenario.sentiment.consumer_ai_discount.get(s, 0.0)
            demand_multiplier = max(0.0, 1.0 - discount * avg_adoption)

            self._sector_demand[s] *= (1.0 + (base_growth + extra) * demand_multiplier)

    def _update_productivity(self, q: int, cap_idx: float, kw_exposure: np.ndarray) -> None:
        # Interface realization: fraction of theoretical AI gain that materialises
        # given current tooling maturity. Improves over the scenario horizon.
        # Does NOT apply to robotics (different interaction model).
        realization = self.scenario.interface_realization.at(q)

        for s in SECTORS:
            sector_def = SECTOR_DEFINITIONS[s]
            resistance = self.scenario.sentiment.workforce_resistance.get(s, 0.0)

            if s == "knowledge_work":
                avg_adoption = np.mean([f.ai_adoption_level for f in self.firms[s]]) if self.firms[s] else 0.0
                avg_exposure = float(kw_exposure.mean())
                # Effective adoption reduced by workforce resistance; gain further
                # scaled by how much of the theoretical benefit workers can actually
                # capture given current interface maturity.
                effective_adoption = avg_adoption * (1.0 - resistance)
                mult = 1.0 + effective_adoption * avg_exposure * realization
                self._productivity[s] = mult

            elif s == "ai_compute":
                chip_growth = self.scenario.compute.chip_supply_growth_annual / QUARTERS_PER_YEAR
                self._productivity[s] = self._productivity[s] * (1.0 + chip_growth)

            else:
                spillover = AI_SPILLOVER.get(s, 0.0)
                avg_adoption = np.mean([f.ai_adoption_level for f in self.firms[s]]) if self.firms[s] else 0.0
                effective_adoption = avg_adoption * (1.0 - resistance)
                quarterly_growth = spillover * cap_idx * effective_adoption * realization / QUARTERS_PER_YEAR
                ai_mult = self._productivity.get(s, 1.0) * (1.0 + quarterly_growth)

                # Robotics boost is separate — physical task automation doesn't
                # share the cognitive interface friction of knowledge tools.
                rob_adoption = self._robotics_adoption.get(s, 0.0)
                robotics_mult = 1.0 / max(0.01, 1.0 - rob_adoption)

                self._productivity[s] = ai_mult * robotics_mult

    def _step_firms(self, cap_idx: float) -> None:
        for s in SECTORS:
            if not self.firms[s]:
                continue
            peer_adoption = float(np.mean([f.ai_adoption_level for f in self.firms[s]]))
            sector_wage = self._sector_median_wage(s)
            regulatory_friction = self.scenario.regulation_friction(s)
            productivity_mult = self._productivity.get(s, 1.0)

            for firm in self.firms[s]:
                firm.step_update_expectations(
                    sector_demand=self._sector_demand[s],
                    own_wage=sector_wage,
                    ai_capability=cap_idx,
                    peer_adoption=peer_adoption,
                )
                firm.step_ai_adoption(self.rng, regulatory_friction, cap_idx)
                firm.step_hiring(self.rng, sector_wage, productivity_mult)
                demand_pressure = (self._sector_demand[s] - 1.0) * 0.5
                firm.step_pricing(demand_pressure)
                firm.compute_financials(sector_wage)

    def _step_firm_dynamics(self) -> None:
        """
        Firm entry and exit.

        Exit: micro/small firms with 4+ consecutive loss quarters exit with
        probability proportional to how long they've been losing money.
        Workers become unemployed. Larger firms need more losses to exit.

        Entry: when sector demand is sustainably high (>1.08), one small
        entrant firm is spawned per quarter per 20 firms already in sector,
        drawn with 1-3 initial vacancies (workers hired in matching step).
        """
        self._firm_exits_this_quarter = 0
        self._firm_entries_this_quarter = 0

        # Exit pass
        for s in SECTORS:
            surviving = []
            for firm in self.firms[s]:
                # Exit threshold varies by size (large firms more resilient)
                threshold = {"micro": 3, "small": 4, "medium": 6, "large": 8}.get(firm.size_tier, 4)
                exit_prob = {"micro": 0.35, "small": 0.20, "medium": 0.10, "large": 0.05}.get(firm.size_tier, 0.20)
                if firm.consecutive_losses >= threshold and self.rng.random() < exit_prob:
                    # Lay off all workers
                    for w in self.workers.get(s, []):
                        if w.is_employed and w.employer_id == firm.id:
                            w.lose_job()
                    self._firm_exits_this_quarter += 1
                else:
                    surviving.append(firm)
            self.firms[s] = surviving

        # Entry pass
        for s in SECTORS:
            demand = self._sector_demand.get(s, 1.0)
            if demand < 1.08:
                continue
            n_existing = max(1, len(self.firms[s]))
            # One entrant per 20 existing firms when demand is strong
            entry_prob = min(0.5, (demand - 1.08) * 5.0) * (n_existing / max(1, n_existing))
            if self.rng.random() > entry_prob:
                continue

            sector_wage = self._sector_median_wage(s)
            new_id = self._next_firm_id[s]
            self._next_firm_id[s] += 1

            from ai_econ_sim.agents.firm import _size_tier
            n_init = int(self.rng.integers(1, 4))
            cap = float(n_init) * 50_000 * self.rng.uniform(0.7, 1.1)
            peer_adopt = float(np.mean([f.ai_adoption_level for f in self.firms[s]])) if self.firms[s] else 0.05
            cap_idx = float(self.scenario.capability.capability_at(self.quarter).mean())

            new_firm = Firm(
                id=new_id,
                sector=s,
                size_tier=_size_tier(n_init),
                n_workers=0,
                capital_stock=cap,
                ai_capital=0.0,
                ai_adoption_level=peer_adopt * self.rng.uniform(0.5, 1.0),
            )
            new_firm.init_expectations(self._sector_demand[s], sector_wage, cap_idx, peer_adopt)
            new_firm.vacancies = n_init
            new_firm.wage_offer = sector_wage * self.rng.uniform(0.95, 1.10)
            self.firms[s].append(new_firm)
            self._firm_entries_this_quarter += 1

    def _step_demographics(self) -> None:
        """
        Quarterly demographic flows:
          - Age all workers by 0.25 years
          - Retire workers aged 65+ with probability (scaled by annual retirement rate)
          - Add new entrant workers to partially replace retirees

        New entrants are young (age 22–24), assigned to sectors proportional to
        current employment distribution. They start unemployed and are picked up
        by the matching step.
        """
        self._retirements_this_quarter = 0
        self._new_entrants_this_quarter = 0

        retirement_rate_quarterly = self.scenario.demographics.retirement_rate_annual / 4.0

        # Age all workers and retire the old ones
        for s in SECTORS:
            surviving = []
            for w in self.workers.get(s, []):
                w.age += 0.25
                if w.age >= 65 and self.rng.random() < retirement_rate_quarterly:
                    # Worker retires: ensure firm headcount is synced later
                    if w.is_employed and w.employer_id is not None:
                        w.lose_job()  # signal to reconcile
                    self._retirements_this_quarter += 1
                    # Do not append — this worker leaves the model
                else:
                    surviving.append(w)
            self.workers[s] = surviving

        # Add new entrants to replace ~50% of retirees
        n_entrants = max(0, round(self._retirements_this_quarter * 0.5))
        if n_entrants == 0:
            return

        # Distribute entrants proportionally to sector employment
        total_workers = sum(len(ws) for ws in self.workers.values())
        if total_workers == 0:
            return

        from ai_econ_sim.config import WAGE_SKILL_PREMIUM, WAGE_IDIOSYNCRATIC_STD, WAGE_BASE_BY_SECTOR
        for _ in range(n_entrants):
            # Pick sector proportional to size
            sizes = [len(self.workers[s]) for s in SECTORS]
            probs = np.array(sizes, dtype=float)
            probs /= probs.sum()
            s = str(self.rng.choice(SECTORS, p=probs))

            age = float(self.rng.integers(22, 25))
            skill = max(1, int(self.rng.integers(1, 4)))  # new entrants: skill 1-3
            occ = _sample_occupation(s, self.rng)
            base_wage = WAGE_BASE_BY_SECTOR.get(s, 50_000)
            log_wage = (
                np.log(base_wage)
                + WAGE_SKILL_PREMIUM * (skill - 1)
                + self.rng.normal(0, WAGE_IDIOSYNCRATIC_STD)
            )
            wage = float(np.exp(log_wage))

            w = Worker(
                id=self._next_worker_id,
                age=age,
                sector=s,
                occupation=occ,
                skill_level=skill,
                current_wage=wage,
                employer_id=None,
                generation=_assign_generation(age),
                is_employed=False,
                is_in_labor_force=True,
            )
            w.init_expectations(
                float(len(self.workers[s])),
                self._sector_median_wage(s),
            )
            self._next_worker_id += 1
            self.workers[s].append(w)
            self._new_entrants_this_quarter += 1

    def _reconcile_layoffs(self) -> None:
        """
        Two-way sync between firm.n_workers and actual employed worker counts.

        Direction 1 (firm shrank): firm.n_workers < roster → fire excess workers.
        Direction 2 (workers left): roster < firm.n_workers → decrement firm count.
        Direction 2 handles workers who retrained or were fired externally.
        """
        for s in SECTORS:
            firm_rosters: dict[int, list[Worker]] = {}
            for w in self.workers.get(s, []):
                if w.is_employed and w.employer_id is not None:
                    firm_rosters.setdefault(w.employer_id, []).append(w)

            for firm in self.firms[s]:
                roster = firm_rosters.get(firm.id, [])
                diff = len(roster) - firm.n_workers
                if diff > 0:
                    # Firm headcount dropped: fire the excess workers
                    to_fire = self.rng.choice(len(roster), size=diff, replace=False)
                    for idx in sorted(to_fire, reverse=True):
                        roster[idx].lose_job()
                elif diff < 0:
                    # Workers left the sector (retraining, etc.): sync firm count down
                    firm.n_workers = max(0, len(roster))

    def _step_workers(self) -> None:
        # Identify declining and growing sectors
        declining = {s for s in SECTORS if self._sector_demand.get(s, 1.0) < 0.95}
        growing = [s for s in SECTORS if self._sector_demand.get(s, 1.0) > 1.02]

        self._retraining_initiations = 0
        self._retraining_successes = 0
        self._retraining_failures = 0
        self._policy_retraining_subsidized = 0
        self._policy_ubi_disbursed = 0.0

        policy = self.scenario.policy
        ubi_quarterly = policy.ubi_annual / 4.0

        all_workers = [w for ws in self.workers.values() for w in ws]

        for w in all_workers:
            sector = w.sector
            if sector is None:
                sector_emp = 0.0
                sector_wage = 0.0
            else:
                sector_emp = float(len([x for x in self.workers.get(sector, []) if x.is_employed]))
                sector_wage = self._sector_median_wage(sector)

            w.step_update_expectations(sector_emp, sector_wage)

            # Advance retraining — detect completions (success vs. failure)
            was_retraining = w.retraining is not None
            completed = w.step_retraining_tick(self.rng)
            if was_retraining and w.retraining is None:
                # Retraining episode ended this quarter
                if completed:
                    self._retraining_successes += 1
                    # Move worker to new sector's list
                    old_sector = None
                    for s, ws in self.workers.items():
                        if w in ws and s != completed:
                            old_sector = s
                            break
                    if old_sector and old_sector != completed:
                        self.workers[old_sector].remove(w)
                        self.workers.setdefault(completed, []).append(w)
                else:
                    self._retraining_failures += 1

            # Retraining decision (base)
            subsidized_this_worker = False
            if not w.is_employed and w.retraining is None and sector in declining:
                initiated = w.step_retraining_decision(self.rng, declining, growing)
                if initiated:
                    self._retraining_initiations += 1
                elif policy.retraining_subsidy_rate > 0.0:
                    # Subsidy: second-chance roll for workers who declined
                    age_factor = max(0.1, 1.0 - 0.02 * max(0, w.age - 30))
                    skill_factor = 0.3 + 0.1 * w.skill_level
                    base_prob = age_factor * skill_factor * 0.15
                    subsidy_prob = base_prob * policy.retraining_subsidy_rate * 0.6
                    if growing and self.rng.random() < subsidy_prob:
                        target_sector = self.rng.choice(growing)
                        target_occ = _sector_default_occupation(target_sector)
                        duration = self.rng.integers(RETRAINING_MIN_QUARTERS, RETRAINING_MAX_QUARTERS + 1)
                        success_prob = _retraining_success_prob(w.age, w.skill_level)
                        w.retraining = RetrainingState(
                            target_sector=target_sector,
                            target_occupation=target_occ,
                            quarters_remaining=int(duration),
                            success_probability=success_prob,
                        )
                        w.is_employed = False
                        w.employer_id = None
                        self._retraining_initiations += 1
                        self._policy_retraining_subsidized += 1
                        subsidized_this_worker = True

            # LFP dynamics
            improving = sector in growing if sector else False
            w.step_labor_force_participation(self.rng, improving)

            # UBI effect on LFP: income floor reduces exits, boosts re-entry
            if ubi_quarterly > 0.0:
                # Track disbursement for non-employed workers
                if not w.is_employed:
                    self._policy_ubi_disbursed += ubi_quarterly
                # Extra re-entry chance for OLF workers when UBI provides income support
                if not w.is_in_labor_force:
                    ref_wage = sector_wage if sector_wage > 0 else 44_000.0
                    extra_reentry = min(0.15, ubi_quarterly / (ref_wage / 4.0) * 0.3)
                    if self.rng.random() < extra_reentry:
                        w.is_in_labor_force = True
                        w.quarters_unemployed = 0

    def _step_labor_matching(self) -> None:
        """Simple stochastic matching: unemployed workers -> firm vacancies."""
        from ai_econ_sim.config import JOB_OFFER_PROBABILITY_BASE, WAGE_ACCEPTANCE_THRESHOLD

        for s in SECTORS:
            # Collect vacant firms and unemployed workers in sector
            vacant_firms = [f for f in self.firms[s] if f.vacancies > 0]
            unemployed = [
                w for w in self.workers.get(s, [])
                if w.is_in_labor_force and not w.is_employed
                and w.retraining is None
            ]

            # Random matching
            self.rng.shuffle(unemployed)  # type: ignore[arg-type]
            for worker in unemployed:
                if not vacant_firms:
                    break
                firm_idx = int(self.rng.integers(0, len(vacant_firms)))
                firm = vacant_firms[firm_idx]
                if self.rng.random() < JOB_OFFER_PROBABILITY_BASE:
                    worker.accept_job(firm.id, s, firm.wage_offer, worker.occupation)
                    firm.hire_worker()
                    if firm.vacancies == 0:
                        vacant_firms.pop(firm_idx)

            # Employed workers: occasional outside offers
            employed = [w for w in self.workers.get(s, []) if w.is_employed]
            for worker in employed:
                if self.rng.random() < 0.05:  # 5% chance of outside offer
                    if vacant_firms:
                        firm = vacant_firms[int(self.rng.integers(0, len(vacant_firms)))]
                        if firm.wage_offer > worker.current_wage * (1 + WAGE_ACCEPTANCE_THRESHOLD):
                            # Accept the outside offer
                            worker.accept_job(firm.id, s, firm.wage_offer, worker.occupation)
                            firm.hire_worker()
                            if firm.vacancies == 0:
                                vacant_firms = [f for f in self.firms[s] if f.vacancies > 0]

    def _compute_macro(self) -> MacroAccounts:
        sector_labor_income: dict[str, float] = {}
        sector_capital_income: dict[str, float] = {}
        sector_employment: dict[str, int] = {}
        sector_wages: dict[str, list[float]] = {}
        sector_price_indices: dict[str, float] = {}
        sector_ai_adoption: dict[str, float] = {}

        for s in SECTORS:
            sector_def = SECTOR_DEFINITIONS[s]
            firms = self.firms[s]
            workers_s = self.workers.get(s, [])

            employed_workers = [w for w in workers_s if w.is_employed]
            wages = [w.current_wage for w in employed_workers]

            n_employed = len(employed_workers)
            sector_employment[s] = n_employed
            sector_wages[s] = wages

            # Total payroll
            total_payroll = sum(wages) / QUARTERS_PER_YEAR

            # Total output (sum of firm revenues)
            total_revenue = sum(f.revenue for f in firms)

            # Labor share of value-added
            ls = sector_def.labor_share
            sector_labor_income[s] = total_revenue * ls
            sector_capital_income[s] = total_revenue * sector_def.capital_share

            # Price index: weighted average of firm price levels
            if firms:
                sector_price_indices[s] = float(np.mean([f.price_level for f in firms]))
            else:
                sector_price_indices[s] = 1.0

            # Average AI adoption
            if firms:
                sector_ai_adoption[s] = float(np.mean([f.ai_adoption_level for f in firms]))
            else:
                sector_ai_adoption[s] = 0.0

        all_workers_flat = [w for ws in self.workers.values() for w in ws]
        total_employed = sum(1 for w in all_workers_flat if w.is_employed)
        total_unemployed = sum(1 for w in all_workers_flat if w.is_in_labor_force and not w.is_employed)
        total_olf = sum(1 for w in all_workers_flat if not w.is_in_labor_force)
        all_incomes = [w.current_wage for w in all_workers_flat if w.is_employed]

        # Generational employment counts
        gen_employment: dict[str, int] = {}
        for w in all_workers_flat:
            if w.is_employed:
                gen_employment[w.generation] = gen_employment.get(w.generation, 0) + 1

        # Knowledge-work median wages by occupation bucket
        from ai_econ_sim.config import KW_OCCUPATIONS
        kw_occupation_wages: dict[str, float] = {}
        for occ in KW_OCCUPATIONS:
            occ_wages = [
                w.current_wage for w in self.workers.get("knowledge_work", [])
                if w.is_employed and w.occupation == occ
            ]
            kw_occupation_wages[occ] = float(np.median(occ_wages)) if occ_wages else 0.0

        accts = self.accounting.compute(
            quarter=self.quarter,
            sector_labor_income=sector_labor_income,
            sector_capital_income=sector_capital_income,
            sector_employment=sector_employment,
            sector_wages=sector_wages,
            sector_price_indices=sector_price_indices,
            sector_ai_adoption=sector_ai_adoption,
            total_employed=total_employed,
            total_unemployed=total_unemployed,
            total_out_of_lf=total_olf,
            all_worker_incomes=all_incomes,
            sector_robotics_adoption=self._robotics_adoption,
            interface_realization=self.scenario.interface_realization.at(self.quarter),
            corporate_tax_rate=CORPORATE_TAX_RATE,
            payroll_tax_rate=PAYROLL_TAX_RATE,
            income_tax_rate=INCOME_TAX_RATE,
        )
        accts.capability_index = float(self.scenario.capability.capability_at(self.quarter).mean())
        accts.retirements = self._retirements_this_quarter
        accts.new_entrants = self._new_entrants_this_quarter
        accts.firm_exits = self._firm_exits_this_quarter
        accts.firm_entries = self._firm_entries_this_quarter
        accts.gen_employment = gen_employment
        accts.kw_occupation_wages = kw_occupation_wages
        accts.retraining_initiations = self._retraining_initiations
        accts.retraining_successes = self._retraining_successes
        accts.retraining_failures = self._retraining_failures

        # Policy accounting
        policy = self.scenario.policy
        windfall_tax = policy.ai_windfall_tax_rate * accts.ai_sector_capital_income
        accts.policy_windfall_tax_collected = windfall_tax
        accts.tax_revenue += windfall_tax
        accts.policy_ubi_disbursed = self._policy_ubi_disbursed
        accts.policy_retraining_subsidized = self._policy_retraining_subsidized

        return accts

    def _sector_median_wage(self, sector: str) -> float:
        wages = [w.current_wage for w in self.workers.get(sector, []) if w.is_employed]
        if wages:
            return float(np.median(wages))
        return WAGE_BASE_BY_SECTOR.get(sector, 50_000)

    # ------------------------------------------------------------------
    # Invariant checks
    # ------------------------------------------------------------------

    def _assert_invariants(self, accts: MacroAccounts) -> None:
        all_workers = [w for ws in self.workers.values() for w in ws]
        total = len(all_workers)
        emp = sum(1 for w in all_workers if w.is_employed)
        unemp = sum(1 for w in all_workers if w.is_in_labor_force and not w.is_employed)
        olf = sum(1 for w in all_workers if not w.is_in_labor_force)

        # Invariant 1: labor accounting
        assert emp + unemp + olf == total, f"Q{self.quarter}: labor accounting violation {emp}+{unemp}+{olf}!={total}"

        # Invariant 2: sectoral employment roughly matches firm workforce
        # Allow up to 20% discrepancy due to matching lag and retraining flows
        for s in SECTORS:
            firm_total = sum(f.n_workers for f in self.firms[s])
            worker_employed = sum(1 for w in self.workers.get(s, []) if w.is_employed)
            tolerance = max(10, round(0.20 * firm_total + 1))
            assert abs(firm_total - worker_employed) <= tolerance, \
                f"Q{self.quarter}: sector {s} employment mismatch: firms={firm_total} workers={worker_employed}"

        # Invariant 4: non-negativity
        assert accts.nominal_gdp >= 0, f"Q{self.quarter}: negative GDP"
        for s in SECTORS:
            assert accts.sector_employment.get(s, 0) >= 0
            assert accts.sector_median_wage.get(s, 0) >= 0

        # Invariant 8: no NaN/Inf
        import math
        assert math.isfinite(accts.nominal_gdp), f"Q{self.quarter}: non-finite GDP"
        assert math.isfinite(accts.gini), f"Q{self.quarter}: non-finite Gini"


def _assign_employer_ids(firms: list[Firm], n_workers: int) -> list[int]:
    """Assign workers to firms proportional to firm size."""
    employer_ids = []
    for firm in firms:
        employer_ids.extend([firm.id] * firm.n_workers)
    # Trim or pad to match n_workers
    if len(employer_ids) > n_workers:
        employer_ids = employer_ids[:n_workers]
    elif len(employer_ids) < n_workers:
        last_id = firms[-1].id if firms else 0
        employer_ids.extend([last_id] * (n_workers - len(employer_ids)))
    return employer_ids
