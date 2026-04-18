"""Worker agent: state and quarterly decision logic."""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from ai_econ_sim.agents.expectations import WorkerExpectations
from ai_econ_sim.config import (
    SKILL_MIN, SKILL_MAX,
    JOB_OFFER_PROBABILITY_BASE, OUTSIDE_OFFER_PROBABILITY, WAGE_ACCEPTANCE_THRESHOLD,
    RETRAINING_MIN_QUARTERS, RETRAINING_MAX_QUARTERS,
    RETRAINING_BASE_SUCCESS_RATE, RETRAINING_AGE_PENALTY, RETRAINING_SKILL_BONUS,
    LFP_EXIT_THRESHOLD_QUARTERS, LFP_EXIT_PROBABILITY, LFP_REENTRY_BASE,
    KW_OCCUPATIONS, EXPECTATION_HALFLIFE_QUARTERS,
)

# Generation labels assigned from birth year (2024 baseline)
GENERATION_LABELS = ("boomer", "genx", "millennial", "genz")

# Multiplier on retraining initiation probability per generation
# GenZ is most mobile; Boomers least likely to retrain
GENERATION_RETRAINING_MULT: dict[str, float] = {
    "boomer":     0.30,
    "genx":       0.65,
    "millennial": 1.10,
    "genz":       1.30,
}

# LFP exit probability multiplier per generation when long-term unemployed
# GenZ more likely to exit to gig/informal economy or schooling
GENERATION_LFP_EXIT_MULT: dict[str, float] = {
    "boomer":     1.20,  # early retirement if prospects bad
    "genx":       0.90,
    "millennial": 0.85,
    "genz":       1.40,  # more likely to NEET or side-hustle out
}


def _assign_generation(age: float) -> str:
    """Assign generation label based on age at simulation start (2024)."""
    if age >= 60:
        return "boomer"
    if age >= 44:
        return "genx"
    if age >= 28:
        return "millennial"
    return "genz"


@dataclass
class RetrainingState:
    target_sector: str
    target_occupation: str
    quarters_remaining: int
    success_probability: float


@dataclass
class Worker:
    id: int
    age: float                           # years (advances by 0.25 each quarter)
    sector: Optional[str]                # None if out_of_labor_force
    occupation: str                      # occupation bucket
    skill_level: int                     # 1..5
    current_wage: float                  # annual
    employer_id: Optional[int]          # None if unemployed
    generation: str = "millennial"       # boomer / genx / millennial / genz
    is_employed: bool = True
    is_in_labor_force: bool = True
    quarters_unemployed: int = 0

    retraining: Optional[RetrainingState] = field(default=None, repr=False)
    expectations: Optional[WorkerExpectations] = field(default=None, repr=False)

    def init_expectations(self, sector_employment: float, sector_median_wage: float) -> None:
        self.expectations = WorkerExpectations(sector_employment, sector_median_wage)

    # ------------------------------------------------------------------
    # Quarterly decision steps
    # ------------------------------------------------------------------

    def step_update_expectations(
        self,
        sector_employment: float,
        sector_median_wage: float,
    ) -> None:
        if self.expectations is None:
            self.init_expectations(sector_employment, sector_median_wage)
        else:
            self.expectations.update(sector_employment, sector_median_wage)

    def step_retraining_tick(self, rng: np.random.Generator) -> Optional[str]:
        """
        Advance retraining by one quarter.
        Returns new sector name if retraining completes, else None.
        """
        if self.retraining is None:
            return None
        self.retraining.quarters_remaining -= 1
        if self.retraining.quarters_remaining <= 0:
            success = rng.random() < self.retraining.success_probability
            target = self.retraining.target_sector
            target_occ = self.retraining.target_occupation
            self.retraining = None
            if success:
                self.sector = target
                self.occupation = target_occ
                self.is_employed = False
                self.employer_id = None
                self.quarters_unemployed = 0
                return target
        return None

    def step_retraining_decision(
        self,
        rng: np.random.Generator,
        declining_sectors: set[str],
        growing_sectors: list[str],
    ) -> bool:
        """
        Decide whether to enter retraining.
        Returns True if retraining is initiated.
        """
        if self.retraining is not None:
            return False
        if self.sector not in declining_sectors:
            return False
        if not growing_sectors:
            return False

        # Probability of initiating retraining (higher for younger, higher-skill workers)
        age_factor = max(0.1, 1.0 - 0.02 * max(0, self.age - 30))
        skill_factor = 0.3 + 0.1 * self.skill_level
        gen_mult = GENERATION_RETRAINING_MULT.get(self.generation, 1.0)
        prob = age_factor * skill_factor * 0.15 * gen_mult  # base 15% if conditions met

        if rng.random() > prob:
            return False

        target_sector = rng.choice(growing_sectors)
        target_occ = _sector_default_occupation(target_sector)
        duration = rng.integers(RETRAINING_MIN_QUARTERS, RETRAINING_MAX_QUARTERS + 1)
        success_prob = _retraining_success_prob(self.age, self.skill_level)

        self.retraining = RetrainingState(
            target_sector=target_sector,
            target_occupation=target_occ,
            quarters_remaining=int(duration),
            success_probability=success_prob,
        )
        self.is_employed = False
        self.employer_id = None
        return True

    def step_labor_force_participation(
        self,
        rng: np.random.Generator,
        sector_improving: bool,
    ) -> None:
        """Handle labor force exit and re-entry."""
        if self.retraining is not None:
            return

        if self.is_in_labor_force and not self.is_employed:
            self.quarters_unemployed += 1
            if self.quarters_unemployed >= LFP_EXIT_THRESHOLD_QUARTERS:
                gen_mult = GENERATION_LFP_EXIT_MULT.get(self.generation, 1.0)
                if rng.random() < LFP_EXIT_PROBABILITY * gen_mult:
                    self.is_in_labor_force = False
                    self.sector = None
                    self.quarters_unemployed = 0
        elif not self.is_in_labor_force and sector_improving:
            if rng.random() < LFP_REENTRY_BASE:
                self.is_in_labor_force = True
                self.quarters_unemployed = 0
        elif self.is_employed:
            self.quarters_unemployed = 0

    def accept_job(self, employer_id: int, sector: str, wage: float, occupation: str) -> None:
        self.employer_id = employer_id
        self.sector = sector
        self.current_wage = wage
        self.occupation = occupation
        self.is_employed = True
        self.is_in_labor_force = True
        self.quarters_unemployed = 0

    def lose_job(self) -> None:
        self.employer_id = None
        self.is_employed = False


def _retraining_success_prob(age: int, skill_level: int) -> float:
    prob = RETRAINING_BASE_SUCCESS_RATE
    if age > 40:
        prob -= RETRAINING_AGE_PENALTY * (age - 40)
    prob += RETRAINING_SKILL_BONUS * max(0, skill_level - 2)
    return max(0.1, min(0.95, prob))


def _sector_default_occupation(sector: str) -> str:
    mapping = {
        "ai_compute": "technical_specialist",
        "knowledge_work": "routine_analytical",
        "services": "service_worker",
        "goods": "production_worker",
        "infrastructure": "infrastructure_worker",
    }
    return mapping.get(sector, "service_worker")


def create_workers(
    sector: str,
    n_workers: int,
    sector_base_wage: float,
    rng: np.random.Generator,
    employer_ids: list[int] | None = None,
) -> list[Worker]:
    """Initialize a population of workers for a sector."""
    workers = []
    for i in range(n_workers):
        age = float(rng.integers(22, 65))
        skill = int(rng.integers(SKILL_MIN, SKILL_MAX + 1))

        # Occupation depends on sector
        occupation = _sample_occupation(sector, rng)

        # Wage: base + skill premium + noise
        from ai_econ_sim.config import WAGE_SKILL_PREMIUM, WAGE_IDIOSYNCRATIC_STD
        log_wage = (
            np.log(sector_base_wage)
            + WAGE_SKILL_PREMIUM * (skill - 1)
            + rng.normal(0, WAGE_IDIOSYNCRATIC_STD)
        )
        wage = float(np.exp(log_wage))

        emp_id = employer_ids[i] if employer_ids and i < len(employer_ids) else None

        w = Worker(
            id=i,
            age=age,
            sector=sector,
            occupation=occupation,
            skill_level=skill,
            current_wage=wage,
            employer_id=emp_id,
            generation=_assign_generation(age),
            is_employed=(emp_id is not None),
        )
        workers.append(w)

    return workers


def _sample_occupation(sector: str, rng: np.random.Generator) -> str:
    if sector == "knowledge_work":
        # DESIGN CHOICE: roughly equal distribution across KW occupations
        weights = [0.40, 0.20, 0.25, 0.15]  # routine_analytical has most workers
        return str(rng.choice(KW_OCCUPATIONS, p=weights))
    return _sector_default_occupation(sector)
