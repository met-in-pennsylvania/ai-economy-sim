"""Load scenario YAML into a typed Scenario object."""

from __future__ import annotations
import yaml
from dataclasses import dataclass, field
from pathlib import Path

from ai_econ_sim.capability.trajectory import CapabilityTrajectory, _interpolate
from ai_econ_sim.config import SECTORS

_DEFAULT_CONSUMER_DISCOUNT = {
    "ai_compute": 0.00,
    "knowledge_work": 0.08,
    "services": 0.18,
    "goods": 0.03,
    "infrastructure": 0.01,
}

_DEFAULT_WORKFORCE_RESISTANCE = {
    "ai_compute": 0.02,
    "knowledge_work": 0.15,
    "services": 0.20,
    "goods": 0.08,
    "infrastructure": 0.10,
}

_DEFAULT_INTERFACE_TRAJECTORY = [
    {"quarter": 0,  "value": 0.35},   # today: ~35% realization
    {"quarter": 20, "value": 0.60},   # better tooling in 5 years
    {"quarter": 40, "value": 0.80},   # mature interfaces in 10 years
]


@dataclass
class ComputeParams:
    cost_per_unit_trajectory: list[dict]
    chip_supply_growth_annual: float = 0.15


@dataclass
class EnergyParams:
    price_trajectory: list[dict]


@dataclass
class DemographicsParams:
    labor_force_growth_annual: float = 0.005
    retirement_rate_annual: float = 0.02


@dataclass
class SentimentParams:
    """
    Human-factors parameters capturing how people actually respond to AI.

    consumer_ai_discount: per-sector [0,1]. Damps sector demand growth
        proportionally to how AI-saturated the sector becomes. Captures
        customers preferring human delivery (bedside manner, restaurant
        experience, personal services).

    workforce_resistance: per-sector [0,1]. Reduces effective AI adoption
        regardless of firm-level adoption. Cultural/generational friction —
        workers who don't want to use AI tools, route around them, or comply
        only nominally. Distinct from interface friction: this doesn't improve
        automatically as tooling matures.
    """
    consumer_ai_discount: dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_CONSUMER_DISCOUNT)
    )
    workforce_resistance: dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_WORKFORCE_RESISTANCE)
    )


@dataclass
class InterfaceRealization:
    """
    Trajectory of how much of theoretical AI productivity actually materializes
    due to interface maturity and workflow integration quality.

    Captures the gap between "AI is capable" and "workers can fluidly recruit it
    for tasks." Even willing workers face overhead: knowing how to prompt,
    context-switching, verifying outputs, recovering from errors. Starts low
    (~0.35) and improves as tooling matures. Configured as keyframes like the
    capability trajectory.

    Does NOT apply to robotics (physical task automation has different
    interaction ergonomics than cognitive AI tools).
    """
    _keyframes: list[tuple[int, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self._keyframes:
            self._keyframes = [(f["quarter"], f["value"]) for f in _DEFAULT_INTERFACE_TRAJECTORY]

    def at(self, quarter: int) -> float:
        return float(_interpolate(self._keyframes, quarter))


@dataclass
class RoboticsParams:
    """Parameters governing AI-powered physical robots (e.g. Optimus-class)."""
    deployment_start_quarter: int = 999      # quarter when robots become available (default: never)
    diffusion_rate: float = 0.04             # robotics adoption gained per quarter after deployment
    max_labor_displacement: float = 0.35     # ceiling on how much labour robots can displace per sector
    affected_sectors: list[str] = field(default_factory=lambda: ["services", "goods", "infrastructure"])


@dataclass
class PolicyParams:
    """
    Government policy levers that can be toggled independently of the scenario.

    ubi_annual: annual per-capita UBI payment in dollars (0 = disabled).
        Reduces LFP exit and slightly boosts re-entry by providing an income
        floor. Does not directly add to worker wages; tracked separately.

    retraining_subsidy_rate: fraction in [0, 1]. Multiplies the extra
        probability of initiating retraining for workers who would otherwise
        not have done so. 0.5 = 50% subsidy → ~50% more retraining initiations
        at the margin.

    ai_windfall_tax_rate: fraction in [0, 0.5] of AI-sector capital income
        collected as a windfall levy each quarter. Tracked in MacroAccounts;
        does not currently flow back into the economy (fiscal use is out of scope).
    """
    ubi_annual: float = 0.0
    retraining_subsidy_rate: float = 0.0
    ai_windfall_tax_rate: float = 0.0


@dataclass
class InitialConditions:
    gdp_shares: dict[str, float] = field(default_factory=dict)
    employment_shares: dict[str, float] = field(default_factory=dict)
    initial_wages: dict[str, float] = field(default_factory=dict)
    nominal_gdp: float = 25_000_000_000_000.0  # ~$25T stylized US GDP


@dataclass
class Scenario:
    name: str
    horizon_quarters: int
    capability: CapabilityTrajectory
    regulation: dict[str, float]         # sector -> friction [0,1]
    compute: ComputeParams
    energy: EnergyParams
    oss_frontier_gap: float
    demographics: DemographicsParams
    initial_conditions: InitialConditions
    robotics: RoboticsParams = field(default_factory=RoboticsParams)
    sentiment: SentimentParams = field(default_factory=SentimentParams)
    interface_realization: InterfaceRealization = field(default_factory=InterfaceRealization)
    policy: PolicyParams = field(default_factory=PolicyParams)
    seed: int = 42

    def regulation_friction(self, sector: str) -> float:
        return self.regulation.get(sector, 0.0)


def load_scenario(path: str | Path) -> Scenario:
    """Parse a scenario YAML file and return a Scenario object."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    traj = raw.get("capability_trajectory", {})
    floors = raw.get("reliability_floors", {})
    capability = CapabilityTrajectory(traj, floors)

    compute_raw = raw.get("compute", {})
    compute = ComputeParams(
        cost_per_unit_trajectory=compute_raw.get("cost_per_unit_trajectory", [{"quarter": 0, "value": 1.0}]),
        chip_supply_growth_annual=compute_raw.get("chip_supply_growth_annual", 0.15),
    )

    energy_raw = raw.get("energy", {})
    energy = EnergyParams(
        price_trajectory=energy_raw.get("price_trajectory", [{"quarter": 0, "value": 1.0}]),
    )

    demo_raw = raw.get("demographics", {})
    demographics = DemographicsParams(
        labor_force_growth_annual=demo_raw.get("labor_force_growth_annual", 0.005),
        retirement_rate_annual=demo_raw.get("retirement_rate_annual", 0.02),
    )

    ic_raw = raw.get("initial_conditions", {})
    initial_conditions = InitialConditions(
        gdp_shares=ic_raw.get("gdp_shares", {}),
        employment_shares=ic_raw.get("employment_shares", {}),
        initial_wages=ic_raw.get("initial_wages", {}),
        nominal_gdp=ic_raw.get("nominal_gdp", 25_000_000_000_000.0),
    )

    # Fill missing sector shares with defaults
    default_gdp_shares = {"ai_compute": 0.02, "knowledge_work": 0.25, "services": 0.35, "goods": 0.25, "infrastructure": 0.13}
    default_emp_shares = {"ai_compute": 0.02, "knowledge_work": 0.22, "services": 0.40, "goods": 0.25, "infrastructure": 0.11}
    for s in SECTORS:
        initial_conditions.gdp_shares.setdefault(s, default_gdp_shares[s])
        initial_conditions.employment_shares.setdefault(s, default_emp_shares[s])

    rob_raw = raw.get("robotics", {})
    robotics = RoboticsParams(
        deployment_start_quarter=rob_raw.get("deployment_start_quarter", 999),
        diffusion_rate=rob_raw.get("diffusion_rate", 0.04),
        max_labor_displacement=rob_raw.get("max_labor_displacement", 0.35),
        affected_sectors=rob_raw.get("affected_sectors", ["services", "goods", "infrastructure"]),
    )

    sent_raw = raw.get("sentiment", {})
    consumer_discount = dict(_DEFAULT_CONSUMER_DISCOUNT)
    consumer_discount.update(sent_raw.get("consumer_ai_discount", {}))
    workforce_resistance = dict(_DEFAULT_WORKFORCE_RESISTANCE)
    workforce_resistance.update(sent_raw.get("workforce_resistance", {}))
    sentiment = SentimentParams(
        consumer_ai_discount=consumer_discount,
        workforce_resistance=workforce_resistance,
    )

    iface_raw = raw.get("interface_realization", {})
    iface_frames_raw = iface_raw.get("trajectory", _DEFAULT_INTERFACE_TRAJECTORY)
    iface_frames = [(f["quarter"], f["value"]) for f in iface_frames_raw]
    interface_realization = InterfaceRealization(_keyframes=iface_frames)

    policy_raw = raw.get("policy", {})
    policy = PolicyParams(
        ubi_annual=policy_raw.get("ubi_annual", 0.0),
        retraining_subsidy_rate=policy_raw.get("retraining_subsidy_rate", 0.0),
        ai_windfall_tax_rate=policy_raw.get("ai_windfall_tax_rate", 0.0),
    )

    return Scenario(
        name=raw.get("scenario_name", "unnamed"),
        horizon_quarters=raw.get("horizon_quarters", 40),
        capability=capability,
        regulation=raw.get("regulation", {}),
        compute=compute,
        energy=energy,
        oss_frontier_gap=raw.get("oss_frontier_gap", 1.0),
        demographics=demographics,
        initial_conditions=initial_conditions,
        robotics=robotics,
        sentiment=sentiment,
        interface_realization=interface_realization,
        policy=policy,
        seed=raw.get("seed", 42),
    )
