"""Monte Carlo runner: sweep seeds (and optionally a parameter) across N runs."""

from __future__ import annotations
import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from ai_econ_sim.scenarios.loader import Scenario
from ai_econ_sim.model import Model
from ai_econ_sim.analysis.outputs import build_time_series

log = logging.getLogger(__name__)

# Quantiles stored for every numeric column
QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]


@dataclass
class MCResults:
    """Aggregated results from a Monte Carlo run."""
    scenario_name: str
    n_runs: int
    runs: list[pd.DataFrame]          # one df per seed, raw time series

    # Aggregated: MultiIndex df indexed by (quarter, quantile)
    bands: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Convenience accessors
    def median(self) -> pd.DataFrame:
        return self.bands.xs(0.50, level="quantile")

    def band(self, lo: float = 0.10, hi: float = 0.90) -> tuple[pd.DataFrame, pd.DataFrame]:
        return self.bands.xs(lo, level="quantile"), self.bands.xs(hi, level="quantile")

    def summary_stats(self) -> pd.DataFrame:
        """Final-quarter stats across runs (mean ± std for key indicators)."""
        finals = pd.concat([df.iloc[[-1]] for df in self.runs], ignore_index=True)
        return finals.describe().T[["mean", "std", "min", "max"]]


@dataclass
class SweepResults:
    """Results from a parameter sweep: one MCResults per parameter value."""
    param_name: str
    param_values: list[Any]
    mc_results: list[MCResults]       # same order as param_values

    def median_series(self, column: str) -> pd.DataFrame:
        """Return DataFrame with one column per param_value, rows = quarters."""
        frames = {}
        for val, mc in zip(self.param_values, self.mc_results):
            frames[val] = mc.median()[column]
        return pd.DataFrame(frames)


def run_monte_carlo(
    scenario: Scenario,
    n_runs: int = 20,
    population_scale: float = 0.1,
    seeds: list[int] | None = None,
    param_override: Callable[[Scenario, int], None] | None = None,
) -> MCResults:
    """
    Run the model N times with different seeds, aggregate results.

    param_override: optional callable(scenario, run_index) for per-run
    parameter variation beyond seed. Used by run_parameter_sweep().
    """
    if seeds is None:
        seeds = [scenario.seed + i for i in range(n_runs)]

    runs: list[pd.DataFrame] = []
    for i, seed in enumerate(seeds):
        s = copy.deepcopy(scenario)
        s.seed = seed
        if param_override:
            param_override(s, i)
        model = Model(s, population_scale=population_scale, dev_assertions=False)
        history = model.run()
        df = build_time_series(history)
        df["run"] = i
        df["seed"] = seed
        runs.append(df)
        log.debug("MC run %d/%d complete (seed=%d)", i + 1, n_runs, seed)

    log.info("MC complete: %d runs, scenario=%s", n_runs, scenario.name)
    bands = _compute_bands(runs)
    return MCResults(scenario_name=scenario.name, n_runs=n_runs, runs=runs, bands=bands)


def run_parameter_sweep(
    scenario: Scenario,
    param_name: str,
    param_values: list[Any],
    setter: Callable[[Scenario, Any], None],
    n_runs_per_value: int = 10,
    population_scale: float = 0.1,
) -> SweepResults:
    """
    Run MC for each value in param_values, varying param via setter.

    setter(scenario, value) should mutate the scenario in-place.

    Example:
        run_parameter_sweep(
            scenario, "robotics_start",
            [8, 16, 24, 999],
            setter=lambda s, v: setattr(s.robotics, "deployment_start_quarter", v),
        )
    """
    mc_results = []
    for val in param_values:
        s = copy.deepcopy(scenario)
        setter(s, val)
        log.info("Sweep: %s=%s", param_name, val)
        mc = run_monte_carlo(s, n_runs=n_runs_per_value, population_scale=population_scale)
        mc_results.append(mc)

    return SweepResults(
        param_name=param_name,
        param_values=list(param_values),
        mc_results=mc_results,
    )


def save_mc_outputs(
    mc: MCResults,
    output_dir: str,
    run_name: str = "mc",
) -> dict[str, str]:
    """Save MC bands CSV and per-run CSV. Returns dict of paths."""
    from pathlib import Path
    import json

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    bands_path = out / f"{run_name}_mc_bands.csv"
    mc.bands.to_csv(bands_path)

    stats_path = out / f"{run_name}_mc_summary.json"
    stats = mc.summary_stats()
    stats.to_json(stats_path, indent=2)

    return {"bands": str(bands_path), "summary": str(stats_path)}


def _compute_bands(runs: list[pd.DataFrame]) -> pd.DataFrame:
    """Compute quantile bands across runs, indexed by (quarter, quantile)."""
    numeric_cols = [
        c for c in runs[0].columns
        if c not in ("quarter", "run", "seed")
        and pd.api.types.is_numeric_dtype(runs[0][c])
    ]
    combined = pd.concat(runs, ignore_index=True)
    result = (
        combined
        .groupby("quarter")[numeric_cols]
        .quantile(QUANTILES)
    )
    result.index.names = ["quarter", "quantile"]
    return result
