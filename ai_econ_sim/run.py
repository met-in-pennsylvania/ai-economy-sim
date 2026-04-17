"""CLI entry point: python -m ai_econ_sim.run --scenario <path>"""

import argparse
import logging
import sys
from pathlib import Path

from ai_econ_sim.config import LOG_FORMAT, LOG_DATEFMT, DEV_POPULATION_SCALE
from ai_econ_sim.scenarios.loader import load_scenario
from ai_econ_sim.model import Model
from ai_econ_sim.analysis.outputs import save_outputs, build_time_series
from ai_econ_sim.analysis.plots import plot_standard_set


def main() -> None:
    parser = argparse.ArgumentParser(description="AI-Economy Macro Simulation v0")
    parser.add_argument("--scenario", required=True, help="Path to scenario YAML")
    parser.add_argument("--output-dir", default="outputs", help="Directory for outputs")
    parser.add_argument("--dev", action="store_true", help="Use small population for fast iteration")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    parser.add_argument("--no-assertions", action="store_true", help="Disable invariant assertions")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    args = parser.parse_args()

    logging.basicConfig(format=LOG_FORMAT, datefmt=LOG_DATEFMT, level=getattr(logging, args.log_level))
    log = logging.getLogger(__name__)

    scenario_path = Path(args.scenario)
    if not scenario_path.exists():
        log.error("Scenario file not found: %s", scenario_path)
        sys.exit(1)

    log.info("Loading scenario: %s", scenario_path)
    scenario = load_scenario(scenario_path)

    scale = DEV_POPULATION_SCALE if args.dev else 1.0
    log.info("Population scale: %.1f (dev=%s)", scale, args.dev)

    model = Model(scenario, population_scale=scale, dev_assertions=not args.no_assertions)

    log.info("Running %d quarters...", scenario.horizon_quarters)
    history = model.run()

    run_name = scenario.name.replace(" ", "_").lower()
    paths = save_outputs(history, model, args.output_dir, run_name)
    log.info("Outputs saved: %s", {k: str(v) for k, v in paths.items()})

    if not args.no_plots:
        df = build_time_series(history)
        plot_paths = plot_standard_set(df, Path(args.output_dir), run_name)
        log.info("Plots saved: %d files", len(plot_paths))

    log.info("Done.")


if __name__ == "__main__":
    main()
