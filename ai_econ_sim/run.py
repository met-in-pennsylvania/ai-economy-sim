"""CLI entry point: python -m ai_econ_sim.run --scenario <path>"""

import argparse
import logging
import sys
from pathlib import Path

from ai_econ_sim.config import LOG_FORMAT, LOG_DATEFMT, DEV_POPULATION_SCALE
from ai_econ_sim.scenarios.loader import load_scenario
from ai_econ_sim.model import Model
from ai_econ_sim.analysis.outputs import save_outputs, build_time_series, build_run_report
from ai_econ_sim.analysis.plots import plot_standard_set


def main() -> None:
    parser = argparse.ArgumentParser(description="AI-Economy Macro Simulation v1")
    parser.add_argument("--scenario", required=True, help="Path to scenario YAML")
    parser.add_argument("--output-dir", default="outputs", help="Directory for outputs")
    parser.add_argument("--population-scale", type=float, default=1.0,
                        help="Scale factor for agent population (default: 1.0)")
    parser.add_argument("--dev", action="store_true",
                        help="Use small population for fast iteration (sets population-scale=0.1)")
    parser.add_argument("--mc-runs", type=int, default=1, metavar="N",
                        help="Number of Monte Carlo runs (default: 1 = single run)")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    parser.add_argument("--no-assertions", action="store_true", help="Disable invariant assertions")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    # Legacy alias kept for backward compatibility
    parser.add_argument("--monte-carlo", type=int, default=0, metavar="N",
                        help=argparse.SUPPRESS)
    parser.add_argument("--calibration-report", action="store_true",
                        help="Print BEA/BLS calibration comparison after single run.")
    args = parser.parse_args()

    logging.basicConfig(format=LOG_FORMAT, datefmt=LOG_DATEFMT,
                        level=getattr(logging, args.log_level), stream=sys.stderr)
    log = logging.getLogger(__name__)

    scenario_path = Path(args.scenario)
    if not scenario_path.exists():
        log.error("Scenario file not found: %s", scenario_path)
        sys.exit(1)

    log.info("Loading scenario: %s", scenario_path)
    scenario = load_scenario(scenario_path)

    # --dev overrides --population-scale
    if args.dev:
        scale = DEV_POPULATION_SCALE
    else:
        scale = args.population_scale
    log.info("Population scale: %.3f (dev=%s)", scale, args.dev)

    run_name = scenario.name.replace(" ", "_").lower()

    # Resolve MC run count: prefer --mc-runs, fall back to legacy --monte-carlo
    mc_runs = args.mc_runs if args.mc_runs > 1 else max(args.mc_runs, args.monte_carlo)

    if mc_runs > 1:
        _run_mc(scenario, args, scale, run_name, mc_runs, log)
    else:
        _run_single(scenario, args, scale, run_name, log)

    log.info("Done.")


def _run_single(scenario, args, scale, run_name, log) -> None:
    model = Model(scenario, population_scale=scale, dev_assertions=not args.no_assertions)
    log.info("Running %d quarters...", scenario.horizon_quarters)
    history = model.run()

    paths = save_outputs(history, model, args.output_dir, run_name)
    log.info("Outputs saved: %s", {k: str(v) for k, v in paths.items()})

    if getattr(args, "calibration_report", False):
        from ai_econ_sim.calibration import calibration_report
        print(calibration_report(history))

    # Print the run report to stdout
    report = build_run_report(history, model)
    print(report)

    if not args.no_plots:
        df = build_time_series(history)
        plot_paths = plot_standard_set(df, Path(args.output_dir), run_name)
        log.info("Plots saved: %d files", len(plot_paths))


def _run_mc(scenario, args, scale, run_name, mc_runs, log) -> None:
    from ai_econ_sim.monte_carlo import run_monte_carlo, save_mc_outputs
    from ai_econ_sim.analysis.plots import plot_mc_bands

    log.info("Running Monte Carlo: %d runs x %d quarters (scale=%.3f)...",
             mc_runs, scenario.horizon_quarters, scale)

    mc = run_monte_carlo(scenario, n_runs=mc_runs, population_scale=scale)

    mc_run_name = f"{run_name}_mc{mc_runs}"
    paths = save_mc_outputs(mc, args.output_dir, mc_run_name)
    log.info("MC outputs saved: %s", paths)

    if not args.no_plots:
        plot_paths = plot_mc_bands(mc, Path(args.output_dir), mc_run_name)
        log.info("MC plots saved: %d files", len(plot_paths))

    # Print quick summary to stdout
    stats = mc.summary_stats()
    key_cols = ["nominal_gdp", "unemployment_rate", "labor_share", "gini"]
    print(f"\nMonte Carlo Summary ({mc_runs} runs):")
    for col in key_cols:
        if col in stats.index:
            row = stats.loc[col]
            print(f"  {col}: mean={row['mean']:.4f} std={row['std']:.4f}")


if __name__ == "__main__":
    main()
