# AI-Economy Macro Simulation v0

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ai-economy-sim-xsakje3xbmb8l9sfjpkiq6.streamlit.app/)

A stylized agent-based macroeconomic simulation of a US-like economy undergoing AI-driven transformation over a ~10-year horizon. Built as a **thinking tool**, not a forecasting model.

> **v0 is a prototype.** Absolute numbers are meaningless. What matters is relative dynamics across scenarios.

## Quickstart

```bash
pip install -e .

# Run reference scenario, produce CSVs and plots in outputs/
python -m ai_econ_sim.run --scenario scenarios/fragmented.yaml

# Fast dev run (small population, ~seconds)
python -m ai_econ_sim.run --scenario scenarios/fragmented.yaml --dev

# Launch interactive UI
streamlit run app/streamlit_app.py

# Run tests
pytest
```

## Architecture

Three layers composed together:

1. **Agent layer** (~1000 firms, ~10,000 workers) with adaptive expectations, quarterly decisions
2. **Structural layer** (5-sector I-O model with intermediate-input linkages)
3. **Macro accounting layer** (GDP, factor shares, prices, tax revenue, inequality)

Driven by a **scenario YAML** specifying AI capability trajectories, regulation, compute costs, energy prices, and OSS proximity to frontier.

### Sectors

| Sector | AI Exposure |
|--------|-------------|
| AI & Compute | Self-producing |
| Knowledge Work | Task-level (10 task categories) |
| Services | Scalar (low spillover) |
| Goods | Scalar (moderate) |
| Infrastructure | Scalar (very low) |

### Scenario inputs (YAML-configurable)

- Per-task AI capability trajectory (keyframe interpolation)
- Reliability floors per task (gates deployment)
- Compute cost and chip supply growth
- Energy price trajectory
- Regulatory friction per sector [0=none, 1=blocking]
- OSS frontier gap (affects capital concentration)

## Scenarios

Five scenarios are included:

| File | Description |
|------|-------------|
| `fragmented.yaml` | **Reference.** Capable but reliability-gated. OSS close to frontier. |
| `plateau.yaml` | Capability plateaus at quarter 10. Limited deployment gains. |
| `continued_exponential.yaml` | Exponential growth across all tasks. Low regulation. |
| `broadly_agentic.yaml` | Coordination/orchestration crosses reliability floors. High regulation response. |
| `transformative.yaml` | Near-AGI: most tasks crossed within 20 quarters. OSS lags far behind. |

## Outputs

Per run:
- `outputs/<name>_timeseries.csv` — 40 quarterly rows × ~40 columns
- `outputs/<name>_summary.json` — run-level summary statistics
- `outputs/<name>_*.png` — matplotlib plots (employment, wages, shares, adoption, labor market)

## Known Limitations

- **Not calibrated.** All parameters are stylized. Do not interpret absolute GDP, wage, or employment numbers as predictions.
- **Closed economy.** No trade or external sector.
- **No financial sector.** No interest rates, stock prices, or credit.
- **No government policy response.** Tax rates are fixed; no UBI, retraining subsidies, or fiscal stabilizers.
- **No spatial dimension.** No regional variation.
- **Fixed firm population.** No entry/exit in v0.
- **Simplified expectations.** EMA/trend extrapolation, not rational expectations.
- **5 sectors only.** Professional services not broken into legal/software/consulting/etc.

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run with small population for fast iteration
python -m ai_econ_sim.run --scenario scenarios/fragmented.yaml --dev --log-level DEBUG

# Run tests
pytest -v

# Run specific test file
pytest tests/test_integration.py -v
```

### Design choices flagged in code

Search for `# DESIGN CHOICE:` in the source for places where the implementation made a judgment call not fully specified in the project plan. These are v1 refinement targets.

## Project structure

```
ai_econ_sim/         Python package
  model.py           Main Model class, quarterly step orchestration
  sectors.py         Sector definitions and I-O matrix
  config.py          Default constants
  agents/            Firm and worker agents + expectations
  capability/        Task categories and capability trajectory
  macro/             Accounting and government modules
  scenarios/         YAML loader and reference scenario
  analysis/          Time series output and plots
  run.py             CLI entry point
app/
  streamlit_app.py   Interactive Streamlit UI
scenarios/           User-editable scenario YAMLs
tests/               pytest test suite
outputs/             Generated outputs (gitignored)
```

## Extending

- **New scenario:** Copy any YAML from `scenarios/`, modify capability trajectories and regulation, run.
- **New sector:** Add to `config.py`, `sectors.py`, and update I-O matrix.
- **Richer agent behavior:** Each agent module has clear step methods; extend decision logic there.
- **Calibration (v1):** Replace stylized initial conditions in scenario YAML with BEA/BLS data.
