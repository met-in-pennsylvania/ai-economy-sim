# AI-Economy Macro Simulation — v0 Project Plan

**Status:** Scoped and ready to build.
**Implementer:** Handing off to Claude Code for build + test.
**Purpose of v0:** Working stylized prototype demonstrating the full architecture. Not calibrated for forecasting — it's a thinking tool whose structure is the value. Treat it as a prototype, not a product.

---

## 1. Project goal

Build a US-focused, ~10-year-horizon macroeconomic simulation of an economy undergoing AI-driven transformation. The model is intended as a reusable engine that takes scenario inputs (AI capability trajectory, regulation, compute/energy, etc.) and produces coherent time series of employment, wages, sectoral output, capital-labor share, and other macro aggregates. It is built to be exploratory (illuminates mechanisms) rather than predictive (no claim to numerical accuracy).

v0 is the **stylized prototype** — aggressive simplifications, full architecture present. v1 will add calibration, sector detail, and richer agent heterogeneity. Don't build for v1 yet; build v0 with clean interfaces so v1 can extend rather than rewrite.

---

## 2. Architecture overview

Three layers composed together:

1. **Structural backbone (I-O style).** 5 sectors with intermediate-input linkages. Defines sector interdependencies and final demand composition. In v0, simplified input-output coefficients rather than full BEA data.

2. **Agent-based micro layer.** ~1000 firms and ~10,000 workers, heterogeneous by sector/size/skill. Firms make quarterly hiring, pricing, and AI-adoption decisions. Workers search for jobs, retrain, switch sectors. Both populations use **adaptive expectations** — extrapolate recent trends rather than perfect foresight.

3. **Macro accounting layer (stock-and-flow).** Aggregates agent behavior into GDP, sectoral value-added, labor/capital share, price indices, government revenue, household consumption. Ensures accounting consistency.

Driven by a **scenario layer** (external inputs) and produces outputs via an **analysis/visualization layer**.

```
┌─────────────────────────────────────────────────────┐
│  Scenario inputs (YAML config)                      │
│  - AI capability trajectory (per task category)     │
│  - Reliability floor                                │
│  - Compute & energy costs                           │
│  - Regulatory stance per sector                     │
│  - OSS proximity to frontier                        │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│  Model engine (quarterly time step)                 │
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │ Agent layer (firms, workers)                │   │
│  │ - adaptive expectations                     │   │
│  │ - task-level AI exposure (knowledge work)   │   │
│  │ - scalar productivity (other sectors)       │   │
│  └─────────────────────────────────────────────┘   │
│                      │                              │
│                      ▼                              │
│  ┌─────────────────────────────────────────────┐   │
│  │ Structural layer (I-O, sector linkages)     │   │
│  └─────────────────────────────────────────────┘   │
│                      │                              │
│                      ▼                              │
│  ┌─────────────────────────────────────────────┐   │
│  │ Macro accounting (GDP, shares, prices, tax) │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│  Outputs                                            │
│  - Time series (CSV, JSON)                          │
│  - Matplotlib/Plotly plots                          │
│  - Streamlit interactive UI                         │
└─────────────────────────────────────────────────────┘
```

---

## 3. Sectors (5 in v0)

| # | Sector | Role in model | AI exposure |
|---|--------|---------------|-------------|
| 1 | AI & Compute | Produces AI capability. Consumes chips, energy, specialist labor. Owns most AI capital stock. | Self-producing |
| 2 | Knowledge Work | Legal, consulting, software, finance, media aggregated. High AI substitutability. | Task-level model |
| 3 | Services | Healthcare, education, retail, food, personal services. Trust/in-person heavy. | Scalar (low) |
| 4 | Goods | Manufacturing, construction, agriculture, logistics. | Scalar (moderate) |
| 5 | Infrastructure | Utilities, transport, real estate, government. Slow-moving, regulated. | Scalar (very low) |

Each sector has: a representative production function, a workforce composition (distribution across occupations/skill levels), a capital stock, and I-O coefficients specifying intermediate inputs from other sectors.

---

## 4. Agents

### Firms (~1000 total)

Distributed across sectors roughly proportional to real US sector shares, with **power-law size distribution** within each sector (many small, few large). Sizes: micro (1-5), small (6-50), medium (51-500), large (500+).

**Firm state:**
- `sector`, `size_tier`, `id`
- `capital_stock` (generic capital + AI capital)
- `workforce` (dict: occupation → count, skill levels)
- `ai_adoption_level` ∈ [0, 1] (fraction of potentially-automatable tasks actually automated)
- `price_level` (sector-relative)
- `revenue`, `costs`, `profit` (flow variables, updated quarterly)
- `expectation_state` (rolling trend estimates for key variables)

**Quarterly decisions:**
1. **Update expectations.** Compute 4-quarter trends in: sector demand, wages for each occupation employed, AI capability applicable to this sector, competitor adoption levels.
2. **AI adoption decision.** Probabilistic function of: cost-benefit of further adoption given current capability × reliability, sector regulatory friction, firm size (larger firms slower), recent trend in competitor adoption. Output: delta to `ai_adoption_level`.
3. **Hiring/firing decision.** Given expected demand and current AI-augmented productivity, compute desired workforce. Adjust toward desired, with friction (can't instantly hire/fire). Wage offers set by sector wage + skill premium + small idiosyncratic noise.
4. **Pricing.** Markup over marginal cost. Markup adapts slowly to demand pressure.

### Workers (~10,000 total)

Distributed across sectors by current employment shares. Each worker has:

**Worker state:**
- `id`, `age`, `sector` (or `unemployed`/`out_of_labor_force`), `occupation_bucket`, `skill_level` ∈ {1..5}
- `current_wage`, `employer_id` (if employed)
- `retraining_state` (None, or `{target_sector, target_occupation, quarters_remaining}`)
- `expectation_state` (sector employment/wage trends)

**Quarterly decisions:**
1. **Update expectations.** Observe sector-level employment changes and median wages over last 4 quarters.
2. **Job search / acceptance.** Unemployed workers receive offers from hiring firms (probability weighted by skill match). Employed workers occasionally receive outside offers; accept if wage gain exceeds threshold.
3. **Retraining decision.** If current sector showing sustained decline in both employment and real wages, probabilistically enter retraining. Retraining takes 4-8 quarters, succeeds with probability < 1 (age/skill dependent), and transitions worker to new sector/occupation.
4. **Labor force participation.** If unemployed for extended period with poor prospects, probabilistically exit labor force. Re-entry possible when sector conditions improve.

**Occupation buckets within Knowledge Work** (for task-level modeling):
- `routine_analytical` (e.g., bookkeeping, basic legal review, junior coding, data entry)
- `creative_synthesis` (e.g., design, writing, senior consulting)
- `relational` (e.g., client relationship, sales, management)
- `technical_specialist` (e.g., senior engineering, research, domain expertise)

Each occupation has a task mix (weights over the 10 task categories below). AI exposure for an occupation = weighted sum of task exposures.

---

## 5. Task-level capability model (Knowledge Work only)

Define 10 task categories:

1. Routine information processing
2. Pattern recognition / classification
3. Structured writing / communication
4. Creative synthesis
5. Quantitative analysis
6. Judgment under uncertainty
7. Interpersonal relationship / trust building
8. Physical presence / embodied action
9. Novel problem solving
10. Coordination / orchestration

**Capability representation:** At each time t, AI capability is a vector `c(t) ∈ [0, 1]^10`, where `c_i(t)` is AI's capability level at task i. Each task also has a **required reliability threshold** `r_i` — AI only gets deployed for that task if `c_i(t) ≥ r_i`. The reliability threshold is a scenario parameter (higher for high-stakes tasks like judgment, lower for routine processing).

**Occupation exposure:** For an occupation with task weights `w`, its AI exposure at time t is:

```
exposure(t) = sum_i w_i * I(c_i(t) ≥ r_i) * min(1, c_i(t) / 1.0)
```

(The indicator gates on reliability; the continuous term captures the fraction of the task AI can actually handle.)

**Firm effect:** A firm's effective AI-augmented labor productivity multiplier for an occupation = 1 + adoption_level × exposure(t) × capability_factor. This feeds into hiring decisions: if AI can do X% of an occupation's tasks reliably, the firm can get the same output from (1 - X%) as many workers (modulo friction).

---

## 6. Scalar AI augmentation (non-knowledge-work sectors)

For Services, Goods, Infrastructure: AI increases labor productivity by a sector-specific scalar that grows under the scenario:

```
productivity_growth_rate_s(t) = base_growth_s + ai_spillover_s * capability_index(t)
```

Where `capability_index(t)` is a weighted scalar summary of the capability vector (reflecting the general level of AI usefulness for non-knowledge tasks). `ai_spillover_s` is low for Infrastructure, moderate for Goods, slightly higher for Services.

---

## 7. AI & Compute sector

Treated as a distinct sector with these properties:

- **Output:** "AI capability" delivered as a service to other sectors. Priced per "unit of effective work" (conceptually like cloud compute hours, but bundled with the model capability).
- **Inputs:** Specialist labor (drawn from a small high-skill pool), compute capital (a distinct capital type with fast depreciation, ~25% annually), energy (exogenous price), and generic capital.
- **Revenue:** Flows from all other sectors proportional to their AI adoption level × their output.
- **Capital concentration:** Ownership of AI-sector capital is more concentrated than other sectors. In v0, track this via a `capital_ownership_concentration` parameter that affects how capital income distributes in the accounting layer.
- **Capacity constraint:** AI-sector output is constrained by compute stock. Compute stock grows via investment, but investment is bounded by exogenous chip supply parameter.

This is where the capital-labor share story plays out: as AI substitutes for labor elsewhere, surplus flows to the AI sector, accumulating in capital accounts.

---

## 8. Scenario inputs (YAML-driven)

A scenario is a YAML file specifying time-varying and static parameters:

```yaml
scenario_name: "fragmented_capable_unreliable"
horizon_quarters: 40

capability_trajectory:
  # Per-task capability c_i(t), specified as keyframes, linear interpolation
  routine_information_processing:
    - {quarter: 0, value: 0.85}
    - {quarter: 40, value: 0.98}
  judgment_under_uncertainty:
    - {quarter: 0, value: 0.35}
    - {quarter: 40, value: 0.55}
  # ... (all 10 tasks)

reliability_floors:
  # r_i — how high c_i must be for deployment
  routine_information_processing: 0.75
  judgment_under_uncertainty: 0.90
  # ... (all 10 tasks)

compute:
  cost_per_unit_trajectory:
    - {quarter: 0, value: 1.0}
    - {quarter: 40, value: 0.3}
  chip_supply_growth_annual: 0.15

energy:
  price_trajectory:
    - {quarter: 0, value: 1.0}
    - {quarter: 40, value: 1.2}

regulation:
  # Per-sector friction to AI adoption, 0 = no friction, 1 = near-blocking
  ai_compute: 0.10
  knowledge_work: 0.20
  services: 0.50   # healthcare/education slow
  goods: 0.15
  infrastructure: 0.70   # regulated, slow

oss_frontier_gap: 1.5
  # Generations behind frontier. 0 = OSS matches frontier,
  # affects pricing power of AI sector and capital concentration.

demographics:
  labor_force_growth_annual: 0.005
  retirement_rate_annual: 0.02

initial_conditions:
  # Sectoral shares of GDP, employment, initial wages, etc.
  # Stylized in v0, not calibrated.
  gdp_shares: {ai_compute: 0.02, knowledge_work: 0.25, services: 0.35, goods: 0.25, infrastructure: 0.13}
  # ... etc
```

**v0 reference scenario: "Fragmented — capable but unreliable"**

- Capability grows (especially routine tasks) but reliability floors stay high for judgment/interpersonal tasks
- OSS gap is small (~0.5 generations) — AI capability is diffuse, not monopolized
- Regulation moderate in knowledge work, high in services/infrastructure
- Compute costs fall moderately; energy prices rise slightly
- Deployment lags capability because reliability thresholds gate it

This scenario should show: partial knowledge-work compression, substantial productivity gains without mass displacement, limited capital concentration (because OSS diffuses), deployment bottlenecks.

---

## 9. Output schema

Per simulation run, produce:

**Time series (40 quarterly points):**
- Sectoral employment (5 columns)
- Sectoral value-added (5 columns)
- Sectoral median wages (5 columns)
- Median wages by occupation bucket within knowledge work (4 columns)
- Aggregate labor share of income
- Aggregate capital share of income (split: AI-sector vs other)
- Aggregate GDP (nominal + real)
- Price index per sector
- Labor force participation rate
- Unemployment rate
- Gini coefficient of income (quarterly)
- Government tax revenue
- AI adoption level per sector
- AI capability index (composite)

**Run-level summaries:**
- Firm size distribution at end vs. start (per sector)
- Retraining attempts, successes, failures (aggregate counts)
- Workers who exited labor force (count)
- Peak and trough of key indicators

**Formats:**
- Canonical: CSV per time series, JSON for run metadata
- Plots: matplotlib PNGs for quick inspection, plotly HTML for interactive
- Streamlit UI: dynamic plots driven by parameter sliders

---

## 10. Repository structure

```
ai_economy_sim/
├── README.md                  # Overview, quickstart, caveats about v0
├── PROJECT_PLAN.md            # This document
├── pyproject.toml             # Package config, dependencies
├── requirements.txt           # Alternative dependency list
├── LICENSE
│
├── ai_econ_sim/
│   ├── __init__.py
│   ├── model.py               # Main Model class, orchestrates quarterly step
│   ├── sectors.py             # Sector definitions, I-O coefficients
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── firm.py            # Firm agent, decision logic
│   │   ├── worker.py          # Worker agent, decision logic
│   │   └── expectations.py    # Adaptive expectation utility
│   ├── capability/
│   │   ├── __init__.py
│   │   ├── tasks.py           # Task categories, occupation-task weights
│   │   └── trajectory.py      # Capability trajectory from scenario
│   ├── macro/
│   │   ├── __init__.py
│   │   ├── accounting.py      # GDP, shares, price indices
│   │   └── government.py      # Tax revenue
│   ├── scenarios/
│   │   ├── __init__.py
│   │   ├── loader.py          # YAML → scenario object
│   │   └── reference.yaml     # Fragmented reference scenario
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── outputs.py         # Time series collection, serialization
│   │   └── plots.py           # Standard plot generation
│   └── config.py              # Default parameters, constants
│
├── app/
│   └── streamlit_app.py       # Interactive UI
│
├── scenarios/                 # Additional scenario YAMLs (user-editable)
│   ├── plateau.yaml
│   ├── continued_exponential.yaml
│   ├── broadly_agentic.yaml
│   ├── fragmented.yaml        # Reference for v0
│   └── transformative.yaml
│
├── tests/
│   ├── test_sectors.py        # I-O consistency, sector sums
│   ├── test_firm.py           # Firm decision logic
│   ├── test_worker.py         # Worker decision logic
│   ├── test_capability.py     # Task/exposure calculations
│   ├── test_accounting.py     # Aggregate consistency
│   ├── test_integration.py    # Full-run smoke test
│   └── test_determinism.py    # Same seed → same output
│
├── notebooks/
│   └── v0_walkthrough.ipynb   # Narrative walkthrough of reference run
│
└── outputs/                   # Generated, gitignored
    └── .gitkeep
```

---

## 11. Implementation sequence

Build in this order. Each step should be independently testable and produce something runnable before moving on.

### Phase 1 — Skeleton and scaffolding
1. Repo setup, `pyproject.toml`, dependencies (`numpy`, `pandas`, `pyyaml`, `matplotlib`, `plotly`, `streamlit`, `pytest`, `mesa` optional — see note below)
2. `config.py` with default constants
3. `sectors.py` with 5 sectors and stylized I-O matrix (5x5, rows-sum-to-1-ish, stylized)
4. `capability/tasks.py` with 10 task categories and occupation-task weight matrix (illustrative, hand-tuned)
5. `scenarios/loader.py` and `scenarios/reference.yaml`
6. Test: load scenario, instantiate sector structure, assert dimensional consistency.

### Phase 2 — Macro accounting layer
7. `macro/accounting.py` — given sector outputs and factor payments, compute GDP, labor/capital share, prices. No agents yet; use stylized sector-level aggregates.
8. Integrate with scenario loader: "empty" run that just produces trivial time series.
9. Test: accounting identities hold (GDP = sum of value-added = sum of factor income).

### Phase 3 — Firms
10. `agents/firm.py` with state and decision methods
11. `agents/expectations.py` for adaptive expectation logic (EMA or simple trend extrapolation)
12. Firm initialization: power-law size distribution per sector, stylized starting capital/labor
13. Firm decision quarterly loop: expectations → adoption → hiring → pricing
14. Test: firm decisions produce reasonable outputs under fixed inputs (no runaway, no NaNs, monotonicity where expected).

### Phase 4 — Workers
15. `agents/worker.py` with state and decision methods
16. Worker initialization matched to firm workforce needs
17. Job matching mechanism: matching unemployed workers to firm vacancies (simple stochastic matching)
18. Retraining mechanism
19. Labor force participation dynamics
20. Test: worker flows balance (employed + unemployed + out = total), retraining transitions work correctly.

### Phase 5 — Task/capability layer
21. `capability/trajectory.py` — compute c(t) per task from scenario keyframes
22. Occupation exposure computation
23. Hook exposure into firm hiring/adoption logic
24. Test: under a "plateau" scenario (capability frozen), exposure stays constant; under aggressive scenario, exposure rises monotonically.

### Phase 6 — AI & Compute sector
25. Specialized logic for AI-Compute sector: revenue from other sectors' adoption, compute stock investment, concentration of capital ownership
26. Test: AI-sector revenue grows with aggregate adoption; capital accumulates as expected.

### Phase 7 — Integration and macro closure
27. Full model step: firms + workers + capability + macro accounting all wired together
28. Run full 40-quarter simulation under reference scenario
29. Verify aggregate consistency: no accounting violations, no NaNs, no runaway dynamics
30. Test: integration smoke test runs end-to-end; outputs non-degenerate.

### Phase 8 — Analysis and outputs
31. `analysis/outputs.py` — collect time series across run, serialize to CSV/JSON
32. `analysis/plots.py` — standard plot set (employment by sector, labor share, etc.)
33. `notebooks/v0_walkthrough.ipynb` — guided tour of reference scenario
34. Test: outputs match expected schema, plots render without error.

### Phase 9 — Streamlit UI
35. `app/streamlit_app.py` — sidebar with scenario selection and key parameter overrides (capability growth rates, regulation levels, OSS gap, reliability floors)
36. Main panel: run button, time-series plots, summary stats
37. Test: app launches, scenario runs complete in acceptable time (target: <30 seconds per run).

### Phase 10 — Documentation and polish
38. README with quickstart, architecture summary, "what this is and isn't" section
39. Clear caveat about v0 being exploratory, not calibrated
40. Contribution guide for users who want to modify scenarios

---

## 12. Technology decisions

- **Python 3.11+**
- **No GPU needed.** This is CPU-bound agent simulation and dataframe work.
- **ABM framework:** Start without `mesa`. Plain Python classes and a central `Model` object stepping agent lists is simpler and gives more control. `mesa` can be adopted later if scheduling becomes complex.
- **Random number generation:** `numpy.random.default_rng(seed)` passed through the model. Seed configurable via scenario for reproducibility.
- **Data structures:** `pandas.DataFrame` for time series, lightweight dataclasses for agent state.
- **Plots:** `matplotlib` for static, `plotly` for Streamlit interactive.
- **UI:** `streamlit` — fastest path to shareable interactive UI. Can be deployed free on Streamlit Community Cloud or Hugging Face Spaces.
- **Testing:** `pytest`. Every module gets a test file. Integration test runs a short simulation and asserts basic invariants.
- **Type hints:** Yes, throughout. Makes the code legible to people reading to modify it.
- **Logging:** Standard `logging` module, INFO level for major events (quarterly step, scenario load), DEBUG for agent-level detail.

---

## 13. Invariants to enforce (and test)

These are sanity checks the model should satisfy at every time step. Violations indicate bugs.

1. **Labor accounting:** employed + unemployed + out_of_labor_force = total workers (modulo deaths/births from demographics)
2. **Sectoral employment:** sum of firm workforce sizes in sector = sector employment
3. **Accounting identity:** GDP = sum of sectoral value-added = sum of labor income + capital income + net taxes
4. **Non-negativity:** no negative employment, wages, output, prices, capital stocks
5. **Ownership totals:** capital ownership shares sum to 1
6. **Capability monotonicity:** under a non-decreasing capability trajectory, no sector's effective AI productivity falls (absent other shocks)
7. **Reproducibility:** same seed + same scenario → bitwise identical outputs
8. **No NaN or Inf:** assertion at end of each quarter that all state is finite

---

## 14. Explicit non-goals for v0

Document these clearly in README so users don't misinterpret outputs.

- **Not calibrated to real data.** Initial conditions and parameters are stylized. Absolute numbers are meaningless; *relative dynamics* across scenarios are what v0 illuminates.
- **No trade / external sector.** Closed economy.
- **No financial sector.** No stock market, no interest rates, no credit. Asset prices not modeled.
- **No government policy response.** Tax rates fixed, no UBI, no retraining subsidies. Government is a passive revenue collector in v0.
- **No spatial dimension.** No regional breakdown.
- **No demographics beyond flat labor force growth and retirement.** No cohort effects, no education pipeline, no migration.
- **Only 5 sectors.** Not 20. Professional services not broken out into legal/software/consulting/etc.
- **Representative agents within size tiers.** Not deep individual heterogeneity.
- **Simplified expectations.** EMA/trend extrapolation, no learning theory, no rational-expectations fixed point.
- **No uncertainty quantification.** Runs are deterministic given seed. Monte Carlo across seeds/parameter ranges is a v1 extension.

---

## 15. Handoff notes for implementer (Claude Code or otherwise)

- **Build incrementally.** Get each phase working and tested before moving on. Integration bugs in simulations are nightmares to debug if you build everything at once.
- **Print / log aggressively during development.** Visibility into agent state during development saves hours.
- **Use small populations during development.** 100 firms, 1000 workers while iterating. Scale up to 1000/10000 only when the logic is stable.
- **Test invariants early.** Wire the invariant checks (section 13) into the model itself as optional assertions. Turn them on during development, can be disabled for production runs.
- **Don't optimize prematurely.** v0 doesn't need to be fast. Readability over cleverness.
- **Flag design choices you make on the fly.** Anywhere you make a judgment call about a detail not specified in this plan, add a `# DESIGN CHOICE:` comment explaining what you chose and why. These become review points.
- **Keep scenario YAML examples in-repo and valid.** A user's first experience will be editing YAML; broken examples will be frustrating.
- **README should include a "known limitations" section.** Not a future-nice-to-have. v0 will be misinterpreted without it.

---

## 16. Acceptance criteria for v0

The build is done when:

1. A user can `git clone`, `pip install -e .`, and `python -m ai_econ_sim.run --scenario scenarios/fragmented.yaml` to get a complete simulation run with output CSVs and plots.
2. A user can `streamlit run app/streamlit_app.py` to interact with scenarios via UI.
3. `pytest` passes all tests.
4. Reference "fragmented" scenario produces plausible dynamics: partial knowledge-work wage compression, moderate productivity gains, no runaway pathologies.
5. README clearly communicates what the model is and is not.
6. At least three alternative scenario YAMLs (plateau, continued exponential, transformative) exist and run without error.

---

## 17. Open design questions (flag during implementation)

These are things not fully specified in this plan; implementer should either make a reasonable choice and flag, or ask:

- Exact functional form for firm AI-adoption probability (logistic? threshold?)
- Exact wage-setting mechanism (Nash bargaining? markup? posted wages?)
- Skill-level distribution initialization (uniform? empirical-ish?)
- Retraining success probability function (age, skill, target sector)
- How quickly expectations update (EMA half-life? trend window length?)
- Treatment of firm entry/exit (fixed firm population in v0? or allow exit on bankruptcy?)
- How AI-sector prices adjust (cost-plus? market-clearing given compute constraint?)

Default: make a reasonable simple choice, add `# DESIGN CHOICE:` comment, move on. These become v1 refinement targets.

---

*End of project plan. Hand off to implementer.*
