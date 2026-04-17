"""Streamlit interactive UI for the AI-Economy Macro Simulation."""

import sys
import time
from pathlib import Path

# Ensure the repo root is on the path so ai_econ_sim is importable
# whether running locally (installed) or on Streamlit Community Cloud.
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import numpy as np

# Resolve paths relative to repo root
REPO_ROOT = Path(__file__).parent.parent
SCENARIO_DIR = REPO_ROOT / "scenarios"

st.set_page_config(
    page_title="AI-Economy Simulation",
    page_icon="🤖",
    layout="wide",
)

st.title("AI-Economy Macro Simulation v0")
st.caption(
    "Stylized prototype of a US-focused ~10-year macroeconomic simulation of AI-driven transformation. "
    "**Not calibrated to real data.** Relative dynamics across scenarios are what matter, not absolute numbers."
)

# ------------------------------------------------------------------
# Sidebar: scenario selection + parameter overrides
# ------------------------------------------------------------------

with st.sidebar:
    st.header("Scenario")

    scenario_files = sorted(SCENARIO_DIR.glob("*.yaml"))
    scenario_names = [f.stem for f in scenario_files]
    selected_name = st.selectbox("Base scenario", scenario_names, index=0)
    selected_path = SCENARIO_DIR / f"{selected_name}.yaml"

    st.divider()
    st.header("Parameter Overrides")

    dev_mode = st.checkbox("Dev mode (fast, small population)", value=True)
    disable_assertions = st.checkbox("Disable invariant assertions", value=False)

    st.subheader("Regulation")
    reg_kw = st.slider("Knowledge Work friction", 0.0, 1.0, 0.20, 0.05)
    reg_svc = st.slider("Services friction", 0.0, 1.0, 0.50, 0.05)
    reg_infra = st.slider("Infrastructure friction", 0.0, 1.0, 0.70, 0.05)

    st.subheader("AI Capability")
    capability_scale = st.slider(
        "Capability growth speed (1x = as in YAML)",
        0.5, 2.0, 1.0, 0.1,
        help="Scales the endpoint capability values. 2x = all tasks reach 2x their final YAML value (capped at 1.0)."
    )

    st.subheader("OSS gap")
    oss_gap = st.slider("OSS frontier gap (generations)", 0.0, 4.0, 0.5, 0.5,
                        help="0 = OSS matches frontier. Higher = more capital concentration.")

    st.subheader("Human Factors")
    consumer_discount_services = st.slider(
        "Consumer AI discount — Services",
        0.0, 0.5, 0.20, 0.05,
        help="How much demand growth is suppressed per unit of AI adoption in services. "
             "Captures preference for human delivery (healthcare, restaurants, personal care)."
    )
    consumer_discount_kw = st.slider(
        "Consumer AI discount — Knowledge Work",
        0.0, 0.3, 0.10, 0.05,
        help="Clients who prefer human lawyers, consultants, advisors."
    )
    workforce_resistance_services = st.slider(
        "Workforce resistance — Services",
        0.0, 0.6, 0.22, 0.05,
        help="Fraction of theoretical AI productivity lost to workers unwilling or unable "
             "to integrate AI tools. Cultural/generational friction."
    )
    workforce_resistance_kw = st.slider(
        "Workforce resistance — Knowledge Work",
        0.0, 0.5, 0.18, 0.05,
        help="Gen Z and others who route around AI tools in knowledge work."
    )
    iface_start = st.slider(
        "Interface realization today (Q0)",
        0.1, 0.9, 0.35, 0.05,
        help="Fraction of theoretical AI productivity that workers can actually capture "
             "given current tool maturity. Even the best tools today feel strange."
    )
    iface_end = st.slider(
        "Interface realization at year 10 (Q40)",
        0.1, 1.0, 0.78, 0.05,
        help="How natural and fluid human-AI collaboration becomes by end of simulation."
    )

    st.subheader("AI-Powered Robots (Optimus-class)")
    robotics_start = st.slider(
        "Robot deployment start (quarter)",
        0, 40, 999,
        help="Quarter when generally-useful robots become available. Set to 40 to disable."
    )
    robotics_diffusion = st.slider(
        "Robot diffusion rate (per quarter)",
        0.01, 0.20, 0.04, 0.01,
        help="How fast robots spread through services/goods/infrastructure after deployment. "
             "0.04 = ~9 quarters to reach 35% displacement."
    )

    st.subheader("Uncertainty")
    mc_runs = st.slider(
        "Monte Carlo runs",
        1, 30, 1,
        help="1 = single deterministic run. >1 = run N times with different seeds "
             "and show median + uncertainty bands. 10–20 gives stable bands; takes ~10× longer."
    )

    run_btn = st.button("▶ Run Simulation", type="primary", use_container_width=True)

# ------------------------------------------------------------------
# Main panel
# ------------------------------------------------------------------

if "last_run" not in st.session_state:
    st.session_state.last_run = None

if run_btn:
    from ai_econ_sim.scenarios.loader import load_scenario
    from ai_econ_sim.model import Model
    from ai_econ_sim.analysis.outputs import build_time_series
    from ai_econ_sim.analysis.plots import build_plotly_dashboard
    from ai_econ_sim.config import DEV_POPULATION_SCALE

    with st.spinner("Loading scenario..."):
        scenario = load_scenario(selected_path)

    # Apply overrides
    scenario.regulation["knowledge_work"] = reg_kw
    scenario.regulation["services"] = reg_svc
    scenario.regulation["infrastructure"] = reg_infra
    scenario.oss_frontier_gap = oss_gap
    scenario.robotics.deployment_start_quarter = robotics_start
    scenario.robotics.diffusion_rate = robotics_diffusion
    scenario.sentiment.consumer_ai_discount["services"] = consumer_discount_services
    scenario.sentiment.consumer_ai_discount["knowledge_work"] = consumer_discount_kw
    scenario.sentiment.workforce_resistance["services"] = workforce_resistance_services
    scenario.sentiment.workforce_resistance["knowledge_work"] = workforce_resistance_kw
    scenario.interface_realization._keyframes = [
        (0, iface_start),
        (40, iface_end),
    ]

    # Scale capability endpoints
    if capability_scale != 1.0:
        traj = scenario.capability._keyframes
        for task_name, frames in traj.items():
            if frames:
                last_q, last_v = frames[-1]
                new_v = min(1.0, last_v * capability_scale)
                frames[-1] = (last_q, new_v)

    scale = DEV_POPULATION_SCALE if dev_mode else 1.0

    if mc_runs > 1:
        from ai_econ_sim.monte_carlo import run_monte_carlo
        from ai_econ_sim.analysis.plots import build_plotly_mc_dashboard
        label = f"Running {mc_runs} MC simulations × {scenario.horizon_quarters} quarters..."
        with st.spinner(label):
            t0 = time.time()
            mc = run_monte_carlo(scenario, n_runs=mc_runs, population_scale=scale)
            elapsed = time.time() - t0
        df = mc.median()
        df["quarter"] = df.index
        st.session_state.last_run = {
            "df": df,
            "mc": mc,
            "history": None,
            "model": None,
            "elapsed": elapsed,
            "scenario_name": scenario.name,
        }
    else:
        with st.spinner(f"Running {scenario.horizon_quarters} quarters (scale={scale})..."):
            t0 = time.time()
            model = Model(scenario, population_scale=scale, dev_assertions=not disable_assertions)
            history = model.run()
            elapsed = time.time() - t0
        df = build_time_series(history)
        st.session_state.last_run = {
            "df": df,
            "mc": None,
            "history": history,
            "model": model,
            "elapsed": elapsed,
            "scenario_name": scenario.name,
        }

if st.session_state.last_run is not None:
    run = st.session_state.last_run
    df = run["df"]
    mc = run.get("mc")
    history = run["history"]

    n_label = f"{mc.n_runs} MC runs" if mc else "1 run"
    quarters_label = mc.runs[0].shape[0] if mc else (len(history) if history else "?")
    st.success(f"Ran **{run['scenario_name']}** — {quarters_label} quarters × {n_label} in {run['elapsed']:.1f}s")

    # Summary metrics from median (MC) or single run
    col1, col2, col3, col4, col5 = st.columns(5)
    if mc:
        med = mc.median()
        col1.metric("Real GDP growth (median)", f"{(med['real_gdp'].iloc[-1] / med['real_gdp'].iloc[0] - 1) * 100:.1f}%")
        col2.metric("Unemployment (median, Q40)", f"{med['unemployment_rate'].iloc[-1] * 100:.1f}%")
        col3.metric("LFP rate (median, Q40)", f"{med['lfp_rate'].iloc[-1] * 100:.1f}%")
        col4.metric("Labor share (median, Q40)", f"{med['labor_share'].iloc[-1] * 100:.1f}%")
        col5.metric("Gini (median, Q40)", f"{med['gini'].iloc[-1]:.3f}")
    elif history:
        final = history[-1]
        lf = final.total_employed + final.total_unemployed
        total_all = lf + final.total_out_of_lf
        col1.metric("Real GDP growth", f"{(final.real_gdp / max(1, history[0].real_gdp) - 1) * 100:.1f}%")
        col2.metric("Unemployment rate", f"{final.total_unemployed / max(1, lf) * 100:.1f}%")
        col3.metric("LFP rate", f"{lf / max(1, total_all) * 100:.1f}%")
        col4.metric("Labor share of GDP", f"{final.total_labor_income / max(1.0, final.nominal_gdp) * 100:.1f}%")
        col5.metric("Gini coefficient", f"{final.gini:.3f}")

    # Plots
    if mc:
        from ai_econ_sim.analysis.plots import build_plotly_mc_dashboard
        fig = build_plotly_mc_dashboard(mc)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        # Also show the full dashboard on median
        st.subheader("Median trajectory (all sectors)")
        from ai_econ_sim.analysis.plots import build_plotly_dashboard
        fig2 = build_plotly_dashboard(df)
        if fig2:
            st.plotly_chart(fig2, use_container_width=True)
    else:
        from ai_econ_sim.analysis.plots import build_plotly_dashboard
        fig = build_plotly_dashboard(df)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)

    # Raw data expander
    with st.expander("Raw time series data"):
        st.dataframe(df.round(4), use_container_width=True)

    # Download button
    csv_bytes = df.to_csv(index=False).encode()
    st.download_button(
        "Download time series CSV",
        data=csv_bytes,
        file_name=f"{run['scenario_name']}_timeseries.csv",
        mime="text/csv",
    )

else:
    st.info("Configure parameters in the sidebar and click **Run Simulation**.")

    with st.expander("About this model"):
        st.markdown("""
**AI-Economy Macro Simulation v0** is a stylized agent-based model of a US-like economy
undergoing AI-driven transformation over a ~10-year horizon.

**Architecture:**
- **5 sectors**: AI & Compute, Knowledge Work, Services, Goods, Infrastructure
- **~1000 firms** with power-law size distribution making quarterly adoption/hiring/pricing decisions
- **~10,000 workers** with adaptive expectations, job search, retraining, and LFP dynamics
- **Task-level AI exposure** for Knowledge Work (10 task categories with reliability gates)
- **Macro accounting** layer ensures GDP = labor + capital income

**What this is NOT:**
- Not calibrated to real data — absolute numbers are meaningless
- No financial sector, no trade, no government policy response
- Simplified expectations (EMA), not rational expectations
- Exploratory prototype, not a forecasting tool

See `PROJECT_PLAN.md` for full architecture documentation.
        """)
