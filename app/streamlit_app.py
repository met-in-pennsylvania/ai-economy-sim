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

    # Import all plot helpers once at top of results block — avoids partial-deployment
    # races where a deferred import hits a stale cached module on Streamlit Cloud.
    from ai_econ_sim.analysis.plots import (
        build_plotly_dashboard,
        build_plotly_mc_dashboard,
    )
    try:
        from ai_econ_sim.analysis.plots import build_plotly_demography_dashboard
    except ImportError:
        build_plotly_demography_dashboard = None  # type: ignore[assignment]

    try:
        from ai_econ_sim.calibration import calibration_report as _calibration_report
    except ImportError:
        _calibration_report = None  # type: ignore[assignment]

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
        fig = build_plotly_mc_dashboard(mc)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        # Also show the full dashboard on median
        st.subheader("Median trajectory (all sectors)")
        fig2 = build_plotly_dashboard(df)
        if fig2:
            st.plotly_chart(fig2, use_container_width=True)
    else:
        fig = build_plotly_dashboard(df)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)

        if build_plotly_demography_dashboard is not None:
            fig_demo = build_plotly_demography_dashboard(df)
            if fig_demo is not None:
                st.subheader("Workforce Demographics & Firm Dynamics")
                st.plotly_chart(fig_demo, use_container_width=True)

    # Calibration report (single runs only)
    if history and run["scenario_name"] == "calibrated_baseline_2023" and _calibration_report:
        with st.expander("BEA/BLS Calibration Report"):
            st.code(_calibration_report(history), language=None)

    # ------------------------------------------------------------------
    # Scenario comparison
    # ------------------------------------------------------------------
    with st.expander("📊 Compare against another scenario", expanded=False):
        st.caption(
            "Run a second scenario with the **same parameter overrides** applied on top, "
            "then diff the key indicators against the primary run."
        )
        cmp_names = [n for n in scenario_names if n != run["scenario_name"]]
        cmp_selected = st.selectbox("Comparison scenario", cmp_names, key="cmp_scenario")
        cmp_btn = st.button("▶ Run comparison", key="cmp_run")

        if cmp_btn or "cmp_run_result" in st.session_state:
            if cmp_btn:
                from ai_econ_sim.scenarios.loader import load_scenario as _ls
                from ai_econ_sim.model import Model as _Model
                from ai_econ_sim.analysis.outputs import build_time_series as _bts
                from ai_econ_sim.config import DEV_POPULATION_SCALE as _DPS

                cmp_path = SCENARIO_DIR / f"{cmp_selected}.yaml"
                with st.spinner(f"Running comparison: {cmp_selected}..."):
                    cmp_scen = _ls(cmp_path)
                    # Apply same overrides as primary
                    cmp_scen.regulation["knowledge_work"] = reg_kw
                    cmp_scen.regulation["services"] = reg_svc
                    cmp_scen.regulation["infrastructure"] = reg_infra
                    cmp_scen.oss_frontier_gap = oss_gap
                    cmp_scen.robotics.deployment_start_quarter = robotics_start
                    cmp_scen.robotics.diffusion_rate = robotics_diffusion
                    cmp_scen.sentiment.consumer_ai_discount["services"] = consumer_discount_services
                    cmp_scen.sentiment.consumer_ai_discount["knowledge_work"] = consumer_discount_kw
                    cmp_scen.sentiment.workforce_resistance["services"] = workforce_resistance_services
                    cmp_scen.sentiment.workforce_resistance["knowledge_work"] = workforce_resistance_kw
                    cmp_scen.interface_realization._keyframes = [(0, iface_start), (40, iface_end)]
                    _scale = _DPS if dev_mode else 1.0
                    cmp_model = _Model(cmp_scen, population_scale=_scale, dev_assertions=False)
                    cmp_history = cmp_model.run()
                cmp_df = _bts(cmp_history)
                st.session_state["cmp_run_result"] = {
                    "df": cmp_df,
                    "name": cmp_selected,
                }

            if "cmp_run_result" in st.session_state:
                cmp_res = st.session_state["cmp_run_result"]
                cmp_df = cmp_res["df"]
                cmp_name = cmp_res["name"]

                st.success(f"Comparison: **{run['scenario_name']}** vs **{cmp_name}**")

                # Delta metrics (final quarter)
                KEY_METRICS = [
                    ("real_gdp",          "Real GDP (Δ%)",            lambda a, b: (b / max(a, 1) - 1) * 100),
                    ("unemployment_rate", "Unemployment (Δpp)",        lambda a, b: (b - a) * 100),
                    ("labor_share",       "Labor share (Δpp)",         lambda a, b: (b - a) * 100),
                    ("gini",              "Gini (Δ)",                   lambda a, b: b - a),
                    ("lfp_rate",          "LFP rate (Δpp)",            lambda a, b: (b - a) * 100),
                ]
                dcols = st.columns(len(KEY_METRICS))
                for col_ui, (key, label, fn) in zip(dcols, KEY_METRICS):
                    try:
                        a_val = float(df[key].iloc[-1])
                        b_val = float(cmp_df[key].iloc[-1])
                        delta = fn(a_val, b_val)
                        col_ui.metric(label, f"{b_val:.3g}", f"{delta:+.2f}")
                    except Exception:
                        pass

                # Overlay chart: key trajectories side-by-side
                try:
                    import plotly.graph_objects as go
                    from plotly.subplots import make_subplots

                    OVERLAY_METRICS = [
                        ("real_gdp",          "Real GDP (index)",    lambda s: s / s.iloc[0]),
                        ("unemployment_rate", "Unemployment %",      lambda s: s * 100),
                        ("labor_share",       "Labor Share %",       lambda s: s * 100),
                        ("gini",              "Gini",                lambda s: s),
                    ]
                    fig_cmp = make_subplots(rows=2, cols=2,
                        subplot_titles=[m[1] for m in OVERLAY_METRICS])
                    for idx, (key, label, transform) in enumerate(OVERLAY_METRICS):
                        r, c = divmod(idx, 2)
                        if key in df.columns and key in cmp_df.columns:
                            fig_cmp.add_trace(go.Scatter(
                                x=df["quarter"], y=transform(df[key]),
                                name=run["scenario_name"], line=dict(color="#1a5276", width=2),
                                showlegend=(idx == 0),
                            ), row=r + 1, col=c + 1)
                            fig_cmp.add_trace(go.Scatter(
                                x=cmp_df["quarter"], y=transform(cmp_df[key]),
                                name=cmp_name, line=dict(color="#922b21", width=2, dash="dash"),
                                showlegend=(idx == 0),
                            ), row=r + 1, col=c + 1)
                    fig_cmp.update_layout(height=550,
                        title_text=f"{run['scenario_name']}  vs  {cmp_name}")
                    st.plotly_chart(fig_cmp, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not render comparison chart: {e}")

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
