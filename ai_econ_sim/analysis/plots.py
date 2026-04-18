"""Standard plot generation (matplotlib static + plotly interactive)."""

from __future__ import annotations
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from ai_econ_sim.config import SECTORS

SECTOR_COLORS = {
    "ai_compute": "#e84393",
    "knowledge_work": "#3498db",
    "services": "#2ecc71",
    "goods": "#f39c12",
    "infrastructure": "#9b59b6",
}

SECTOR_LABELS = {
    "ai_compute": "AI & Compute",
    "knowledge_work": "Knowledge Work",
    "services": "Services",
    "goods": "Goods",
    "infrastructure": "Infrastructure",
}


def plot_standard_set(df: pd.DataFrame, output_dir: Path, run_name: str = "run") -> list[Path]:
    """Generate standard PNG plots. Returns list of paths created."""
    if not HAS_MPL:
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    paths.append(_plot_employment(df, output_dir, run_name))
    paths.append(_plot_wages(df, output_dir, run_name))
    paths.append(_plot_gdp_shares(df, output_dir, run_name))
    paths.append(_plot_labor_share(df, output_dir, run_name))
    paths.append(_plot_ai_adoption(df, output_dir, run_name))
    paths.append(_plot_labor_market(df, output_dir, run_name))
    paths.append(_plot_robotics(df, output_dir, run_name))

    return [p for p in paths if p is not None]


# ------------------------------------------------------------------
# Monte Carlo band plots
# ------------------------------------------------------------------

def plot_mc_bands(mc_results: Any, output_dir: Path, run_name: str = "mc") -> list[Path]:
    """
    Generate MC uncertainty-band PNGs for key indicators.
    Median line + shaded 10th-90th and 25th-75th percentile bands.
    """
    if not HAS_MPL:
        return []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    indicators = [
        ("nominal_gdp",          "Nominal GDP",              1e12, "$T"),
        ("unemployment_rate",    "Unemployment Rate",        100,  "%"),
        ("labor_share",          "Labor Share of GDP",       100,  "%"),
        ("gini",                 "Gini Coefficient",         1,    ""),
        ("employment_services",  "Services Employment",      1,    "workers"),
        ("employment_knowledge_work", "KW Employment",       1,    "workers"),
        ("ai_adoption_knowledge_work", "KW AI Adoption",     100,  "%"),
        ("interface_realization","Interface Realization",    100,  "%"),
    ]

    for col, label, scale, unit in indicators:
        p = _plot_one_mc_band(mc_results, col, label, scale, unit, output_dir, run_name)
        if p:
            paths.append(p)

    return paths


def _plot_one_mc_band(
    mc: Any, col: str, label: str, scale: float, unit: str,
    out: Path, name: str,
) -> Optional[Path]:
    try:
        bands = mc.bands
        if col not in bands.columns:
            return None
        quarters = bands.xs(0.50, level="quantile").index

        p10 = bands.xs(0.10, level="quantile")[col] * scale
        p25 = bands.xs(0.25, level="quantile")[col] * scale
        p50 = bands.xs(0.50, level="quantile")[col] * scale
        p75 = bands.xs(0.75, level="quantile")[col] * scale
        p90 = bands.xs(0.90, level="quantile")[col] * scale

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.fill_between(quarters, p10, p90, alpha=0.15, color="#2980b9", label="10–90th pct")
        ax.fill_between(quarters, p25, p75, alpha=0.30, color="#2980b9", label="25–75th pct")
        ax.plot(quarters, p50, color="#1a5276", linewidth=2, label="Median")
        ax.set_xlabel("Quarter")
        ylabel = f"{label} ({unit})" if unit else label
        ax.set_ylabel(ylabel)
        ax.set_title(f"{label} — MC uncertainty bands (n={mc.n_runs})")
        ax.legend(fontsize=9)
        path = out / f"{name}_mc_{col}.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return path
    except Exception:
        return None


def plot_sweep_comparison(sweep: Any, column: str, label: str, output_dir: Path, run_name: str = "sweep") -> Optional[Path]:
    """Plot median trajectories for each parameter value in a sweep."""
    if not HAS_MPL:
        return None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.cm.viridis  # type: ignore[attr-defined]
    colors = [cmap(i / max(1, len(sweep.param_values) - 1)) for i in range(len(sweep.param_values))]

    for val, mc, color in zip(sweep.param_values, sweep.mc_results, colors):
        if column not in mc.bands.columns:
            continue
        median = mc.median()[column]
        ax.plot(median.index, median, label=f"{sweep.param_name}={val}", color=color)

    ax.set_xlabel("Quarter")
    ax.set_ylabel(label)
    ax.set_title(f"{label} by {sweep.param_name}")
    ax.legend(fontsize=9)
    path = output_dir / f"{run_name}_sweep_{column}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def build_plotly_mc_dashboard(mc_results: Any) -> Any:
    """Plotly figure with MC band traces for embedding in Streamlit."""
    if not HAS_PLOTLY:
        return None

    indicators = [
        ("nominal_gdp",       "GDP ($)",          1e9),
        ("unemployment_rate", "Unemployment %",   100),
        ("labor_share",       "Labor Share %",    100),
        ("gini",              "Gini",             1),
    ]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[label for _, label, _ in indicators],
    )

    for idx, (col, label, scale) in enumerate(indicators):
        row, col_pos = divmod(idx, 2)
        row += 1; col_pos += 1

        try:
            bands = mc_results.bands
            if col not in bands.columns:
                continue
            quarters = list(bands.xs(0.50, level="quantile").index)
            p10 = (bands.xs(0.10, level="quantile")[col] * scale).tolist()
            p50 = (bands.xs(0.50, level="quantile")[col] * scale).tolist()
            p90 = (bands.xs(0.90, level="quantile")[col] * scale).tolist()

            fig.add_trace(go.Scatter(
                x=quarters + quarters[::-1], y=p90 + p10[::-1],
                fill="toself", fillcolor="rgba(41,128,185,0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                name="10–90th pct", showlegend=(idx == 0),
            ), row=row, col=col_pos)
            fig.add_trace(go.Scatter(
                x=quarters, y=p50,
                line=dict(color="#1a5276", width=2),
                name="Median", showlegend=(idx == 0),
            ), row=row, col=col_pos)
        except Exception:
            continue

    fig.update_layout(height=600, title_text=f"MC Uncertainty Bands (n={mc_results.n_runs})")
    return fig


def _plot_employment(df: pd.DataFrame, out: Path, name: str) -> Optional[Path]:
    fig, ax = plt.subplots(figsize=(10, 5))
    quarters = df["quarter"]
    for s in SECTORS:
        col = f"employment_{s}"
        if col in df.columns:
            ax.plot(quarters, df[col], label=SECTOR_LABELS[s], color=SECTOR_COLORS[s])
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Workers")
    ax.set_title("Employment by Sector")
    ax.legend()
    path = out / f"{name}_employment.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_wages(df: pd.DataFrame, out: Path, name: str) -> Optional[Path]:
    fig, ax = plt.subplots(figsize=(10, 5))
    quarters = df["quarter"]
    for s in SECTORS:
        col = f"median_wage_{s}"
        if col in df.columns:
            ax.plot(quarters, df[col] / 1000, label=SECTOR_LABELS[s], color=SECTOR_COLORS[s])
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Median Annual Wage ($k)")
    ax.set_title("Median Wages by Sector")
    ax.legend()
    path = out / f"{name}_wages.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_gdp_shares(df: pd.DataFrame, out: Path, name: str) -> Optional[Path]:
    fig, ax = plt.subplots(figsize=(10, 5))
    quarters = df["quarter"]
    total_va = sum(df.get(f"value_added_{s}", 0) for s in SECTORS)
    total_va = total_va.replace(0, np.nan)
    for s in SECTORS:
        col = f"value_added_{s}"
        if col in df.columns:
            share = df[col] / (df["nominal_gdp"].replace(0, np.nan))
            ax.plot(quarters, share * 100, label=SECTOR_LABELS[s], color=SECTOR_COLORS[s])
    ax.set_xlabel("Quarter")
    ax.set_ylabel("% of GDP")
    ax.set_title("Sectoral Value-Added Share of GDP")
    ax.legend()
    path = out / f"{name}_gdp_shares.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_labor_share(df: pd.DataFrame, out: Path, name: str) -> Optional[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    q = df["quarter"]

    axes[0].plot(q, df["labor_share"] * 100, label="Labor share", color="#2c3e50")
    axes[0].plot(q, df["capital_share"] * 100, label="Capital share", color="#e74c3c")
    axes[0].plot(q, df["ai_capital_share"] * 100, label="AI capital share", color=SECTOR_COLORS["ai_compute"], linestyle="--")
    axes[0].set_xlabel("Quarter")
    axes[0].set_ylabel("% of GDP")
    axes[0].set_title("Factor Income Shares")
    axes[0].legend()

    axes[1].plot(q, df["gini"], color="#8e44ad")
    axes[1].set_xlabel("Quarter")
    axes[1].set_ylabel("Gini Coefficient")
    axes[1].set_title("Income Inequality (Gini)")

    path = out / f"{name}_income_shares.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_ai_adoption(df: pd.DataFrame, out: Path, name: str) -> Optional[Path]:
    fig, ax = plt.subplots(figsize=(10, 5))
    quarters = df["quarter"]
    for s in SECTORS:
        col = f"ai_adoption_{s}"
        if col in df.columns:
            ax.plot(quarters, df[col] * 100, label=SECTOR_LABELS[s], color=SECTOR_COLORS[s])
    ax.set_xlabel("Quarter")
    ax.set_ylabel("AI Adoption Level (%)")
    ax.set_title("AI Adoption by Sector")
    ax.legend()
    path = out / f"{name}_ai_adoption.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_robotics(df: pd.DataFrame, out: Path, name: str) -> Optional[Path]:
    rob_cols = [f"robotics_adoption_{s}" for s in SECTORS if f"robotics_adoption_{s}" in df.columns]
    if not rob_cols or df[rob_cols].max().max() == 0:
        return None
    fig, ax = plt.subplots(figsize=(10, 5))
    quarters = df["quarter"]
    for s in SECTORS:
        col = f"robotics_adoption_{s}"
        if col in df.columns:
            ax.plot(quarters, df[col] * 100, label=SECTOR_LABELS[s], color=SECTOR_COLORS[s])
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Robotics Adoption (% of physical tasks displaced)")
    ax.set_title("AI-Powered Robotics Adoption by Sector")
    ax.legend()
    path = out / f"{name}_robotics.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_labor_market(df: pd.DataFrame, out: Path, name: str) -> Optional[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    q = df["quarter"]

    axes[0].plot(q, df["unemployment_rate"] * 100, color="#e74c3c")
    axes[0].set_xlabel("Quarter")
    axes[0].set_ylabel("Rate (%)")
    axes[0].set_title("Unemployment Rate")

    axes[1].plot(q, df["lfp_rate"] * 100, color="#2980b9")
    axes[1].set_xlabel("Quarter")
    axes[1].set_ylabel("Rate (%)")
    axes[1].set_title("Labor Force Participation Rate")

    path = out / f"{name}_labor_market.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def build_plotly_dashboard(df: pd.DataFrame) -> Any:
    """Return a plotly Figure with multiple subplots for Streamlit embedding."""
    if not HAS_PLOTLY:
        return None

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            "Employment by Sector",
            "Median Wages by Sector",
            "Factor Income Shares",
            "AI Adoption by Sector",
            "Labor Market",
            "GDP & Price Level",
        ],
    )
    q = df["quarter"]

    # Employment
    for s in SECTORS:
        col = f"employment_{s}"
        if col in df.columns:
            fig.add_trace(go.Scatter(x=q, y=df[col], name=SECTOR_LABELS[s],
                                     line=dict(color=SECTOR_COLORS[s]), legendgroup=s), row=1, col=1)

    # Wages
    for s in SECTORS:
        col = f"median_wage_{s}"
        if col in df.columns:
            fig.add_trace(go.Scatter(x=q, y=df[col] / 1000, name=SECTOR_LABELS[s],
                                     line=dict(color=SECTOR_COLORS[s]), legendgroup=s, showlegend=False), row=1, col=2)

    # Factor shares
    fig.add_trace(go.Scatter(x=q, y=df["labor_share"] * 100, name="Labor share",
                              line=dict(color="#2c3e50")), row=2, col=1)
    fig.add_trace(go.Scatter(x=q, y=df["capital_share"] * 100, name="Capital share",
                              line=dict(color="#e74c3c")), row=2, col=1)
    fig.add_trace(go.Scatter(x=q, y=df["gini"], name="Gini",
                              line=dict(color="#8e44ad", dash="dot")), row=2, col=1)

    # AI Adoption
    for s in SECTORS:
        col = f"ai_adoption_{s}"
        if col in df.columns:
            fig.add_trace(go.Scatter(x=q, y=df[col] * 100, name=SECTOR_LABELS[s],
                                     line=dict(color=SECTOR_COLORS[s]), legendgroup=s, showlegend=False), row=2, col=2)

    # Labor market
    fig.add_trace(go.Scatter(x=q, y=df["unemployment_rate"] * 100, name="Unemployment %",
                              line=dict(color="#e74c3c")), row=3, col=1)
    fig.add_trace(go.Scatter(x=q, y=df["lfp_rate"] * 100, name="LFP Rate %",
                              line=dict(color="#2980b9")), row=3, col=1)

    # GDP
    if "real_gdp" in df.columns:
        gdp_normalized = df["real_gdp"] / df["real_gdp"].iloc[0]
        fig.add_trace(go.Scatter(x=q, y=gdp_normalized, name="Real GDP (index)",
                                  line=dict(color="#27ae60")), row=3, col=2)
    fig.add_trace(go.Scatter(x=q, y=df["aggregate_price_index"], name="Price Index",
                              line=dict(color="#f39c12", dash="dot")), row=3, col=2)

    fig.update_layout(height=900, showlegend=True, title_text="AI-Economy Simulation Dashboard")
    return fig


KW_OCC_COLORS = {
    "routine_analytical":   "#3498db",
    "creative_synthesis":   "#e74c3c",
    "relational":           "#2ecc71",
    "technical_specialist": "#f39c12",
}
KW_OCC_LABELS = {
    "routine_analytical":   "Routine Analytical",
    "creative_synthesis":   "Creative Synthesis",
    "relational":           "Relational",
    "technical_specialist": "Technical Specialist",
}


def build_plotly_kw_dashboard(df: pd.DataFrame) -> Any:
    """
    Plotly figure showing knowledge-work occupation wage divergence
    and capability index trajectory.  Rendered as a separate panel
    below the main dashboard for single runs.
    """
    if not HAS_PLOTLY:
        return None

    kw_cols = [f"kw_wage_{o}" for o in KW_OCC_COLORS]
    if not all(c in df.columns for c in kw_cols):
        return None

    subplot_titles = [
        "KW Median Wages by Occupation ($k/yr)",
        "KW Wage Divergence (index, Q0=1)",
        "AI Capability Index",
        "KW AI Adoption & Interface Realization",
    ]
    fig = make_subplots(rows=2, cols=2, subplot_titles=subplot_titles)
    q = df["quarter"]

    # Absolute wages
    for occ, color in KW_OCC_COLORS.items():
        col = f"kw_wage_{occ}"
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=q, y=df[col] / 1_000,
                name=KW_OCC_LABELS[occ],
                line=dict(color=color),
                legendgroup=occ,
            ), row=1, col=1)

    # Indexed divergence (Q0 = 1.0)
    for occ, color in KW_OCC_COLORS.items():
        col = f"kw_wage_{occ}"
        if col in df.columns and df[col].iloc[0] > 0:
            indexed = df[col] / df[col].iloc[0]
            fig.add_trace(go.Scatter(
                x=q, y=indexed,
                name=KW_OCC_LABELS[occ],
                line=dict(color=color),
                legendgroup=occ,
                showlegend=False,
            ), row=1, col=2)

    # Capability index
    if "capability_index" in df.columns:
        fig.add_trace(go.Scatter(
            x=q, y=df["capability_index"],
            name="Capability index",
            line=dict(color="#8e44ad", width=2),
        ), row=2, col=1)
    if "interface_realization" in df.columns:
        fig.add_trace(go.Scatter(
            x=q, y=df["interface_realization"],
            name="Interface realization",
            line=dict(color="#16a085", dash="dot"),
        ), row=2, col=1)

    # KW AI adoption
    if "ai_adoption_knowledge_work" in df.columns:
        fig.add_trace(go.Scatter(
            x=q, y=df["ai_adoption_knowledge_work"] * 100,
            name="KW AI adoption %",
            line=dict(color="#2980b9", width=2),
        ), row=2, col=2)

    fig.update_layout(height=600, showlegend=True,
                      title_text="Knowledge Work: Occupation Wages & AI Capability")
    return fig


GENERATION_COLORS = {
    "boomer":     "#e67e22",
    "genx":       "#2980b9",
    "millennial": "#27ae60",
    "genz":       "#8e44ad",
}

GENERATION_LABELS = {
    "boomer":     "Boomers (60+)",
    "genx":       "Gen X (44–59)",
    "millennial": "Millennials (28–43)",
    "genz":       "Gen Z (22–27)",
}


def build_plotly_demography_dashboard(df: pd.DataFrame) -> Any:
    """
    Plotly figure showing generational employment composition and firm dynamics.
    Only rendered when the time-series contains the Phase 3 columns.
    """
    if not HAS_PLOTLY:
        return None

    gen_cols = [f"employed_{g}" for g in ("boomer", "genx", "millennial", "genz")]
    if not all(c in df.columns for c in gen_cols):
        return None

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Employment by Generation",
            "Gen Share of Employment (%)",
            "Firm Entry & Exit (per quarter)",
            "Demographic Flows (per quarter)",
        ],
    )
    q = df["quarter"]

    # Absolute generational employment
    for g in ("boomer", "genx", "millennial", "genz"):
        col = f"employed_{g}"
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=q, y=df[col], name=GENERATION_LABELS[g],
                line=dict(color=GENERATION_COLORS[g]),
                legendgroup=g,
            ), row=1, col=1)

    # Generational share
    total_gen = sum(df.get(f"employed_{g}", 0) for g in ("boomer", "genx", "millennial", "genz"))
    total_gen = total_gen.replace(0, float("nan"))
    for g in ("boomer", "genx", "millennial", "genz"):
        col = f"employed_{g}"
        if col in df.columns:
            share = df[col] / total_gen * 100
            fig.add_trace(go.Scatter(
                x=q, y=share, name=GENERATION_LABELS[g],
                line=dict(color=GENERATION_COLORS[g]),
                legendgroup=g, showlegend=False,
            ), row=1, col=2)

    # Firm entry/exit
    if "firm_entries" in df.columns:
        fig.add_trace(go.Bar(
            x=q, y=df["firm_entries"], name="Entries",
            marker_color="#27ae60", opacity=0.7,
        ), row=2, col=1)
    if "firm_exits" in df.columns:
        fig.add_trace(go.Bar(
            x=q, y=-df["firm_exits"], name="Exits",
            marker_color="#e74c3c", opacity=0.7,
        ), row=2, col=1)

    # Demographic flows
    if "new_entrants" in df.columns:
        fig.add_trace(go.Bar(
            x=q, y=df["new_entrants"], name="New entrants",
            marker_color="#2980b9", opacity=0.7,
        ), row=2, col=2)
    if "retirements" in df.columns:
        fig.add_trace(go.Bar(
            x=q, y=-df["retirements"], name="Retirements",
            marker_color="#e67e22", opacity=0.7,
        ), row=2, col=2)

    fig.update_layout(
        height=700,
        barmode="overlay",
        showlegend=True,
        title_text="Workforce Demographics & Firm Dynamics",
    )
    return fig
