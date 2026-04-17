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
