"""Collect time series and serialize to CSV/JSON."""

from __future__ import annotations
import json
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ai_econ_sim.macro.accounting import MacroAccounts
from ai_econ_sim.config import SECTORS


def build_time_series(history: list[MacroAccounts]) -> pd.DataFrame:
    """Convert history list to a tidy DataFrame."""
    rows = []
    for accts in history:
        row: dict[str, Any] = {"quarter": accts.quarter}

        for s in SECTORS:
            row[f"employment_{s}"] = accts.sector_employment.get(s, 0)
            row[f"value_added_{s}"] = accts.sector_value_added.get(s, 0.0)
            row[f"median_wage_{s}"] = accts.sector_median_wage.get(s, 0.0)
            row[f"price_index_{s}"] = accts.sector_price_index.get(s, 1.0)
            row[f"ai_adoption_{s}"] = accts.sector_ai_adoption.get(s, 0.0)
            row[f"robotics_adoption_{s}"] = accts.sector_robotics_adoption.get(s, 0.0)

        row["interface_realization"] = accts.interface_realization

        # Knowledge-work median wages by occupation bucket
        for occ in ("routine_analytical", "creative_synthesis", "relational", "technical_specialist"):
            row[f"kw_wage_{occ}"] = accts.kw_occupation_wages.get(occ, 0.0)

        # Retraining flows
        row["retraining_initiations"] = accts.retraining_initiations
        row["retraining_successes"] = accts.retraining_successes
        row["retraining_failures"] = accts.retraining_failures

        # Demographic and firm dynamics flows
        row["retirements"] = accts.retirements
        row["new_entrants"] = accts.new_entrants
        row["firm_exits"] = accts.firm_exits
        row["firm_entries"] = accts.firm_entries
        for gen in ("boomer", "genx", "millennial", "genz"):
            row[f"employed_{gen}"] = accts.gen_employment.get(gen, 0)
        row["labor_share"] = accts.total_labor_income / max(1.0, accts.nominal_gdp)
        row["capital_share"] = accts.total_capital_income / max(1.0, accts.nominal_gdp)
        row["ai_capital_share"] = accts.ai_sector_capital_income / max(1.0, accts.nominal_gdp)
        row["nominal_gdp"] = accts.nominal_gdp
        row["real_gdp"] = accts.real_gdp
        row["aggregate_price_index"] = accts.aggregate_price_index
        row["total_employed"] = accts.total_employed
        row["total_unemployed"] = accts.total_unemployed
        row["total_out_of_lf"] = accts.total_out_of_lf
        total_lf = accts.total_employed + accts.total_unemployed
        total_all = total_lf + accts.total_out_of_lf
        row["unemployment_rate"] = accts.total_unemployed / max(1, total_lf)
        row["lfp_rate"] = total_lf / max(1, total_all)
        row["gini"] = accts.gini
        row["tax_revenue"] = accts.tax_revenue

        rows.append(row)

    return pd.DataFrame(rows)


def build_run_summary(history: list[MacroAccounts], model: Any) -> dict[str, Any]:
    """Compute run-level summary dict (used for JSON export and report)."""
    if not history:
        return {}

    df = build_time_series(history)
    last = history[-1]
    first = history[0]

    all_workers = [w for ws in model.workers.values() for w in ws]
    olf = sum(1 for w in all_workers if not w.is_in_labor_force)

    summary: dict[str, Any] = {
        "scenario": model.scenario.name,
        "quarters_run": len(history),
        "initial_gdp": first.nominal_gdp,
        "final_gdp": last.nominal_gdp,
        "gdp_growth_total_pct": (last.nominal_gdp / max(1.0, first.nominal_gdp) - 1.0) * 100,
        "real_gdp_growth_total_pct": (df["real_gdp"].iloc[-1] / max(1.0, df["real_gdp"].iloc[0]) - 1.0) * 100,
        "final_unemployment_rate": df["unemployment_rate"].iloc[-1],
        "peak_unemployment_rate": df["unemployment_rate"].max(),
        "final_lfp_rate": df["lfp_rate"].iloc[-1],
        "final_labor_share": df["labor_share"].iloc[-1],
        "initial_labor_share": df["labor_share"].iloc[0],
        "final_gini": df["gini"].iloc[-1],
        "initial_gini": df["gini"].iloc[0],
        "workers_out_of_lf_final": olf,
        "total_retirements": int(df["retirements"].sum()),
        "total_new_entrants": int(df["new_entrants"].sum()),
        "total_firm_exits": int(df["firm_exits"].sum()),
        "total_firm_entries": int(df["firm_entries"].sum()),
        "total_retraining_initiations": int(df["retraining_initiations"].sum()),
        "total_retraining_successes": int(df["retraining_successes"].sum()),
        "total_retraining_failures": int(df["retraining_failures"].sum()),
        "final_ai_adoption_by_sector": {
            s: round(last.sector_ai_adoption.get(s, 0.0), 4) for s in SECTORS
        },
        "final_employment_by_sector": {
            s: last.sector_employment.get(s, 0) for s in SECTORS
        },
        "final_kw_occupation_wages": dict(last.kw_occupation_wages),
    }
    return summary


def build_run_report(history: list[MacroAccounts], model: Any) -> str:
    """
    Generate a human-readable plain-text run report.
    Covers macro outcomes, sectoral detail, labor market, retraining,
    firm dynamics, and generational employment.
    Suitable for download as a .txt file.
    """
    if not history:
        return "No simulation history available."

    s = build_run_summary(history, model)
    df = build_time_series(history)
    last = history[-1]
    n_q = s["quarters_run"]
    n_yr = n_q / 4

    retraining_success_rate = (
        s["total_retraining_successes"] /
        max(1, s["total_retraining_successes"] + s["total_retraining_failures"])
    ) * 100

    lines = [
        "=" * 68,
        f"  AI-Economy Simulation Run Report",
        f"  Scenario : {s['scenario']}",
        f"  Horizon  : {n_q} quarters ({n_yr:.1f} years)",
        "=" * 68,
        "",
        "── MACRO OUTCOMES ─────────────────────────────────────────────",
        f"  Nominal GDP:  ${s['initial_gdp']/1e9:.1f}B  →  ${s['final_gdp']/1e9:.1f}B"
        f"  ({s['gdp_growth_total_pct']:+.1f}%)",
        f"  Real GDP growth (deflated):  {s['real_gdp_growth_total_pct']:+.1f}%",
        f"  Aggregate price index (final): {df['aggregate_price_index'].iloc[-1]:.3f}",
        "",
        "── LABOR MARKET ────────────────────────────────────────────────",
        f"  Unemployment rate:  {df['unemployment_rate'].iloc[0]*100:.1f}%  →  "
        f"{s['final_unemployment_rate']*100:.1f}%  "
        f"(peak: {s['peak_unemployment_rate']*100:.1f}%)",
        f"  LFP rate:           {df['lfp_rate'].iloc[0]*100:.1f}%  →  "
        f"{s['final_lfp_rate']*100:.1f}%",
        f"  Workers out of LF (final): {s['workers_out_of_lf_final']:,}",
        "",
        "── INCOME DISTRIBUTION ─────────────────────────────────────────",
        f"  Labor share of GDP:  {s['initial_labor_share']*100:.1f}%  →  "
        f"{s['final_labor_share']*100:.1f}%  "
        f"(Δ {(s['final_labor_share']-s['initial_labor_share'])*100:+.1f}pp)",
        f"  Capital share:       {(1-s['initial_labor_share'])*100:.1f}%  →  "
        f"{(1-s['final_labor_share'])*100:.1f}%",
        f"  Gini coefficient:    {s['initial_gini']:.3f}  →  {s['final_gini']:.3f}  "
        f"(Δ {s['final_gini']-s['initial_gini']:+.3f})",
        "",
        "── SECTORAL DETAIL ─────────────────────────────────────────────",
        f"  {'Sector':<20} {'Employ (final)':>14} {'AI adoption':>12} {'Median wage':>13}",
        f"  {'-'*20} {'-'*14} {'-'*12} {'-'*13}",
    ]
    for sec in SECTORS:
        emp = s["final_employment_by_sector"].get(sec, 0)
        adopt = s["final_ai_adoption_by_sector"].get(sec, 0.0)
        wage = last.sector_median_wage.get(sec, 0.0)
        lines.append(f"  {sec:<20} {emp:>14,} {adopt*100:>11.1f}% {wage/1000:>11.1f}k")
    lines.append("")

    # KW occupation wages
    kw_wages = s.get("final_kw_occupation_wages", {})
    if kw_wages:
        lines += [
            "── KNOWLEDGE-WORK OCCUPATION WAGES (final quarter) ────────────",
            f"  {'Occupation':<26} {'Median wage':>12}",
            f"  {'-'*26} {'-'*12}",
        ]
        occ_labels = {
            "routine_analytical":   "Routine Analytical",
            "creative_synthesis":   "Creative Synthesis",
            "relational":           "Relational",
            "technical_specialist": "Technical Specialist",
        }
        for occ, label in occ_labels.items():
            w_val = kw_wages.get(occ, 0.0)
            if w_val > 0:
                lines.append(f"  {label:<26} ${w_val/1000:>9.1f}k/yr")
        lines.append("")

    lines += [
        "── RETRAINING FLOWS (cumulative over run) ──────────────────────",
        f"  Initiations : {s['total_retraining_initiations']:,}",
        f"  Successes   : {s['total_retraining_successes']:,}",
        f"  Failures    : {s['total_retraining_failures']:,}",
        f"  Success rate: {retraining_success_rate:.1f}%",
        "",
        "── FIRM DYNAMICS (cumulative over run) ─────────────────────────",
        f"  Firm exits   : {s['total_firm_exits']:,}",
        f"  Firm entries : {s['total_firm_entries']:,}",
        f"  Net change   : {s['total_firm_entries'] - s['total_firm_exits']:+,}",
        "",
        "── DEMOGRAPHIC FLOWS (cumulative over run) ─────────────────────",
        f"  Retirements  : {s['total_retirements']:,}",
        f"  New entrants : {s['total_new_entrants']:,}",
    ]

    # Generational employment (final quarter)
    gen_labels = {"boomer": "Boomers", "genx": "Gen X",
                  "millennial": "Millennials", "genz": "Gen Z"}
    gen_total = sum(last.gen_employment.get(g, 0) for g in gen_labels)
    if gen_total > 0:
        lines += [
            "",
            "── GENERATIONAL EMPLOYMENT SHARE (final quarter) ───────────────",
            f"  {'Generation':<14} {'Workers':>10} {'Share':>8}",
            f"  {'-'*14} {'-'*10} {'-'*8}",
        ]
        for gen, label in gen_labels.items():
            cnt = last.gen_employment.get(gen, 0)
            share = cnt / gen_total * 100
            lines.append(f"  {label:<14} {cnt:>10,} {share:>7.1f}%")

    lines += [
        "",
        "=" * 68,
        "  Not calibrated for forecasting. Relative dynamics matter, not absolute values.",
        "=" * 68,
    ]
    return "\n".join(lines)


def save_outputs(
    history: list[MacroAccounts],
    model: Any,
    output_dir: str | Path,
    run_name: str = "run",
) -> dict[str, Path]:
    """Save time series CSV, summary JSON. Returns dict of output paths."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = build_time_series(history)
    ts_path = out / f"{run_name}_timeseries.csv"
    df.to_csv(ts_path, index=False)

    summary = build_run_summary(history, model)
    meta_path = out / f"{run_name}_summary.json"
    with open(meta_path, "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)

    return {"timeseries": ts_path, "summary": meta_path}


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not serializable: {type(obj)}")
