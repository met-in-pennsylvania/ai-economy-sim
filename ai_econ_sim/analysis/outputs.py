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
    """Compute run-level summary statistics."""
    if not history:
        return {}

    df = build_time_series(history)
    last = history[-1]

    # Retraining stats
    all_workers = [w for ws in model.workers.values() for w in ws]
    retrained = sum(1 for w in all_workers if w.sector != w.occupation.split("_")[0] if hasattr(w, "_original_sector"))
    olf = sum(1 for w in all_workers if not w.is_in_labor_force)

    summary = {
        "scenario": model.scenario.name,
        "quarters_run": len(history),
        "initial_gdp": history[0].nominal_gdp,
        "final_gdp": last.nominal_gdp,
        "gdp_growth_total": (last.nominal_gdp / max(1.0, history[0].nominal_gdp)) - 1.0,
        "final_unemployment_rate": df["unemployment_rate"].iloc[-1],
        "final_lfp_rate": df["lfp_rate"].iloc[-1],
        "final_labor_share": df["labor_share"].iloc[-1],
        "final_gini": df["gini"].iloc[-1],
        "peak_unemployment": df["unemployment_rate"].max(),
        "workers_out_of_lf": olf,
        "final_ai_adoption_by_sector": {
            s: last.sector_ai_adoption.get(s, 0.0) for s in SECTORS
        },
    }
    return summary


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
