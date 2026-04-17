"""
2023 BEA/BLS calibration data for the 5-sector model.

Sector mapping (private-sector only, excluding government and real estate):
  ai_compute       → BEA Information industry ($2.1T, 3.1M workers)
  knowledge_work   → BEA Finance+Insurance + Professional/Scientific/Technical ($5.9T, 17.1M)
  services         → BEA Healthcare+Social Assistance + Retail + Accommodation+Food +
                     Administrative/Support + Other Personal Services ($6.6T, 66.2M)
  goods            → BEA Manufacturing + Agriculture + Mining ($3.7T, 16.2M)
  infrastructure   → BEA Construction + Utilities + Transportation+Warehousing ($2.2T, 15.0M)

Modeled economy ≈ $20.4T (75% of $27.4T nominal GDP; excludes ~$3.5T govt + ~$3.2T RE/rental).
Employment base ≈ 117.6M private-sector workers (excludes ~22M government).

Sources:
  GDP shares  — BEA NIPA Table 1.3.5 (2023 current dollar value added)
  Employment  — BLS Current Employment Statistics (2023 annual average nonfarm payroll)
  Wages       — BLS Occupational Employment & Wage Statistics (May 2023 median)
  Growth      — BLS/BEA historical 2015–2019 CAGR (pre-AI baseline)
"""

from __future__ import annotations
from dataclasses import dataclass

from ai_econ_sim.scenarios.loader import InitialConditions


# ---------------------------------------------------------------------------
# Raw sector-mapped data (all values are private-economy shares, not total US)
# ---------------------------------------------------------------------------

GDP_SHARES: dict[str, float] = {
    "ai_compute":      0.101,   # Information: $2.1T / $20.4T
    "knowledge_work":  0.289,   # Finance+Insurance+Prof/Sci/Tech: $5.9T
    "services":        0.324,   # Healthcare+Retail+Food+Admin+Personal: $6.6T
    "goods":           0.179,   # Manufacturing+Ag+Mining: $3.7T
    "infrastructure":  0.107,   # Construction+Utilities+Transport: $2.2T
}

EMPLOYMENT_SHARES: dict[str, float] = {
    "ai_compute":      0.026,   # 3.1M / 117.6M
    "knowledge_work":  0.145,   # 17.1M (Finance+Insurance+Prof/Sci/Tech)
    "services":        0.563,   # 66.2M (HC+Retail+Food+Admin+Personal)
    "goods":           0.138,   # 16.2M
    "infrastructure":  0.128,   # 15.0M
}

MEDIAN_WAGES: dict[str, float] = {
    "ai_compute":      135_000,  # BLS Information $108k; AI/tech roles skew higher
    "knowledge_work":   88_000,  # Finance $85k, Prof/Tech $88k — blended
    "services":         44_000,  # HC $60k, Retail $39k, Food $30k, Admin $43k — pop-weighted
    "goods":            52_000,  # Manufacturing $54k, Ag $35k, Mining $78k — pop-weighted
    "infrastructure":   59_000,  # Construction $58k, Utilities $95k, Transport $56k — pop-weighted
}

# Pre-AI base growth rates (annual real CAGR, 2015–2019)
BASE_GROWTH_RATES: dict[str, float] = {
    "ai_compute":     0.080,   # Information sector tech boom
    "knowledge_work": 0.040,   # Finance + professional services
    "services":       0.025,   # Healthcare + retail (population-driven)
    "goods":          0.015,   # Manufacturing (flat/slight reshoring)
    "infrastructure": 0.025,   # Construction + utilities
}

# Modeled economy nominal GDP ($20.4T = private sector ex-govt ex-real-estate)
NOMINAL_GDP: float = 20_400_000_000_000.0

# Reference labels for reporting
SECTOR_BLS_LABELS: dict[str, str] = {
    "ai_compute":     "Information",
    "knowledge_work": "Finance+Ins+Prof/Sci/Tech",
    "services":       "HC+Retail+Food+Admin+Personal",
    "goods":          "Manufacturing+Ag+Mining",
    "infrastructure": "Construction+Utilities+Transport",
}


@dataclass
class CalibrationBenchmarks:
    """Snapshot of real-world benchmarks for model comparison."""
    gdp_shares: dict[str, float]
    employment_shares: dict[str, float]
    median_wages: dict[str, float]
    nominal_gdp: float
    source_year: int = 2023


BENCHMARKS_2023 = CalibrationBenchmarks(
    gdp_shares=GDP_SHARES,
    employment_shares=EMPLOYMENT_SHARES,
    median_wages=MEDIAN_WAGES,
    nominal_gdp=NOMINAL_GDP,
)


def calibrated_initial_conditions() -> InitialConditions:
    """Return InitialConditions populated from 2023 BEA/BLS data."""
    return InitialConditions(
        gdp_shares=dict(GDP_SHARES),
        employment_shares=dict(EMPLOYMENT_SHARES),
        initial_wages=dict(MEDIAN_WAGES),
        nominal_gdp=NOMINAL_GDP,
    )


def compare_to_model(history: list) -> "dict[str, dict[str, float]]":
    """
    Compare final model state to BEA/BLS benchmarks.

    Returns nested dict: {metric: {sector: (model_value, benchmark_value, pct_error)}}.
    Requires history list of MacroAccounts objects.
    """
    import pandas as pd
    from ai_econ_sim.analysis.outputs import build_time_series

    df = build_time_series(history)
    final = df.iloc[-1]
    nom_gdp = final.get("nominal_gdp", 1.0)
    total_emp = sum(
        final.get(f"employment_{s}", 0) for s in GDP_SHARES
    )

    result: dict[str, dict] = {"gdp_share": {}, "employment_share": {}, "wage": {}}

    for s in GDP_SHARES:
        va_col = f"value_added_{s}"
        emp_col = f"employment_{s}"
        wage_col = f"median_wage_{s}"

        model_gdp_share = final.get(va_col, 0) / max(nom_gdp, 1)
        bench_gdp = GDP_SHARES[s]
        result["gdp_share"][s] = {
            "model": round(model_gdp_share, 4),
            "benchmark": bench_gdp,
            "pct_error": round((model_gdp_share - bench_gdp) / bench_gdp * 100, 1),
        }

        model_emp_share = final.get(emp_col, 0) / max(total_emp, 1)
        bench_emp = EMPLOYMENT_SHARES[s]
        result["employment_share"][s] = {
            "model": round(model_emp_share, 4),
            "benchmark": bench_emp,
            "pct_error": round((model_emp_share - bench_emp) / bench_emp * 100, 1),
        }

        model_wage = final.get(wage_col, 0)
        bench_wage = MEDIAN_WAGES[s]
        result["wage"][s] = {
            "model": round(model_wage, 0),
            "benchmark": bench_wage,
            "pct_error": round((model_wage - bench_wage) / bench_wage * 100, 1) if bench_wage else 0,
        }

    return result


def calibration_report(history: list) -> str:
    """Return a formatted text report comparing model output to BEA/BLS benchmarks."""
    cmp = compare_to_model(history)
    lines = [
        "=" * 64,
        "  Calibration Report: Model vs. BEA/BLS 2023",
        "=" * 64,
    ]

    for metric, label in [
        ("gdp_share", "GDP Share"),
        ("employment_share", "Employment Share"),
        ("wage", "Median Wage ($)"),
    ]:
        lines.append(f"\n{label}")
        lines.append(f"  {'Sector':<22} {'Model':>10} {'BEA/BLS':>10} {'Error%':>8}")
        lines.append(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*8}")
        for s, vals in cmp[metric].items():
            fmt_m = f"{vals['model']:.3f}" if metric != "wage" else f"{vals['model']:,.0f}"
            fmt_b = f"{vals['benchmark']:.3f}" if metric != "wage" else f"{vals['benchmark']:,.0f}"
            lines.append(f"  {s:<22} {fmt_m:>10} {fmt_b:>10} {vals['pct_error']:>7.1f}%")

    lines.append("\n" + "=" * 64)
    return "\n".join(lines)
