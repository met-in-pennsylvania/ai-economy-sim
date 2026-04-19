"""Tests for CLI entry point (run.py) and firm size distribution in summary/report."""

import subprocess
import sys
import pytest
from pathlib import Path

from ai_econ_sim.scenarios.loader import load_scenario
from ai_econ_sim.model import Model
from ai_econ_sim.analysis.outputs import build_run_summary, build_run_report
from ai_econ_sim.config import SECTORS

REFERENCE_YAML = Path(__file__).parent.parent / "scenarios" / "fragmented.yaml"


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def short_run():
    scenario = load_scenario(REFERENCE_YAML)
    scenario.horizon_quarters = 4
    model = Model(scenario, population_scale=0.1, dev_assertions=True)
    history = model.run()
    return model, history


# ---------------------------------------------------------------------------
# CLI import and --help
# ---------------------------------------------------------------------------

def test_cli_module_imports():
    """run.py should be importable without side-effects."""
    import ai_econ_sim.run  # noqa: F401 — just verifying no import error


def test_cli_help():
    """python -m ai_econ_sim.run --help should exit 0 and mention --scenario."""
    result = subprocess.run(
        [sys.executable, "-m", "ai_econ_sim.run", "--help"],
        capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "--scenario" in result.stdout


def test_cli_dev_smoke_run(tmp_path):
    """--dev smoke run should complete and write a timeseries CSV and summary JSON."""
    result = subprocess.run(
        [
            sys.executable, "-m", "ai_econ_sim.run",
            "--scenario", str(REFERENCE_YAML),
            "--dev",
            "--output-dir", str(tmp_path),
            "--no-plots",
        ],
        capture_output=True, text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"CLI failed:\nstdout={result.stdout}\nstderr={result.stderr}"

    csv_files = list(tmp_path.glob("*_timeseries.csv"))
    json_files = list(tmp_path.glob("*_summary.json"))
    assert len(csv_files) == 1, f"Expected 1 timeseries CSV, found: {csv_files}"
    assert len(json_files) == 1, f"Expected 1 summary JSON, found: {json_files}"


def test_cli_population_scale_flag(tmp_path):
    """--population-scale 0.05 should produce output files without error."""
    result = subprocess.run(
        [
            sys.executable, "-m", "ai_econ_sim.run",
            "--scenario", str(REFERENCE_YAML),
            "--population-scale", "0.05",
            "--output-dir", str(tmp_path),
            "--no-plots",
        ],
        capture_output=True, text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"CLI failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    assert any(tmp_path.glob("*_timeseries.csv"))


def test_cli_report_printed_to_stdout(tmp_path):
    """Run report should be printed to stdout (not just logged)."""
    result = subprocess.run(
        [
            sys.executable, "-m", "ai_econ_sim.run",
            "--scenario", str(REFERENCE_YAML),
            "--dev",
            "--output-dir", str(tmp_path),
            "--no-plots",
        ],
        capture_output=True, text=True,
        timeout=120,
    )
    assert result.returncode == 0
    # Report contains these section headers
    assert "MACRO OUTCOMES" in result.stdout
    assert "LABOR MARKET" in result.stdout
    assert "FIRM SIZE DISTRIBUTION" in result.stdout


# ---------------------------------------------------------------------------
# Firm size distribution in summary
# ---------------------------------------------------------------------------

def test_firm_size_dist_in_summary(short_run):
    model, history = short_run
    summary = build_run_summary(history, model)
    assert "firm_size_dist_final" in summary, "firm_size_dist_final missing from summary"


def test_firm_size_dist_has_all_sectors(short_run):
    model, history = short_run
    summary = build_run_summary(history, model)
    dist = summary["firm_size_dist_final"]
    for sector in SECTORS:
        assert sector in dist, f"Sector {sector!r} missing from firm_size_dist_final"


def test_firm_size_dist_has_all_tiers(short_run):
    model, history = short_run
    summary = build_run_summary(history, model)
    dist = summary["firm_size_dist_final"]
    for sector in SECTORS:
        for tier in ("micro", "small", "medium", "large"):
            assert tier in dist[sector], f"Tier {tier!r} missing for sector {sector!r}"


def test_firm_size_dist_counts_are_non_negative(short_run):
    model, history = short_run
    summary = build_run_summary(history, model)
    dist = summary["firm_size_dist_final"]
    for sector, counts in dist.items():
        for tier, n in counts.items():
            assert n >= 0, f"{sector}/{tier} count is negative: {n}"


def test_firm_size_dist_total_matches_firm_count(short_run):
    """Sum of tier counts per sector should equal number of firms in that sector."""
    model, history = short_run
    summary = build_run_summary(history, model)
    dist = summary["firm_size_dist_final"]
    for sector in SECTORS:
        expected = len(model.firms.get(sector, []))
        actual = sum(dist[sector].values())
        assert actual == expected, (
            f"Sector {sector!r}: tier counts sum to {actual}, "
            f"but model.firms has {expected} firms"
        )


# ---------------------------------------------------------------------------
# Firm size distribution in report text
# ---------------------------------------------------------------------------

def test_firm_size_dist_in_report(short_run):
    model, history = short_run
    report = build_run_report(history, model)
    assert "FIRM SIZE DISTRIBUTION" in report


def test_firm_size_dist_report_has_sectors(short_run):
    model, history = short_run
    report = build_run_report(history, model)
    # At least one sector name should appear in the firm size section
    dist_section_start = report.find("FIRM SIZE DISTRIBUTION")
    assert dist_section_start != -1
    dist_section = report[dist_section_start:]
    found = any(sec in dist_section for sec in SECTORS)
    assert found, "No sector names found in FIRM SIZE DISTRIBUTION section of report"
