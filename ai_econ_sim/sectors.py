"""Sector definitions and I-O coefficients."""

from dataclasses import dataclass, field
import numpy as np

from ai_econ_sim.config import SECTORS


@dataclass
class Sector:
    name: str
    gdp_share: float          # share of initial GDP
    employment_share: float   # share of initial employment
    ai_exposure_type: str     # "task_level", "scalar", or "self_producing"

    # Production function parameters
    labor_share: float        # labor share of value-added
    capital_share: float      # capital share of value-added (1 - labor_share)

    # AI spillover for scalar sectors (0 for task-level / self-producing)
    ai_spillover: float = 0.0


# Stylized sector definitions matching US rough proportions
SECTOR_DEFINITIONS = {
    "ai_compute": Sector(
        name="ai_compute",
        gdp_share=0.02,
        employment_share=0.02,
        ai_exposure_type="self_producing",
        labor_share=0.40,
        capital_share=0.60,  # capital-intensive
        ai_spillover=0.0,
    ),
    "knowledge_work": Sector(
        name="knowledge_work",
        gdp_share=0.25,
        employment_share=0.22,
        ai_exposure_type="task_level",
        labor_share=0.70,
        capital_share=0.30,
        ai_spillover=0.0,
    ),
    "services": Sector(
        name="services",
        gdp_share=0.35,
        employment_share=0.40,
        ai_exposure_type="scalar",
        labor_share=0.65,
        capital_share=0.35,
        ai_spillover=0.05,
    ),
    "goods": Sector(
        name="goods",
        gdp_share=0.25,
        employment_share=0.25,
        ai_exposure_type="scalar",
        labor_share=0.55,
        capital_share=0.45,
        ai_spillover=0.10,
    ),
    "infrastructure": Sector(
        name="infrastructure",
        gdp_share=0.13,
        employment_share=0.11,
        ai_exposure_type="scalar",
        labor_share=0.50,
        capital_share=0.50,
        ai_spillover=0.03,
    ),
}


# I-O coefficient matrix: io_matrix[i][j] = fraction of sector j's inputs from sector i
# Rows = supplying sector, Columns = receiving sector
# Order: ai_compute, knowledge_work, services, goods, infrastructure
# DESIGN CHOICE: stylized coefficients, not calibrated to BEA data
IO_MATRIX = np.array([
    #  ai_c  kw    svc   gds   infra   (column = buying sector)
    [0.05, 0.08, 0.03, 0.02, 0.02],  # ai_compute sells to
    [0.15, 0.10, 0.08, 0.05, 0.05],  # knowledge_work sells to
    [0.10, 0.15, 0.10, 0.08, 0.08],  # services sells to
    [0.20, 0.05, 0.10, 0.15, 0.20],  # goods sells to
    [0.15, 0.10, 0.10, 0.12, 0.10],  # infrastructure sells to
], dtype=float)

# Value-added share = 1 - sum of intermediate inputs (per column)
_va_shares = 1.0 - IO_MATRIX.sum(axis=0)
assert np.all(_va_shares > 0), f"Negative value-added shares: {_va_shares}"


def get_value_added_shares() -> dict[str, float]:
    """Value-added fraction of gross output per sector."""
    return {s: float(_va_shares[i]) for i, s in enumerate(SECTORS)}


def get_io_matrix() -> np.ndarray:
    """Return the 5x5 I-O coefficient matrix (rows=supplier, cols=buyer)."""
    return IO_MATRIX.copy()
