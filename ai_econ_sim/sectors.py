"""
Sector definitions and I-O coefficients.

Calibrated to 2023 BEA data (see calibration/bea_bls.py for source mapping).

Sector mapping:
  ai_compute      → BEA Information industry
  knowledge_work  → Finance+Insurance + Professional/Scientific/Technical
  services        → Healthcare+Social Assistance + Retail + Accommodation+Food +
                    Administrative/Support + Other Personal Services
  goods           → Manufacturing + Agriculture + Mining
  infrastructure  → Construction + Utilities + Transportation+Warehousing

Labor/capital shares: BEA NIPA GDP-by-industry compensation ratios, 2022–2023.
I-O coefficients: derived from BEA Use Table proportions, simplified to 5×5.
"""

from dataclasses import dataclass, field
import numpy as np

from ai_econ_sim.config import SECTORS


@dataclass
class Sector:
    name: str
    gdp_share: float          # share of initial GDP
    employment_share: float   # share of initial employment
    ai_exposure_type: str     # "task_level", "scalar", or "self_producing"

    # Production function parameters (BEA compensation / gross-operating-surplus splits)
    labor_share: float        # labor share of value-added
    capital_share: float      # capital share of value-added (1 - labor_share)

    # AI spillover for scalar sectors (0 for task-level / self-producing)
    ai_spillover: float = 0.0


# Sector definitions with BEA-calibrated GDP/employment shares and factor shares.
# GDP and employment shares from calibration/bea_bls.py (2023 private-sector).
# Labor share of value-added from BEA NIPA GDP-by-industry (2022–2023 avg):
#   ai_compute:    ~40% labor (capital-intensive tech infrastructure)
#   knowledge_work: ~63% labor (professional and financial services)
#   services:      ~70% labor (healthcare, retail, food — very labor-intensive)
#   goods:         ~53% labor (manufacturing mix of capital and labour)
#   infrastructure: ~58% labor (construction + transport; utilities are capital-heavy)
SECTOR_DEFINITIONS = {
    "ai_compute": Sector(
        name="ai_compute",
        gdp_share=0.101,        # BEA Information: $2.1T / $20.4T modeled economy
        employment_share=0.026,  # BLS: 3.1M / 117.6M private-sector
        ai_exposure_type="self_producing",
        labor_share=0.40,       # BEA: Information compensation ~40% of value-added
        capital_share=0.60,
        ai_spillover=0.0,
    ),
    "knowledge_work": Sector(
        name="knowledge_work",
        gdp_share=0.289,        # BEA: Finance+Insurance+Prof/Sci/Tech
        employment_share=0.145,
        ai_exposure_type="task_level",
        labor_share=0.63,       # BEA: Professional/financial services ~63%
        capital_share=0.37,
        ai_spillover=0.0,
    ),
    "services": Sector(
        name="services",
        gdp_share=0.324,        # BEA: HC+Retail+Food+Admin+Personal
        employment_share=0.563,
        ai_exposure_type="scalar",
        labor_share=0.70,       # BEA: Services (ex-finance) ~70%; healthcare ~72%
        capital_share=0.30,
        ai_spillover=0.05,
    ),
    "goods": Sector(
        name="goods",
        gdp_share=0.179,        # BEA: Manufacturing+Ag+Mining
        employment_share=0.138,
        ai_exposure_type="scalar",
        labor_share=0.53,       # BEA: Manufacturing compensation ~53%
        capital_share=0.47,
        ai_spillover=0.10,
    ),
    "infrastructure": Sector(
        name="infrastructure",
        gdp_share=0.107,        # BEA: Construction+Utilities+Transport
        employment_share=0.128,
        ai_exposure_type="scalar",
        labor_share=0.58,       # BEA: Blended construction/utilities/transport
        capital_share=0.42,
        ai_spillover=0.03,
    ),
}


# I-O coefficient matrix: io_matrix[i][j] = fraction of sector j's gross output
# purchased as intermediate inputs from sector i.
# Rows = supplying sector, Columns = buying sector.
# Order: ai_compute, knowledge_work, services, goods, infrastructure
#
# Calibrated from BEA 2022 Use Table (Producers' Prices), aggregated to 5 sectors.
# Each column represents what that sector buys as intermediate inputs; value-added
# share = 1 - column sum must be > 0.
IO_MATRIX = np.array([
    #  ai_c   kw     svc    gds    infra   (column = buying sector)
    [0.08,  0.05,  0.04,  0.03,  0.02],  # ai_compute sells to     (software, cloud)
    [0.12,  0.08,  0.06,  0.04,  0.05],  # knowledge_work sells to (legal, acctg, mgmt)
    [0.04,  0.05,  0.08,  0.04,  0.04],  # services sells to       (admin, food, health)
    [0.03,  0.03,  0.12,  0.18,  0.22],  # goods sells to          (materials, equipment)
    [0.02,  0.04,  0.06,  0.07,  0.08],  # infrastructure sells to (utilities, transport)
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
