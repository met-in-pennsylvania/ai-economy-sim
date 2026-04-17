"""Tests for sector definitions and I-O consistency."""

import numpy as np
import pytest
from ai_econ_sim.sectors import (
    SECTOR_DEFINITIONS, IO_MATRIX, get_value_added_shares, get_io_matrix, SECTORS
)
from ai_econ_sim.config import SECTORS as CFG_SECTORS


def test_sector_keys_match_config():
    assert set(SECTOR_DEFINITIONS.keys()) == set(CFG_SECTORS)


def test_gdp_shares_sum_to_one():
    total = sum(s.gdp_share for s in SECTOR_DEFINITIONS.values())
    assert abs(total - 1.0) < 1e-6, f"GDP shares sum to {total}"


def test_employment_shares_sum_to_one():
    total = sum(s.employment_share for s in SECTOR_DEFINITIONS.values())
    assert abs(total - 1.0) < 1e-6, f"Employment shares sum to {total}"


def test_labor_capital_shares_sum_to_one():
    for name, s in SECTOR_DEFINITIONS.items():
        assert abs(s.labor_share + s.capital_share - 1.0) < 1e-9, f"{name}: labor+capital != 1"


def test_io_matrix_shape():
    assert IO_MATRIX.shape == (5, 5)


def test_io_matrix_non_negative():
    assert np.all(IO_MATRIX >= 0)


def test_value_added_shares_positive():
    va = get_value_added_shares()
    for s, v in va.items():
        assert v > 0, f"{s}: value-added share {v} <= 0"


def test_io_matrix_columns_leave_room_for_va():
    va = get_value_added_shares()
    for v in va.values():
        assert v < 1.0, "Value-added share >= 1 means no intermediate inputs"
        assert v > 0.0
