"""Tests for task/exposure calculations and capability trajectory."""

import numpy as np
import pytest
from ai_econ_sim.capability.tasks import (
    TASK_NAMES, N_TASKS, OCCUPATION_TASK_WEIGHTS, compute_occupation_exposure
)
from ai_econ_sim.capability.trajectory import CapabilityTrajectory, _interpolate


def test_task_count():
    assert N_TASKS == 10


def test_occupation_weight_rows_sum_to_one():
    row_sums = OCCUPATION_TASK_WEIGHTS.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-9)


def test_exposure_all_zeros_when_no_capability():
    cap = np.zeros(N_TASKS)
    floors = np.full(N_TASKS, 0.5)
    exposure = compute_occupation_exposure(cap, floors)
    assert np.all(exposure == 0.0)


def test_exposure_positive_when_all_capable():
    cap = np.ones(N_TASKS)
    floors = np.zeros(N_TASKS)
    exposure = compute_occupation_exposure(cap, floors)
    assert np.all(exposure > 0)
    assert np.all(exposure <= 1.0)


def test_exposure_gated_by_reliability_floor():
    # Only task 0 exceeds floor
    cap = np.zeros(N_TASKS)
    cap[0] = 1.0
    floors = np.zeros(N_TASKS)
    floors[1:] = 2.0  # impossible to meet for tasks 1-9
    exposure = compute_occupation_exposure(cap, floors)
    # Only task 0 contributes; exposure = w[:,0] * 1.0
    expected = OCCUPATION_TASK_WEIGHTS[:, 0] * 1.0
    np.testing.assert_allclose(exposure, expected, atol=1e-9)


def test_trajectory_interpolation():
    frames = [(0, 0.0), (10, 1.0)]
    assert _interpolate(frames, 0) == 0.0
    assert _interpolate(frames, 10) == 1.0
    assert abs(_interpolate(frames, 5) - 0.5) < 1e-9


def test_trajectory_extrapolation_edges():
    frames = [(5, 0.3), (15, 0.8)]
    assert _interpolate(frames, 0) == 0.3   # clamp at start
    assert _interpolate(frames, 20) == 0.8  # clamp at end


def test_capability_trajectory_non_decreasing_under_growing_scenario():
    traj_spec = {t: [{"quarter": 0, "value": 0.5}, {"quarter": 40, "value": 0.9}] for t in TASK_NAMES}
    floors_spec = {t: 0.0 for t in TASK_NAMES}
    ct = CapabilityTrajectory(traj_spec, floors_spec)
    prev = ct.capability_index(0)
    for q in range(1, 41):
        curr = ct.capability_index(q)
        assert curr >= prev - 1e-9, f"Capability declined at q={q}: {prev} -> {curr}"
        prev = curr


def test_capability_trajectory_frozen_under_plateau():
    # All tasks at constant 0.7
    traj_spec = {t: [{"quarter": 0, "value": 0.7}] for t in TASK_NAMES}
    floors_spec = {t: 0.5 for t in TASK_NAMES}
    ct = CapabilityTrajectory(traj_spec, floors_spec)
    c0 = ct.capability_at(0)
    c20 = ct.capability_at(20)
    np.testing.assert_allclose(c0, c20, atol=1e-9)
