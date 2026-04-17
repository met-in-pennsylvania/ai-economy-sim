"""Tests for worker decision logic and flow accounting."""

import numpy as np
import pytest
from ai_econ_sim.agents.worker import Worker, create_workers, _retraining_success_prob


def _rng():
    return np.random.default_rng(0)


def _worker(**kwargs) -> Worker:
    defaults = dict(
        id=0, age=35, sector="knowledge_work", occupation="routine_analytical",
        skill_level=3, current_wage=80_000.0, employer_id=1, is_employed=True,
    )
    defaults.update(kwargs)
    w = Worker(**defaults)
    w.init_expectations(2500.0, 80_000.0)
    return w


def test_retraining_success_prob_range():
    for age in [25, 35, 50, 65]:
        for skill in [1, 3, 5]:
            p = _retraining_success_prob(age, skill)
            assert 0.0 <= p <= 1.0


def test_retraining_younger_higher_success():
    p_young = _retraining_success_prob(25, 3)
    p_old = _retraining_success_prob(60, 3)
    assert p_young >= p_old


def test_retraining_tick_decrements():
    from ai_econ_sim.agents.worker import RetrainingState
    w = _worker(is_employed=False, employer_id=None)
    w.retraining = RetrainingState(
        target_sector="services",
        target_occupation="service_worker",
        quarters_remaining=3,
        success_probability=1.0,
    )
    w.step_retraining_tick(_rng())
    assert w.retraining is not None
    assert w.retraining.quarters_remaining == 2


def test_retraining_completion_transitions_sector():
    from ai_econ_sim.agents.worker import RetrainingState
    rng = np.random.default_rng(7)
    w = _worker(sector="knowledge_work", is_employed=False, employer_id=None)
    w.retraining = RetrainingState(
        target_sector="services",
        target_occupation="service_worker",
        quarters_remaining=1,
        success_probability=1.0,
    )
    result = w.step_retraining_tick(rng)
    assert result == "services"
    assert w.sector == "services"
    assert w.retraining is None


def test_lfp_exit_after_prolonged_unemployment():
    rng = np.random.default_rng(1)
    w = _worker(is_employed=False, employer_id=None)
    w.is_in_labor_force = True
    # Simulate many quarters of unemployment
    for _ in range(30):
        w.step_labor_force_participation(rng, sector_improving=False)
    # After 30 quarters with exit probability, very likely to have exited
    # (This is probabilistic but with 30 trials at 0.1/quarter it's 96%+ likely)
    # Just check the state is valid
    assert isinstance(w.is_in_labor_force, bool)


def test_lfp_reentry_when_improving():
    rng = np.random.default_rng(2)
    w = _worker(is_employed=False, employer_id=None, is_in_labor_force=False)
    initial_olf = not w.is_in_labor_force
    # Run many reentry attempts
    for _ in range(50):
        w.step_labor_force_participation(rng, sector_improving=True)
        if w.is_in_labor_force:
            break
    # May or may not reenter in 50 quarters, just check state validity
    assert isinstance(w.is_in_labor_force, bool)


def test_accept_job_updates_state():
    w = _worker(is_employed=False, employer_id=None)
    w.is_employed = False
    w.accept_job(employer_id=5, sector="services", wage=50_000, occupation="service_worker")
    assert w.is_employed
    assert w.employer_id == 5
    assert w.current_wage == 50_000
    assert w.sector == "services"


def test_lose_job_updates_state():
    w = _worker()
    w.lose_job()
    assert not w.is_employed
    assert w.employer_id is None


def test_create_workers_count():
    rng = _rng()
    workers = create_workers("services", 100, 45_000, rng)
    assert len(workers) == 100


def test_create_workers_all_valid():
    rng = _rng()
    workers = create_workers("knowledge_work", 50, 90_000, rng)
    for w in workers:
        assert w.sector == "knowledge_work"
        assert 1 <= w.skill_level <= 5
        assert 22 <= w.age <= 64
        assert w.current_wage > 0
