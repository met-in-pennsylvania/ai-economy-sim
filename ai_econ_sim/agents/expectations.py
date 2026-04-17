"""Adaptive expectation utilities (EMA-based trend extrapolation)."""

from collections import deque
import numpy as np


class ExpectationState:
    """
    Tracks a rolling history of a scalar signal and estimates trend.
    Uses Exponential Moving Average with configurable half-life.
    """

    def __init__(self, halflife_quarters: int = 4, initial_value: float = 0.0):
        # DESIGN CHOICE: EMA with 4-quarter half-life as baseline adaptive expectation
        alpha = 1.0 - 0.5 ** (1.0 / halflife_quarters)
        self._alpha = alpha
        self._ema = initial_value
        self._prev_ema = initial_value
        self._history: deque[float] = deque(maxlen=8)
        self._history.append(initial_value)

    def update(self, new_value: float) -> None:
        self._prev_ema = self._ema
        self._ema = self._alpha * new_value + (1.0 - self._alpha) * self._ema
        self._history.append(new_value)

    @property
    def level(self) -> float:
        return self._ema

    @property
    def trend(self) -> float:
        """Estimated quarterly change (positive = rising)."""
        return self._ema - self._prev_ema

    @property
    def is_declining(self) -> bool:
        """True if trend has been negative for last 4 observations."""
        if len(self._history) < 4:
            return False
        recent = list(self._history)[-4:]
        return recent[-1] < recent[0]


class FirmExpectations:
    """Collection of expectation states for a firm."""

    def __init__(self, sector_demand: float, own_wage: float, ai_capability: float, peer_adoption: float):
        self.sector_demand = ExpectationState(initial_value=sector_demand)
        self.own_wage = ExpectationState(initial_value=own_wage)
        self.ai_capability = ExpectationState(initial_value=ai_capability)
        self.peer_adoption = ExpectationState(initial_value=peer_adoption)

    def update(self, sector_demand: float, own_wage: float, ai_capability: float, peer_adoption: float) -> None:
        self.sector_demand.update(sector_demand)
        self.own_wage.update(own_wage)
        self.ai_capability.update(ai_capability)
        self.peer_adoption.update(peer_adoption)


class WorkerExpectations:
    """Collection of expectation states for a worker."""

    def __init__(self, sector_employment: float, sector_median_wage: float):
        self.sector_employment = ExpectationState(initial_value=sector_employment)
        self.sector_median_wage = ExpectationState(initial_value=sector_median_wage)

    def update(self, sector_employment: float, sector_median_wage: float) -> None:
        self.sector_employment.update(sector_employment)
        self.sector_median_wage.update(sector_median_wage)
