"""Capability trajectory: compute c(t) per task from scenario keyframes."""

import numpy as np
from ai_econ_sim.capability.tasks import TASK_NAMES, N_TASKS, TASK_INDEX


class CapabilityTrajectory:
    """
    Stores per-task capability keyframes and interpolates to any quarter.
    Also computes reliability floors and the scalar capability_index.
    """

    def __init__(self, trajectory_spec: dict, reliability_floors_spec: dict):
        self._keyframes: dict[str, list[tuple[int, float]]] = {}
        self._reliability_floors = np.zeros(N_TASKS)

        for task_name in TASK_NAMES:
            frames = trajectory_spec.get(task_name, [{"quarter": 0, "value": 0.5}])
            self._keyframes[task_name] = [(f["quarter"], f["value"]) for f in frames]

        for task_name, floor in reliability_floors_spec.items():
            if task_name in TASK_INDEX:
                self._reliability_floors[TASK_INDEX[task_name]] = float(floor)

    def capability_at(self, quarter: int) -> np.ndarray:
        """Return capability vector c(t) at given quarter, linearly interpolated."""
        c = np.zeros(N_TASKS)
        for task_name, frames in self._keyframes.items():
            idx = TASK_INDEX[task_name]
            c[idx] = _interpolate(frames, quarter)
        return c

    def reliability_floors(self) -> np.ndarray:
        """Return reliability floor vector r."""
        return self._reliability_floors.copy()

    def capability_index(self, quarter: int) -> float:
        """Scalar summary of capability (mean across tasks)."""
        return float(self.capability_at(quarter).mean())


def _interpolate(keyframes: list[tuple[int, float]], t: int) -> float:
    """Linearly interpolate (or extrapolate at edges) a list of (quarter, value) keyframes."""
    if not keyframes:
        return 0.0
    keyframes = sorted(keyframes, key=lambda x: x[0])
    if t <= keyframes[0][0]:
        return keyframes[0][1]
    if t >= keyframes[-1][0]:
        return keyframes[-1][1]
    for i in range(len(keyframes) - 1):
        q0, v0 = keyframes[i]
        q1, v1 = keyframes[i + 1]
        if q0 <= t <= q1:
            frac = (t - q0) / (q1 - q0)
            return v0 + frac * (v1 - v0)
    return keyframes[-1][1]
