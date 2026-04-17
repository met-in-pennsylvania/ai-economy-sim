"""Task categories and occupation-task weight matrix for Knowledge Work."""

import numpy as np

# 10 task categories (indices 0-9)
TASK_NAMES = [
    "routine_information_processing",   # 0
    "pattern_recognition_classification", # 1
    "structured_writing_communication",  # 2
    "creative_synthesis",               # 3
    "quantitative_analysis",            # 4
    "judgment_under_uncertainty",       # 5
    "interpersonal_relationship",       # 6
    "physical_presence",               # 7
    "novel_problem_solving",           # 8
    "coordination_orchestration",      # 9
]

N_TASKS = len(TASK_NAMES)
TASK_INDEX = {name: i for i, name in enumerate(TASK_NAMES)}

# Occupation buckets within Knowledge Work
KW_OCCUPATION_NAMES = [
    "routine_analytical",
    "creative_synthesis",
    "relational",
    "technical_specialist",
]
N_KW_OCCUPATIONS = len(KW_OCCUPATION_NAMES)
KW_OCCUPATION_INDEX = {name: i for i, name in enumerate(KW_OCCUPATION_NAMES)}

# Occupation-task weight matrix: shape (N_KW_OCCUPATIONS, N_TASKS)
# Rows = occupation, Columns = tasks
# DESIGN CHOICE: hand-tuned illustrative weights. Each row need not sum to 1;
# weights represent relative task intensity.
_WEIGHTS_RAW = np.array([
    #  ri_p  pat   sw    cr    qa    jud   ipr   phy   nps   coo
    [0.50, 0.30, 0.30, 0.10, 0.40, 0.15, 0.10, 0.00, 0.10, 0.15],  # routine_analytical
    [0.10, 0.20, 0.40, 0.60, 0.20, 0.20, 0.25, 0.00, 0.35, 0.20],  # creative_synthesis
    [0.10, 0.15, 0.35, 0.20, 0.10, 0.30, 0.70, 0.10, 0.15, 0.40],  # relational
    [0.20, 0.40, 0.20, 0.30, 0.60, 0.40, 0.15, 0.05, 0.60, 0.30],  # technical_specialist
], dtype=float)

# Normalize each row so weights sum to 1
ROW_SUMS = _WEIGHTS_RAW.sum(axis=1, keepdims=True)
OCCUPATION_TASK_WEIGHTS = _WEIGHTS_RAW / ROW_SUMS  # shape: (4, 10)


def compute_occupation_exposure(
    capability: np.ndarray,
    reliability_floors: np.ndarray,
) -> np.ndarray:
    """
    Compute AI exposure for each KW occupation.

    capability: shape (N_TASKS,), c_i in [0,1]
    reliability_floors: shape (N_TASKS,), r_i in [0,1]

    Returns: shape (N_KW_OCCUPATIONS,), each value in [0,1]
    """
    deployed = (capability >= reliability_floors).astype(float)  # I(c_i >= r_i)
    task_exposure = deployed * np.minimum(1.0, capability)       # per task: 0 or c_i
    # Weighted sum per occupation
    return OCCUPATION_TASK_WEIGHTS @ task_exposure               # (4, 10) @ (10,) -> (4,)
