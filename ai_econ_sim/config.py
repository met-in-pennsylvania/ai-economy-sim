"""Default constants and model parameters."""

from dataclasses import dataclass, field

SECTORS = ["ai_compute", "knowledge_work", "services", "goods", "infrastructure"]

# Default firm counts per sector (sums to ~1000)
FIRM_COUNTS = {
    "ai_compute": 30,
    "knowledge_work": 280,
    "services": 350,
    "goods": 250,
    "infrastructure": 90,
}

# Default worker counts per sector (sums to ~10000)
WORKER_COUNTS = {
    "ai_compute": 200,
    "knowledge_work": 2500,
    "services": 3500,
    "goods": 2500,
    "infrastructure": 1300,
}

# Knowledge-work occupation buckets
KW_OCCUPATIONS = [
    "routine_analytical",
    "creative_synthesis",
    "relational",
    "technical_specialist",
]

# Development population scale (set to 1.0 for full, 0.1 for dev mode)
DEFAULT_POPULATION_SCALE = 1.0
DEV_POPULATION_SCALE = 0.1

# Quarterly time step
QUARTERS_PER_YEAR = 4

# Firm size tiers: (label, min_workers, max_workers)
SIZE_TIERS = [
    ("micro", 1, 5),
    ("small", 6, 50),
    ("medium", 51, 500),
    ("large", 501, 5000),
]

# Power-law exponent for firm size distribution
FIRM_SIZE_PARETO_ALPHA = 1.5

# Base sector growth rates (annual, before AI)
BASE_GROWTH_RATES = {
    "ai_compute": 0.08,
    "knowledge_work": 0.02,
    "services": 0.02,
    "goods": 0.01,
    "infrastructure": 0.01,
}

# AI spillover coefficient for non-knowledge-work sectors
AI_SPILLOVER = {
    "ai_compute": 0.0,   # modeled separately
    "knowledge_work": 0.0,  # task-level model
    "services": 0.05,
    "goods": 0.10,
    "infrastructure": 0.03,
}

# Compute depreciation rate (annual)
COMPUTE_DEPRECIATION_ANNUAL = 0.25

# Skill level range
SKILL_MIN = 1
SKILL_MAX = 5

# Adaptive expectations EMA half-life (quarters)
EXPECTATION_HALFLIFE_QUARTERS = 4

# Job search parameters
JOB_OFFER_PROBABILITY_BASE = 0.3  # probability unemployed worker gets offer per quarter
OUTSIDE_OFFER_PROBABILITY = 0.05  # probability employed worker gets outside offer
WAGE_ACCEPTANCE_THRESHOLD = 0.05  # accept outside offer if wage gain > 5%

# Retraining parameters
RETRAINING_MIN_QUARTERS = 4
RETRAINING_MAX_QUARTERS = 8
RETRAINING_BASE_SUCCESS_RATE = 0.6
RETRAINING_AGE_PENALTY = 0.01  # per year above 40
RETRAINING_SKILL_BONUS = 0.05  # per skill level above 2

# Labor force participation
LFP_EXIT_THRESHOLD_QUARTERS = 8  # unemployed this long → consider exit
LFP_EXIT_PROBABILITY = 0.1  # quarterly probability of exiting LF
LFP_REENTRY_BASE = 0.05  # quarterly reentry probability when conditions improve

# AI adoption parameters
ADOPTION_BASE_RATE = 0.05  # maximum quarterly adoption delta
ADOPTION_FIRM_SIZE_FRICTION = {
    "micro": 0.8,   # small firms adopt faster (less bureaucracy)
    "small": 0.9,
    "medium": 1.0,
    "large": 0.7,   # DESIGN CHOICE: large firms face organizational friction
}
ADOPTION_PEER_WEIGHT = 0.3  # weight on peer adoption in adoption decision

# Pricing parameters
MARKUP_BASE = 0.15
MARKUP_ADJUSTMENT_SPEED = 0.1  # fraction of gap closed per quarter

# Hiring friction
HIRING_SPEED = 0.3  # fraction of desired adjustment made per quarter

# Wage setting
WAGE_BASE_BY_SECTOR = {
    "ai_compute": 120_000,
    "knowledge_work": 90_000,
    "services": 45_000,
    "goods": 55_000,
    "infrastructure": 65_000,
}
WAGE_SKILL_PREMIUM = 0.15  # per skill level above 1 (annual)
WAGE_IDIOSYNCRATIC_STD = 0.05  # noise std on log wages

# Tax parameters
CORPORATE_TAX_RATE = 0.21
PAYROLL_TAX_RATE = 0.15
INCOME_TAX_RATE = 0.22

# Capital income concentration for AI sector
AI_CAPITAL_CONCENTRATION = 0.8  # fraction of AI capital held by top 10%

# Logging
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
LOG_DATEFMT = "%H:%M:%S"
