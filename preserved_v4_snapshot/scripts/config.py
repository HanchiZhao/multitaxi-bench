"""Central configuration for MultiTaxi-Bench dynamic two-hour operations.

The model is deliberately explicit about assumptions. TLC trip records observe successful
pickups and dropoffs, but do not directly observe the number of vacant taxis competing for
orders. The code therefore reports a calibrated latent competition proxy and low/medium/high
sensitivity scenarios rather than claiming exact idle-fleet counts.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = PROJECT_ROOT / "processed_data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

MONTHS = ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06"]
MONTH_TAG = f"{MONTHS[0]}_{MONTHS[-1]}"
ZONE_SHP = DATA_DIR / "taxi_zones.shp"
ZONE_LOOKUP = DATA_DIR / "taxi_zone_lookup.csv"

# Historical data resolution and decision-model resolution are separate. Historical demand
# and OD statistics use 15-minute bins. The semi-Markov decision model uses a 5-minute grid
# and interpolates values when a trip/repositioning action ends between grid points.
TIME_BIN_MINUTES = 15
N_TIME_BINS = 24 * 60 // TIME_BIN_MINUTES
DECISION_TIME_STEP_MINUTES = 5
WAIT_REEVALUATION_MINUTES = 5
HORIZON_MINUTES = 120
HORIZON_STEPS = HORIZON_MINUTES // DECISION_TIME_STEP_MINUTES

# Trip cleaning.
MIN_FARE = 2.50
MAX_FARE = 500.0
MIN_DISTANCE_MILES = 0.01
MAX_DISTANCE_MILES = 100.0
MIN_DURATION_MIN = 1.0
MAX_DURATION_MIN = 240.0

# Revenue / operating-cost assumptions. These are model parameters, not accounting claims.
DRIVER_REVENUE_SHARE = 1.00
OCCUPIED_COST_PER_MILE = 0.35
EMPTY_COST_PER_MILE = 0.35

# Passenger-demand / latent vacant-taxi competition model.
REFERENCE_WAIT_MINUTES = 12.0
MIN_EXPECTED_WAIT_MINUTES = 1.0
MAX_EXPECTED_WAIT_MINUTES = 75.0
VACANT_STOCK_PERSISTENCE = 0.72
VACANT_HISTORY_BINS = 8
VACANT_DEMAND_ATTRACTION = 0.45
VACANT_PICKUP_DEPLETION = 0.72
NEIGHBOR_SUPPLY_DIFFUSION = 0.22
SUPPLY_BASE_FLOOR = 0.35
COMPETITION_SCENARIO_MULTIPLIERS = {"low": 0.70, "medium": 1.00, "high": 1.35}
DEFAULT_COMPETITION_SCENARIO = "medium"
AIRPORT_QUEUE_MULTIPLIER = 1.25

# Spatial / routing assumptions.
FALLBACK_SPEED_MPH = 15.0
MIN_SPEED_MPH = 5.0
MAX_SPEED_MPH = 35.0
MAX_REPOSITION_MINUTES = 45.0
MAX_REPOSITION_MILES = 25.0
ADJACENCY_BUFFER_FEET = 250.0

# Dynamic OD model. Passenger simulation stores a compact probability support for each
# origin/time cell. estimate_od() still supports every valid origin-destination-time query
# through observed, temporal, global, spatial and gravity fallbacks.
MAX_DESTINATIONS_PER_ORIGIN = 20
OD_DIRICHLET_STRENGTH = 8.0
OD_OBS_PRIOR_STRENGTH = 10.0
NODE_PRIOR_STRENGTH = 12.0
TEMPORAL_WINDOW_BINS = 4
TEMPORAL_DECAY_BINS = 1.5
SPATIAL_NEIGHBOR_COUNT = 4
SPATIAL_DECAY_MILES = 2.0
GLOBAL_OD_DISCOUNT = 0.35
GRAVITY_PRIOR_STRENGTH = 2.0
MIN_CONFIDENCE = 0.05
FULL_OD_CACHE_MAX_SIZE = 200_000

# Candidate-action / Shapley defaults.
POLICY_CANDIDATE_ZONES = 12
DYNAMIC_CANDIDATE_CATEGORY_QUOTA = 4
SHAPLEY_CANDIDATE_ZONES = 10
SHAPLEY_POOL_MULTIPLIER = 4
SHAPLEY_EXACT_MAX_PLAYERS = 10
SHAPLEY_PERMUTATIONS = 512
TOP_K_FOR_REPORTING = 10

# Simulation / learning defaults.
SCENARIO_COUNT = 300
RANDOM_SEED = 20260712
Q_LEARNING_EPISODES = 10_000
Q_LEARNING_ALPHA = 0.12
Q_LEARNING_GAMMA = 0.985
Q_LEARNING_EPSILON_START = 1.0
Q_LEARNING_EPSILON_END = 0.03
Q_LEARNING_FALLBACK = "greedy_net_earnings"
DQN_EPISODES = 8_000
DQN_DYNAMIC_CANDIDATES = POLICY_CANDIDATE_ZONES
DQN_REPLAY_CAPACITY = 120_000
DQN_REPLAY_WARMUP = 1_000
DQN_BATCH_SIZE = 128
DQN_TARGET_UPDATE_STEPS = 500
DQN_VALIDATION_EPISODES = 60
DQN_VALIDATION_INTERVAL = 500
DQN_LEARNING_RATE = 5e-4
DQN_TRAIN_EVERY_STEPS = 2

# Duration-tail calibration. Sparse OD cells are estimated on a log-duration scale and then
# corrected with time-of-day, distance-band, borough-pair and airport factors learned from
# observed trips. This reduces the previous tendency to shrink long trips too aggressively.
DURATION_DISTANCE_BINS_MILES = (0.0, 1.0, 2.0, 4.0, 7.0, 12.0, 20.0, 100.0)
DURATION_CALIBRATION_MIN_COUNT = 50
DURATION_CALIBRATION_SHRINKAGE = 150.0
DURATION_MIN_TAIL_MULTIPLIER = 0.70
DURATION_MAX_TAIL_MULTIPLIER = 1.80

# Validation.
LOW_CONFIDENCE_THRESHOLD = 0.35
AXIOM_TOLERANCE = 1e-7
DIM_REVENUE_INCREASE = 0.05
DME_REVENUE_STEPS = (0.05, 0.10, 0.15)
OD_COVERAGE_TEST_PAIRS = 5_000
PAIRED_COMPARISON_BASELINE = "wait_only"
REPRESENTATIVE_EARNINGS_TOLERANCE_SD = 0.35


@dataclass(frozen=True)
class RunConfig:
    start_zone: int
    start_time: str = "08:00"
    horizon_minutes: int = HORIZON_MINUTES
    scenario_count: int = SCENARIO_COUNT
    random_seed: int = RANDOM_SEED
    competition_scenario: str = DEFAULT_COMPETITION_SCENARIO


def ensure_directories() -> None:
    for path in (PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)
