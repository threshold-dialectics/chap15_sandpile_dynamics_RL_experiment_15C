#config.py
import numpy as np

# --- Main Simulation Parameters ---
GRID_SIZE = (30, 30)
INITIAL_K_TH = 4  # For sandpile model init, continuous beta_lever_p will be based on this
G_LEVER_P_INITIAL = 1.0 # For sandpile model init, continuous g_lever_p will be based on this
MAX_K_TH = 6
MIN_K_TH = 3
MAX_P_TOPPLE = 1.0
MIN_P_TOPPLE = 0.1


# --- TD Lever Parameters for Theta_T Calculation ---
W_G_P = 0.2 # Weight for g_lever_p
W_BETA_P = 0.3 # Weight for beta_lever_p (k_th)
W_FCRIT_P = 0.5 # Weight for f_crit_p
C_P_SCALE = 1.0 # System-specific constant for Theta_T
THETA_T_SCALING_FACTOR = 0.003 # Tuned to bring Theta_T into a reasonable range

# --- Sandpile System's Internal Adaptive Logic Parameters (from experiment 15B&C base) ---
# These will run unless overridden by RL agent's intervention
ADAPTIVE_BETA_LEVER_ENABLED = True
AVG_DELTA_P_LOWER_THRESHOLD_FOR_K_TH_ADAPT = 0.2 # System tries to increase k_th if strain is low
AVG_DELTA_P_UPPER_THRESHOLD_FOR_K_TH_ADAPT = 0.8 # System tries to decrease k_th if strain is high
K_TH_ADAPT_FREQUENCY = 1 # How often k_th adaptation logic runs
K_TH_ADAPT_RATE_CONTINUOUS = 0.1 # How much beta_lever_p_continuous changes

ADAPTIVE_G_LEVER_ENABLED = True
AVG_DELTA_P_LOWER_THRESHOLD_FOR_G_ADAPT = 0.15 # System tries to decrease p_topple if strain is low
AVG_DELTA_P_UPPER_THRESHOLD_FOR_G_ADAPT = 0.6  # System tries to increase p_topple if strain is high
G_LEVER_ADAPT_FREQUENCY = 1
G_LEVER_ADAPT_RATE = 0.05

# --- TD Diagnostic Calculation Parameters ---
STRAIN_AVG_WINDOW = 20
DERIVATIVE_WINDOW_LEN = 11 # For savgol filter
COUPLE_WINDOW_LEN = 25   # For Pearson correlation

# --- RL Environment Parameters ---
MAX_EPISODE_STEPS = 1000
BURN_IN_STEPS = 200 # Steps to run in reset() before agent starts
LARGE_AVALANCHE_THRESH = GRID_SIZE[0] * GRID_SIZE[1] * 0.05 # 5% of grid size

# Intervention Parameters (fixed for simplicity for now)
INTERVENTION_G_PULSE_TARGET_PTOPPLE = 0.95
INTERVENTION_G_PULSE_DURATION = 10
INTERVENTION_BETA_ADJUST_TARGET_KTH_REDUCTION = 1 # Reduce k_th by this much
INTERVENTION_BETA_ADJUST_DURATION = 15
INTERVENTION_FCRIT_BOLSTER_GRAINS_TO_REMOVE = int(0.01 * GRID_SIZE[0] * GRID_SIZE[1]) # e.g. 1% of total cells worth of grains
INTERVENTION_FCRIT_BOLSTER_CELLS_TO_TARGET = int(0.05 * GRID_SIZE[0] * GRID_SIZE[1]) # target 5% of cells

# Reward Function Weights
R_SURVIVAL_STEP = 0.01
R_LARGE_AVALANCHE_PENALTY = -10.0
R_SAFETY_MARGIN_FACTOR = 0.02 # Reward for (G_p - target_G_p)
R_TARGET_SAFETY_MARGIN = 0.1 # Target for G_p
R_THETA_T_BREACH_PENALTY = -0.5
R_DANGER_ZONE_PENALTY = -0.2 # Penalty per step in danger zone
# Danger Zone Definition (example, needs calibration)
DANGER_SPEED_THRESHOLD = 0.6 # Example: If Speed_p > this
DANGER_COUPLE_THRESHOLD_POSITIVE = 0.4 # Example: AND Couple_p > this (for detrimental positive coupling)
# Or a more complex danger zone logic might be needed

# Intervention Costs for Reward Function
COST_ACTION_BASE = -0.02 # Small cost for any intervention
COST_G_PULSE_ACTION = -0.05
COST_BETA_ADJUST_ACTION = -0.1
COST_FCRIT_BOLSTER_ACTION = -0.25


# Observation Space Normalization Bounds (approximate, need tuning)
# These are [min, max] pairs for each feature
OBS_BOUNDS = {
    'g_lever_p_topple_prob': np.array([MIN_P_TOPPLE, MAX_P_TOPPLE], dtype=np.float32),
    'beta_lever_p_continuous': np.array([MIN_K_TH, MAX_K_TH], dtype=np.float32),
    'actual_k_th': np.array([MIN_K_TH, MAX_K_TH], dtype=np.float32),
    'f_crit_p': np.array([0, GRID_SIZE[0]*GRID_SIZE[1]*MAX_K_TH * 1.2], dtype=np.float32), # Slack can be high
    'avg_delta_P_p': np.array([0, GRID_SIZE[0]*GRID_SIZE[1]*0.5], dtype=np.float32), # Strain can be high
    'theta_T_p': np.array([0, 10.0], dtype=np.float32), # Theta_T is scaled
    'safety_margin_p': np.array([-50.0, 10.0], dtype=np.float32), # Can be very negative
    'speed_p': np.array([0, 5.0], dtype=np.float32), # Derivatives can be large
    'couple_p': np.array([-1.0, 1.0], dtype=np.float32),
    'dot_fcrit_p': np.array([-GRID_SIZE[0]*GRID_SIZE[1]*0.5, GRID_SIZE[0]*GRID_SIZE[1]*0.5], dtype=np.float32),
    'dot_beta_p': np.array([-1.0, 1.0], dtype=np.float32), # Rate of change of k_th
    'dot_g_p': np.array([-0.5, 0.5], dtype=np.float32), # Rate of change of p_topple
    'num_unstable_pre_relax': np.array([0, GRID_SIZE[0]*GRID_SIZE[1]], dtype=np.float32),
    'last_avalanche_size': np.array([0, GRID_SIZE[0]*GRID_SIZE[1]], dtype=np.float32),
    'time_since_last_large_avalanche': np.array([0, MAX_EPISODE_STEPS], dtype=np.float32),
    'is_intervention_active': np.array([0, 1], dtype=np.float32), # Binary
    'active_intervention_type_g_pulse': np.array([0, 1], dtype=np.float32), # Binary
    'active_intervention_type_beta_adjust': np.array([0, 1], dtype=np.float32), # Binary
    'active_intervention_type_fcrit_bolster': np.array([0, 1], dtype=np.float32), # Binary
    'intervention_duration_remaining_norm': np.array([0, 1.0], dtype=np.float32) # Normalized
}
# Ordered list of observation keys (must match _calculate_observation order)
OBS_KEYS = [
    'g_lever_p_topple_prob', 'beta_lever_p_continuous', 'actual_k_th', 'f_crit_p',
    'avg_delta_P_p', 'theta_T_p', 'safety_margin_p', 'speed_p', 'couple_p',
    'dot_fcrit_p', 'dot_beta_p', 'dot_g_p', 'num_unstable_pre_relax',
    'last_avalanche_size', 'time_since_last_large_avalanche',
    'is_intervention_active', 'active_intervention_type_g_pulse',
    'active_intervention_type_beta_adjust', 'active_intervention_type_fcrit_bolster',
    'intervention_duration_remaining_norm'
]
N_OBS_FEATURES = len(OBS_KEYS)

# Parameters for the heuristic agent used in evaluation. These roughly
# follow TD principles and can be tweaked when running experiments.
HEURISTIC_AGENT_PARAMS = {
    'SAFETY_MARGIN_THRESHOLD': 0.05,  # Intervene if G_p drops below this
    'SPEED_THRESHOLD': 0.7,           # Consider intervention if speed_p exceeds this
    'COUPLE_THRESHOLD_POSITIVE': 0.5, # and couple_p is above this
    'INTERVENTION_COOLDOWN': 15       # Minimum steps between heuristic interventions
}

# Max duration for any pulse intervention, used for normalizing remaining duration
MAX_PULSE_DURATION = max(INTERVENTION_G_PULSE_DURATION, INTERVENTION_BETA_ADJUST_DURATION)