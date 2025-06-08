#sandpile_rl_env_experiment_15C.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
import random

# Assuming td_core_extended.py and config_experiment_15C.py are in the same directory or accessible
from td_core_extended import SandpileBTW, calculate_raw_instantaneous_strain_p, \
                             calculate_energetic_slack_p, calculate_tolerance_sheet_p, \
                             calculate_derivatives_savgol
import config as cfg

class SandpileInterventionRLEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, config_dict=None):
        super().__init__()

        # Load config (can be overridden by passing config_dict)
        self.cfg = cfg # Default to module
        if config_dict:
            # Simple way to update:
            for key, value in config_dict.items():
                setattr(self.cfg, key, value)
        
        self.sandpile_model = SandpileBTW(
            grid_size=self.cfg.GRID_SIZE,
            k_th=self.cfg.INITIAL_K_TH, # Initial k_th
            p_topple=self.cfg.G_LEVER_P_INITIAL # Initial p_topple
        )

        # Observation space
        lows = np.array([self.cfg.OBS_BOUNDS[key][0] for key in self.cfg.OBS_KEYS], dtype=np.float32)
        highs = np.array([self.cfg.OBS_BOUNDS[key][1] for key in self.cfg.OBS_KEYS], dtype=np.float32)
        self.observation_space = spaces.Box(low=lows, high=highs, dtype=np.float32)

        # Action space: 0: Do Nothing, 1: Pulse G, 2: Adjust Beta, 3: Bolster Fcrit
        self.action_space = spaces.Discrete(4)

        # TD lever states
        self.beta_lever_p_continuous = float(self.cfg.INITIAL_K_TH)
        self.current_p_topple = self.cfg.G_LEVER_P_INITIAL
        self.sandpile_model.k_th = int(round(self.beta_lever_p_continuous))
        self.sandpile_model.p_topple = self.current_p_topple
        
        # Histories for diagnostics
        self.simulation_log_hist = deque(maxlen=max(self.cfg.DERIVATIVE_WINDOW_LEN, self.cfg.COUPLE_WINDOW_LEN) + 5)
        self.unstable_history_for_strain_calc = deque(maxlen=self.cfg.STRAIN_AVG_WINDOW)

        # Intervention state
        self.active_intervention_type = 0 # 0:None, 1:G, 2:Beta, 3:Fcrit
        self.intervention_steps_remaining = 0
        self.original_p_topple_before_pulse = self.current_p_topple
        self.original_beta_target_before_adjust = self.beta_lever_p_continuous

        # Episode state
        self.current_step = 0
        self.time_since_last_large_avalanche = 0
        self.last_avalanche_size_metric = 0 # For observation

        # Ensure OBS_KEYS order is fixed for constructing observation
        if len(self.cfg.OBS_KEYS) != self.cfg.N_OBS_FEATURES:
            raise ValueError("OBS_KEYS length mismatch with N_OBS_FEATURES")


    def _normalize_obs(self, obs_dict):
        normalized_obs = np.zeros(self.cfg.N_OBS_FEATURES, dtype=np.float32)
        for i, key in enumerate(self.cfg.OBS_KEYS):
            val = obs_dict.get(key, 0.0) # Default to 0.0 if a key is missing (should not happen)
            min_val, max_val = self.cfg.OBS_BOUNDS[key]
            if max_val == min_val: # Avoid division by zero if bounds are the same (e.g. for binary flags already 0-1)
                 normalized_obs[i] = np.clip(val, min_val, max_val) if min_val == 0 and max_val == 1 else 0.0
            else:
                normalized_obs[i] = 2 * (val - min_val) / (max_val - min_val) - 1 # Normalize to [-1, 1]
            normalized_obs[i] = np.clip(normalized_obs[i], -1.0, 1.0)
        return normalized_obs

    def _calculate_observation(self, current_metrics_dict):
        # This dict should contain all raw values needed for OBS_KEYS
        obs_raw = {key: current_metrics_dict.get(key, np.nan) for key in self.cfg.OBS_KEYS} # Use NaN for missing

        # Fill in intervention-specific observation parts
        obs_raw['is_intervention_active'] = 1.0 if self.active_intervention_type != 0 else 0.0
        obs_raw['active_intervention_type_g_pulse'] = 1.0 if self.active_intervention_type == 1 else 0.0
        obs_raw['active_intervention_type_beta_adjust'] = 1.0 if self.active_intervention_type == 2 else 0.0
        obs_raw['active_intervention_type_fcrit_bolster'] = 1.0 if self.active_intervention_type == 3 else 0.0
        
        if self.cfg.MAX_PULSE_DURATION > 0:
             obs_raw['intervention_duration_remaining_norm'] = float(self.intervention_steps_remaining) / self.cfg.MAX_PULSE_DURATION
        else:
             obs_raw['intervention_duration_remaining_norm'] = 0.0
        obs_raw['intervention_duration_remaining_norm'] = np.clip(obs_raw['intervention_duration_remaining_norm'],0.0,1.0)

        # Fill other observation parts that might not be in current_metrics_dict directly
        obs_raw['g_lever_p_topple_prob'] = self.current_p_topple
        obs_raw['beta_lever_p_continuous'] = self.beta_lever_p_continuous
        obs_raw['actual_k_th'] = self.sandpile_model.k_th
        obs_raw['last_avalanche_size'] = self.last_avalanche_size_metric
        obs_raw['time_since_last_large_avalanche'] = self.time_since_last_large_avalanche
        
        # Handle NaNs from derivatives at the start of episode by replacing with 0
        for key in ['speed_p', 'couple_p', 'dot_fcrit_p', 'dot_beta_p', 'dot_g_p']:
            if np.isnan(obs_raw.get(key, 0.0)):
                obs_raw[key] = 0.0
        
        return self._normalize_obs(obs_raw)

    def _get_current_metrics(self, avalanche_size_current_event, num_unstable_pre_relax_val):
        # This function consolidates metric calculation, similar to the main loop in Study 1/2
        # It assumes sandpile levers (k_th, p_topple) are already set for the current step (either by internal logic or RL intervention)
        k_th_current = self.sandpile_model.k_th
        
        f_crit_p_val = calculate_energetic_slack_p(self.sandpile_model.grid, k_th_current)
        
        # Strain calculation uses history
        raw_strain_this_step = calculate_raw_instantaneous_strain_p(self.sandpile_model.grid, k_th_current) # Pre-relax strain
        # Note: The prompt has strain calculated *before* grain add and relax sometimes.
        # For RL state, strain *after* relax but *before* next grain might be more informative of current stability.
        # Let's assume avg_delta_P_p represents the "pressure" just before the decision to relax.
        # The Study 1/2 code calculates raw_strain_this_step *after* add_grain, *before* relax.
        # Then appends to unstable_history_for_strain_calc. Let's follow that.
        # For simplicity, let's assume the strain history for avg_delta_P_p is based on pre-relaxation states.
        # This part needs to be consistent with how avg_delta_P_p is used in internal adaptive logic.
        # The existing code adds raw_strain_this_step (calculated AFTER add_grain, BEFORE relax) to history.
        self.unstable_history_for_strain_calc.append(
             calculate_raw_instantaneous_strain_p(self.sandpile_model.grid, k_th_current) # Strain on the current grid
        )
        avg_delta_P_p_val = np.mean(list(self.unstable_history_for_strain_calc)) if self.unstable_history_for_strain_calc else 0.0

        current_metrics = {
            'time_step': self.current_step, # Env step, not sandpile global time_step_counter
            'g_lever_p_topple_prob': self.current_p_topple,
            'beta_lever_p_continuous': self.beta_lever_p_continuous,
            'actual_k_th': k_th_current,
            'f_crit_p': f_crit_p_val,
            'avg_delta_P_p': avg_delta_P_p_val,
            'avalanche_size': avalanche_size_current_event,
            'num_unstable_pre_relax': num_unstable_pre_relax_val,
            # Costs are handled by reward function based on action
        }
        current_metrics['theta_T_p'] = calculate_tolerance_sheet_p(
            current_metrics['g_lever_p_topple_prob'],
            current_metrics['beta_lever_p_continuous'],
            current_metrics['f_crit_p'],
            self.cfg.W_G_P, self.cfg.W_BETA_P, self.cfg.W_FCRIT_P,
            self.cfg.C_P_SCALE, self.cfg.THETA_T_SCALING_FACTOR
        )
        current_metrics['safety_margin_p'] = current_metrics['theta_T_p'] - current_metrics['avg_delta_P_p']

        # Add current metrics to history for derivative calculation
        # Only add specific keys needed for derivatives to avoid excessive memory
        log_for_deriv = {
            'f_crit_p': current_metrics['f_crit_p'],
            'beta_lever_p': current_metrics['beta_lever_p_continuous'], # Use continuous for smooth derivatives
            'g_lever_p_topple_prob': current_metrics['g_lever_p_topple_prob']
        }
        self.simulation_log_hist.append(log_for_deriv)
        
        # Calculate derivatives (Speed/Couple)
        if len(self.simulation_log_hist) >= self.cfg.DERIVATIVE_WINDOW_LEN:
            f_crit_hist = np.array([log['f_crit_p'] for log in self.simulation_log_hist])
            beta_hist = np.array([log['beta_lever_p'] for log in self.simulation_log_hist])
            g_hist = np.array([log['g_lever_p_topple_prob'] for log in self.simulation_log_hist])

            dot_f_crit_p_full = calculate_derivatives_savgol(f_crit_hist, self.cfg.DERIVATIVE_WINDOW_LEN)
            dot_beta_p_full = calculate_derivatives_savgol(beta_hist, self.cfg.DERIVATIVE_WINDOW_LEN)
            dot_g_p_full = calculate_derivatives_savgol(g_hist, self.cfg.DERIVATIVE_WINDOW_LEN)

            current_metrics['dot_fcrit_p'] = dot_f_crit_p_full[-1] if not np.all(np.isnan(dot_f_crit_p_full)) else np.nan
            current_metrics['dot_beta_p'] = dot_beta_p_full[-1] if not np.all(np.isnan(dot_beta_p_full)) else np.nan
            current_metrics['dot_g_p'] = dot_g_p_full[-1] if not np.all(np.isnan(dot_g_p_full)) else np.nan
            
            if not np.isnan(current_metrics['dot_fcrit_p']) and not np.isnan(current_metrics['dot_beta_p']):
                current_metrics['speed_p'] = np.sqrt(current_metrics['dot_fcrit_p']**2 + current_metrics['dot_beta_p']**2)
            else:
                current_metrics['speed_p'] = np.nan

            # Update history in self.simulation_log_hist with these calculated derivatives for couple calculation
            # This ensures past logs used for couple_p have their derivatives if they were calculable
            self.simulation_log_hist[-1]['dot_fcrit_p'] = current_metrics['dot_fcrit_p']
            self.simulation_log_hist[-1]['dot_beta_p'] = current_metrics['dot_beta_p']


            if len(self.simulation_log_hist) >= self.cfg.COUPLE_WINDOW_LEN:
                # Use the updated history that now includes derivatives
                # Ensure we take the last COUPLE_WINDOW_LEN elements from the deque
                hist_list_for_couple = list(self.simulation_log_hist)[-self.cfg.COUPLE_WINDOW_LEN:]
                
                dot_f_crit_past = np.array([log.get('dot_fcrit_p', np.nan) for log in hist_list_for_couple])
                dot_beta_past = np.array([log.get('dot_beta_p', np.nan) for log in hist_list_for_couple])

                valid_mask = ~np.isnan(dot_f_crit_past) & ~np.isnan(dot_beta_past)
                segment_f = dot_f_crit_past[valid_mask]
                segment_beta = dot_beta_past[valid_mask]

                # Ensure enough valid points for correlation
                if len(segment_f) >= max(2, self.cfg.COUPLE_WINDOW_LEN * 0.8): # Need at least 2 points for correlation
                    # Check for variance before calculating correlation
                    if np.var(segment_f) > 1e-9 and np.var(segment_beta) > 1e-9:
                        # Using np.corrcoef for robustness with potentially constant inputs
                        with np.errstate(divide='ignore', invalid='ignore'): # Suppress warnings for constant inputs
                            corr_matrix = np.corrcoef(segment_f, segment_beta)
                        
                        # np.corrcoef returns a 2x2 matrix if inputs are 1D arrays.
                        # If inputs are identical or perfectly anti-correlated, or one is constant, it can be tricky.
                        if isinstance(corr_matrix, np.ndarray) and corr_matrix.ndim == 2:
                            current_metrics['couple_p'] = corr_matrix[0, 1]
                        elif isinstance(corr_matrix, (float, np.float_)): # if it returned a scalar NaN due to issues
                             current_metrics['couple_p'] = 0.0 if np.isnan(corr_matrix) else corr_matrix
                        else: # Fallback if structure is unexpected
                            current_metrics['couple_p'] = 0.0
                    elif np.var(segment_f) < 1e-9 and np.var(segment_beta) < 1e-9 : 
                        # If both have no variance, they are "perfectly" (though trivially) correlated
                        current_metrics['couple_p'] = 0.0 # Or 1.0, but 0.0 is safer
                    else: # If only one has no variance, correlation is undefined or 0
                        current_metrics['couple_p'] = 0.0 
                else: # Not enough valid points
                    current_metrics['couple_p'] = np.nan # Or 0.0, depending on preference for downstream
            else: # Not enough history for COUPLE_WINDOW_LEN
                current_metrics['couple_p'] = np.nan
        else: # Not enough history for DERIVATIVE_WINDOW_LEN
            current_metrics['dot_fcrit_p'] = np.nan
            current_metrics['dot_beta_p'] = np.nan
            current_metrics['dot_g_p'] = np.nan
            current_metrics['speed_p'] = np.nan
            current_metrics['couple_p'] = np.nan
        
        # Ensure couple_p is not NaN for the observation, default to 0.0
        if np.isnan(current_metrics.get('couple_p', 0.0)):
            current_metrics['couple_p'] = 0.0
            
        return current_metrics

    def _apply_rl_intervention(self, action):
        intervention_cost_component = 0.0
        self.active_intervention_type = 0 # Reset first

        if action == 1: # Pulse G
            self.active_intervention_type = 1
            self.intervention_steps_remaining = self.cfg.INTERVENTION_G_PULSE_DURATION
            self.original_p_topple_before_pulse = self.current_p_topple # Store for reversion
            self.current_p_topple = self.cfg.INTERVENTION_G_PULSE_TARGET_PTOPPLE
            self.sandpile_model.p_topple = self.current_p_topple
            intervention_cost_component = self.cfg.COST_G_PULSE_ACTION
        elif action == 2: # Adjust Beta
            self.active_intervention_type = 2
            self.intervention_steps_remaining = self.cfg.INTERVENTION_BETA_ADJUST_DURATION
            self.original_beta_target_before_adjust = self.beta_lever_p_continuous # Store for reversion
            # Target a k_th that is lower by the reduction amount
            target_k_th_reduced = max(self.cfg.MIN_K_TH, int(round(self.beta_lever_p_continuous)) - self.cfg.INTERVENTION_BETA_ADJUST_TARGET_KTH_REDUCTION)
            self.beta_lever_p_continuous = float(target_k_th_reduced) # RL directly sets beta_lever_p
            self.sandpile_model.k_th = int(round(self.beta_lever_p_continuous)) # Update sandpile
            intervention_cost_component = self.cfg.COST_BETA_ADJUST_ACTION
        elif action == 3: # Bolster Fcrit
            self.active_intervention_type = 3
            self.sandpile_model.bolster_capacity(
                self.cfg.INTERVENTION_FCRIT_BOLSTER_GRAINS_TO_REMOVE,
                self.cfg.INTERVENTION_FCRIT_BOLSTER_CELLS_TO_TARGET
            )
            self.intervention_steps_remaining = 0 # Instantaneous
            intervention_cost_component = self.cfg.COST_FCRIT_BOLSTER_ACTION
        
        if action > 0 : # If any intervention was taken
             intervention_cost_component += self.cfg.COST_ACTION_BASE
        
        return intervention_cost_component

    def _manage_intervention_duration_and_revert(self):
        if self.intervention_steps_remaining > 0:
            self.intervention_steps_remaining -= 1
            if self.intervention_steps_remaining == 0:
                # Revert the intervention
                if self.active_intervention_type == 1: # G pulse ended
                    self.current_p_topple = self.original_p_topple_before_pulse 
                    # Internal adaptive logic for p_topple will take over next step
                elif self.active_intervention_type == 2: # Beta adjust ended
                    self.beta_lever_p_continuous = self.original_beta_target_before_adjust
                    # Internal adaptive logic for beta_lever_p will take over next step
                self.active_intervention_type = 0 # Reset

    def _apply_internal_adaptive_logic(self, avg_delta_P_p_val, is_large_avalanche_flag):
        # --- Internal K_th (Beta Lever) Adaptation (from Study 2/3) ---
        if self.cfg.ADAPTIVE_BETA_LEVER_ENABLED and self.active_intervention_type != 2: # Don't adapt if RL is controlling beta
            if self.current_step % self.cfg.K_TH_ADAPT_FREQUENCY == 0:
                current_k_th_for_sandpile = int(round(self.beta_lever_p_continuous))
                # Check if grid mean is high enough to justify k_th reduction for slack (similar to Study1 logic)
                grid_mean_density_factor = self.sandpile_model.grid.mean() / (current_k_th_for_sandpile if current_k_th_for_sandpile > 0 else 1)

                if avg_delta_P_p_val < self.cfg.AVG_DELTA_P_LOWER_THRESHOLD_FOR_K_TH_ADAPT and grid_mean_density_factor > 0.5 : # If strain low, consider increasing k_th
                    self.beta_lever_p_continuous = min(float(self.cfg.MAX_K_TH), self.beta_lever_p_continuous + self.cfg.K_TH_ADAPT_RATE_CONTINUOUS)
                elif avg_delta_P_p_val > self.cfg.AVG_DELTA_P_UPPER_THRESHOLD_FOR_K_TH_ADAPT or is_large_avalanche_flag: # If strain high or large avalanche, decrease k_th
                    self.beta_lever_p_continuous = max(float(self.cfg.MIN_K_TH), self.beta_lever_p_continuous - self.cfg.K_TH_ADAPT_RATE_CONTINUOUS)
        
        self.sandpile_model.k_th = np.clip(int(round(self.beta_lever_p_continuous)), self.cfg.MIN_K_TH, self.cfg.MAX_K_TH)

        # --- Internal P_topple (G Lever) Adaptation (from Study 2/3) ---
        if self.cfg.ADAPTIVE_G_LEVER_ENABLED and self.active_intervention_type != 1: # Don't adapt if RL is controlling G
            if self.current_step % self.cfg.G_LEVER_ADAPT_FREQUENCY == 0:
                if avg_delta_P_p_val < self.cfg.AVG_DELTA_P_LOWER_THRESHOLD_FOR_G_ADAPT:
                    self.current_p_topple = max(self.cfg.MIN_P_TOPPLE, self.current_p_topple - self.cfg.G_LEVER_ADAPT_RATE)
                elif avg_delta_P_p_val > self.cfg.AVG_DELTA_P_UPPER_THRESHOLD_FOR_G_ADAPT:
                    self.current_p_topple = min(self.cfg.MAX_P_TOPPLE, self.current_p_topple + self.cfg.G_LEVER_ADAPT_RATE)
        
        self.sandpile_model.p_topple = np.clip(self.current_p_topple, self.cfg.MIN_P_TOPPLE, self.cfg.MAX_P_TOPPLE)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Important for reproducibility
        
        self.sandpile_model = SandpileBTW(grid_size=self.cfg.GRID_SIZE, k_th=self.cfg.INITIAL_K_TH, p_topple=self.cfg.G_LEVER_P_INITIAL)
        self.beta_lever_p_continuous = float(self.cfg.INITIAL_K_TH)
        self.current_p_topple = self.cfg.G_LEVER_P_INITIAL
        self.sandpile_model.k_th = int(round(self.beta_lever_p_continuous))
        self.sandpile_model.p_topple = self.current_p_topple

        self.simulation_log_hist.clear()
        self.unstable_history_for_strain_calc.clear()
        
        self.active_intervention_type = 0
        self.intervention_steps_remaining = 0
        self.original_p_topple_before_pulse = self.current_p_topple
        self.original_beta_target_before_adjust = self.beta_lever_p_continuous

        self.current_step = 0
        self.time_since_last_large_avalanche = 0
        self.last_avalanche_size_metric = 0

        # Burn-in phase
        for _ in range(self.cfg.BURN_IN_STEPS):
            # During burn-in, use internal adaptive logic only
            # Calculate strain based on current grid BEFORE adding grain for adaptive logic consistency
            current_grid_strain_for_adapt = calculate_raw_instantaneous_strain_p(self.sandpile_model.grid, self.sandpile_model.k_th)
            self.unstable_history_for_strain_calc.append(current_grid_strain_for_adapt)
            avg_delta_P_p_burn_in = np.mean(list(self.unstable_history_for_strain_calc)) if self.unstable_history_for_strain_calc else 0.0
            
            self._apply_internal_adaptive_logic(avg_delta_P_p_burn_in, False) # No large avalanches in burn-in for adapt logic
            
            self.sandpile_model.add_grain()
            _ = self.sandpile_model.topple_and_relax()
            
            # Add to simulation_log_hist for derivative warmup if needed, or just fill with NaNs later
            # For simplicity, we'll just let the first few RL steps have NaN derivatives
            # This is similar to how Study 1/2 handles it.
            # We do need to log some initial values though for the deque to fill.
            log_entry = {
                'f_crit_p': calculate_energetic_slack_p(self.sandpile_model.grid, self.sandpile_model.k_th),
                'beta_lever_p': self.beta_lever_p_continuous,
                'g_lever_p_topple_prob': self.current_p_topple
            }
            self.simulation_log_hist.append(log_entry)


        # Calculate initial observation after burn-in
        # For the very first observation, we need placeholder metrics
        initial_metrics_dict = {
            'avg_delta_P_p': avg_delta_P_p_burn_in if self.unstable_history_for_strain_calc else 0.0,
            'f_crit_p': calculate_energetic_slack_p(self.sandpile_model.grid, self.sandpile_model.k_th),
            'theta_T_p': 0.0, # Will be calculated properly in _get_current_metrics
            'safety_margin_p': 0.0,
            'speed_p': np.nan, 'couple_p': np.nan, 'dot_fcrit_p': np.nan,
            'dot_beta_p': np.nan, 'dot_g_p': np.nan,
            'num_unstable_pre_relax': 0, # Placeholder for first obs
        }
        # Now get fuller metrics to populate observation correctly
        current_k_th = self.sandpile_model.k_th
        num_unstable_init = np.sum(self.sandpile_model.grid >= current_k_th)
        full_initial_metrics = self._get_current_metrics(0, num_unstable_init) # avalanche=0, num_unstable

        observation = self._calculate_observation(full_initial_metrics)
        return observation, {}

    def step(self, action):
        self.current_step += 1
        self.time_since_last_large_avalanche +=1
        reward = 0.0

        # 1. Manage ongoing RL intervention duration and potential reversion
        self._manage_intervention_duration_and_revert()

        # 2. Apply new RL action (intervention) if chosen
        intervention_cost_component = self._apply_rl_intervention(action)
        reward += intervention_cost_component # Add cost of taking action

        # 3. Apply sandpile's internal adaptive logic for levers
        #    (This uses current system state. If RL intervention is active for a lever,
        #     that lever's internal adaptation is skipped for this step by _apply_internal_adaptive_logic)
        #    Need avg_delta_P_p *before* this step's grain add for internal logic
        avg_delta_P_p_for_internal_adapt = np.mean(list(self.unstable_history_for_strain_calc)) if self.unstable_history_for_strain_calc else 0.0
        is_large_avalanche_prev_step = self.last_avalanche_size_metric > self.cfg.LARGE_AVALANCHE_THRESH # From *previous* step
        
        self._apply_internal_adaptive_logic(avg_delta_P_p_for_internal_adapt, is_large_avalanche_prev_step)
        # Sandpile k_th and p_topple are now set based on either RL intervention or internal logic

        # 4. Simulate Sandpile Step
        self.sandpile_model.add_grain()
        
        # Calculate pre-relaxation unstable count based on current k_th
        k_th_for_current_step = self.sandpile_model.k_th
        num_unstable_pre_relax_val = np.sum(self.sandpile_model.grid >= k_th_for_current_step)

        # Calculate strain for avg_delta_P_p *before* relaxation based on current grid
        # This strain is what drives system adaptation and is part of observation.
        current_raw_strain = calculate_raw_instantaneous_strain_p(self.sandpile_model.grid, k_th_for_current_step)
        self.unstable_history_for_strain_calc.append(current_raw_strain)
        # avg_delta_P_p for observation will be calculated in _get_current_metrics based on this updated history

        avalanche_size = self.sandpile_model.topple_and_relax()
        self.last_avalanche_size_metric = avalanche_size # Store for next observation

        # 5. Calculate current system metrics for observation and reward
        current_metrics_dict = self._get_current_metrics(avalanche_size, num_unstable_pre_relax_val)
        
        # 6. Calculate Reward
        reward += self.cfg.R_SURVIVAL_STEP

        is_large_avalanche_this_step = avalanche_size > self.cfg.LARGE_AVALANCHE_THRESH
        if is_large_avalanche_this_step:
            reward += self.cfg.R_LARGE_AVALANCHE_PENALTY
            self.time_since_last_large_avalanche = 0
        
        # Safety margin reward
        safety_margin = current_metrics_dict.get('safety_margin_p', 0.0)
        reward += self.cfg.R_SAFETY_MARGIN_FACTOR * (safety_margin - self.cfg.R_TARGET_SAFETY_MARGIN)

        if current_metrics_dict.get('avg_delta_P_p', 0.0) > current_metrics_dict.get('theta_T_p', 0.0):
            reward += self.cfg.R_THETA_T_BREACH_PENALTY
        
        # Danger zone penalty
        speed_p = current_metrics_dict.get('speed_p', 0.0)
        couple_p = current_metrics_dict.get('couple_p', 0.0)
        if not np.isnan(speed_p) and not np.isnan(couple_p):
             if speed_p > self.cfg.DANGER_SPEED_THRESHOLD and couple_p > self.cfg.DANGER_COUPLE_THRESHOLD_POSITIVE:
                 reward += self.cfg.R_DANGER_ZONE_PENALTY
        
        self.last_raw_metrics = current_metrics_dict.copy()
        # 7. Construct Next Observation
        observation = self._calculate_observation(current_metrics_dict)

        # 8. Determine terminated / truncated
        terminated = False
        if is_large_avalanche_this_step and avalanche_size > self.cfg.LARGE_AVALANCHE_THRESH * 2: # Catastrophic
            terminated = True
        # Could add termination if Fcrit_p is critically low for too long
        # if current_metrics_dict.get('f_crit_p', 100) < 0.01 * self.cfg.OBS_BOUNDS['f_crit_p'][1]: # e.g. <1% of max
        #     terminated = True
            
        truncated = self.current_step >= self.cfg.MAX_EPISODE_STEPS
        
        return observation, reward, terminated, truncated, {}

    def render(self):
        # Optional: could visualize the grid or plot key metrics
        pass

    def close(self):
        pass



