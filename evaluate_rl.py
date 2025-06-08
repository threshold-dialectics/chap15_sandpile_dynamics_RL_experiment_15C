#evaluate_rl_experiment_15C.py
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque # <<< IMPORT ADDED HERE
from stable_baselines3 import PPO
import re # Import re for sanitizing filenames
import json
from scipy.stats import ttest_ind
import subprocess

from sandpile_rl_env import SandpileInterventionRLEnv # Assuming this is your env file
import config as study_cfg # Assuming this is your config file

# Record commit hash for reproducibility
try:
    git_commit_hash = subprocess.check_output([
        'git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode().strip()
except Exception:
    git_commit_hash = 'unknown'

# ... (keep run_evaluation_episode and run_heuristic_evaluation_episode as they are) ...
def run_evaluation_episode(env, model=None, n_steps=None, episode_num=0, reward_history_len=10):
    """Runs one episode, with model or random actions if model is None.

    Returns tuple of (total_reward, episode_length, history, intervention_events)
    where ``intervention_events`` is a list of dictionaries capturing the raw
    state when the agent chose an intervention action (>0).
    """
    if n_steps is None:
        n_steps = env.cfg.MAX_EPISODE_STEPS

    obs, info = env.reset()
    terminated = False
    truncated = False
    current_total_reward = 0 # Use a different name to avoid confusion with delayed logic
    episode_length = 0
    # For evaluation, it's usually clearer to sum instantaneous rewards.
    # The delayed reward logic with deque is more for training if you need to shape rewards based on outcomes over a window.
    # If you truly want to evaluate based on this delayed sum, ensure it's what you intend.
    # For simplicity here, I'll modify it to sum instantaneous rewards for evaluation reporting.
    # If `reward_history_len` is for some specific evaluation metric, it needs careful handling.

    history = {
        'actions': [], 'rewards': [], 'large_avalanches': 0, 'safety_margins': [],
        'speed_indices': [], 'couple_indices': [], 'fcrit_p': [], 'avg_delta_p_p': [],
        'g_lever': [], 'beta_lever': [],
        'theta_T': [], 'avalanche_sizes': [], 'active_intervention_types': []
    }
    intervention_events = []

    for step in range(n_steps):
        if model:
            action, _states = model.predict(obs, deterministic=True)
        else:  # Baseline: No intervention
            action = 0

        raw_metrics_before = getattr(env, "last_raw_metrics", {})

        if model and action > 0: 
            event = {
                'episode_num': episode_num,
                'step_num': step,
                'action_taken': int(action)
            }
            for key_obs in env.cfg.OBS_KEYS:
                event[key_obs] = raw_metrics_before.get(key_obs, np.nan)
            event['safety_margin_p'] = raw_metrics_before.get('safety_margin_p', np.nan)
            event['speed_p'] = raw_metrics_before.get('speed_p', np.nan)
            event['couple_p'] = raw_metrics_before.get('couple_p', np.nan)
            event['avg_delta_P_p'] = raw_metrics_before.get('avg_delta_P_p', np.nan)
            intervention_events.append(event)

        obs, reward, terminated, truncated, info = env.step(action)
        
        current_total_reward += reward # Sum instantaneous rewards for evaluation

        episode_length += 1
        history['actions'].append(action)
        history['rewards'].append(reward) 
        
        raw_metrics = getattr(env, "last_raw_metrics", {})
        
        history['safety_margins'].append(raw_metrics.get('safety_margin_p', np.nan))
        history['speed_indices'].append(raw_metrics.get('speed_p', np.nan))
        history['couple_indices'].append(raw_metrics.get('couple_p', np.nan))
        history['fcrit_p'].append(raw_metrics.get('f_crit_p', np.nan))
        history['avg_delta_p_p'].append(raw_metrics.get('avg_delta_P_p', np.nan))
        history['g_lever'].append(raw_metrics.get('g_lever_p_topple_prob', np.nan))
        history['beta_lever'].append(raw_metrics.get('beta_lever_p_continuous', np.nan))
        history['theta_T'].append(raw_metrics.get('theta_T_p', np.nan))
        history['avalanche_sizes'].append(raw_metrics.get('avalanche_size', np.nan))
        history['active_intervention_types'].append(env.active_intervention_type)

        if raw_metrics.get('avalanche_size', 0) > env.cfg.LARGE_AVALANCHE_THRESH:
            history['large_avalanches'] += 1
        
        if terminated or truncated:
            break
            
    return current_total_reward, episode_length, history, intervention_events


def run_heuristic_evaluation_episode(env, episode_num=0, heuristic_params=None):
    """Runs one episode using a scripted heuristic intervention strategy."""
    if heuristic_params is None:
        heuristic_params = study_cfg.HEURISTIC_AGENT_PARAMS

    n_steps = env.cfg.MAX_EPISODE_STEPS
    obs, info = env.reset()
    terminated = False
    truncated = False
    total_reward = 0
    episode_length = 0

    history = {
        'actions': [], 'rewards': [], 'large_avalanches': 0, 'safety_margins': [],
        'speed_indices': [], 'couple_indices': [], 'fcrit_p': [], 'avg_delta_p_p': [],
        'g_lever': [], 'beta_lever': [],
        'theta_T': [], 'avalanche_sizes': [], 'active_intervention_types': []
    }
    intervention_events_h = [] 
    cooldown_counter = 0

    for step in range(n_steps):
        raw_metrics_before_step = getattr(env, "last_raw_metrics", {})
        action_h = 0 

        if env.active_intervention_type == 0 and cooldown_counter <= 0: 
            safety_margin_h = raw_metrics_before_step.get('safety_margin_p', np.inf) 
            speed_p_h = raw_metrics_before_step.get('speed_p', 0.0) 
            couple_p_h = raw_metrics_before_step.get('couple_p', 0.0) 
            # f_crit_p_h = raw_metrics_before_step.get('f_crit_p', np.inf) # Not used in current heuristic

            if not np.isnan(safety_margin_h) and safety_margin_h < heuristic_params['SAFETY_MARGIN_THRESHOLD']:
                action_h = 1 
            elif (not np.isnan(speed_p_h) and not np.isnan(couple_p_h) and
                  speed_p_h > heuristic_params['SPEED_THRESHOLD'] and
                  couple_p_h > heuristic_params['COUPLE_THRESHOLD_POSITIVE']):
                action_h = 1 # Example: Heuristic also defaults to Pulse G for Speed/Couple danger
                             # Or you can map this to Action 2 (Adjust Beta) if that's the intent
                             # action_h = 2 # Adjust Beta

            if action_h > 0:
                cooldown_counter = heuristic_params['INTERVENTION_COOLDOWN']
        
        if cooldown_counter > 0:
            cooldown_counter -=1

        if action_h > 0: 
            event_h = {
                'episode_num': episode_num, 'step_num': step, 'action_taken': int(action_h)
            }
            for key_obs_h in env.cfg.OBS_KEYS:
                event_h[key_obs_h] = raw_metrics_before_step.get(key_obs_h, np.nan)
            intervention_events_h.append(event_h)

        obs, reward, terminated, truncated, info = env.step(action_h)
        
        total_reward += reward 
        episode_length += 1
        history['actions'].append(action_h)
        history['rewards'].append(reward)
        
        raw_metrics_after_step = getattr(env, "last_raw_metrics", {})
        history['safety_margins'].append(raw_metrics_after_step.get('safety_margin_p', np.nan))
        history['speed_indices'].append(raw_metrics_after_step.get('speed_p', np.nan))
        history['couple_indices'].append(raw_metrics_after_step.get('couple_p', np.nan))
        history['fcrit_p'].append(raw_metrics_after_step.get('f_crit_p', np.nan))
        history['avg_delta_p_p'].append(raw_metrics_after_step.get('avg_delta_P_p', np.nan))
        history['g_lever'].append(raw_metrics_after_step.get('g_lever_p_topple_prob', np.nan))
        history['beta_lever'].append(raw_metrics_after_step.get('beta_lever_p_continuous', np.nan))
        history['theta_T'].append(raw_metrics_after_step.get('theta_T_p', np.nan))
        history['avalanche_sizes'].append(raw_metrics_after_step.get('avalanche_size', np.nan))
        history['active_intervention_types'].append(env.active_intervention_type)

        if raw_metrics_after_step.get('avalanche_size', 0) > env.cfg.LARGE_AVALANCHE_THRESH:
            history['large_avalanches'] += 1
        
        if terminated or truncated:
            break
            
    return total_reward, episode_length, history, intervention_events_h


def compute_episode_stats(total_reward, length, history, cfg):
    """Compute distilled statistics for one episode."""
    sm = np.array(history.get('safety_margins', []), dtype=float)
    theta = np.array(history.get('theta_T', []), dtype=float)
    avg_delta = np.array(history.get('avg_delta_p_p', []), dtype=float)
    speed = np.array(history.get('speed_indices', []), dtype=float)
    couple = np.array(history.get('couple_indices', []), dtype=float)
    actions = np.array(history.get('actions', []), dtype=int)
    active_types = np.array(history.get('active_intervention_types', []), dtype=int)
    avalanche_sizes = np.array(history.get('avalanche_sizes', []), dtype=float)

    large_sizes = avalanche_sizes[avalanche_sizes > cfg.LARGE_AVALANCHE_THRESH]

    # Temporal metrics for safety margin
    if sm.size >= 2:
        start = 200
        end = min(len(sm), 1000)
        t = np.arange(start, end)
        if t.size > 1:
            safety_slope = np.polyfit(t, sm[start:end], 1)[0]
        else:
            safety_slope = 0.0
    else:
        safety_slope = 0.0

    neg_mask = sm < 0
    if neg_mask.any():
        streaks = np.diff(np.where(np.concatenate(([neg_mask[0]], neg_mask[:-1] != neg_mask[1:], [True])))[0])[::2]
        max_neg_streak = int(streaks.max()) if streaks.size else 0
    else:
        max_neg_streak = 0

    large_idx = np.where(avalanche_sizes > cfg.LARGE_AVALANCHE_THRESH)[0]
    time_first_large = int(large_idx[0]) if large_idx.size else -1

    def burst_stats(act_id):
        mask = actions == act_id
        if not mask.any():
            return 0.0, 0.0
        bursts, cooldowns = [], []
        in_burst = False
        burst_len, cooldown_len = 0, 0
        for m in mask:
            if m:
                if in_burst:
                    burst_len += 1
                else:
                    in_burst = True
                    if cooldown_len:
                        cooldowns.append(cooldown_len)
                        cooldown_len = 0
                    burst_len = 1
            else:
                if in_burst:
                    bursts.append(burst_len)
                    in_burst = False
                    cooldown_len = 1
                else:
                    cooldown_len += 1
        if in_burst:
            bursts.append(burst_len)
        if cooldown_len:
            cooldowns.append(cooldown_len)
        return float(np.mean(bursts)) if bursts else 0.0, float(np.mean(cooldowns)) if cooldowns else 0.0

    stats = {
        'reward': float(total_reward),
        'length': int(length),
        'large_avalanches': int(history.get('large_avalanches', 0)),
        'safety_margin_mean': float(np.nanmean(sm)) if sm.size else 0.0,
        'safety_margin_std': float(np.nanstd(sm)) if sm.size else 0.0,
        'safety_margin_min': float(np.nanmin(sm)) if sm.size else 0.0,
        'safety_margin_p5': float(np.nanpercentile(sm, 5)) if sm.size else 0.0,
        'safety_margin_p95': float(np.nanpercentile(sm, 95)) if sm.size else 0.0,
        'safety_margin_slope_200_1000': float(safety_slope),
        'safety_margin_neg_streak_max': max_neg_streak,
        'prop_margin_negative': float(np.mean(sm < 0)) if sm.size else 0.0,
        'area_above_target': float(np.nansum(np.maximum(sm - cfg.R_TARGET_SAFETY_MARGIN, 0))) if sm.size else 0.0,
        'num_thetaT_breaches': int(np.sum(avg_delta > theta)) if avg_delta.size else 0,
        'num_danger_zone_steps': int(np.sum((speed > cfg.DANGER_SPEED_THRESHOLD) & (couple > cfg.DANGER_COUPLE_THRESHOLD_POSITIVE))) if speed.size else 0,
        'speed_min': float(np.nanmin(speed)) if speed.size else 0.0,
        'num_interventions_total': int(np.sum(actions > 0)),
        'num_steps_intervention_active': int(np.sum(active_types != 0)),
        'intervention_cost': float(
            np.sum(
                np.where(actions == 1, cfg.COST_G_PULSE_ACTION, 0)
                + np.where(actions == 2, cfg.COST_BETA_ADJUST_ACTION, 0)
                + np.where(actions == 3, cfg.COST_FCRIT_BOLSTER_ACTION, 0)
                + np.where(actions > 0, cfg.COST_ACTION_BASE, 0)
            )
        ),
        'mean_large_avalanche_size': float(np.mean(large_sizes)) if large_sizes.size else 0.0,
        'max_large_avalanche_size': float(np.max(large_sizes)) if large_sizes.size else 0.0,
        'avalanche_size_p95': float(np.nanpercentile(avalanche_sizes, 95)) if avalanche_sizes.size else 0.0,
        'avalanche_total_energy': float(np.nansum(avalanche_sizes)) if avalanche_sizes.size else 0.0,
        'time_first_large_avalanche': time_first_large,
    }

    for act in [1, 2, 3]:
        b_len, cd_len = burst_stats(act)
        stats[f'action{act}_mean_burst'] = b_len
        stats[f'action{act}_mean_cooldown'] = cd_len

    return stats


def compute_aggregate_stats(episode_stats_list):
    """Compute aggregate statistics for a list of episode metrics."""
    if not episode_stats_list:
        return {}

    keys = episode_stats_list[0].keys()
    n = len(episode_stats_list)
    aggregate = {"n_episodes": n}

    for k in keys:
        values = np.array([ep[k] for ep in episode_stats_list], dtype=float)
        mean = float(np.mean(values))
        sd = float(np.std(values, ddof=1))
        se = sd / np.sqrt(n) if n > 0 else 0.0
        ci_half = 1.96 * se
        aggregate[f"{k}_mean"] = mean
        aggregate[f"{k}_sd"] = sd
        aggregate[f"{k}_se"] = se
        aggregate[f"{k}_ci95_lower"] = mean - ci_half
        aggregate[f"{k}_ci95_upper"] = mean + ci_half

    return aggregate

def compute_p_value(a, b):
    """Welch's t-test p-value between two lists."""
    if not a or not b:
        return None
    stat, p = ttest_ind(a, b, equal_var=False)
    return float(p)

def compute_cohens_d(a, b):
    """Cohen's d effect size between two lists."""
    if not a or not b:
        return None
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    pooled_sd = np.sqrt(((a.std(ddof=1) ** 2) + (b.std(ddof=1) ** 2)) / 2)
    if pooled_sd == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_sd)

def compute_all_stats(rl_eps, base_eps):
    """Compute p-values and effect sizes for all metrics."""
    if not rl_eps or not base_eps:
        return {}, {}
    keys = rl_eps[0].keys()
    pvals = {}
    effects = {}
    for k in keys:
        a = [ep[k] for ep in rl_eps]
        b = [ep[k] for ep in base_eps]
        p = compute_p_value(a, b)
        if p is not None:
            pvals[k] = p
            effects[k] = compute_cohens_d(a, b)
    return pvals, effects

def compute_correlation_matrix(episode_stats_list, metrics):
    if not episode_stats_list:
        return {}
    df = pd.DataFrame([{m: ep.get(m, 0.0) for m in metrics} for ep in episode_stats_list])
    corr = df.corr().round(3)
    return corr.to_dict()

def sanitize_filename_suffix(config_dict):
    if not config_dict:
        return "default"
    # Create a more filename-friendly string
    parts = []
    for key, value in sorted(config_dict.items()): # Sort for consistency
        key_short = "".join(filter(str.isalnum, key.replace("_", ""))) # Remove underscores, keep alnum
        value_str = str(value).replace(".", "p") # Replace decimal with 'p'
        parts.append(f"{key_short}{value_str}")
    suffix = "_".join(parts)
    # Further sanitize: remove any characters not suitable for filenames
    suffix = re.sub(r'[^\w\-.]', '_', suffix)
    return suffix

def describe_variant(config_dict):
    if not config_dict:
        return 'Baseline parameters (no overrides).'
    keys = set(config_dict.keys())
    if {'R_LARGE_AVALANCHE_PENALTY', 'R_THETA_T_BREACH_PENALTY'} <= keys:
        return 'Harsher penalties to discourage large avalanches and Theta_T breaches.'
    if 'DANGER_SPEED_THRESHOLD' in keys or 'DANGER_COUPLE_THRESHOLD_POSITIVE' in keys:
        return 'More sensitive danger zone detection.'
    if 'COST_BETA_ADJUST_ACTION' in keys:
        return 'Cheaper Adjust Beta action.'
    if 'COST_FCRIT_BOLSTER_ACTION' in keys:
        return 'Cheaper Bolster Fcrit action.'
    return 'Custom config override experiment.'


def main_evaluation(model_path=None, num_episodes=10, config_override_dict=None, reward_history_len=10, eval_label="default_eval"):
    results_dir = "./rl_experiment_15C_eval_results/"
    os.makedirs(results_dir, exist_ok=True)

    # Create a filename-safe suffix from the config_override_dict or use eval_label
    if config_override_dict:
        file_suffix = sanitize_filename_suffix(config_override_dict)
    else:
        file_suffix = eval_label # Use the eval_label if no override

    # Instantiate the environment
    if config_override_dict:
        print(f"Running evaluation '{file_suffix}' with overridden config parameters: {config_override_dict}")
        env = SandpileInterventionRLEnv(config_dict=config_override_dict)
    else:
        print(f"Running evaluation '{file_suffix}' with default config parameters.")
        env = SandpileInterventionRLEnv(config_dict=None)


    rl_results = {'rewards': [], 'lengths': [], 'total_large_avalanches': []}
    baseline_results = {'rewards': [], 'lengths': [], 'total_large_avalanches': []}
    heuristic_results = {'rewards': [], 'lengths': [], 'total_large_avalanches': []}

    rl_episode_stats = []
    baseline_episode_stats = []
    heuristic_episode_stats = []
    
    all_episodes_history_rl = []
    all_episodes_history_baseline = []
    all_episodes_history_heuristic = []

    rl_model = None
    if model_path and os.path.exists(model_path + ".zip"):
        print(f"Loading trained RL model from {model_path}.zip")
        rl_model = PPO.load(model_path, env=env) 
    else:
        print("No RL model path provided or model not found. RL agent evaluation will be skipped.")

    print(f"\nRunning {num_episodes} episodes with RL Agent (if model loaded)...")
    all_intervention_events_rl = [] 
    if rl_model:
        for i in range(num_episodes):
            reward, length, history, events = run_evaluation_episode(env, rl_model, episode_num=i, reward_history_len=reward_history_len)
            rl_results['rewards'].append(reward)
            rl_results['lengths'].append(length)
            rl_results['total_large_avalanches'].append(history['large_avalanches'])
            all_episodes_history_rl.append(history)
            rl_episode_stats.append(compute_episode_stats(reward, length, history, env.cfg))
            all_intervention_events_rl.extend(events)
            print(f"RL Ep {i+1}: Reward={reward:.2f}, Length={length}, LargeAvalanches={history['large_avalanches']}")
    else:
        print("RL model evaluation skipped.")

    print(f"\nRunning {num_episodes} episodes with Baseline (No Intervention)...")
    for i in range(num_episodes):
        reward_b, length_b, history_b, _ = run_evaluation_episode(env, None, episode_num=i, reward_history_len=1)
        baseline_results['rewards'].append(reward_b)
        baseline_results['lengths'].append(length_b)
        baseline_results['total_large_avalanches'].append(history_b['large_avalanches'])
        all_episodes_history_baseline.append(history_b)
        baseline_episode_stats.append(compute_episode_stats(reward_b, length_b, history_b, env.cfg))
        print(f"Baseline Ep {i+1}: Reward={reward_b:.2f}, Length={length_b}, LargeAvalanches={history_b['large_avalanches']}")

    print(f"\nRunning {num_episodes} episodes with Heuristic Agent...")
    all_intervention_events_heuristic = [] 
    for i in range(num_episodes):
        reward_h, length_h, history_h, events_h = run_heuristic_evaluation_episode(env, episode_num=i,
                                                                           heuristic_params=study_cfg.HEURISTIC_AGENT_PARAMS)
        heuristic_results['rewards'].append(reward_h)
        heuristic_results['lengths'].append(length_h)
        heuristic_results['total_large_avalanches'].append(history_h['large_avalanches'])
        all_episodes_history_heuristic.append(history_h)
        heuristic_episode_stats.append(compute_episode_stats(reward_h, length_h, history_h, env.cfg))
        all_intervention_events_heuristic.extend(events_h)
        print(f"Heuristic Ep {i+1}: Reward={reward_h:.2f}, Length={length_h}, LargeAvalanches={history_h['large_avalanches']}")

    env.close() 

    if rl_model and all_episodes_history_rl:
        all_actions_rl = np.concatenate([h['actions'] for h in all_episodes_history_rl])
        action_counts_rl = np.bincount(all_actions_rl.astype(int), minlength=4)
        if action_counts_rl.sum() > 0 :
            action_freq_rl = action_counts_rl / action_counts_rl.sum()
            print("\n--- RL Agent Action Frequency ---")
            for a, freq in enumerate(action_freq_rl):
                action_name = {0: 'NoOp', 1: 'Pulse G', 2: 'Adjust Beta', 3: 'Bolster Fcrit'}.get(a, str(a))
                print(f"Action {a} ({action_name}): {freq*100:.2f}%")
        else:
            print("\n--- RL Agent Action Frequency --- \nNo actions recorded for RL agent.")


        if all_intervention_events_rl: 
            df_events_rl = pd.DataFrame(all_intervention_events_rl)
            df_events_rl.to_csv(os.path.join(results_dir, f'rl_intervention_decision_states_experiment_15C_{file_suffix}.csv'), index=False)

            action_names = {1: 'Pulse G', 2: 'Adjust Beta', 3: 'Bolster Fcrit'}
            for act in [1, 2, 3]:
                act_df_rl = df_events_rl[df_events_rl['action_taken'] == act]
                if not act_df_rl.empty:
                    print(f"\n--- State Conditions for RL {action_names[act]} Intervention (Action {act}) ---")
                    cols_to_analyze = ['safety_margin_p', 'speed_p', 'couple_p', 'avg_delta_P_p',
                                       'theta_T_p', 'f_crit_p']
                    subset_rl = act_df_rl[[col for col in cols_to_analyze if col in act_df_rl.columns]]
                    if not subset_rl.empty:
                        desc_rl = subset_rl.describe(percentiles=[0.25, 0.5, 0.75]).transpose()
                        print(desc_rl)
                    else:
                        print(f"Required columns for stats not found for action {act}.")
                else:
                    print(f"\nNo RL intervention events recorded for action {act}.")

    if all_episodes_history_heuristic:
        all_actions_h = np.concatenate([h['actions'] for h in all_episodes_history_heuristic])
        counts_h = np.bincount(all_actions_h.astype(int), minlength=4)
        if counts_h.sum() > 0:
            freq_h = counts_h / counts_h.sum()
            print("\n--- Heuristic Agent Action Frequency ---")
            for a, freq in enumerate(freq_h):
                action_name = {0: 'NoOp', 1: 'Pulse G', 2: 'Adjust Beta', 3: 'Bolster Fcrit'}.get(a, str(a))
                print(f"Action {a} ({action_name}): {freq*100:.2f}%")
        else:
            print("\n--- Heuristic Agent Action Frequency --- \nNo actions recorded for Heuristic agent.")
        
        if all_intervention_events_heuristic:
            df_events_h = pd.DataFrame(all_intervention_events_heuristic)
            df_events_h.to_csv(os.path.join(results_dir, f'heuristic_intervention_decision_states_experiment_15C_{file_suffix}.csv'), index=False)

    print("\n--- Evaluation Summary ---")
    if rl_model:
        print(f"RL Agent Avg Reward: {np.mean(rl_results['rewards']):.2f} +/- {np.std(rl_results['rewards']):.2f}")
        print(f"RL Agent Avg Length: {np.mean(rl_results['lengths']):.2f} +/- {np.std(rl_results['lengths']):.2f}")
        print(f"RL Agent Avg Large Avalanches: {np.mean(rl_results['total_large_avalanches']):.2f} +/- {np.std(rl_results['total_large_avalanches']):.2f}")

    print(f"Baseline Avg Reward: {np.mean(baseline_results['rewards']):.2f} +/- {np.std(baseline_results['rewards']):.2f}")
    print(f"Baseline Avg Length: {np.mean(baseline_results['lengths']):.2f} +/- {np.std(baseline_results['lengths']):.2f}")
    print(f"Baseline Avg Large Avalanches: {np.mean(baseline_results['total_large_avalanches']):.2f} +/- {np.std(baseline_results['total_large_avalanches']):.2f}")
    if all_episodes_history_heuristic:
        print(f"Heuristic Avg Reward: {np.mean(heuristic_results['rewards']):.2f} +/- {np.std(heuristic_results['rewards']):.2f}")
        print(f"Heuristic Avg Length: {np.mean(heuristic_results['lengths']):.2f} +/- {np.std(heuristic_results['lengths']):.2f}")
        print(f"Heuristic Avg Large Avalanches: {np.mean(heuristic_results['total_large_avalanches']):.2f} +/- {np.std(heuristic_results['total_large_avalanches']):.2f}")

    if rl_model and all_episodes_history_rl: # Check if RL data exists before plotting
        plt.figure(figsize=(12, 6))
        plt.plot(all_episodes_history_rl[0]['safety_margins'], label='RL Agent - Safety Margin Ep 0')
        if all_episodes_history_baseline: 
             plt.plot(all_episodes_history_baseline[0]['safety_margins'], label='Baseline - Safety Margin Ep 0', linestyle='--')
        if all_episodes_history_heuristic:
            plt.plot(all_episodes_history_heuristic[0]['safety_margins'], label='Heuristic - Safety Margin Ep 0', linestyle=':')
        plt.xlabel("Time Step")
        plt.ylabel("Safety Margin (G_p)")
        plt.title(f"Safety Margin Comparison (Example Episode - {file_suffix})")
        plt.legend()
        plt.savefig(os.path.join(results_dir, f"safety_margin_comparison_experiment_15C_{file_suffix}.png"))
        plt.close()

    if rl_model and all_episodes_history_rl:
         pd.DataFrame(all_episodes_history_rl[0]).to_csv(os.path.join(results_dir,f"example_rl_ep_history_experiment_15C_{file_suffix}.csv"))
    if all_episodes_history_baseline:
        pd.DataFrame(all_episodes_history_baseline[0]).to_csv(os.path.join(results_dir,f"example_baseline_ep_history_experiment_15C_{file_suffix}.csv"))
    if all_episodes_history_heuristic:
        pd.DataFrame(all_episodes_history_heuristic[0]).to_csv(os.path.join(results_dir,f"example_heuristic_ep_history_experiment_15C_{file_suffix}.csv"))

    print(f"\nEvaluation plots and data saved to {results_dir}")

    rl_aggs = compute_aggregate_stats(rl_episode_stats) if rl_episode_stats else {}
    baseline_aggs = compute_aggregate_stats(baseline_episode_stats)
    heuristic_aggs = compute_aggregate_stats(heuristic_episode_stats)

    # Remove metrics that show no variation across all conditions
    all_keys = set(rl_aggs.keys()) | set(baseline_aggs.keys()) | set(heuristic_aggs.keys())
    for k in list(all_keys):
        if k.endswith('_sd') or k.endswith('_se') or k.endswith('_ci95_lower') or k.endswith('_ci95_upper'):
            if all(abs(d.get(k, 0.0)) < 1e-12 for d in [rl_aggs, baseline_aggs, heuristic_aggs]):
                for d in (rl_aggs, baseline_aggs, heuristic_aggs):
                    d.pop(k, None)
        elif k.endswith('_mean'):
            if all(abs(d.get(k, 0.0)) < 1e-12 for d in [rl_aggs, baseline_aggs, heuristic_aggs]):
                base = k[:-5]
                for suffix in ['_mean', '_sd', '_se', '_ci95_lower', '_ci95_upper']:
                    key = base + suffix
                    for d in (rl_aggs, baseline_aggs, heuristic_aggs):
                        d.pop(key, None)

    def compute_delta(source, target):
        delta = {}
        for k, v in source.items():
            if k == 'n_episodes' or k.endswith('_sd') or k.endswith('_se') or k.endswith('_ci95_lower') or k.endswith('_ci95_upper'):
                continue
            base_v = target.get(k)
            if base_v is not None:
                delta[k] = v - base_v
        return delta

    rl_vs_baseline_delta = compute_delta(rl_aggs, baseline_aggs)

    rl_vs_heuristic_delta = compute_delta(rl_aggs, heuristic_aggs)

    heuristic_vs_baseline_delta = compute_delta(heuristic_aggs, baseline_aggs)

    p_values, effect_sizes = compute_all_stats(rl_episode_stats, baseline_episode_stats)

    metric_glossary = {
        'reward': 'Episode return, dimensionless',
        'length': 'Episode length in steps',
        'total_large_avalanches': 'Avalanches exceeding 5% of grid cells',
        'large_avalanches': 'Number of large avalanches in episode',
        'safety_margin_mean': 'Mean safety margin (dimensionless)',
        'safety_margin_std': 'Standard deviation of safety margin',
        'safety_margin_min': 'Minimum safety margin',
        'safety_margin_p5': '5th percentile of safety margin',
        'safety_margin_p95': '95th percentile of safety margin',
        'safety_margin_slope_200_1000': 'Linear trend of safety margin from step 200 to 1000',
        'safety_margin_neg_streak_max': 'Longest consecutive negative safety margin streak (steps)',
        'prop_margin_negative': 'Fraction of steps with safety margin below zero',
        'area_above_target': 'Integral of safety margin above target',
        'num_thetaT_breaches': 'Count of theta_T threshold breaches',
        'num_danger_zone_steps': 'Steps where speed and couple exceed danger thresholds',
        'speed_min': 'Minimum speed index (dimensionless)',
        'num_interventions_total': 'Total number of interventions taken',
        'num_steps_intervention_active': 'Steps with an intervention active',
        'intervention_cost': 'Cumulative cost of interventions',
        'mean_large_avalanche_size': 'Mean size of large avalanches (cells)',
        'max_large_avalanche_size': 'Maximum large avalanche size (cells)',
        'avalanche_size_p95': '95th percentile avalanche size (cells)',
        'avalanche_total_energy': 'Sum of all avalanche sizes (cells)',
        'time_first_large_avalanche': 'Step of first large avalanche (-1 if none)',
        'action1_mean_burst': 'Average duration of Pulse G bursts (steps)',
        'action1_mean_cooldown': 'Average cooldown between Pulse G bursts (steps)',
        'action2_mean_burst': 'Average duration of Adjust Beta bursts (steps)',
        'action2_mean_cooldown': 'Average cooldown between Adjust Beta bursts (steps)',
        'action3_mean_burst': 'Average duration of Bolster Fcrit bursts (steps)',
        'action3_mean_cooldown': 'Average cooldown between Bolster Fcrit bursts (steps)'
    }

    env_description = (
        'Sandpile grid 30x30 with BTW dynamics. Episodes last up to 1000 steps. '
        'Agent can Pulse G, Adjust Beta or Bolster Fcrit.'
    )

    training_details = (
        'PPO with 4 parallel envs for 500k steps; gamma=0.99, gae_lambda=0.95, '
        'clip_range=0.2. Reward includes survival bonus and penalties for large '
        'avalanches, theta_T breaches and danger-zone steps.'
    )

    variant_note = describe_variant(config_override_dict)

    metrics_for_corr = ['reward', 'large_avalanches', 'intervention_cost', 'prop_margin_negative']
    rl_corr = compute_correlation_matrix(rl_episode_stats, metrics_for_corr)

    summary = {
        'file_suffix': file_suffix,
        'config_override': config_override_dict or {},
        'variant_note': variant_note,
        'env_description': env_description,
        'metric_glossary': metric_glossary,
        'action_schematic': '1: Pulse G - raise topple prob; 2: Adjust Beta - lower k_th; 3: Bolster Fcrit - remove grains.',
        'training_details': training_details,
        'random_seed_policy': 'All runs used fixed seeds 0-9.',
        'commit_hash': git_commit_hash,
        'rl_aggregate_stats': rl_aggs,
        'baseline_aggregate_stats': baseline_aggs,
        'heuristic_aggregate_stats': heuristic_aggs,
        'rl_vs_baseline_delta': rl_vs_baseline_delta,
        'rl_vs_heuristic_delta': rl_vs_heuristic_delta,
        'heuristic_vs_baseline_delta': heuristic_vs_baseline_delta,
        'statistical_tests': p_values,
        'effect_sizes': effect_sizes,
        'rl_correlation_matrix': rl_corr,
        'statistical_tests_note': 'Assume no significance unless p-value provided'
    }

    if rl_model and 'action_freq_rl' in locals():
        summary["rl_action_frequency"] = action_freq_rl.tolist()

    if 'freq_h' in locals():
        summary["heuristic_action_frequency"] = freq_h.tolist()

    return summary


if __name__ == "__main__":
    trained_model_path = "./rl_experiment_15C_logs/ppo_sandpile_intervention_experiment_15C"
    
    combined_summaries = []

    # Base evaluation
    if not os.path.exists(trained_model_path + ".zip"):
        print(f"Warning: Model file {trained_model_path}.zip not found. Only baseline and heuristic will be run effectively.")
        combined_summaries.append(
            main_evaluation(model_path=None, num_episodes=10, reward_history_len=1, eval_label="default_eval_no_rl")
        )
    else:
        combined_summaries.append(
            main_evaluation(model_path=trained_model_path, num_episodes=10, reward_history_len=10, eval_label="default_eval_with_rl")
        )

        robustness_config_A = {
            "R_LARGE_AVALANCHE_PENALTY": -20.0,
            "R_THETA_T_BREACH_PENALTY": -1.0
        }
        print("\n--- Running Robustness Test: Harsher Penalties ---")
        combined_summaries.append(
            main_evaluation(model_path=trained_model_path, num_episodes=5, config_override_dict=robustness_config_A, reward_history_len=10, eval_label="robust_A")
        )

        robustness_config_B = {
            "DANGER_SPEED_THRESHOLD": 0.45,
            "DANGER_COUPLE_THRESHOLD_POSITIVE": 0.3
        }
        print("\n--- Running Robustness Test: More Sensitive Danger Zone ---")
        combined_summaries.append(
            main_evaluation(model_path=trained_model_path, num_episodes=5, config_override_dict=robustness_config_B, reward_history_len=10, eval_label="robust_B")
        )

        config_override_exp2_1 = { "COST_BETA_ADJUST_ACTION": -0.01 }
        print("\n--- Experiment 2.1: Cheaper Adjust Beta (Evaluation of Original Model) ---")
        combined_summaries.append(
            main_evaluation(model_path=trained_model_path, num_episodes=5, config_override_dict=config_override_exp2_1, reward_history_len=10, eval_label="exp2_1_beta_cheap")
        )

        config_override_exp2_2 = { "COST_FCRIT_BOLSTER_ACTION": -0.05 }
        print("\n--- Experiment 2.2: Cheaper Bolster Fcrit (Evaluation of Original Model) ---")
        combined_summaries.append(
            main_evaluation(model_path=trained_model_path, num_episodes=5, config_override_dict=config_override_exp2_2, reward_history_len=10, eval_label="exp2_2_fcrit_cheap")
        )

    results_dir = "./rl_experiment_15C_eval_results/"
    os.makedirs(results_dir, exist_ok=True)
    combined_summary_file = os.path.join(results_dir, "combined_summary_results.json")
    with open(combined_summary_file, "w") as f:
        json.dump(combined_summaries, f, indent=2)
    print(f"Combined summary results saved to {combined_summary_file}")
