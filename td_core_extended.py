#td_core_extended.py
import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from typing import Sequence, Tuple, Set, List, Dict, Deque
from collections import deque

class SandpileBTW:
    """Bak–Tang–Wiesenfeld sandpile model with adjustable k_th and p_topple."""

    def __init__(self, grid_size: Tuple[int, int] = (50, 50), k_th: int = 4, p_topple: float = 1.0):
        self.grid_size = grid_size
        self.grid = np.zeros(grid_size, dtype=int)
        self.k_th = k_th # This can be updated each step by the environment
        self.p_topple = p_topple # This can be updated each step by the environment
        self.time_step_counter = 0
        self.total_grains_lost = 0
        self.last_event_scar_coords: Set[Tuple[int, int]] = set()

    def add_grain(self, pos: Tuple[int, int] = None) -> None:
        if pos is None:
            pos = (np.random.randint(0, self.grid_size[0]), np.random.randint(0, self.grid_size[1]))
        self.grid[pos] += 1
        self.time_step_counter += 1

    def get_unstable_coords(self) -> np.ndarray:
        return np.argwhere(self.grid >= self.k_th)

    def topple_and_relax(self) -> int:
        current_event_avalanche_size = 0
        _current_cascade_scar_coords: Set[Tuple[int, int]] = set()
        unstable_coords = self.get_unstable_coords()
        
        # Limit iterations to prevent infinite loops in edge cases
        max_iterations = self.grid_size[0] * self.grid_size[1] * (self.k_th + 2) # Heuristic limit
        relaxation_iterations = 0

        while unstable_coords.shape[0] > 0:
            relaxation_iterations += 1
            if relaxation_iterations > max_iterations:
                # print(f"Warning: Max relaxation iterations reached at timestep {self.time_step_counter}")
                break # Safety break

            coords_to_topple_this_sub_step = []
            for r_unstable, c_unstable in unstable_coords:
                if np.random.rand() < self.p_topple:
                    coords_to_topple_this_sub_step.append((r_unstable, c_unstable))
            
            if not coords_to_topple_this_sub_step: # No cells chose to topple this sub-step
                # This can happen if p_topple is low and all unstable cells "decide" not to topple.
                # If there are still unstable cells, this could lead to issues if not handled.
                # For now, if no cells topple, we break the inner loop. If unstable_coords is still non-empty,
                # the outer loop's condition (unstable_coords.shape[0] > 0) might lead to another iteration.
                # However, if p_topple is very low, this could be slow. The `max_iterations` above is a safeguard.
                break


            for r, c in coords_to_topple_this_sub_step:
                if self.grid[r, c] >= self.k_th: # Check again, as a cell might have received grains and become stable
                    current_event_avalanche_size += 1
                    _current_cascade_scar_coords.add((r, c))

                    grains_to_distribute = self.k_th 
                    self.grid[r, c] -= grains_to_distribute # Can go negative if k_th is large relative to grid[r,c] before topple
                                                            # Should be self.grid[r,c] -= self.k_th if it's always k_th grains
                                                            # Or self.grid[r,c] -= self.grid[r,c] if all grains above k_th-1 topple.
                                                            # The BTW model topples by k_th grains.
                    
                    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)] # N, S, W, E
                    
                    # Distribute k_th grains, one to each neighbor up to 4.
                    # If k_th is > 4, the model description implies only 4 neighbors get grains.
                    # Standard BTW distributes to 4 neighbors. Some variations might distribute more.
                    # The prompt's code `min(grains_to_distribute, len(dirs))` ensures only up to 4.
                    for i in range(min(grains_to_distribute, len(dirs))): 
                        dr, dc = dirs[i]
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < self.grid_size[0] and 0 <= cc < self.grid_size[1]:
                            self.grid[rr, cc] += 1
                        else:
                            self.total_grains_lost += 1
            
            unstable_coords = self.get_unstable_coords()
        
        self.last_event_scar_coords = _current_cascade_scar_coords
        return current_event_avalanche_size

    def bolster_capacity(self, num_grains_to_remove_total: int, num_cells_to_target: int):
        """
        Removes grains from stable cells that are nearly full (height k_th - 1)
        to increase their capacity, effectively increasing aggregate Fcrit_p.
        """
        if self.k_th <= 1: # Cannot bolster if k_th is 1 or less (no stable full cells)
            return 0
            
        stable_nearly_full_cells_indices = np.argwhere(self.grid == (self.k_th - 1))
        
        if stable_nearly_full_cells_indices.shape[0] == 0:
            return 0 # No cells to bolster

        num_actually_targeted = min(num_cells_to_target, stable_nearly_full_cells_indices.shape[0])
        
        if num_actually_targeted == 0:
            return 0

        # Determine grains to remove per cell. Ensure at least 1 if total > 0.
        grains_per_cell_to_remove = num_grains_to_remove_total // num_actually_targeted
        if grains_per_cell_to_remove == 0 and num_grains_to_remove_total > 0:
            grains_per_cell_to_remove = 1
        
        if grains_per_cell_to_remove == 0: # Nothing to remove
            return 0

        actual_grains_removed_total = 0
        
        # Randomly select cells to target from the identified nearly_full_cells
        target_cell_row_indices = np.random.choice(stable_nearly_full_cells_indices.shape[0], 
                                                   size=num_actually_targeted, 
                                                   replace=False)
        
        for row_idx in target_cell_row_indices:
            r, c = stable_nearly_full_cells_indices[row_idx]
            
            # Ensure we don't remove more grains than exist or make cell height negative
            # Since cells are at k_th-1, removing more than k_th-1 would make them negative.
            # We should only remove up to (k_th-1) grains, usually just a few.
            # The intent is to increase capacity, so removing 1 or 2 is typical.
            grains_this_cell = min(self.grid[r,c], grains_per_cell_to_remove) # Max remove what's there
            self.grid[r,c] -= grains_this_cell
            actual_grains_removed_total += grains_this_cell
            
        return actual_grains_removed_total

# --- TD Calculation Functions (from prompt) ---
def calculate_raw_instantaneous_strain_p(grid: np.ndarray, k_th: int) -> float:
    if k_th <= 0: return np.sum(grid) # Or handle as an error, k_th shouldn't be <=0
    unstable_mask = grid >= k_th
    if np.sum(unstable_mask) == 0:
        return 0.0
    # Strain is sum of grains above (k_th - 1) for unstable cells
    return float(np.sum(grid[unstable_mask] - (k_th - 1)))

def calculate_energetic_slack_p(grid: np.ndarray, k_th: int) -> float:
    if k_th <= 0: return 0.0 # Or handle error
    stable_mask = grid < k_th
    # Slack is sum of capacity (k_th - current_height) for stable cells
    # but each cell can hold k_th-1 grains before toppling. So capacity is (k_th-1) - grid[cell]
    # No, Fcrit isslack = np.sum(k_th - grid[stable_mask]) -- as per book, tolerance capacity
    # However, for sandpile, capacity per cell before toppling is k_th-1.
    # So slack should be sum over ( (k_th-1) - grid[stable_mask])
    # Let's stick to the provided function from the original context for consistency:
    slack = np.sum(k_th - grid[stable_mask]) # This might overestimate if k_th is the toppling threshold
                                             # A cell at k_th-1 has 1 unit of slack if it topples at k_th.
                                             # So (k_th - grid[stable_mask]) seems correct. Example: k_th=4, grid=0, slack=4. Grid=3, slack=1.
    return float(slack)


def calculate_tolerance_sheet_p(
    g_lever_p_val: float,
    beta_lever_p_val: float, # This is the continuous k_th proxy
    f_crit_p_val: float,
    w_g: float,
    w_beta: float,
    w_fcrit: float,
    C_p: float = 1.0,
    scaling: float = 1.0,
) -> float:
    g_eff = max(g_lever_p_val, 1e-9) # Avoid log(0) or power of 0 if an exponent is 0
    beta_eff = max(beta_lever_p_val, 1e-9) # e.g. if k_th could be 0
    fcrit_eff = max(f_crit_p_val, 1e-9) # Slack can be 0
    
    # Handle cases where exponents are zero
    term_g = g_eff ** w_g if w_g > 0 else (1.0 if w_g == 0 else np.nan) # if w_g <0, result is complex/nan
    term_beta = beta_eff ** w_beta if w_beta > 0 else (1.0 if w_beta == 0 else np.nan)
    term_fcrit = fcrit_eff ** w_fcrit if w_fcrit > 0 else (1.0 if w_fcrit == 0 else np.nan)
    
    if np.isnan(term_g) or np.isnan(term_beta) or np.isnan(term_fcrit):
        # This case should ideally not happen with w_k > 0 constraints
        return 0.0 # Or raise error

    return scaling * C_p * term_g * term_beta * term_fcrit


def calculate_derivatives_savgol(timeseries_arr: Sequence[float], window_length: int = 5, polyorder: int = 2) -> np.ndarray:
    ts_array = np.array(timeseries_arr, dtype=float) # Ensure it's a numpy array of floats
    if len(ts_array) < window_length:
        return np.full_like(ts_array, np.nan, dtype=float)
    
    # Ensure window_length is odd and less than or equal to the length of timeseries_arr
    wl = min(window_length if window_length % 2 != 0 else window_length + 1, len(ts_array))
    if wl <= polyorder: # polyorder must be less than window_length
        # print(f"Warning: Savgol filter polyorder ({polyorder}) >= window_length ({wl}). Cannot compute derivative.")
        return np.full_like(ts_array, np.nan, dtype=float)

    # Pad for edge effects if window is large relative to series, though savgol handles edges.
    # For consistency with original code, padding is not explicitly done here before savgol_filter
    # as savgol_filter can handle boundaries. The original did pad for some reason.
    # Let's keep it simple: if not enough points, return NaNs.
    # Savgol itself will raise error if len(ts_array) < wl. Let's ensure wl is at least polyorder + 1.
    if wl < polyorder + 1 + (polyorder % 2): # A common heuristic for minimum window size relative to polyorder
        return np.full_like(ts_array, np.nan, dtype=float)

    try:
        derivatives = savgol_filter(ts_array, window_length=wl, polyorder=polyorder, deriv=1, delta=1.0)
    except ValueError as e:
        # print(f"SavGol filter error: {e}. Returning NaNs.")
        derivatives = np.full_like(ts_array, np.nan, dtype=float)
    return derivatives