# Threshold Dialectics - Experiment 15C: Sandpile Interventions

This repository contains the complete source code and configuration for **Experiment 15C** from the book *Threshold Dialectics: Understanding Complex Systems and Enabling Active Robustness* by Axel Pond.

This experiment investigates whether a Reinforcement Learning (RL) agent can learn effective, proactive intervention strategies to manage the stability of a complex, self-organizing system, using the Bak-Tang-Wiesenfeld (BTW) sandpile model as a microcosm.

**Note:** This repository exclusively contains the code for Experiment 15C. The preceding experiments in this series, 15A (Core TD Mechanics) and 15B (Self-Organized Tolerance Creep), are detailed in the book and their code can be found in a separate repository.

## Overview

> The sandpile, deceptively simple, proves a fertile ground for testing the complex truths of adaptive endurance.
>
> — *Threshold Dialectics, Chapter 15*

The BTW sandpile model is a canonical example of Self-Organized Criticality (SOC), where a system naturally evolves to a critical state where small perturbations can trigger "avalanches" of all sizes. While traditional SOC theory describes the statistical signatures of this state, Threshold Dialectics (TD) provides a mechanistic lens to understand the underlying dynamics of adaptive capacity that drive the system towards criticality.

Experiment 15C applies the principles of adaptive management explored in Chapter 9 to this non-biological, rule-based system. The core objective is to train an RL agent to monitor the sandpile's state through TD-informed diagnostics and apply timely interventions to prevent catastrophic, system-wide avalanches. The performance of this learned policy is compared against a no-intervention baseline and a TD-informed heuristic agent.

## Context from "Threshold Dialectics"

In the TD framework, the viability of a complex system is determined by the dynamic interplay of its core adaptive capacities (levers) and the systemic strain it faces. The key concepts are operationalized in the sandpile model as follows:

| TD Concept | Sandpile Proxy | Description |
| :--- | :--- | :--- |
| **Systemic Strain** ($\avgDeltaPtau^p$) | "avg_delta_P_p" | Time-averaged sum of "excess" grains in unstable cells. Represents the aggregate pressure for toppling. |
| **Policy Precision** ($\betaLever_p$) | "k_th" | The global toppling threshold. A higher threshold signifies a "greedier" or more precise policy of accumulating stress before acting. |
| **Energetic Slack** ($\FEcrit^p$) | "f_crit_p" | The total capacity of stable cells to absorb new grains without toppling. A direct measure of the system's buffer. |
| **Perception Gain** ($\gLever_p$) | "p_topple" | The probability that an unstable cell will topple in a given time step. Represents the system's sensitivity to local instability. |
| **Tolerance Sheet** ($\ThetaT^p$) | "theta_T_p" | The system's calculated capacity to withstand strain, derived from the three lever proxies. A breach occurs when $\avgDeltaPtau^p > \ThetaT^p$. |

The sandpile model, with these proxies, becomes a dynamic entity whose "adaptive economy"—its balancing of grain accumulation against toppling capacity—can be meaningfully interpreted and managed through the principles of Threshold Dialectics.

## Experiment 15C: TD-Informed RL Interventions

This experiment frames the management of the sandpile as a Reinforcement Learning problem.

*   **Objective:** To train an agent that can successfully manage the sandpile system to maximize its survival time and prevent catastrophic avalanches (defined as an avalanche involving >5% of the grid cells).
*   **Environment:** A custom Gymnasium environment ("SandpileInterventionRLEnv") that simulates the adaptive sandpile model. The sandpile has its own internal adaptive logic (e.g., for "k_th" and "p_topple"), which the RL agent can override with its interventions.
*   **Agent:** A Proximal Policy Optimization (PPO) agent from the "stable-baselines3" library. The agent observes the sandpile's state via TD-derived metrics and chooses from a set of discrete actions.
*   **Actions:** The agent can choose to:
    1.  **Do Nothing (No-Op):** Allow the sandpile's internal adaptive logic to run.
    2.  **Pulse G:** Temporarily increase the probability of toppling ("p_topple"), increasing system reactivity.
    3.  **Adjust Beta:** Temporarily decrease the toppling threshold ("k_th"), making the system less "greedy."
    4.  **Bolster Fcrit:** Directly remove a small number of grains from nearly-full cells, increasing the system's slack.
*   **Reward Function:** The agent is rewarded for system survival and for maintaining a healthy "safety margin" ($G_p = \ThetaT^p - \avgDeltaPtau^p$). It is penalized for costly interventions, large avalanches, and for allowing the system to enter a "danger zone" defined by TD diagnostics.

## Repository Structure

This repository contains the following key files:

*   "sandpile_rl_env_experiment_15C.py": Defines the custom "SandpileInterventionRLEnv" Gymnasium environment, which wraps the core sandpile model and provides the interface for the RL agent.
*   "td_core_extended.py": Implements the core Bak-Tang-Wiesenfeld (BTW) sandpile model, including its adaptive logic and the functions for calculating TD proxies and diagnostics.
*   "config.py": A central configuration file containing all simulation parameters, RL environment settings, reward function weights, and intervention parameters.
*   "train_rl_experiment_15C.py": The script used to train the PPO agent. It sets up the vectorized environment, callbacks, and the PPO model, then initiates the training loop.
*   "evaluate_rl_experiment_15C.py": The script to evaluate the trained agent. It runs evaluation episodes for the RL agent, a no-intervention baseline, and a TD-informed heuristic agent, then generates summary statistics, plots, and a "combined_summary_results.json" file.

## Getting Started

### Prerequisites

You will need Python 3.8+ and the following packages. You can install them using the provided "requirements.txt" file.

*   "numpy"
*   "pandas"
*   "matplotlib"
*   "scipy"
*   "stable-baselines3[extra]"
*   "gymnasium"

### Installation

1.  Clone the repository:
    """bash
    git clone https://github.com/your-username/threshold-dialectics-exp15c.git
    cd threshold-dialectics-exp15c
    """
2.  Install the required packages:
    """bash
    pip install -r requirements.txt
    """
    (Note: A "requirements.txt" file would need to be created with the package list. For now, users can install them manually.)

### How to Run

The experiment is conducted in two main stages: training the agent and evaluating its performance.

#### 1. Training the RL Agent

To train the PPO agent, run the training script from the root of the repository:

"""bash
python train_rl_experiment_15C.py
"""

This will:
*   Create a "rl_experiment_15C_logs/" directory.
*   Instantiate a vectorized training environment.
*   Train the PPO agent for 500,000 timesteps, saving periodic checkpoints and evaluation logs.
*   Save the final trained model as "rl_experiment_15C_logs/ppo_sandpile_intervention_experiment_15C.zip".

#### 2. Evaluating the Agent

Once the model is trained, you can evaluate its performance against the baseline and heuristic agents by running the evaluation script:

"""bash
python evaluate_rl_experiment_15C.py
"""

This script will:
*   Create a "rl_experiment_15C_eval_results/" directory.
*   Load the trained PPO model.
*   Run evaluation episodes for three policies: the trained RL agent, a baseline "do nothing" agent, and a TD-informed heuristic agent.
*   Perform robustness tests by altering environment parameters.
*   Print a summary of the performance metrics to the console.
*   Save detailed evaluation data, including example episode histories, intervention decision states, and plots (e.g., "safety_margin_comparison.png"), to the results directory.
*   Generate a "combined_summary_results.json" file containing aggregated statistics and comparisons.

## Key Findings from Experiment 15C

The evaluation reveals a nuanced hierarchy of performance among the intervention strategies.

*   **RL Agent Performance:** The trained RL agent learned a highly effective policy, significantly outperforming the no-intervention baseline in episode length, total reward, and prevention of large avalanches. It learned to aggressively maintain a high safety margin ($G_p$), demonstrating a sophisticated understanding of the system's long-term dynamics.
*   **Heuristic Agent Performance:** Surprisingly, the simple TD-informed heuristic agent achieved the highest total reward and a perfect record of preventing large avalanches, doing so with much lower intervention costs than the RL agent. This highlights that in a well-understood system, a well-tuned heuristic can be exceptionally effective.
*   **The Nuance of "Optimal":** The RL agent's strategy, while more costly, resulted in a higher average safety margin. This suggests it learned a different kind of "optimal" policy—one focused on maximizing stability at a higher operational cost. This contrasts with the heuristic's "efficient prevention" strategy and serves as a powerful lesson on the importance of reward function design in shaping agent behavior.
*   **Robustness:** Robustness tests (e.g., altering the cost of specific actions) showed the RL agent had internalized a complex model of the system's dynamics, not just a simple cost-minimization strategy.

## Citation

If you use or refer to this code or the concepts from Threshold Dialectics, please cite the accompanying book:

@book{pond2025threshold,
  author    = {Axel Pond},
  title     = {Threshold Dialectics: Understanding Complex Systems and Enabling Active Robustness},
  year      = {2025},
  isbn      = {978-82-693862-2-6},
  publisher = {Amazon Kindle Direct Publishing},
  url       = {https://www.thresholddialectics.com},
  note      = {Code repository: \url{https://github.com/threshold-dialectics}}
}

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

