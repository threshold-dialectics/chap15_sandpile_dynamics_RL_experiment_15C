#train_rl_experiment_15C.py
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from sandpile_rl_env import SandpileInterventionRLEnv
import config as study_cfg # Renamed for clarity

def main():
    log_dir = "./rl_experiment_15C_logs/"
    model_save_path = os.path.join(log_dir, "ppo_sandpile_intervention_experiment_15C")
    tensorboard_log_path = os.path.join(log_dir, "tensorboard/")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(tensorboard_log_path, exist_ok=True)

    # Instantiate the environment
    # Using a lambda to allow passing config if needed, though here it uses module directly
    env_id = "SandpileIntervention-v0"
    
    # It's good practice to register the env if using make_vec_env with string id,
    # but for direct lambda, it's fine. Let's make it explicit:
    if env_id not in gym.envs.registry:
         gym.register(
             id=env_id,
             entry_point='sandpile_rl_env_experiment_15C:SandpileInterventionRLEnv', # if file is sandpile_rl_env_experiment_15C.py
         )

    # Create a function to instantiate the environment
    def make_env():
        env = SandpileInterventionRLEnv(config_dict=None) # Pass your config here if needed
        env = Monitor(env, log_dir) # SB3 Monitor for rewards and episode stats
        return env

    # Vectorized environments for parallel training
    num_cpu = 4 # Adjust as per your CPU
    vec_env = make_vec_env(make_env, n_envs=num_cpu)

    # Callbacks
    # Save a checkpoint every N steps
    checkpoint_callback = CheckpointCallback(save_freq=50000, save_path=log_dir, name_prefix="rl_model_experiment_15C")
    
    # Separate evaluation env
    eval_env = make_env() # Single env for evaluation
    eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir,
                                 log_path=log_dir, eval_freq=max(25000 // num_cpu, 1),
                                 deterministic=True, render=False)

    # Define the PPO agent
    # Hyperparameters can be tuned. These are common defaults.
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=tensorboard_log_path,
        learning_rate=3e-4,
        n_steps=2048 // num_cpu, # Adjusted for vec_env
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0, # No entropy bonus
        vf_coef=0.5,
        max_grad_norm=0.5,
        # policy_kwargs=dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])]) # Example larger network
    )

    print("Starting RL agent training for Study 15C...")
    model.learn(
        total_timesteps=500_000, # Adjust total timesteps as needed
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True
    )

    model.save(model_save_path)
    print(f"Training complete. Model saved to {model_save_path}")

    vec_env.close()
    eval_env.close()

if __name__ == "__main__":
    main()