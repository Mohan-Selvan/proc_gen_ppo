import constants
import game_world

import pickle
import numpy as np

import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3.common import env_checker

def evaluate_model(model, env, num_episodes=10):
    """
    Evaluate the trained model on the environment.

    Parameters:
    - model: The trained PPO model.
    - env: The environment instance (should be a vectorized environment).
    - num_episodes: Number of episodes to evaluate.

    Returns:
    - average_reward: Average reward obtained during the evaluation.
    """
    total_reward = 0

    for episode in range(num_episodes):
        obs, info = env.reset()  # Reset the environment and get the initial observation
        done = [False] * 1 #env.num_envs  # List of done flags for each environment
        episode_reward = 0
        
        while not all(done):  # Continue until all environments are done
            # Get action from the model
            action, _ = model.predict(obs, deterministic=True)

            # Apply the action to the environment
            # Here, action is a batch of actions for each environment
            obs, reward, terminated, truncated, info = env.step(action)
            done[0] = terminated or truncated

            episode_reward += np.sum(reward)  # Accumulate rewards across all environments

        total_reward += episode_reward
        print(f"Episode {episode + 1} reward: {episode_reward}")

    average_reward = total_reward / num_episodes
    print(f"Average reward over {num_episodes} episodes: {average_reward}")
    return average_reward


model_file_path = "./saves/model"
player_path_file_path = './saves/path_list'

player_path = []
with open (player_path_file_path, 'rb') as fp:
    player_path = pickle.load(fp)
    print("Path loaded from 'path_list'")

# Create the environment
env = game_world.GameWorld(width=constants.GRID_SIZE[0], 
                           height=constants.GRID_SIZE[1], 
                           player_path=player_path,
                           mask_size=constants.MASK_SIZE, 
                           num_tile_actions=constants.NUMBER_OF_ACTIONS_PER_CELL
                           )

# env_checker.check_env(env=env, warn=True, skip_render_check=False)

# Create a directory to save logs
log_dir = "./ppo_training_logs"
os.makedirs(log_dir, exist_ok=True)

# Define the PPO model with a CNN policy for processing grid-based inputs
model = PPO("CnnPolicy", env, verbose=1)

# # Train the model
model.learn(total_timesteps=1000)

# Save the model
model.save(model_file_path)

# Load the model later for evaluation
loaded_model = PPO.load(model_file_path)

# Evaluate the model
evaluate_model(loaded_model, env)