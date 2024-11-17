import constants
import game_world

import pickle
import numpy as np
import helper

import os
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecTransposeImage

from stable_baselines3.common import env_checker

import gymnasium as gym

import custom_policy_lstm
import custom_policy_ppo


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
        env.save_screen_image(f"./saves/levels/test_iter_{(episode + 1)}.png")
        env.set_player_path(env.generate_player_path(max_turns=1000, randomness=env.path_randomness))

    average_reward = total_reward / num_episodes
    print(f"Average reward over {num_episodes} episodes: {average_reward}")
    return average_reward


model_file_path = "./saves/model"
player_path_file_path = './saves/path_list'

player_path = []
with open (player_path_file_path, 'rb') as fp:
    player_path = pickle.load(fp)
    print("Path loaded from 'path_list'")


def create_env():

    # Create the environment
    env = game_world.GameWorld(width=constants.GRID_SIZE[0], 
                            height=constants.GRID_SIZE[1], 
                            player_path=player_path,
                            observation_window_shape=constants.OBSERVATION_WINDOW_SHAPE,
                            mask_shape=constants.ACTION_MASK_SHAPE, 
                            num_tile_actions=constants.NUMBER_OF_ACTIONS_PER_CELL,
                            path_randomness=0.5,
                            random_seed=constants.RANDOM_SEED
                            ) 

    return env


def check_env():
    env_checker.check_env(env=create_env(), warn=True, skip_render_check=False)


class RewardLoggingCallback(BaseCallback):
    def __init__(self, log_dir, verbose=1):
        super(RewardLoggingCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.episode_rewards = []

    def _on_step(self):

        # Collect episode reward data after each episode
        if len(self.locals.get("infos", [])) > 0:
            for info in self.locals["infos"]:
                if "episode" in info.keys():
                    reward = info["episode"]["r"]
                    self.episode_rewards.append(reward)
                    print(f"Episode : {len(self.episode_rewards)}, Reward : {reward}")
        #helper.plot(self.episode_rewards)
        return True

    def on_training_end(self):

        # Save rewards to file for later analysis
        np.save(os.path.join(self.log_dir, "episode_rewards.npy"), self.episode_rewards)

        # Load the rewards data
        episode_rewards = np.load(os.path.join(self.log_dir, "episode_rewards.npy"))

        # Plot the episode rewards
        plt.figure(figsize=(10, 5))
        plt.plot(episode_rewards, label="Episode Reward")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Reward per Episode during Training")
        plt.legend()
        plt.show()
        plt.tight_layout()
        plt.savefig("./saves/train_plot.png")
        plt.close()


def train(device):

    # Create a directory to save logs
    log_dir = "./ppo_training_logs"
    os.makedirs(log_dir, exist_ok=True)

    reward_callback = RewardLoggingCallback(log_dir)


    env = make_vec_env(lambda: create_env(), n_envs=4, vec_env_cls=SubprocVecEnv)  # Assumes "YourCustomEnv-v0" is registered in Gym

    # Define the PPO model with a CNN policy for processing grid-based inputs
    # model = PPO("CnnPolicy", env, verbose=1, gamma=0.95, n_epochs=20, seed=2)
    # "CnnLstmPolicy"
    model = RecurrentPPO(custom_policy_lstm.CustomCnnLstmPolicy, env, verbose=1, gamma=0.95, n_epochs=200, learning_rate=3e-4, batch_size=64, seed=constants.RANDOM_SEED, device=device)
    # model = PPO(custom_policy_ppo.CustomCnnPolicy, env, verbose=1, gamma=0.95, n_epochs=100, learning_rate=0.01, batch_size=32, seed=constants.RANDOM_SEED, device=device)

    print("Training : Start")

    # # Train the model
    model.learn(total_timesteps=10000, progress_bar=True, callback=reward_callback, reset_num_timesteps=True)

    print("Training : Complete")

    # Save the model
    model.save(model_file_path)

def test(device):
    
    print("Testing : Start")

    # Load the model later for evaluation
    # loaded_model = PPO.load(model_file_path)
    loaded_model = RecurrentPPO.load(model_file_path, device=device)
    # loaded_model = PPO.load(model_file_path, device=device)

    # Evaluate the model
    evaluate_model(loaded_model, create_env())

    print("Testing : Complete")

def load_and_predict(env):
    
    model = RecurrentPPO.load(model_file_path)
    # model = PPO.load(model_file_path)

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
        env._update(True)

    print(f"Test episode reward: {episode_reward}")
    env.save_screen_image(f"./saves/custom_path_levels/test.png")

DEVICE = 'cuda:0'
if(__name__ == "__main__"):
    check_env()
    train(device=DEVICE)
    test(device=DEVICE)