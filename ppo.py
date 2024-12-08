import constants
import game_world

import pickle
import numpy as np
import helper
import time

import os
import json
import pygame

import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from sb3_contrib import TRPO

from torch import nn

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecTransposeImage
from stable_baselines3.common.vec_env import VecMonitor, VecTransposeImage
from stable_baselines3.common.logger import configure

from torch.utils.tensorboard import SummaryWriter

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
            action, _ = model.predict(obs, deterministic=False)

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

player_path = []
with open (constants.DEFAULT_PLAYER_PATH_FILE_PATH, 'rb') as fp:
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
                            random_seed=constants.RANDOM_SEED,
                            force_move_agent_forward=False
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
                    #print(f"Episode : {len(self.episode_rewards)}, Reward : {reward}")
        
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


class EventsSequenceCallback(BaseCallback):
    def __init__(self, log_dir, verbose=1):
        super(EventsSequenceCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.training_start_time = None
        self.training_end_time = None
        self.timings = []  # List to store logs
        self.level_count = 0

    def _on_training_start(self):
        # Log the start time of training
        self.training_start_time = time.time()
        if self.verbose:
            print(f"Training started at {self.training_start_time}")

    def _on_step(self) -> bool:
        """
        Called at every step to log terminated or truncated times.
        """
        # Get the infos, terminated, and truncated flags
        infos = self.locals.get("infos", [])

        current_time = time.time()

        for idx, info in enumerate(infos):
            log_entry = {"time": current_time, "env_index": idx}

            terminated = False
            truncated = False
            data = None
            img = None

            if "term" in info:
                terminated = info["term"]
            else:
                print("!!!!!!!!!!!!------Terminated : Key not found------!!!!!!!!!!!!!")

            if "trunc" in info:
                truncated = info["trunc"]
            else:
                print("!!!!!!!!!!!!------Truncated : Key not found------!!!!!!!!!!!!!")

            if "data" in info:
                data = info["data"]
            else:
                print("!!!!!!!!!!!!------Data : Key not found------!!!!!!!!!!!!!")

            if "img" in info:
                img = info["img"]
            else:
                print("!!!!!!!!!!!!------Img : Key not found------!!!!!!!!!!!!!")

            # Check if terminated or truncated occurred
            if terminated:

                self.level_count += 1

                base_directory = "./saves/train_levels/"
                level_id = self.level_count

                pygame.image.save(pygame.image.fromstring(img, (1024, 768), 'RGBA'), os.path.join(base_directory, f"level_{level_id}_img.png"))
                with open(os.path.join(base_directory, f'level_{level_id}_path'), 'wb') as fp:
                    np.save(fp, data["player_path"])
                with open(os.path.join(base_directory, f'level_{level_id}_grid'), 'wb') as fp:
                    np.save(fp, data["grid"])
                with open(os.path.join(base_directory, f'level_{level_id}_data'), 'w') as write_file:
                    json.dump(data, write_file)

                details = {k: info[k] for k in ('path_progress', 'term', 'trunc', 'data', 'episode')}
                details['level_id'] = level_id

                log_entry["event"] = "terminated"
                log_entry["details"] = details
                self.timings.append(log_entry)

            # if truncated:
            #     log_entry["event"] = "truncated"
            #     log_entry["details"] = 0
            #     self.timings.append(log_entry)

        return True

    def _on_training_end(self):
        # Log the end time of training
        self.training_end_time = time.time()
        if self.verbose:
            print(f"Training ended at {self.training_end_time}")

        # Append training start and end time to the logs
        self.timings.append({"time": self.training_start_time, "event": "training_start"})
        self.timings.append({"time": self.training_end_time, "event": "training_end"})

        # Save the timings to a JSON file
        os.makedirs(self.log_dir, exist_ok=True)
        timings_file = os.path.join(self.log_dir, "training_timings.json")

        with open(timings_file, "w") as f:
            json.dump(self.timings, f, indent=4)

        if self.verbose:
            print(f"Training timings saved to {timings_file}")

        # Generate a plot of the events
        self._plot_events()

    def _plot_events(self):
        """
        Generates a plot of the sequence of events (start, termination, truncation, end).
        """
        times = [entry["time"] for entry in self.timings]
        events = [entry["event"] for entry in self.timings]

        # Convert UNIX timestamps to relative time (in seconds)
        start_time = self.training_start_time
        relative_times = [t - start_time for t in times]

        # Assign a numeric value to each event type for plotting
        event_mapping = {"training_start": 0, "terminated": 1, "truncated": 2, "training_end": 3}
        event_labels = {v: k.replace("_", " ").capitalize() for k, v in event_mapping.items()}
        event_values = [event_mapping[event] for event in events]

        # Create the plot
        plt.figure(figsize=(12, 6))
        plt.scatter(relative_times, event_values, c=event_values, cmap="viridis", label="Events")
        plt.yticks(list(event_labels.keys()), list(event_labels.values()))
        plt.xlabel("Time (seconds from training start)")
        plt.ylabel("Event Type")
        plt.title("Sequence of Events During Training")
        plt.grid(True)
        plt.tight_layout()

        # Save the plot
        plot_file = os.path.join(self.log_dir, "event_sequence_plot.png")
        plt.savefig(plot_file)
        if self.verbose:
            print(f"Event sequence plot saved to {plot_file}")
        plt.close()

# class RewardLoggingCallback(BaseCallback):
#     def __init__(self, log_dir, verbose=1):
#         super(RewardLoggingCallback, self).__init__(verbose)
#         self.log_dir = log_dir
#         self.episode_rewards = []

#     def _on_step(self):

#         # Collect episode reward data after each episode
#         if len(self.locals.get("infos", [])) > 0:
#             for info in self.locals["infos"]:
#                 if "episode" in info.keys():
#                     reward = info["episode"]["r"]
#                     self.episode_rewards.append(reward)
#                     print(f"Episode : {len(self.episode_rewards)}, Reward : {reward}")
        
#         return True

#     def on_training_end(self):

#         # Save rewards to file for later analysis
#         np.save(os.path.join(self.log_dir, "episode_rewards.npy"), self.episode_rewards)

#         # Load the rewards data
#         episode_rewards = np.load(os.path.join(self.log_dir, "episode_rewards.npy"))

#         # Plot the episode rewards
#         plt.figure(figsize=(10, 5))
#         plt.plot(episode_rewards, label="Episode Reward")
#         plt.xlabel("Episode")
#         plt.ylabel("Total Reward")
#         plt.title("Reward per Episode during Training")
#         plt.legend()
#         plt.show()
#         plt.tight_layout()
#         plt.savefig("./saves/train_plot.png")
#         plt.close()

###################################################################################################

# Define a custom callback to log additional metrics to TensorBoard
# class TensorboardCallback(BaseCallback):
#     def __init__(self, verbose=0):
#         super(TensorboardCallback, self).__init__(verbose)

#     def _on_step(self) -> bool:
#         # Example: Add custom metrics
#         episode_rewards = np.mean(self.training_env.get_attr("last_episode_rewards"))
#         self.logger.record("custom/episode_rewards_mean", episode_rewards)
#         return True

###################################################################################################
    
def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

def train(device):

    # Create a directory to save logs
    tensorboard_log_dir = "./ppo_training_logs"
    os.makedirs(tensorboard_log_dir, exist_ok=True)

    env = make_vec_env(lambda: create_env(), n_envs=4, vec_env_cls=SubprocVecEnv)
    # env = VecMonitor(env)  # For monitoring
    # env = VecTransposeImage(env)  # Convert observation to channel-first

    # Enable custom TensorBoard logging
    logger = configure(tensorboard_log_dir, ["stdout", "tensorboard"])
  
    # "CnnLstmPolicy"
    model = RecurrentPPO("CnnLstmPolicy", env, verbose=1, 
                            #policy_kwargs=dict(normalize_images=False, ortho_init=True, lstm_hidden_size=256),
                            policy_kwargs = dict(
                                normalize_images=False,
                                features_extractor_class=custom_policy_lstm.CustomSmallCnnFeatureExtractor,
                                features_extractor_kwargs=dict(features_dim=1024),
                            ),
                            gamma=0.99, 
                            #gae_lambda=0.95,
                            n_epochs=10, 
                            #ent_coef=0.1,
                            #clip_range=0.3,
                            #max_grad_norm=0.5,
                            #vf_coef=0.5,
                            learning_rate=3e-4,
                            normalize_advantage=True,
                            seed=constants.RANDOM_SEED,
                            device=device,
                            tensorboard_log=tensorboard_log_dir)

    # model = PPO("CnnPolicy", 
    #             env, 
    #             verbose=1,
    #             #policy_kwargs=dict(normalize_images=False, ortho_init=True),
    #             policy_kwargs = dict(
    #                 normalize_images=False,
    #                 features_extractor_class=custom_policy_ppo.CustomCNN,
    #                 features_extractor_kwargs=dict(features_dim=1024),
    #             ),
    #             use_sde=False,
    #             #gamma=0.99,
    #             n_epochs=10,
    #             #ent_coef=0.1,
    #             #clip_range=0.3,
    #             learning_rate=3e-4,
    #             seed=constants.RANDOM_SEED,
    #             device=device,
    #             tensorboard_log=tensorboard_log_dir)
    
    # model = TRPO("CnnPolicy", 
    #             env,
    #             verbose=1,
    #             policy_kwargs = dict(
    #                 normalize_images=False,
    #                 features_extractor_class=custom_policy_ppo.CustomCNN,
    #                 features_extractor_kwargs=dict(features_dim=128),
    #             ),
    #             learning_rate=0.001,
    #             tensorboard_log=tensorboard_log_dir)

    model.set_logger(logger)

    log_dir = "./saves/training_logs/"
    os.makedirs(log_dir, exist_ok=True)

    reward_callback = RewardLoggingCallback(log_dir)
    event_callback = EventsSequenceCallback(log_dir)

    callbacks = CallbackList([reward_callback, event_callback])

    print("Training : Start")
    # # Train the model
    model.learn(total_timesteps=100000, progress_bar=True, callback=callbacks, reset_num_timesteps=True)
    print("Training : Complete")

    # Save the model
    model.save(model_file_path)

def test(device):
    
    print("Testing : Start")

    # Load the model later for evaluation
    # loaded_model = RecurrentPPO.load(model_file_path, device=device)
    loaded_model = PPO.load(model_file_path, device=device)
    # loaded_model = TRPO.load(model_file_path, device=device)

    env = create_env()

    # Evaluate the model
    evaluate_model(loaded_model, env, num_episodes=10)

    print("Testing : Complete")

def load_and_predict(env):
    
    # model = RecurrentPPO.load(model_file_path)
    model = PPO.load(model_file_path)
    # model = TRPO.load(model_file_path)

    obs, info = env.reset()  # Reset the environment and get the initial observation
    done = [False] * 1 #env.num_envs  # List of done flags for each environment
    episode_reward = 0
        
    while not all(done):  # Continue until all environments are done
        # Get action from the model
        action, _ = model.predict(obs, deterministic=False)

        # Apply the action to the environment
        # Here, action is a batch of actions for each environment
        obs, reward, terminated, truncated, info = env.step(action)
        done[0] = terminated or truncated

        episode_reward += np.sum(reward)  # Accumulate rewards across all environments
        env._update(True)

    print(f"Test episode reward: {episode_reward}")
    env.save_screen_image(f"./saves/custom_path_levels/test.png")

DEVICE = 'auto'
if(__name__ == "__main__"):
    # check_env()
    train(device=DEVICE)
    test(device=DEVICE)