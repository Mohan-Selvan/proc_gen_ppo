import gymnasium as gym
import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np

import constants


class CustomCnnExtractor(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for observation space (3, 16, 16).
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 64):
        """
        :param observation_space: (gym.spaces.Box) The observation space of the environment
        :param features_dim: (int) Number of features extracted. This is the size of the output layer.
        """
        super(CustomCnnExtractor, self).__init__(observation_space, features_dim)

        # Sanity check for observation space
        assert observation_space.shape == (3, 16, 16), (
            "This custom feature extractor is designed for (3, 16, 16) input shapes."
        )

        # Define the CNN layers
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output size: (32, 8, 8)

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output size: (64, 4, 4)
        )

        # Calculate the flattened size of the output
        with th.no_grad():
            sample_input = th.zeros((1,) + observation_space.shape)  # (1, 3, 16, 16)
            n_flatten = self.cnn(sample_input).view(-1).shape[0]

        # Define the fully connected layers
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        Forward pass for extracting features.
        :param observations: (torch.Tensor) Input tensor (B, C, H, W)
        :return: Extracted features (B, features_dim)
        """
        x = self.cnn(observations)
        x = th.flatten(x, start_dim=1)
        return self.linear(x)
    


# Custom policy class for PPO
class CustomCnnPolicy(ActorCriticCnnPolicy):
    """
    PPO policy with a custom CNN feature extractor for (3, 16, 16) input.
    """
    def __init__(self, *args, **kwargs):
        super(CustomCnnPolicy, self).__init__(
            *args,
            **kwargs,
            features_extractor_class=CustomCnnExtractor,
            features_extractor_kwargs={"features_dim": 64},  # Customize features_dim if needed
        )