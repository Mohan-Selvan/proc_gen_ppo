import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.spaces import Box
import numpy as np

import constants


class CustomCNNExtractor(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for the small (3, 7, 7) observation space.
    """
    def __init__(self, observation_space: Box, features_dim: int = 128):
        super(CustomCNNExtractor, self).__init__(observation_space, features_dim)

        # Validate input dimensions
        assert observation_space.shape == (3, 7, 7), "Expected input shape (3, 7, 7)."

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # Output: (16, 7, 7)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # Output: (32, 7, 7)
            nn.ReLU(),
            nn.Flatten()                                          # Output: (32 * 7 * 7)
        )

        # Compute the flattened size after convolution layers
        n_flatten = 32 * 7 * 7

        # Final linear layer to map to desired feature dimension
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.cnn(observations)
        return self.linear(x)


class CustomPolicy(ActorCriticCnnPolicy):
    """
    Custom Policy with CNN backbone for small grid observations.
    """
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(
            *args,
            **kwargs,
            features_extractor_class=CustomCNNExtractor,
            features_extractor_kwargs=dict(features_dim=128)
        )
