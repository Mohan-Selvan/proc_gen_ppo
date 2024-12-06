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

class SmallObservationCNN(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for small observation space (3, 15, 15).
    """
    def __init__(self, observation_space: Box, features_dim: int = 128):
        super(SmallObservationCNN, self).__init__(observation_space, features_dim)
        
        # Validate observation space
        obs_shape = observation_space.shape
        assert obs_shape == (3, 7, 7), f"Expected observation space (3, 15, 15), got {obs_shape}"
        
        # Custom CNN for small input
        self.cnn = nn.Sequential(
            nn.Conv2d(obs_shape[0], 8, kernel_size=1, stride=1, padding=0),  # Output: (16, 15, 15)
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=1, stride=1, padding=0),  # Output: (16, 15, 15)
            nn.ReLU(),
            # nn.Conv2d(16, 32, kernel_size=1, stride=2, padding=1),            # Output: (32, 8, 8)
            # nn.ReLU(),
            # nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=1),            # Output: (64, 8, 8)
            # nn.ReLU(),
            nn.Flatten()                                                     # Output: (64 * 8 * 8 = 4096)
        )
        
        # Compute the CNN output size
        with torch.no_grad():
            n_flatten = self.cnn(torch.zeros(1, *obs_shape)).shape[1]
        
        # Fully connected layer to match `features_dim`
        self.fc = nn.Sequential(
            nn.Linear(n_flatten, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.cnn(observations)
        return self.fc(x)


class CustomCnnPolicy(ActorCriticCnnPolicy):
    """
    Custom policy for PPO with a small CNN feature extractor.
    """
    def __init__(self, *args, **kwargs):
        # Use the custom CNN feature extractor
        kwargs['features_extractor_class'] = SmallObservationCNN
        kwargs['features_extractor_kwargs'] = {'features_dim': 128}
        super(CustomCnnPolicy, self).__init__(*args, **kwargs)