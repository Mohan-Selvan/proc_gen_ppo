import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticCnnPolicy, RecurrentActorCriticPolicy
from sb3_contrib import RecurrentPPO

from gymnasium.spaces import Box


class CustomSmallCnnFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for small observation space (5, 15, 15).
    """
    def __init__(self, observation_space: Box, features_dim: int = 128):
        super(CustomSmallCnnFeatureExtractor, self).__init__(observation_space, features_dim)
        
        # Validate the observation space
        obs_shape = observation_space.shape
        #assert obs_shape == (3, 15, 15), f"Expected observation space (5, 15, 15), got {obs_shape}"
        
        # A smaller and simpler CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(obs_shape[0], 16, kernel_size=1, stride=1, padding=1),  # Output: (16, 15, 15)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=1, stride=1, padding=1),            # Output: (32, 15, 15)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=1),            # Output: (64, 8, 8)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=1),            # Output: (64, 8, 8)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=1),            # Output: (64, 8, 8)
            nn.ReLU(),
            nn.Flatten()                                                     # Output: (64 * 8 * 8)
        )
        
        # Calculate the output size of the CNN layers
        with torch.no_grad():
            n_flatten = self.cnn(torch.zeros(1, *obs_shape)).shape[1]
        
        # Fully connected layer to produce features_dim
        self.fc = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.cnn(observations)
        return self.fc(x)


class CustomRecurrentPPOPolicy(RecurrentActorCriticCnnPolicy):
    """
    Custom policy for Recurrent PPO with a custom CNN for small observation spaces.
    """
    def __init__(self, *args, **kwargs):
        # Use the custom CNN extractor
        features_extractor_class = CustomSmallCnnFeatureExtractor
        kwargs['features_extractor_class'] = features_extractor_class
        kwargs['features_extractor_kwargs'] = {'features_dim': 128}
        super(CustomRecurrentPPOPolicy, self).__init__(*args, **kwargs)
