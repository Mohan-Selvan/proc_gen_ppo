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
    def __init__(self, observation_space: Box, features_dim: int = 256):
        super(CustomSmallCnnFeatureExtractor, self).__init__(observation_space, features_dim)
        
        # Validate the observation space
        obs_shape = observation_space.shape
        n_input_channels = obs_shape[0]

        # A smaller and simpler CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        
        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim), 
            nn.ReLU()
        )

        # self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))
