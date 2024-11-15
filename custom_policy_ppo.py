import gymnasium as gym
import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np


# Define the custom CNN feature extractor
class CustomCnnExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        # The base CNN feature extractor
        super().__init__(observation_space, features_dim)
        
        # Define CNN layers
        self.cnn = nn.Sequential(
            nn.Conv2d(observation_space.shape[0], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Calculate the size of the output of the CNN to set up the final layer
        with th.no_grad():
            sample_input = th.as_tensor(observation_space.sample()[np.newaxis]).float()
            n_flatten = self.cnn(sample_input).shape[1]

        # Final linear layer to reduce to features_dim
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
    
    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))
    

# Custom policy class for PPO
class CustomCnnPolicy(ActorCriticCnnPolicy):
    def __init__(self, observation_space: gym.spaces.Box, action_space: gym.spaces.Space, lr_schedule, **kwargs):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=CustomCnnExtractor,
            **kwargs
        )