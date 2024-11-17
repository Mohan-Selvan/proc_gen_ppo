import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np

import constants


# Define the custom CNN feature extractor
class CustomCnnExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        # The base CNN feature extractor
        super().__init__(observation_space, features_dim)

        # Get the number of input channels from the observation space
        n_input_channels = observation_space.shape[2]
        
        # Define CNN layers
        self.cnn = nn.Sequential(
            # First convolutional layer
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (32, 8, 8)
            
            # Second convolutional layer
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (64, 4, 4)
            
            # Third convolutional layer
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)   # Output: (128, 2, 2)
        )

        # Calculate the resulting features_dim
        with torch.no_grad():
            dummy_input = torch.zeros(1, n_input_channels, constants.OBSERVATION_WINDOW_SHAPE[0], constants.OBSERVATION_WINDOW_SHAPE[1])  # Batch size 1
            output = self.cnn(dummy_input)
            extracted_features_dim = output.numel()  # Flattened size of the output
        self._features_dim = extracted_features_dim

        # Fully connected layer to map CNN output to features_dim
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(extracted_features_dim, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations):
        cnn_output = self.cnn(observations)
        return self.linear(cnn_output)
    

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