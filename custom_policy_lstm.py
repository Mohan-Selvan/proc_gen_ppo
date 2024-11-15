import gymnasium as gym
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticCnnPolicy
from sb3_contrib import RecurrentPPO

class CustomCnnExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        # Extract the shape of the input image from observation space
        super(CustomCnnExtractor, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        # Define your custom CNN layers
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Compute the shape of the output tensor after the CNN layers
        with th.no_grad():
            sample_input = th.as_tensor(observation_space.sample()[None]).float()
            cnn_output_dim = self.cnn(sample_input).shape[1]

        # Define a fully connected layer to get the final feature dimensions
        self.linear = nn.Sequential(
            nn.Linear(cnn_output_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # Pass the observation through the CNN and the final fully connected layer
        return self.linear(self.cnn(observations))
    

class CustomCnnLstmPolicy(RecurrentActorCriticCnnPolicy):
    def __init__(self, *args, **kwargs):
        # Pass the custom CNN feature extractor to the policy
        super(CustomCnnLstmPolicy, self).__init__(
            *args,
            **kwargs,
            features_extractor_class=CustomCnnExtractor,
            features_extractor_kwargs=dict(features_dim=256)  # Set the desired feature dimensions
        )
