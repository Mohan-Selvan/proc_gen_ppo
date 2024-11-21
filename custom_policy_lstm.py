import gymnasium as gym
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticCnnPolicy
from sb3_contrib import RecurrentPPO


class CustomCnnExtractor(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for observations with shape (3, 16, 16).
    """

    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)

        # Calculate the number of input channels from the observation space
        n_channels = observation_space.shape[0]

        # Define the CNN layers
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=2, stride=1, padding=1),  # Output: (32, 16, 16)
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),                          # Output: (32, 8, 8)

            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=1),          # Output: (64, 8, 8)
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),                          # Output: (64, 4, 4)

            nn.Flatten(),  # Flatten the spatial dimensions
        )

        # Compute feature dimension after the CNN
        with th.no_grad():
            dummy_input = th.zeros(1, *observation_space.shape)
            n_flatten = self.cnn(dummy_input).shape[1]

        # Define the final fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

        self._features_dim = features_dim

    def forward(self, observations):
        x = self.cnn(observations)
        return self.fc(x)


class CustomRecurrentCnnPolicy(RecurrentActorCriticCnnPolicy):
    """
    Custom Recurrent CNN Policy for Recurrent PPO with observation space of (3, 16, 16).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            features_extractor_class=CustomCnnExtractor,
            features_extractor_kwargs={"features_dim": 256},
            **kwargs,
        )