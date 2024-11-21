import gymnasium as gym
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticCnnPolicy, ActorCriticPolicy
from sb3_contrib import RecurrentPPO


class CustomRecurrentCnnExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super(CustomRecurrentCnnExtractor, self).__init__(observation_space, features_dim)

        # Define CNN layers to process normalized observation space (3x36x36)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # Output: (16, 36, 36)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # Output: (32, 36, 36)
            nn.ReLU(),
            nn.Flatten(),
        )
        # Calculate the output shape after conv layers
        self.conv_output_dim = 32 * 36 * 36

        # LSTM layer for the recurrent part
        self.lstm = nn.LSTM(input_size=self.conv_output_dim, hidden_size=features_dim, batch_first=True)

    def forward(self, observations, rnn_states, masks):
        # Process the observations through CNN layers
        features = self.cnn(observations)
        features = features.view(features.size(0), -1)  # Flatten the features for LSTM

        # Feed through LSTM to handle recurrent part
        lstm_out, rnn_states = self.lstm(features.unsqueeze(1), rnn_states)

        # Return the output of LSTM and the hidden state for the next step
        return lstm_out.squeeze(1), rnn_states

class CustomRecurrentPolicy(RecurrentActorCriticCnnPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomRecurrentPolicy, self).__init__(*args, **kwargs,
                                                   features_extractor_class=CustomRecurrentCnnExtractor,
                                                   features_extractor_kwargs=dict(features_dim=128))