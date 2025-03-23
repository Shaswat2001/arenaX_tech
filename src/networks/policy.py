import torch
import torch.nn as nn
import gymnasium as gym
from typing import Callable
from src.networks.feature_extractor import MLPExtractor, CNNFeatureExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

class CustomCNNMLPBCPolicy(ActorCriticPolicy):
    """
    Custom actor-critic policy combining CNN and MLP feature extractors.
    The CNN extracts features from high-dimensional image observations, and the MLP further processes them.
    
    Args:
        observation_space (gym.spaces.Box): The observation space.
        action_space (gym.spaces.Discrete or gym.spaces.Box): The action space.
        lr_schedule (Callable): Learning rate schedule.
    """
    def __init__(self, observation_space: gym.spaces.Box, action_space: gym.spaces.Discrete, lr_schedule: Callable, **kwargs) -> None:
        super(CustomCNNMLPBCPolicy,self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=CNNFeatureExtractor,
            features_extractor_kwargs={"features_dim": 128},  # CNN output feature size
            **kwargs
        )

        # MLP extractor that processes the features from CNN extractor
        self.mlp_extractor = MLPExtractor(128, hidden_dim=32)  # MLP will process CNN features

        # Action and Value output layers
        self.action_net = nn.Linear(32, action_space.n)  # Action output layer
        self.value_net = nn.Linear(32, 1)  # Value output layer

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the policy.
        
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, C, H, W).
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Action logits (before softmax)
                - Value predictions
        """
        # CNN feature extraction
        cnn_features = self.features_extractor(x)
        
        # MLP processing
        policy, value = self.mlp_extractor(cnn_features)
        
        # Action and Value predictions
        action_probs = self.action_net(policy)
        value_pred = self.value_net(value)
        
        return action_probs, value_pred
