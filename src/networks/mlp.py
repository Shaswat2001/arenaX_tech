import torch
import torch.nn as nn
import numpy as np
from gymnasium import spaces
from torch.distributions import Categorical
from stable_baselines3.common.policies import ActorCriticPolicy

class SoftQNetwork(nn.Module):
    """
    Soft Q-Network for reinforcement learning.
    
    This network estimates Q-values for discrete actions given an observation.
    It also provides a soft value function estimate using the entropy-regularized formulation.
    """
    def __init__(self, observation_space: spaces.Box, action_space: spaces.Discrete, alpha: int = 4):
        """
        Initializes the SoftQNetwork.

        Args:
            observation_space (spaces.Box): The observation space of the environment (assumed to be an image-like tensor).
            action_space (spaces.Discrete): The action space of the environment (assumed to be discrete).
            alpha (int): Temperature parameter for entropy regularization.
        """        
        super(SoftQNetwork, self).__init__()
        
        self.alpha = alpha
        self.n_input_channels: int = observation_space.shape[0] * observation_space.shape[1] * observation_space.shape[2]
        n_actions = action_space.n  # Number of discrete actions

        # Fully connected layers for processing flattened input
        self.fc1 = nn.Linear(self.n_input_channels, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, n_actions)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.
        
        Args:
            x (torch.Tensor): Input tensor representing the observation.
        
        Returns:
            torch.Tensor: Q-values for each action.
        """
        x = torch.flatten(x).reshape(-1, self.n_input_channels)  # Flatten input
        x = self.relu(self.fc1(x))  # First hidden layer
        x = self.relu(self.fc2(x))  # Second hidden layer
        x = self.fc3(x)  # Output layer with Q-values
        return x

    def getVVal(self, q_value: torch.Tensor) -> torch.Tensor:
        """
        Computes the soft value function estimate from Q-values.
        
        Args:
            q_value (torch.Tensor): Q-values for each action.
        
        Returns:
            torch.Tensor: Soft state value estimate.
        """
        v = self.alpha * torch.log(torch.sum(torch.exp(q_value / self.alpha), dim=1, keepdim=True))
        v[v == float("Inf")] = 20  # Clip large values
        v[v == float("-Inf")] = -20  # Clip small values
        
        return v
        
    def choose_action(self, state: np.ndarray) -> int:
        """
        Selects an action using a soft Q-learning policy.
        
        Args:
            state (np.ndarray): The current state of the environment.
        
        Returns:
            int: The chosen action.
        """
        state = torch.FloatTensor(state)  # Convert state to tensor
        with torch.no_grad():  # Disable gradient computation for inference
            q = self.forward(state)  # Compute Q-values
            v = self.getVVal(q).squeeze()  # Compute soft value estimate
            
            # Compute softmax distribution over actions
            dist = torch.exp((q - v) / self.alpha)
            dist = dist / torch.sum(dist)
            
            # Sample an action from the computed distribution
            c = Categorical(dist)
            a = c.sample()
        
        return a.item()