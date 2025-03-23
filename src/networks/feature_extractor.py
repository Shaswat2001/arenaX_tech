import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Custom CNN Feature Extractor
class CNNFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom CNN-based feature extractor for image-based observations.
    This module extracts features from an image input using a series of convolutional layers,
    followed by a fully connected layer to reduce dimensionality.
    
    Args:
        observation_space (gym.spaces.Box): The observation space, expected to be an image with shape (C, H, W).
        features_dim (int): The output feature dimension after extraction.
    """
    def __init__(self, observation_space, features_dim=32):
        super(CNNFeatureExtractor, self).__init__(observation_space, features_dim)

        n_input_channels: int = observation_space.shape[0]  # Number of input channels in the image
        
        # Define the CNN layers
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=n_input_channels, out_channels=32, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.Flatten()  # Flatten the output to pass into a fully connected layer
        )

        # Determine the flattened feature size by passing a sample observation
        with torch.no_grad():
            n_flatten: int = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]
        
        # Fully connected layer to obtain final feature vector
        self.fc = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CNN feature extractor.
        
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, C, H, W).
        
        Returns:
            torch.Tensor: Extracted feature representation of shape (batch_size, features_dim).
        """
        return self.fc(self.cnn(x))
    

class MLPExtractor(nn.Module):
    """
    MLP-based feature extractor.
    This module extracts features from a flattened input using separate networks for policy and value estimation.
    
    Args:
        input_dim (int): The dimensionality of the input feature vector.
        hidden_dim (int): The number of neurons in the hidden layers.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 32) -> None:
        super(MLPExtractor, self).__init__()
        
        # Policy network for action selection
        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        # Value network for state value estimation
        self.value_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for both actor and critic networks.
        
        Args:
            features (torch.Tensor): Input tensor with shape (batch_size, input_dim).
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Actor output (policy network output)
                - Critic output (value network output)
        """
        return self.forward_actor(features), self.forward_critic(features)
    
    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the actor (policy network).
        
        Args:
            features (torch.Tensor): Input feature tensor.
        
        Returns:
            torch.Tensor: Policy network output.
        """
        return self.policy_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the critic (value network).
        
        Args:
            features (torch.Tensor): Input feature tensor.
        
        Returns:
            torch.Tensor: Value network output.
        """
        return self.value_net(features)