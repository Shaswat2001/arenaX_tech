import torch
import torch.nn as nn
from src.networks.feature_extractor import MLPExtractor, CNNFeatureExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

class CustomCNNMLPBCPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
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

        print(type(self.features_extractor))
        print(type(self.mlp_extractor))

    def forward(self, x):
        # CNN feature extraction
        cnn_features = self.features_extractor(x)
        
        # MLP processing
        policy, value = self.mlp_extractor(cnn_features)
        
        # Action and Value predictions
        action_probs = self.action_net(policy)
        value_pred = self.value_net(value)
        
        return action_probs, value_pred
