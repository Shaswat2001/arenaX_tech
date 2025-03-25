import torch
import torch.nn as nn
from torch.distributions import Categorical
from stable_baselines3.common.policies import ActorCriticPolicy
class SoftQNetwork(nn.Module):
    def __init__(self, observation_space, action_space, alpha: int = 4):
        super(SoftQNetwork, self).__init__()
        
        self.alpha = alpha
        self.n_input_channels: int = observation_space.shape[0]*observation_space.shape[1]*observation_space.shape[2]
        n_actions = action_space.n

        self.fc1 = nn.Linear(self.n_input_channels, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 256)
        self.fc3 = nn.Linear(256, n_actions)
        
    def forward(self, x):
        x = torch.flatten(x).reshape(-1,self.n_input_channels)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def getVVal(self, q_value):
        v = self.alpha * torch.logsumexp(q_value / self.alpha, dim=1, keepdim=True)
        return v
        
    def choose_action(self, state):

        state = torch.FloatTensor(state)
        # print('state : ', state)
        with torch.no_grad():
            q = self.forward(state)
            v = self.getVVal(q).squeeze()
            dist = torch.exp((q-v)/self.alpha)
            dist = dist / torch.sum(dist)
            c = Categorical(dist)
            a = c.sample()
        return a.item()
