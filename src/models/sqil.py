import os
import torch 
import torch.nn.functional as F
import pickle
import numpy as np
from typing import Dict
from src.buffers.replay_buffer import ReplayBuffer
from src.networks.mlp import SoftQNetwork
from stable_baselines3.common.vec_env import VecEnv
from typing import (
    Any,
    Dict,
    List,
    Mapping,
)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

class SQIL:

    def __init__(self, env:VecEnv, args: Dict) -> None:

        self.args = args
        self.env = env
        self.reset()

    def learn(self) -> None:

        self.learning_step+=1
        if self.learning_step<self.args["model_parameters"]["batch_size"]:
            return
        
        sample_state,sample_action,sample_reward,sample_next_state,sample_done = self.sample_buffer.shuffle()

        sample_state = torch.Tensor(sample_state)
        sample_action  = torch.Tensor(sample_action)
        sample_reward = torch.Tensor(sample_reward)
        sample_next_state = torch.Tensor(sample_next_state)
        sample_done = torch.Tensor(sample_done)

        with torch.no_grad():
            next_q = self.TargetQNet(sample_next_state)
            next_v = self.TargetQNet.getVVal(next_q)
            y = sample_reward + (1 - sample_done) * self.args["model_parameters"]["gamma"] * next_v

        sample_loss = F.mse_loss(self.QNet(sample_state).gather(1, sample_action.long()), y)

        expert_state,expert_action,expert_reward,expert_next_state,expert_done = self.expert_buffer.shuffle()

        expert_state = torch.Tensor(expert_state)
        expert_action  = torch.Tensor(expert_action)
        expert_reward = torch.Tensor(expert_reward)
        expert_next_state = torch.Tensor(expert_next_state)
        expert_done = torch.Tensor(expert_done)

        with torch.no_grad():
            next_q = self.TargetQNet(expert_next_state)
            next_v = self.TargetQNet.getVVal(next_q)
            y = expert_reward + (1 - expert_done) * self.args["model_parameters"]["gamma"] * next_v

        expert_loss = F.mse_loss(self.QNet(expert_state).gather(1, expert_action.long()), y)

        loss = self.args["model_parameters"]["lambda"]*sample_loss + expert_loss
        self.QOptimizer.zero_grad()
        loss.backward()
        self.QOptimizer.step()

        if self.learning_step%self.args["model_parameters"]["target_update"] == 0:                
            hard_update(self.TargetQNet,self.QNet)

    def predict(self,obs: np.ndarray, epsilon: float = 0.1) -> int:

        if self.args["mode"] == "training":
            if np.random.rand() < epsilon:  
            # Explore: Choose a random action
                action = np.random.randint(0, self.env.action_space.n)
            else:  
                # Exploit: Choose the best action based on Q-values
                action = self.QNet.choose_action(obs)
        else:
            action = self.TargetQNet.choose_action(obs)

        return action
    
    def reset(self) -> None:

        self.expert_buffer = ReplayBuffer(input_shape=self.env.observation_space.shape,mem_size=self.args["model_parameters"]["mem_size"],n_actions=1,batch_size=self.args["model_parameters"]["batch_size"]//2)
        self.fill_demonstrations()
        self.sample_buffer = ReplayBuffer(input_shape=self.env.observation_space.shape,mem_size=self.args["model_parameters"]["mem_size"],n_actions=1,batch_size=self.args["model_parameters"]["batch_size"]//2)
        self.learning_step = 0

        self.QNet = SoftQNetwork(observation_space=self.env.observation_space,action_space=self.env.action_space)
        self.TargetQNet = SoftQNetwork(observation_space=self.env.observation_space,action_space=self.env.action_space)
        self.QOptimizer = torch.optim.Adam(self.QNet.parameters(), lr=self.args["model_parameters"]["lr"])

        hard_update(self.TargetQNet,self.QNet)

    def fill_demonstrations(self):

        transitions = []
        for file in self.args["transition_file"]:
            with open(file, 'rb') as f:
                rollouts = pickle.load(f)
            
            if rollouts:
                transitions += rollouts 
        
        for traj in transitions:
            
            observation = traj.obs
            state = observation[:-1]
            next_state = observation[1:]
            action = traj.acts
            dones = np.zeros(len(traj.acts), dtype=bool)
            dones[-1] = traj.terminal
            for i in range(len(traj.acts)):
                self.expert_buffer.store(state[i],action[i],1,next_state[i],dones[i])
    
    def add(self,s,action,rwd,next_state,done):
        self.sample_buffer.store(s,action,0,next_state,done)

    def save(self):
        print("-------SAVING NETWORK -------")

        os.makedirs("data/models/sqil_weights", exist_ok=True)
        torch.save(self.QNet.state_dict(),"data/models/sqil_weights/actorWeights.pth")
        torch.save(self.TargetQNet.state_dict(),"data/models/sqil_weights/TargetactorWeights.pth")

    def load(self):

        self.QNet.load_state_dict(torch.load("data/models/sqil_weights/actorWeights.pth",map_location=torch.device('cpu')))
        self.TargetQNet.load_state_dict(torch.load("data/models/sqil_weights/TargetactorWeights.pth",map_location=torch.device('cpu')))

