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

def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    """
    Perform a soft update of the target network parameters.
    
    Args:
        target (torch.nn.Module): Target network whose parameters will be updated.
        source (torch.nn.Module): Source network from which parameters are copied.
        tau (float): Interpolation factor for soft update (0 < tau < 1).
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def hard_update(target: torch.nn.Module, source: torch.nn.Module) -> None:
    """
    Perform a hard update by copying parameters from the source network to the target network.
    
    Args:
        target (torch.nn.Module): Target network whose parameters will be updated.
        source (torch.nn.Module): Source network from which parameters are copied.
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

class SQIL:
    """
    Soft Q Imitation Learning (SQIL) implementation for reinforcement learning with expert demonstrations.
    """
    def __init__(self, env: VecEnv, args: Dict[str, Any]) -> None:
        """
        Initialize the SQIL algorithm.
        
        Args:
            env (VecEnv): The vectorized environment.
            args (Dict[str, Any]): Dictionary containing hyperparameters and configuration settings.
        """
        self.args = args
        self.env = env
        self.reset()

    def learn(self) -> None:
        """
        Perform one step of learning using both expert and agent-collected samples.
        """
        self.learning_step += 1
        if self.learning_step < self.args["model_parameters"]["batch_size"]:
            return  # Ensure enough samples are collected before training
        
        # Sample from the agent's replay buffer
        sample_state, sample_action, sample_reward, sample_next_state, sample_done = self.sample_buffer.shuffle()
        
        # Convert to tensors
        sample_state = torch.Tensor(sample_state)
        sample_action = torch.Tensor(sample_action)
        sample_reward = torch.Tensor(sample_reward)
        sample_next_state = torch.Tensor(sample_next_state)
        sample_done = torch.Tensor(sample_done)
        
        # Compute target Q-values using the target network
        with torch.no_grad():
            next_q = self.TargetQNet(sample_next_state)
            next_v = self.TargetQNet.getVVal(next_q)
            y = sample_reward + (1 - sample_done) * self.args["model_parameters"]["gamma"] * next_v
        
        # Compute loss for agent samples
        sample_loss = F.mse_loss(self.QNet(sample_state).gather(1, sample_action.long()), y)
        
        # Sample from the expert buffer
        expert_state, expert_action, expert_reward, expert_next_state, expert_done = self.expert_buffer.shuffle()
        
        # Convert to tensors
        expert_state = torch.Tensor(expert_state)
        expert_action = torch.Tensor(expert_action)
        expert_reward = torch.Tensor(expert_reward)
        expert_next_state = torch.Tensor(expert_next_state)
        expert_done = torch.Tensor(expert_done)
        
        # Compute target Q-values for expert samples
        with torch.no_grad():
            next_q = self.TargetQNet(expert_next_state)
            next_v = self.TargetQNet.getVVal(next_q)
            y = expert_reward + (1 - expert_done) * self.args["model_parameters"]["gamma"] * next_v
        
        # Compute loss for expert samples
        expert_loss = F.mse_loss(self.QNet(expert_state).gather(1, expert_action.long()), y)
        
        # Compute total loss with weighting factor lambda
        loss = self.args["model_parameters"]["lambda"] * sample_loss + expert_loss
        
        # Perform optimization step
        self.QOptimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.QNet.parameters(), max_norm=5)
        self.QOptimizer.step()
        
        # Soft update target network periodically
        if self.learning_step % self.args["model_parameters"]["target_update"] == 0:                
            soft_update(self.TargetQNet, self.QNet, tau=0.005)

    def predict(self, obs: np.ndarray, epsilon: float = 0.1) -> int:
        """
        Select an action using an epsilon-greedy policy.
        
        Args:
            obs (np.ndarray): Observation from the environment.
            epsilon (float): Probability of selecting a random action (for exploration).
        Returns:
            int: Selected action.
        """
        if self.args["mode"] == "training":
            if np.random.rand() < epsilon:
                return np.random.randint(0, self.env.action_space.n)  # Random action
            else:
                return self.QNet.choose_action(obs)  # Greedy action
        else:
            return self.TargetQNet.choose_action(obs)
    
    def reset(self) -> None:
        """
        Reset the agent, initialize buffers and networks.
        """
        self.expert_buffer = ReplayBuffer(
            input_shape=self.env.observation_space.shape,
            mem_size=self.args["model_parameters"]["mem_size"],
            n_actions=1,
            batch_size=self.args["model_parameters"]["batch_size"] // 2
        )
        self.fill_demonstrations()
        
        self.sample_buffer = ReplayBuffer(
            input_shape=self.env.observation_space.shape,
            mem_size=self.args["model_parameters"]["mem_size"],
            n_actions=1,
            batch_size=self.args["model_parameters"]["batch_size"] // 2
        )
        self.learning_step = 0
        
        self.QNet = SoftQNetwork(self.env.observation_space, self.env.action_space)
        self.TargetQNet = SoftQNetwork(self.env.observation_space, self.env.action_space)
        self.QOptimizer = torch.optim.Adam(self.QNet.parameters(), lr=self.args["model_parameters"]["lr"])
        
        hard_update(self.TargetQNet, self.QNet)

    def fill_demonstrations(self):
        """
        Load expert demonstrations into the expert buffer.
        """
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
    
    def add(self,s: np.ndarray, action: int,rwd: Any, next_state: np.ndarray, done: bool):
        """
        Stores the sample to the replay buffer
        """
        self.sample_buffer.store(s,action,0,next_state,done)

    def save(self):
        """
        Saves the weights of the QNet and target QNet
        """
        print("-------SAVING NETWORK -------")

        os.makedirs("data/models/sqil_weights", exist_ok=True)
        torch.save(self.QNet.state_dict(),"data/models/sqil_weights/actorWeights.pth")
        torch.save(self.TargetQNet.state_dict(),"data/models/sqil_weights/TargetactorWeights.pth")

    def load(self):
        """
        Loades the weights of the QNet and target QNet
        """
        self.QNet.load_state_dict(torch.load("data/models/sqil_weights/actorWeights.pth",map_location=torch.device('cpu')))
        self.TargetQNet.load_state_dict(torch.load("data/models/sqil_weights/TargetactorWeights.pth",map_location=torch.device('cpu')))

