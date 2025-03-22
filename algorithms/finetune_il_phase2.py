import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from stable_baselines3 import PPO,HerReplayBuffer, DQN, SAC
import arenax_minigames.coop_puzzle
import arenax_sai
import gymnasium.envs.registration
import gymnasium
from torch.optim import Adam
from typing import Callable
from algorithms.models.coopppo import CoopPPO
total_steps = 10000
warmup_steps = 5000

env = gymnasium.make('CoopPuzzle-v0',grid_size="med",render_mode="human")

ppo_model = CoopPPO.load("data/models/ppo_finetuned_policy4_param2.zip")
ppo_model.set_env(env)
ppo_model.set_schedule_parameters(
    total_steps=10_000_000,
    warmup_steps=3_000_000,
    critic_end_lr=1e-7,
    critic_start_lr =2.5e-4,
    actor_end_lr=1e-7,
    actor_start_lr= 0.0,
    ent_coef= 0.001,
)
# ppo_model.env.reset()
for param in ppo_model.policy.mlp_extractor.policy_net.parameters():
    param.requires_grad = True

for param in ppo_model.policy.mlp_extractor.value_net.parameters():
    param.requires_grad = True

for param in ppo_model.policy.action_net.parameters():
    param.requires_grad = True

for param in ppo_model.policy.value_net.parameters():
    param.requires_grad = True

for param_group in ppo_model.policy.optimizer.param_groups:
    print(param_group['lr'])

# Train PPO with the learning rate schedule
ppo_model.learn(
    total_timesteps=10_000_000, 
    tb_log_name = "phase 2" # 4M steps as per your requirement
)

for param_group in ppo_model.policy.optimizer.param_groups:
    print(f"Final Learning Rate: {param_group['lr']}")

# Save the fine-tuned policy
ppo_model.save("data/models/ppo_finetuned_policy4_phase2_ent_001")