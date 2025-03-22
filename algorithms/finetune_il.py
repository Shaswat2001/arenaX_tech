import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from stable_baselines3 import PPO,HerReplayBuffer, DQN, SAC
import arenax_minigames.coop_puzzle
import arenax_sai
import gymnasium.envs.registration
import gymnasium
import pickle
from imitate.data import rollout
from algorithms.replay_buffer.replay_with_demonstrations import ReplayBufferWithDemonstrations
from algorithms.models.coopppo import CoopPPO

with open("data/models/bc_policy4.pkl", "rb") as f:
    loaded_policy = pickle.load(f)

# Convert to PPO-compatible format
env = gymnasium.make('CoopPuzzle-v0',grid_size="med",render_mode="human")

ppo_model = CoopPPO(
    "MlpPolicy", env, 
    policy_kwargs=dict(net_arch=[32, 32]),  # Match BC architecture
    verbose=1,
    n_epochs=2,                # Fewer updates per batch
    ent_coef=0.0,           # Minimal exploration
    learning_rate=2.5e-4,        # Small updates
    total_steps=4_000_000,
    warmup_steps=4_000_000,
    critic_end_lr=2.5e-4,
    critic_start_lr =2.5e-4,
    actor_end_lr= 0.0,
    actor_start_lr= 0.0,
    tensorboard_log="data/log/finetune/",
    gae_lambda = 0.99,
    clip_range = 0.2,
    vf_coef = 0.5,
    max_grad_norm = 0.2,
    batch_size=64
)

ppo_model.policy.load_state_dict(loaded_policy.state_dict(),strict=False)

for param in ppo_model.policy.mlp_extractor.policy_net.parameters():
    param.requires_grad = False  # Freeze policy_net parameters

for param in ppo_model.policy.mlp_extractor.value_net.parameters():
    param.requires_grad = False  # Freeze value_net parameters

for param in ppo_model.policy.action_net.parameters():
    param.requires_grad = False  # Freeze value_net parameters

for name, param in ppo_model.policy.named_parameters():
    print(f"{name} -> requires_grad: {param.requires_grad}")

with torch.no_grad():
    for param in ppo_model.policy.value_net.parameters():
        param.data *= 0.01  # Scale weights close to zero

# Train PPO
ppo_model.learn(total_timesteps=4_000_000, tb_log_name="phase_1")

# Save the fine-tuned policy
ppo_model.save("data/models/ppo_finetuned_policy4_param2.zip")