from stable_baselines3 import PPO
import arenax_minigames.coop_puzzle
import arenax_sai
import gymnasium.envs.registration
import gymnasium
import pickle
# Load pre-trained imitation learning policy
with open("bc_policy.pkl", "rb") as f:
    loaded_policy = pickle.load(f)

# Convert to PPO-compatible format
env = gymnasium.make('CoopPuzzle-v0',grid_size="med",render_mode="human")

policy_kwargs = dict(net_arch=[32, 32])  # Match BC's architecture

ppo_model = PPO(
    "MlpPolicy", env, 
    policy_kwargs=dict(net_arch=[32, 32]),  # Match BC architecture
    verbose=1,
    n_epochs=3,                # Fewer updates per batch
    ent_coef=0.0001,           # Minimal exploration
    learning_rate=1e-4,        # Small updates
    clip_range=0.05,           # Prevent drastic changes
    vf_coef=0.2,               # Reduce value function influence
    batch_size=128,            # More stable learning
    max_grad_norm=0.1          # Prevent large updates
)

ppo_model.policy.load_state_dict(loaded_policy.state_dict(),strict=False)

# Train PPO
ppo_model.learn(total_timesteps=10000000)

# Save the fine-tuned policy
ppo_model.save("ppo_finetuned_policy1.zip")
