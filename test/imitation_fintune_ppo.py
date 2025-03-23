import arenax_minigames.coop_puzzle
import arenax_sai
import gymnasium.envs.registration
from arenax_minigames import coop_puzzle
from stable_baselines3 import PPO, DQN
from algorithms.models.coopppo import CoopPPO
import gymnasium
import pickle
# Create the environment
env = gymnasium.make('CoopPuzzle-v0',grid_size="med",render_mode="human")
import torch

ppo_model = CoopPPO.load("data/models/ppo_finetuned_policy4_phase2_ent_001.zip")

ppo_model.ent_coef = 0.0

# Set the model to eval mode
ppo_model.policy.eval()  # This is typically used to disable dropout or batch norm, if applicable

obs,_ = env.reset()
i = 0

while True:

    action = ppo_model.predict(obs)
    # user_input = input("Enter action (integer): ")
    # if user_input.isdigit():
    #     action = int(user_input)
    obs, reward, done, trunc, info = env.step(action[0])
    env.render()
    i += 1
    if done or i == 800:
      i = 0
      
      if done:
         print("FINISH")
      obs,_ = env.reset()

    
