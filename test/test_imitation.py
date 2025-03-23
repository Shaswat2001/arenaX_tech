import arenax_minigames.coop_puzzle
import arenax_sai
import gymnasium.envs.registration
from stable_baselines3 import PPO
from arenax_minigames import coop_puzzle
from stable_baselines3 import PPO
import gymnasium
import pickle
import torch
from imitation.algorithms.dagger import SimpleDAggerTrainer
from algorithms.networks.policy import CustomCNNMLPBCPolicy
# Create the environment
env = gymnasium.make('CoopPuzzle-v0',grid_size="med",render_mode="human")

def load_model(name = "bc_policy"):
    with open(f"data/models/{name}.pkl", "rb") as f:
        loaded_policy = pickle.load(f)
    return loaded_policy

loaded_policy = load_model("bc_policy4")

obs,_ = env.reset()
i = 0
while True:

    action = loaded_policy.predict(obs)
    obs, reward, done, trunc, info = env.step(action[0])
    env.render()
    i += 1
    if done or i == 600:
      i = 0
      obs,_ = env.reset()

    
