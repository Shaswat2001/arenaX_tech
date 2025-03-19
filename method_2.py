import pickle 
import numpy as np
import arenax_minigames.coop_puzzle
import arenax_sai
import gymnasium.envs.registration
import gymnasium
import torch
from imitation.algorithms import bc

rng = np.random.default_rng(0)
env = gymnasium.make('CoopPuzzle-v0',grid_size="med",render_mode="human")
with open('data.pkl', 'rb') as f:
    transitions = pickle.load(f)

data = np.load("/Users/shaswatgarg/Downloads/final.npz", allow_pickle=True)
print(data)
num_trajs = len(data["indices"])
fields = [
    # Account for the extra obs in each trajectory
    np.split(data["obs"], data["indices"] + np.arange(num_trajs) + 1),
    np.split(data["acts"], data["indices"]),
    np.split(data["infos"], data["indices"]),
    data["terminal"],
]

print(num_trajs)
print(len(fields))

# obs,_ = env.reset()
# for action in transitions.acts:

#     obs, reward, done, trunc, info = env.step(action[0])
#     env.render()
#     if done:
#       obs,_ = env.reset()
