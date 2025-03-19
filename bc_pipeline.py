import pickle 
from imitation_learning.rollout import flatten_trajectories, Rollout
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
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

bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=transitions,
    rng=rng,
)
bc_trainer.train(n_epochs=10000,)
torch.save(bc_trainer.policy,"bc_policy.zip")

with open("bc_policy.pkl", "wb") as f:
    pickle.dump(bc_trainer.policy, f)
# reward, _ = evaluate_policy(bc_trainer.policy, env, 1000)
# print("Reward:", reward)

