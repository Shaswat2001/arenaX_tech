import pickle 
# from imitate.data.rollout import flatten_trajectories, Rollout
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
import arenax_minigames.coop_puzzle
import arenax_sai
import gymnasium.envs.registration
import gymnasium
import torch
from imitation.algorithms import bc

def load_data(location = "data/trajectory1.pkl"):
    transitions = None
    with open(location, 'rb') as f:
        transitions = pickle.load(f)
    
    return transitions

def save_model(policy,name = "bc_policy"):

    with open(f"data/models/{name}.pkl", "wb") as f:
        pickle.dump(policy, f)

rng = np.random.default_rng(0)
env = gymnasium.make('CoopPuzzle-v0',grid_size="med",render_mode="human")

transitions = load_data()

bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=transitions,
    rng=rng,
)
bc_trainer.train(n_epochs=10000,)
save_model(bc_trainer.policy)

