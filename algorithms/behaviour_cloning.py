import sys
import os

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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from imitate.data import rollout

from algorithms.networks.policy import CustomCNNMLPBCPolicy
from algorithms.networks.cnn import CNNFeatureExtractor
def load_data(location = "data/trajectory3.pkl"):
    rollouts = None
    with open(location, 'rb') as f:
        rollouts = pickle.load(f)
    return rollouts

def save_model(policy,name = "bc_policy4"):

    with open(f"data/models/{name}.pkl", "wb") as f:
        pickle.dump(policy, f)

rng = np.random.default_rng(0)
env = gymnasium.make('CoopPuzzle-v0',grid_size="med",render_mode="human")

rollouts = load_data("data/trajectory3.pkl") + load_data("data/trajectory2.pkl")
transitions = rollout.flatten_trajectories(rollouts)
# policy = CustomCNNMLPBCPolicy(env.observation_space, 
#                               env.action_space,
#                               lr_schedule=lambda _: torch.finfo(torch.float32).max)
bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=transitions,
    rng=rng,
    # policy=policy,
    batch_size=64,
    minibatch_size=2,
    optimizer_kwargs = {
                        "lr": 1.0e-4,  # Learning rate
                        "eps": 1.0e-5  # Epsilon for numerical stability
                    }
)

bc_trainer.train(n_epochs=1000)
save_model(bc_trainer.policy)

