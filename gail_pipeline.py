import pickle 
from imitation.data import rollout
import numpy as np
import arenax_minigames.coop_puzzle
import arenax_sai
import gymnasium.envs.registration
import gymnasium
import torch

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms.adversarial.gail import GAIL
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env


rng = np.random.default_rng(0)
env = gymnasium.make('CoopPuzzle-v0',grid_size="med",render_mode="human")
with open('data.pkl', 'rb') as f:
    transitions = pickle.load(f)

SEED = 42

with open("bc_policy.pkl", "rb") as f:
    loaded_policy = pickle.load(f)

rollouts = rollout.rollout(
    loaded_policy,
    env,
    rollout.make_sample_until(min_timesteps=None, min_episodes=60),
    rng=np.random.default_rng(SEED),
)


learner = PPO(
    env=env,
    policy=MlpPolicy,
    batch_size=64,
    ent_coef=0.0,
    learning_rate=0.0004,
    gamma=0.95,
    n_epochs=5,
    seed=SEED,
)
reward_net = BasicRewardNet(
    observation_space=env.observation_space,
    action_space=env.action_space,
    normalize_input_layer=RunningNorm,
)
gail_trainer = GAIL(
    demonstrations=rollouts,
    demo_batch_size=1024,
    gen_replay_buffer_capacity=512,
    n_disc_updates_per_round=8,
    venv=env,
    gen_algo=learner,
    reward_net=reward_net,
)

# train the learner and evaluate again
gail_trainer.train(20000)  # Train for 800_000 steps to match expert.
gail_trainer.save("gail.zip")

