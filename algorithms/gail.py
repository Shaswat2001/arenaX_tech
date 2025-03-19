import pickle 
from imitate.data import rollout
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
from imitate.util.util import make_vec_env

def save_model(policy,name = "gail_policy"):

    with open(f"data/models/{name}.pkl", "wb") as f:
        pickle.dump(policy, f)

SEED = 42
expert = True
rng = np.random.default_rng(0)
env = make_vec_env(
    "CoopPuzzle-v0",
    grid_size="med",
    render_mode="human",
    rng=np.random.default_rng(SEED),
    n_envs=1,
    post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # to compute rollouts
)

if expert:
    with open("data/models/bc_policy.pkl", "rb") as f:
        loaded_policy = pickle.load(f)

    rollouts = rollout.rollout(
        loaded_policy,
        env,
        rollout.make_sample_until(min_timesteps=None, min_episodes=60),
        rng=np.random.default_rng(SEED),
    )
else:
    pass

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
    demo_batch_size=200,
    gen_replay_buffer_capacity=512,
    n_disc_updates_per_round=8,
    venv=env,
    gen_algo=learner,
    reward_net=reward_net,
)

# train the learner and evaluate again
gail_trainer.train(2000000)  # Train for 800_000 steps to match expert.
save_model(gail_trainer.policy)

