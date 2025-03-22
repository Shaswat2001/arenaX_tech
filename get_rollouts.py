import pickle 
from imitate.data import rollout
import numpy as np
import arenax_minigames.coop_puzzle
import arenax_sai
import gymnasium

from imitation.data.wrappers import RolloutInfoWrapper
from imitate.util.util import make_vec_env

SEED = 42
expert = False
rng = np.random.default_rng(0)
env = make_vec_env(
    "CoopPuzzle-v0",
    grid_size="med",
    render_mode="human",
    rng=np.random.default_rng(SEED),
    n_envs=1,
    post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # to compute rollouts
)

rollouts = rollout.rollout_play(
env,
rollout.make_sample_until(min_timesteps=None, min_episodes=10),
rng=np.random.default_rng(SEED),
)

with open(f"data/trajectory4.pkl", "wb") as f:
    pickle.dump(rollouts, f)