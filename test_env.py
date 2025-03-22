import pickle 
from imitate.data import rollout
import numpy as np
import arenax_minigames.coop_puzzle
import arenax_sai
import gymnasium

from imitation.data.wrappers import RolloutInfoWrapper
from imitate.util.util import make_vec_env
from itertools import product

SEED = 42
expert = False
rng = np.random.default_rng(0)
env = gymnasium.make('CoopPuzzle-v0',grid_size="med",render_mode="human")

class ExpertPolicy:

    def __init__(self,env):

        self.env = env
traj_actions = []

stop = False
sub_traj = []

traj = []
obs,_ = env.reset()

for i in range(obs.shape[-1]):
    print(obs[:,:,i])
# print(env.player_one_start)
# print(env.player_two_start)
    
# print(env.goal_colliders)
# rollouts = rollout.rollout_play(
# env,
# rollout.make_sample_until(min_timesteps=None, min_episodes=5),
# rng=np.random.default_rng(SEED),
# )

# with open(f"data/trajectory3.pkl", "wb") as f:
#     pickle.dump(rollouts, f)