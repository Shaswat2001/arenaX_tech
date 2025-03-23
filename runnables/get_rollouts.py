import pickle 
import argparse
from src.imitate.data import rollout
import numpy as np

from imitation.data.wrappers import RolloutInfoWrapper
from src.imitate.util.util import make_vec_env

def main(args):

    SEED = 42
    env = make_vec_env(
        "CoopPuzzle-v0",
        rng=np.random.default_rng(SEED),
        n_envs=1,
        post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # to compute rollouts
        env_make_kwargs={"grid_size": "med","render_mode": "human"}
    )

    rollouts = rollout.rollout_play(
                                    env,
                                    rollout.make_sample_until(min_timesteps=None, min_episodes=args.min_episodes),
                                    rng=np.random.default_rng(SEED),
                                    )

    with open(args.filename, "wb") as f:
        pickle.dump(rollouts, f)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Modify YAML configuration parameters.")
    parser.add_argument("--filename", type=str, default="data/trajectories/trajectory4.pkl", help="Path to save demonstrations")
    parser.add_argument("--min_episodes", type=int, default=10, help="Minimum number of episodes to collect data")
    args = parser.parse_args()

    main(args)