import pickle 
import argparse
import numpy as np
from imitation.data.wrappers import RolloutInfoWrapper
from src.imitate.data import rollout
from src.imitate.util.util import make_vec_env

def main(args):
    """
    Collects rollouts from the "CoopPuzzle-v0" environment and saves them as a pickle file.
    
    Args:
        args (argparse.Namespace): Command-line arguments containing filename and min_episodes.
    """
    SEED = 42  # Fixed seed for reproducibility
    
    # Create a vectorized environment with the specified settings
    env = make_vec_env(
        "CoopPuzzle-v0",
        rng=np.random.default_rng(SEED),
        n_envs=1,
        post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # Wrapper for rollout computation
        env_make_kwargs={"grid_size": "med", "render_mode": "human"}  # Medium grid size and human-rendered mode
    )
    
    # Generate rollouts using the rollout_play function
    rollouts = rollout.rollout_play(
        env,
        rollout.make_sample_until(min_timesteps=None, min_episodes=args.min_episodes),
        rng=np.random.default_rng(SEED),
    )
    
    # Save rollouts as a pickle file
    with open(args.filename, "wb") as f:
        pickle.dump(rollouts, f)
    print(f"Rollout data saved to {args.filename}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Collect and save trajectory rollouts.")
    parser.add_argument(
        "--filename", 
        type=str, 
        default="data/trajectories/trajectory4.pkl", 
        help="Path to save collected rollouts as a pickle file."
    )
    parser.add_argument(
        "--min_episodes", 
        type=int, 
        default=10, 
        help="Minimum number of episodes to collect before saving."
    )
    args = parser.parse_args()
    
    main(args)
