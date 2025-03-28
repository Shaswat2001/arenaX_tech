import arenax_minigames.coop_puzzle
import arenax_sai
import gymnasium.envs.registration
from stable_baselines3 import PPO
from arenax_minigames import coop_puzzle
from stable_baselines3 import PPO
import gymnasium
import pickle
import argparse
# Create the environment
env = gymnasium.make('CoopPuzzle-v0',grid_size="med",render_mode="human")

def load_model(model_path: str):
    with open(model_path, "rb") as f:
        loaded_policy = pickle.load(f)
    return loaded_policy

def main(model_path: str) -> None:
    loaded_policy = load_model(model_path)

    obs,_ = env.reset()
    i = 0
    while True:

        action = loaded_policy.predict(obs)
        obs, reward, done, trunc, info = env.step(action[0])
        env.render()
        i += 1
        if done or i == 600:
            i = 0
            obs,_ = env.reset()

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Modify YAML configuration parameters.")
    parser.add_argument("--model_file", type=str, default="data/models/imitation/bc_policy_best.pkl", help="Path to YAML config file")
    args = parser.parse_args()
    
    # Run the main training process
    main(args.model_file)
