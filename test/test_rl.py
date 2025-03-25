import arenax_minigames.coop_puzzle
import arenax_sai
import gymnasium.envs.registration
from arenax_minigames import coop_puzzle
from stable_baselines3 import PPO, DQN
from src.models.coopppo import CoopPPO
import gymnasium
import argparse

# Create the environment
env = gymnasium.make('CoopPuzzle-v0',grid_size="med",render_mode="human")

def main(model_path: str) -> None:
    loaded_policy = CoopPPO.load(model_path)

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
    parser.add_argument("--model_file", type=str, default="data/models/ppo_finetuned_policy_phase2_best.zip", help="Path to YAML config file")
    args = parser.parse_args()
    
    # Run the main training process
    main(args.model_file)