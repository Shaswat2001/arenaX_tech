import argparse
import arenax_minigames.coop_puzzle
import arenax_sai
import gymnasium
import numpy as np
from ruamel.yaml import YAML
from typing import Dict, Any
from ruamel.yaml.scalarfloat import ScalarFloat
from src.models.sqil import SQIL

yaml = YAML()

def convert_scalar_floats(obj):
    if isinstance(obj, dict):
        return {k: convert_scalar_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_scalar_floats(v) for v in obj]
    elif isinstance(obj, ScalarFloat):
        return float(obj)
    return obj

def load_config(file_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        file_path (str): Path to the YAML configuration file. Default is "config.yaml".
    
    Returns:
        Dict: Parsed configuration as a dictionary.
    """
    with open(file_path, "r") as file:
        return yaml.load(file)

def main(args: Dict[str, Any]) -> None:

    env = gymnasium.make(args["env"]["name"],**args["env"]["parameters"])
    trainer = SQIL(env,args)

    best_reward = -np.inf
    total_reward = []
    avg_reward_list = []
    total_ep_length = []
    for i in range(args["model_parameters"]["n_episodes"]):
        obs,_ = env.reset()
        reward = 0
        while True:

            action = trainer.predict(obs)
            next_obs,rwd,done,truncated,_ = env.step(action)

            trainer.add(obs,action,rwd,next_obs,done)
            trainer.learn()
            reward+=rwd
            if truncated:
                break
                
            obs = next_obs

        total_reward.append(reward)
        total_ep_length.append(env.frame)
        avg_reward = np.mean(total_reward[-40:])
        avg_ep_length = np.mean(total_ep_length[-40:])
        trainer.save()
        # if avg_reward>best_reward and i > 10:
        #     best_reward=avg_reward
        #     if args["model_parameters"]["save_rl_weights"]:
        #         print("Weights Saved !!!")
        #         trainer.save()

        print("Episode * {} * Avg Reward is ==> {}".format(i, avg_reward))
        print("Episode * {} * Avg EP Length is ==> {}".format(i, avg_ep_length))
        avg_reward_list.append(avg_reward)

if __name__ == "__main__":

    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Modify YAML configuration parameters.")
    parser.add_argument("--config", type=str, default="config/train_sqil.yaml", help="Path to YAML config file")
    args = parser.parse_args()

    # Load the configuration from the provided YAML file
    config = load_config(args.config)
    config = convert_scalar_floats(config)
    print(config)
    main(config)   # Call the main function with the loaded config
