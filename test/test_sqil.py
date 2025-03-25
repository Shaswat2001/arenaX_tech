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

    env = gymnasium.make('CoopPuzzle-v0',grid_size="med",render_mode="human")
    trainer = SQIL(env,args)

    best_reward = -np.inf
    trainer.load()
    for i in range(args["model_parameters"]["n_episodes"]):
        obs,_ = env.reset()
        reward = 0
        while True:

            action = trainer.predict(obs)
            next_obs,rwd,done,truncated,_ = env.step(action)

            reward+=rwd
            if truncated:
                break
            env.render()
            obs = next_obs

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
