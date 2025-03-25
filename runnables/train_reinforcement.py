import argparse
from ruamel.yaml import YAML
from ruamel.yaml.scalarfloat import ScalarFloat
from algorithms.train_rl import TrainRL
from typing import Dict, Any

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
    """
    Main function to train the RL model based on the configuration.

    Args:
        args (Dict): The dictionary containing configurations for training.
    """

    # Initialize and train the RL model based on the provided configuration
    trainer = TrainRL(env=args["env"],
                        model_parameters=args["model_parameters"],
                        pretrain_il=args["pretrain_il"],
                        phase1=args["phase1"],
                        phase2=args["phase2"])  
          
    trainer.train() # Train the model

if __name__ == "__main__":

    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Modify YAML configuration parameters.")
    parser.add_argument("--config", type=str, default="config/train_rl.yaml", help="Path to YAML config file")
    args = parser.parse_args()

    # Load the configuration from the provided YAML file
    config = load_config(args.config)
    
    # Convert ScalarFloat values to float if needed
    # for key, value in config.items():
    #     print(key)
    #     print(value)
    #     if isinstance(value, ScalarFloat):
    #         config[key] = float(value)
    config = convert_scalar_floats(config)
    print(config)
    main(config)   # Call the main function with the loaded config
