import argparse
from ruamel.yaml import YAML
from ruamel.yaml.scalarfloat import ScalarFloat
from algorithms.train_bc import TrainBC
from algorithms.train_dagger import TrainDAgger
from src.networks.policy import CustomCNNMLPBCPolicy
from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Dict, Any

# Initialize YAML parser
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
        file_path (str): Path to the YAML configuration file.
    
    Returns:
        dict: Parsed configuration as a dictionary.
    """
    with open(file_path, "r") as file:
        return yaml.load(file)

def main(args: Dict[str, Any]) -> None:
    """
    Main function to train a model using BC or DAgger based on the provided configuration.
    
    Args:
        args (Dict): Configuration dictionary containing parameters for training.
    """
    
    # Select policy type based on configuration
    if args["policy"] == "CNN":
        policy = CustomCNNMLPBCPolicy
    else:
        policy = ActorCriticPolicy

    # Choose training algorithm based on target
    if args["target"] == "BC":
        trainer = TrainBC(env=args["env"],
                          model_parameters=args["model_parameters"],
                          opt_parameters=args["opt_parameters"],
                          transition_file=args["transition_file"],
                          policy=policy)
        
    elif args["target"] == "DAgger":
        trainer = TrainDAgger(env=args["env"],
                              model_parameters=args["model_parameters"],
                              opt_parameters=args["opt_parameters"],
                              transition_file=args["transition_file"],
                              policy=policy)
    
    # Train the model
    trainer.train()
    # Save the trained model
    trainer.save()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Modify YAML configuration parameters.")
    parser.add_argument("--config", type=str, default="config/train_bc.yaml", help="Path to YAML config file")
    args = parser.parse_args()

    # Load configuration from YAML file
    config = load_config(args.config)
    
    # Convert ScalarFloat values to standard float
    config = convert_scalar_floats(config)
    
    # Run the main training process
    main(config)
