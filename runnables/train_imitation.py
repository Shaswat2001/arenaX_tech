import argparse
from ruamel.yaml import YAML
from ruamel.yaml.scalarfloat import ScalarFloat
from algorithms.train_bc import TrainBC
from algorithms.train_dagger import TrainDAgger
from src.networks.policy import CustomCNNMLPBCPolicy
from stable_baselines3.common.policies import ActorCriticPolicy

yaml = YAML()

def load_config(file_path="config.yaml"):
    """Load YAML configuration file."""
    with open(file_path, "r") as file:
        return yaml.load(file)

def main(args):


    if args["policy"] == "CNN":

        policy = CustomCNNMLPBCPolicy
    else:
        policy = ActorCriticPolicy

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

    trainer.train()
    trainer.save()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Modify YAML configuration parameters.")
    parser.add_argument("--config", type=str, default="config/train_bc.yaml", help="Path to YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)
    for key, value in config.items():
        if isinstance(value, ScalarFloat):
            config[key] = float(value)

    print(config)
    main(config)