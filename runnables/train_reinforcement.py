import argparse
from ruamel.yaml import YAML
from ruamel.yaml.scalarfloat import ScalarFloat
from algorithms.train_rl import TrainRL

yaml = YAML()

def load_config(file_path="config.yaml"):
    """Load YAML configuration file."""
    with open(file_path, "r") as file:
        return yaml.load(file)

def main(args):

    trainer = TrainRL(env=args["env"],
                        model_parameters=args["model_parameters"],
                        pretrain_il=args["pretrain_il"],
                        phase1=args["phase1"],
                        phase2=args["phase2"])        
    trainer.train()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Modify YAML configuration parameters.")
    parser.add_argument("--config", type=str, default="config/train_rl.yaml", help="Path to YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)
    for key, value in config.items():
        if isinstance(value, ScalarFloat):
            config[key] = float(value)

    print(config)
    main(config)