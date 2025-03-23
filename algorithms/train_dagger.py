import pickle 
import numpy as np
import arenax_minigames.coop_puzzle
import arenax_sai

import datetime
import gymnasium
import tempfile
from typing import List, Dict, Optional
from hydra.utils import instantiate

from imitation.algorithms import bc
from src.imitate.data import rollout
from stable_baselines3.common.policies import BasePolicy
from src.imitate.util.util import make_vec_env
from imitation.data.wrappers import RolloutInfoWrapper
from src.imitate.algorithms.dagger import SimpleDAggerTrainerWithHuman

SEED = 42

class TrainDAgger:
    """
    A class to train a behavior cloning (BC) policy using the DAgger algorithm.
    """
    
    def __init__(self, env: Dict, model_parameters: Dict, opt_parameters: Dict, transition_file: List, policy: Optional[BasePolicy] = None) -> None:
        """
        Initializes the DAgger training pipeline.
        
        Args:
            env (Dict): Environment details including name and parameters.
            model_parameters (Dict): Hyperparameters for BC training.
            opt_parameters (Dict): Optimizer parameters (learning rate, epsilon, etc.).
            transition_file (List): List of paths to transition data files.
            policy (Optional[BasePolicy]): Optional pre-initialized policy.
        """
        
        # Create vectorized environment with rollout info wrapper
        self.env = make_vec_env(
                        env["name"],
                        rng=np.random.default_rng(SEED),
                        n_envs=1,
                        post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],
                        env_make_kwargs=env["parameters"]
                    )
        
        self.rng = np.random.default_rng(0)
        self.lr = float(opt_parameters["lr"])
        self.eps = float(opt_parameters["eps"])
        self.batch_size = model_parameters["batch_size"]
        self.minibatch_size = model_parameters["minibatch_size"]
        self.epochs = model_parameters["epochs"]
        self.trainer = None
        self.transitions = None
        self.policy = policy

        self.load_transitions(transition_file)
        self.set_trainer()

    def load_transitions(self, file_path: List) -> None:
        """
        Loads transition data from provided files and flattens them into a usable format.
        
        Args:
            file_path (List): List of file paths containing transition rollouts.
        """
        transitions = []
        for file in file_path:
            with open(file, 'rb') as f:
                rollouts = pickle.load(f)
            
            if rollouts:
                transitions += rollouts

        self.transitions = rollout.flatten_trajectories(transitions)

    def set_env(self, env: Dict) -> None:
        """
        Sets up a new environment and reinitializes the trainer.
        
        Args:
            env (Dict): New environment configuration.
        """
        self.env = make_vec_env(
                        "CoopPuzzle-v0",
                        rng=np.random.default_rng(SEED),
                        n_envs=1,
                        post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],
                        env_make_kwargs=env["parameters"]
                    )
        self.set_trainer()
    
    def set_trainer(self) -> None:
        """
        Initializes the BC trainer with the given environment and transition data.
        """
        self.trainer = bc.BC(
                        observation_space=self.env.observation_space,
                        action_space=self.env.action_space,
                        demonstrations=self.transitions,
                        rng=self.rng,
                        batch_size=self.batch_size,
                        minibatch_size=self.minibatch_size,
                        optimizer_kwargs={"lr": self.lr, "eps": self.eps}
                    )

    def train(self) -> None:
        """
        Trains the policy using the DAgger algorithm with human intervention.
        """
        with tempfile.TemporaryDirectory(prefix="dagger_example_") as tmpdir:
            self.dagger_trainer = SimpleDAggerTrainerWithHuman(
                venv=self.env,
                scratch_dir=tmpdir,
                bc_trainer=self.trainer,
                rng=self.rng,
            )
            self.dagger_trainer.train(total_timesteps=self.epochs)
        
    def save(self, model_path: Optional[str] = None) -> None:
        """
        Saves the trained policy to a file.
        
        Args:
            model_path (Optional[str]): Path where the model should be saved. If not provided, a timestamped filename is used.
        """
        if model_path is None:
            date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Format current date/time
            model_path = f"data/model/bc_policy_{date_str}.pkl"
        
        with open(model_path, "wb") as f:
            pickle.dump(self.dagger_trainer.policy, f)
