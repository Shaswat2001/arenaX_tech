import pickle 
import numpy as np
import arenax_minigames.coop_puzzle
import arenax_sai
import datetime
import torch
from datetime import datetime
import gymnasium
from typing import List, Dict, Optional
from src.models.coopppo import CoopPPO
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import EvalCallback, EventCallback

class TrainRL:
    """
    A class for training reinforcement learning models using CoopPPO in a cooperative puzzle environment.
    """
    def __init__(self, env: Dict, model_parameters: Dict, pretrain_il: Dict, phase1: Dict, phase2: Dict) -> None:
        """
        Initializes the training pipeline.
        
        Args:
            env (Dict): Environment details including name and parameters.
            model_parameters (Dict): Hyperparameters for RL model.
            pretrain_il (Dict): Parameters related to imitation learning pretraining.
            phase1 (Dict): Phase 1 training configuration.
            phase2 (Dict): Phase 2 training configuration.
        """
        self.env = gymnasium.make(env["name"],**env["parameters"])
        self.model_parameters = model_parameters
        self.pretrain_il = pretrain_il
        self.phase1 = phase1
        self.phase2 = phase2

        # Initialize RL trainer (CoopPPO model)
        self.trainer = CoopPPO(
            "MlpPolicy", self.env, 
            policy_kwargs=dict(net_arch=[32, 32]),
            verbose=model_parameters["verbose"],
            n_epochs=model_parameters["n_epochs"], 
            ent_coef=model_parameters["ent_coef"],         
            total_steps=model_parameters["total_steps"],
            warmup_steps=model_parameters["warmup_steps"],
            critic_end_lr=model_parameters["critic_end_lr"],
            critic_start_lr=model_parameters["critic_start_lr"],
            actor_end_lr=model_parameters["actor_end_lr"],
            actor_start_lr=model_parameters["actor_start_lr"],
            gae_lambda=model_parameters["gae_lambda"],
            clip_range=model_parameters["clip_range"],
            vf_coef=model_parameters["vf_coef"],
            max_grad_norm=model_parameters["max_grad_norm"],
            batch_size=model_parameters["batch_size"]
        )

    def train(self) -> None:
        """
        Manages the full training pipeline, including loading pretrained models and executing phase-specific training.
        """

        # Load pre-trained imitation learning model if required
        if self.pretrain_il["load"]:
            if self.pretrain_il["model_path"]:
                self.load_il_model(self.pretrain_il["model_path"])

        # Execute phase-specific training
        self.phase1_setup()
        self.phase2_setup()

        # If no phase-specific training is needed, perform standard RL training
        if not self.phase1["train"] and not self.phase2["train"]:
            self.train_rl_model(total_timesteps = self.model_parameters["total_steps"])
            
            # Save the trained model with a timestamp
            date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Format current date/time
            model_path = f"data/models/cooppo_{date_str}.zip"
            self.save(model_path)

    def phase1_setup(self) -> None:
        """
        Handles Phase 1 of training, which primarily focuses on training the critic while freezing the actor.
        """
        
        if self.phase1["load"]:
            print("LOADING PRETRAINED MODEL -- PHASE1")
            self.trainer = CoopPPO.load(self.phase1["load_model_path"])
            self.reset_parames(self.env,self.phase2["model_parameters"])
        elif self.phase1["train"]:
            print("TRAINING MODEL -- PHASE1")
            self.set_actor_learning(learn=False)  # Freeze actor
            self.set_critic_learning(learn=True)  # Train critic
            self.initialize_critic_weights(multiplier=0.01)

            self.train_rl_model(total_timesteps=self.phase1["model_parameters"]["total_steps"])
            
            # Save trained model
            date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Format current date/time
            model_path = f"data/models/cooppo_finetune_phase1_{date_str}.zip"
            self.save(model_path)
    
    def phase2_setup(self) -> None:
        """
        Handles Phase 2 of training, where both actor and critic are trained.
        """

        if self.phase2["train"]:

            self.set_actor_learning(learn=True)
            self.set_critic_learning(learn=True)

            eval_callback = EvalCallback(self.env, best_model_save_path='data/models/', eval_freq=10000,
                             deterministic=True, render=False)
            
            self.train_rl_model(total_timesteps = self.phase2["model_parameters"]["total_steps"],callback=eval_callback)

            # Save trained model
            date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Format current date/time
            model_path = f"data/models/cooppo_finetune_phase2_{date_str}.zip"
            self.save(model_path)

    def set_critic_learning(self, learn: bool) -> None:
        for param in self.trainer.policy.value_net.parameters():
            param.requires_grad = learn
    
    def set_actor_learning(self, learn: bool) -> None:

        for param in self.trainer.policy.mlp_extractor.policy_net.parameters():
            param.requires_grad = learn

        for param in self.trainer.policy.mlp_extractor.value_net.parameters():
            param.requires_grad = learn

        for param in self.trainer.policy.action_net.parameters():
            param.requires_grad = learn

    def reset_parames(self, env: Optional[VecEnv] = None, model_parameters: Optional[Dict] = None) -> None:
        """
        Resets model parameters and environment if provided.
        
        Args:
            env (Optional[VecEnv]): New environment to set. If None, environment remains unchanged.
            model_parameters (Optional[Dict]): Dictionary of model parameters to update. If None, parameters remain unchanged.
        """

        if env:
            self.trainer.set_env(env)

        if model_parameters:
            self.trainer.set_schedule_parameters(
                total_steps=model_parameters["total_steps"],
                warmup_steps=model_parameters["warmup_steps"],
                critic_end_lr=model_parameters["critic_end_lr"],
                critic_start_lr =model_parameters["critic_start_lr"],
                actor_end_lr=model_parameters["actor_end_lr"],
                actor_start_lr= model_parameters["actor_start_lr"],
                ent_coef= model_parameters["ent_coef"],
            )

    def save(self,model_path: str) ->  None:
        """
        Saves the model to a file.

        Args:
            model_path (str): Path where the model should be saved.
        """
        self.trainer.save(model_path)

    def initialize_critic_weights(self, multiplier: float) -> None:
        """
        Initializes critic weights by scaling them close to zero.

        Args:
            multiplier (float): Used while initializing the weights of critic network
        """
        with torch.no_grad():
            for param in self.trainer.policy.value_net.parameters():
                param.data *= multiplier

    def load_il_model(self, model_path: str) -> None:
        """
        Loads a pre-trained imitation learning model.

        Args:
            model_path (str): Path from where the imitation learning model needs to be loaded from.
        """
        with open(model_path, "rb") as f:
            il_policy = pickle.load(f)
        self.trainer.policy.load_state_dict(il_policy.state_dict(), strict=False)

    def train_rl_model(self, total_timesteps: int, callback: Optional[EventCallback] = None) -> None:
        """
        Trains the RL model for a given number of timesteps.

        Args:
            total_timesteps (int): Number of timesteps to train the RL model.
            callback (Optional[EventCallback]): Optional callback function for training monitoring.
        """
        self.trainer.learn(total_timesteps=total_timesteps, callback=callback)

    def load_rl_model(self, model_path: str) -> None:
        """
        Loads a trained RL model.
        
        Args:
            model_path (str): Path to the RL model file to be loaded.
        """
        self.trainer = CoopPPO.load(model_path)

