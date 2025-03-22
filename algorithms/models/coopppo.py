from stable_baselines3 import PPO
import warnings
from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union, List

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
from stable_baselines3.common.buffers import DictRolloutBuffer

import io
import pathlib
import warnings
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, ClassVar, Dict, Iterable, List, Optional, Tuple, Type, TypeVar, Union

from stable_baselines3.common.save_util import recursive_getattr, save_to_zip_file
from stable_baselines3.common.type_aliases import GymEnv, Schedule

class CoopPPO(PPO):

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[Type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        total_steps: int = 4_000_000, 
        warmup_steps: int = 2_000_000,
        critic_end_lr: float = 1e-5,
        critic_start_lr: float = 2e-4,
        actor_end_lr: float = 1e-5,
        actor_start_lr: float = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super(CoopPPO,self).__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            batch_size = batch_size,
            n_epochs= n_epochs,
            clip_range = clip_range,
            clip_range_vf = clip_range_vf,
            normalize_advantage = normalize_advantage,
            target_kl = target_kl,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
        )

        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.critic_end_lr = critic_end_lr
        self.critic_start_lr = critic_start_lr

        self.actor_end_lr = actor_end_lr
        self.actor_start_lr = actor_start_lr


        if _init_setup_model:
            self._setup_model()
        
    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

        if self.rollout_buffer_class is None:
            if isinstance(self.observation_space, spaces.Dict):
                self.rollout_buffer_class = DictRolloutBuffer
            else:
                self.rollout_buffer_class = RolloutBuffer

        self.rollout_buffer = self.rollout_buffer_class(
            self.n_steps,
            self.observation_space,  # type: ignore[arg-type]
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            **self.rollout_buffer_kwargs,
        )
        self.policy = self.policy_class(  # type: ignore[assignment]
            self.observation_space, self.action_space, self.actor_lr_schedule, use_sde=self.use_sde, **self.policy_kwargs
        )
        self.policy = self.policy.to(self.device)

        actor_params = (
            list(self.policy.mlp_extractor.policy_net.parameters()) +
            list(self.policy.mlp_extractor.value_net.parameters()) +
            list(self.policy.action_net.parameters())
        )
        critic_params = list(self.policy.value_net.parameters())

        # Create optimizer with two parameter groups
        self.policy.optimizer = th.optim.Adam([
            {'params': actor_params, 'lr': self.actor_lr_schedule(1)},
            {'params': critic_params, 'lr': self.critic_lr_schedule(1)}
        ])

    def _setup_lr_schedule(self):
        """Custom learning rate schedules for actor and critic networks."""

        def critic_lr_schedule(progress_remaining):
            """Decay from 2.5e-4 to 1.5e-5 over 2M steps, then remain constant."""
            current_step = (1 - progress_remaining) * self.total_steps
            if current_step < self.warmup_steps:
                return self.critic_start_lr - (current_step / self.warmup_steps) * (self.critic_start_lr - self.critic_end_lr)
            return self.critic_end_lr  # Constant after 2M steps

        def actor_lr_schedule(progress_remaining):
            """Increase from 0 to 1.5e-5 over 2M steps, then remain constant."""
            current_step = (1 - progress_remaining) * self.total_steps
            if current_step < self.warmup_steps:
                return (current_step / self.warmup_steps) * self.actor_end_lr
            return self.actor_end_lr  # Constant after 2M steps

        self.actor_lr_schedule = actor_lr_schedule
        self.critic_lr_schedule = critic_lr_schedule

    def _update_learning_rate(self, optimizer: Union[List[th.optim.Optimizer], th.optim.Optimizer]) -> None:
        """Update learning rates dynamically at each training step."""
        # print(self._current_progress_remaining)
        new_actor_lr = self.actor_lr_schedule(self._current_progress_remaining)
        new_critic_lr = self.critic_lr_schedule(self._current_progress_remaining)
        # print(new_actor_lr)
        self.logger.record("train/actor_learning_rate", new_actor_lr)
        self.logger.record("train/critic_learning_rate", new_critic_lr)

        optimizer.param_groups[0]['lr'] = new_actor_lr  # Actor LR
        optimizer.param_groups[1]['lr'] = new_critic_lr  # Critic LR
    
    def set_schedule_parameters(self,
                                total_steps: int = 4_000_000,
                                warmup_steps: int = 2_000_000,
                                critic_end_lr: float = 1.5e-5,
                                critic_start_lr: float =2.5e-4,
                                actor_end_lr: float = 1.5e-5,
                                actor_start_lr: float = 0.0,
                                ent_coef: float = 0.0):
        
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.critic_end_lr = critic_end_lr
        self.critic_start_lr = critic_start_lr

        self.actor_end_lr = actor_end_lr
        self.actor_start_lr = actor_start_lr
        self.ent_coef = ent_coef

    def save(
        self,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        exclude: Optional[Iterable[str]] = None,
        include: Optional[Iterable[str]] = None,
    ) -> None:
        """
        Save all the attributes of the object and the model parameters in a zip-file.

        :param path: path to the file where the rl agent should be saved
        :param exclude: name of parameters that should be excluded in addition to the default ones
        :param include: name of parameters that might be excluded but should be included anyway
        """
        # Copy parameter list so we don't mutate the original dict
        data = self.__dict__.copy()

        # Exclude is union of specified parameters (if any) and standard exclusions
        if exclude is None:
            exclude = []
        exclude = set(exclude).union(self._excluded_save_params())

        # Do not exclude params if they are specifically included
        if include is not None:
            exclude = exclude.difference(include)

        state_dicts_names, torch_variable_names = self._get_torch_save_params()
        all_pytorch_variables = state_dicts_names + torch_variable_names
        for torch_var in all_pytorch_variables:
            # We need to get only the name of the top most module as we'll remove that
            var_name = torch_var.split(".")[0]
            # Any params that are in the save vars must not be saved by data
            exclude.add(var_name)

        # Remove parameter entries of parameters which are to be excluded
        for param_name in exclude:
            data.pop(param_name, None)

        # Build dict of torch variables
        pytorch_variables = None
        if torch_variable_names is not None:
            pytorch_variables = {}
            for name in torch_variable_names:
                attr = recursive_getattr(self, name)
                pytorch_variables[name] = attr

        # Build dict of state_dicts
        params_to_save = self.get_parameters()
        del data["critic_lr_schedule"]
        del data["actor_lr_schedule"]
        save_to_zip_file(path, data=data, params=params_to_save, pytorch_variables=pytorch_variables)

