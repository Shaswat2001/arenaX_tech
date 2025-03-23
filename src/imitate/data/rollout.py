"""Methods to collect, analyze and manipulate transition and trajectory rollouts."""

import collections
import dataclasses
import logging
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.utils import check_for_correct_spaces
from stable_baselines3.common.vec_env import VecEnv

from imitation.data import types
from imitation.data.rollout import rollout_stats, TrajectoryAccumulator, GenTrajTerminationFn, PolicyCallable, AnyPolicy, make_sample_until

def unwrap_traj(traj: types.TrajectoryWithRew) -> types.TrajectoryWithRew:
    """Uses `RolloutInfoWrapper`-captured `obs` and `rews` to replace fields.

    This can be useful for bypassing other wrappers to retrieve the original
    `obs` and `rews`.

    Fails if `infos` is None or if the trajectory was generated from an
    environment without imitation.data.wrappers.RolloutInfoWrapper

    Args:
        traj: A trajectory generated from `RolloutInfoWrapper`-wrapped Environments.

    Returns:
        A copy of `traj` with replaced `obs` and `rews` fields.

    Raises:
        ValueError: If `traj.infos` is None
    """
    if traj.infos is None:
        raise ValueError("Trajectory must have infos to unwrap")
    ep_info = traj.infos[-1]["rollout"]
    res = dataclasses.replace(traj, obs=ep_info["obs"], rews=ep_info["rews"].astype("float64"))
    assert len(res.obs) == len(res.acts) + 1
    assert len(res.rews) == len(res.acts)
    return res

def generate_trajectories_play(
    venv: VecEnv,
    sample_until: GenTrajTerminationFn,
    rng: np.random.Generator,
    *,
    deterministic_policy: bool = False,
) -> Sequence[types.TrajectoryWithRew]:
    """Generate trajectory dictionaries using a human and an environment.

    Args:
        venv: The vectorized environments to interact with.
        sample_until: A function determining the termination condition.
            It takes a sequence of trajectories, and returns a bool.
            Most users will want to use one of `min_episodes` or `min_timesteps`.
        deterministic_policy: If True, asks policy to deterministically return
            action. Note the trajectories might still be non-deterministic if the
            environment has non-determinism!
        rng: used for shuffling trajectories.

    Returns:
        Sequence of trajectories, satisfying `sample_until`. Additional trajectories
        may be collected to avoid biasing process towards short episodes; the user
        should truncate if required.
    """
    # Collect rollout tuples.
    trajectories = []
    # accumulator for incomplete trajectories
    trajectories_accum = TrajectoryAccumulator()
    obs = venv.reset()
    assert isinstance(
        obs,
        (np.ndarray, dict),
    ), "Tuple observations are not supported."
    wrapped_obs = types.maybe_wrap_in_dictobs(obs)

    # we use dictobs to iterate over the envs in a vecenv
    for env_idx, ob in enumerate(wrapped_obs):
        # Seed with first obs only. Inside loop, we'll only add second obs from
        # each (s,a,r,s') tuple, under the same "obs" key again. That way we still
        # get all observations, but they're not duplicated into "next obs" and
        # "previous obs" (this matters for, e.g., Atari, where observations are
        # really big).
        trajectories_accum.add_step(dict(obs=ob), env_idx)

    # Now, we sample until `sample_until(trajectories)` is true.
    # If we just stopped then this would introduce a bias towards shorter episodes,
    # since longer episodes are more likely to still be active, i.e. in the process
    # of being sampled from. To avoid this, we continue sampling until all epsiodes
    # are complete.
    #
    # To start with, all environments are active.
    active = np.ones(venv.num_envs, dtype=bool)
    state = None
    dones = np.zeros(venv.num_envs, dtype=bool)
    while np.any(active):
        # policy gets unwrapped observations (eg as dict, not dictobs)
        user_input = input("Enter action (integer): ")
        if user_input.isdigit():
            acts = np.array([int(user_input)])
        obs, rews, dones, infos = venv.step(acts)
        venv.render()
        assert isinstance(
            obs,
            (np.ndarray, dict),
        ), "Tuple observations are not supported."
        wrapped_obs = types.maybe_wrap_in_dictobs(obs)

        # If an environment is inactive, i.e. the episode completed for that
        # environment after `sample_until(trajectories)` was true, then we do
        # *not* want to add any subsequent trajectories from it. We avoid this
        # by just making it never done.
        dones &= active

        new_trajs = trajectories_accum.add_steps_and_auto_finish(
            acts,
            wrapped_obs,
            rews,
            dones,
            infos,
        )
        trajectories.extend(new_trajs)

        if sample_until(trajectories):
            # Termination condition has been reached. Mark as inactive any
            # environments where a trajectory was completed this timestep.
            active &= ~dones

    # Note that we just drop partial trajectories. This is not ideal for some
    # algos; e.g. BC can probably benefit from partial trajectories, too.

    # Each trajectory is sampled i.i.d.; however, shorter episodes are added to
    # `trajectories` sooner. Shuffle to avoid bias in order. This is important
    # when callees end up truncating the number of trajectories or transitions.
    # It is also cheap, since we're just shuffling pointers.
    rng.shuffle(trajectories)  # type: ignore[arg-type]

    # Sanity checks.
    for trajectory in trajectories:
        n_steps = len(trajectory.acts)
        # extra 1 for the end
        if isinstance(venv.observation_space, spaces.Dict):
            exp_obs = {}
            for k, v in venv.observation_space.items():
                assert v.shape is not None
                exp_obs[k] = (n_steps + 1,) + v.shape
        else:
            obs_space_shape = venv.observation_space.shape
            assert obs_space_shape is not None
            exp_obs = (n_steps + 1,) + obs_space_shape  # type: ignore[assignment]
        real_obs = trajectory.obs.shape
        assert real_obs == exp_obs, f"expected shape {exp_obs}, got {real_obs}"
        assert venv.action_space.shape is not None
        exp_act = (n_steps,) + venv.action_space.shape
        real_act = trajectory.acts.shape
        assert real_act == exp_act, f"expected shape {exp_act}, got {real_act}"
        exp_rew = (n_steps,)
        real_rew = trajectory.rews.shape
        assert real_rew == exp_rew, f"expected shape {exp_rew}, got {real_rew}"

    return trajectories

def flatten_trajectories(
    trajectories: Iterable[types.Trajectory],
) -> types.Transitions:
    """Flatten a series of trajectory dictionaries into arrays.

    Args:
        trajectories: list of trajectories.

    Returns:
        The trajectories flattened into a single batch of Transitions.
    """

    def all_of_type(key, desired_type):
        return all(
            isinstance(getattr(traj, key), desired_type) for traj in trajectories
        )

    assert all_of_type("obs", types.DictObs) or all_of_type("obs", np.ndarray)
    assert all_of_type("acts", np.ndarray)

    # mypy struggles without Any annotation here.
    # The necessary constraints are enforced above.
    keys = ["obs", "next_obs", "acts", "dones", "infos"]
    parts: Mapping[str, List[Any]] = {key: [] for key in keys}
    for traj in trajectories:
        parts["acts"].append(traj.acts)

        obs = traj.obs
        parts["obs"].append(obs[:-1])
        parts["next_obs"].append(obs[1:])

        dones = np.zeros(len(traj.acts), dtype=bool)
        dones[-1] = traj.terminal
        parts["dones"].append(dones)

        if traj.infos is None:
            infos = np.array([{}] * len(traj))
        else:
            infos = traj.infos
        parts["infos"].append(infos)

    cat_parts = {
        key: types.concatenate_maybe_dictobs(part_list)
        for key, part_list in parts.items()
    }
    lengths = set(map(len, cat_parts.values()))
    assert len(lengths) == 1, f"expected one length, got {lengths}"
    return types.Transitions(**cat_parts)

def demonstration_trajectories(
    trajectories: Iterable[types.Trajectory],
) -> Dict:
    """Flatten a series of trajectory dictionaries into arrays.

    Args:
        trajectories: list of trajectories.

    Returns:
        The trajectories flattened into a single batch of Transitions.
    """

    def all_of_type(key, desired_type):
        return all(
            isinstance(getattr(traj, key), desired_type) for traj in trajectories
        )

    assert all_of_type("obs", types.DictObs) or all_of_type("obs", np.ndarray)
    assert all_of_type("acts", np.ndarray)

    # mypy struggles without Any annotation here.
    # The necessary constraints are enforced above.
    keys = ["obs", "next_obs", "acts", "dones", "rewards"]
    parts: Mapping[str, List[Any]] = {key: [] for key in keys}
    for traj in trajectories:

        dones = np.zeros(len(traj.acts), dtype=bool)
        reward = np.zeros(len(traj.acts), dtype=bool)
        dones[-1] = traj.terminal
        if traj.terminal:
            reward[-1] = 1.0
        for i in range(len(traj.acts)):
            parts["acts"].append(traj.acts[i])

            obs = traj.obs
            parts["obs"].append(obs[:-1][i])
            parts["next_obs"].append(obs[1:][i])

            parts["dones"].append(dones[i])
            parts["rewards"].append(reward[i])

    return parts
    
def rollout_play(
    venv: VecEnv,
    sample_until: GenTrajTerminationFn,
    rng: np.random.Generator,
    *,
    unwrap: bool = True,
    exclude_infos: bool = True,
    verbose: bool = True,
    **kwargs: Any,
) -> Sequence[types.TrajectoryWithRew]:
    """Generate policy rollouts by human playing the environment.

    This method is a wrapper of generate_trajectories_play that allows
    the user to additionally replace the rewards and observations with the original
    values if the environment is wrapped, to exclude the infos from the
    trajectories, and to print summary statistics of the rollout.

    The `.infos` field of each Trajectory is set to `None` to save space.

    Args:
        venv: The vectorized environments.
        sample_until: End condition for rollout sampling.
        rng: Random state to use for sampling.
        unwrap: If True, then save original observations and rewards (instead of
            potentially wrapped observations and rewards) by calling
            `unwrap_traj()`.
        exclude_infos: If True, then exclude `infos` from pickle by setting
            this field to None. Excluding `infos` can save a lot of space during
            pickles.
        verbose: If True, then print out rollout stats before saving.
        **kwargs: Passed through to `generate_trajectories`.

    Returns:
        Sequence of trajectories, satisfying `sample_until`. Additional trajectories
        may be collected to avoid biasing process towards short episodes; the user
        should truncate if required.
    """
    trajs = generate_trajectories_play(
        venv,
        sample_until,
        rng=rng,
        **kwargs,
    )
    if unwrap:
        trajs = [unwrap_traj(traj) for traj in trajs]
    if exclude_infos:
        trajs = [dataclasses.replace(traj, infos=None) for traj in trajs]
    if verbose:
        stats = rollout_stats(trajs)
        logging.info(f"Rollout stats: {stats}")
    return trajs