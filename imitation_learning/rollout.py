import numpy as np
from typing import Iterable, List, Any, Mapping
from imitation.data import types

class Rollout:

    def __init__(self):
        
        self.clear_rollout()

    def save_experience(self,state,action,reward,next_state,done):
        
        self.states.append(state)
        self.acts.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

    def return_trajectory(self):

        return(np.array(self.states),np.array(self.acts),np.array(self.rewards),np.array(self.next_states),np.array(self.dones))

    def clear_rollout(self):

        self.acts = []
        self.states = []
        self.dones = []
        self.rewards = []
        self.next_states = []

def flatten_trajectories(
    trajectories: Iterable[Rollout],
) -> types.Transitions:
    """Flatten a series of trajectory dictionaries into arrays.

    Args:
        trajectories: list of trajectories.

    Returns:
        The trajectories flattened into a single batch of Transitions.
    """

    # mypy struggles without Any annotation here.
    # The necessary constraints are enforced above.
    keys = ["obs", "next_obs", "acts", "dones", "infos"]
    parts: Mapping[str, List[Any]] = {key: [] for key in keys}
    for traj in trajectories:

        state, acts, rewards, next_state, dones = traj.return_trajectory()
        parts["acts"].append(acts)

        parts["obs"].append(state)
        parts["next_obs"].append(next_state)
        parts["dones"].append(dones)
        infos = np.array([{}] * len(acts))
        parts["infos"].append(infos)

    cat_parts = {
        key: types.concatenate_maybe_dictobs(part_list)
        for key, part_list in parts.items()
    }
    lengths = set(map(len, cat_parts.values()))
    assert len(lengths) == 1, f"expected one length, got {lengths}"
    return types.Transitions(**cat_parts)

    


