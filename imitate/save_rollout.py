import arenax_minigames.coop_puzzle
import arenax_sai
import gymnasium.envs.registration
from imitate.data.rollout import Rollout,flatten_trajectories
import gymnasium
import pickle

num_traj = 4
env = gymnasium.make('CoopPuzzle-v0',grid_size="med",render_mode="human")
rollouts = {"obs": [], "acts": [], "infos": [], "terminal": [], }

for i in range(num_traj):

    rolloutI = Rollout()

    # Create the environment

    obs,_ = env.reset()
    while True:

        user_input = input("Enter action (integer): ")
        if user_input.isdigit():
            action = int(user_input)
        next_obs, reward, done, trunc, info = env.step(action)
        env.render()

        rolloutI.save_experience(obs,action,reward,next_obs,done)
        obs = next_obs
        if done:
            rollouts.append(rolloutI)
            break

transitions = flatten_trajectories(rollouts)

with open("data.pkl", "wb") as f:
    pickle.dump(transitions, f)