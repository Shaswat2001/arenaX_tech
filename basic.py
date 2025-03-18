import arenax_minigames.coop_puzzle
import arenax_sai
import gymnasium.envs.registration
from stable_baselines3 import PPO
from arenax_minigames import coop_puzzle
from stable_baselines3 import PPO
import gymnasium

# Create the environment
env = gymnasium.make('CoopPuzzle-v0',grid_size="med",render_mode="human")

obs,_ = env.reset()
i = 0
while True:

    user_input = input("Enter action (integer): ")
    if user_input.isdigit():
        action = int(user_input)
    obs, reward, done, trunc, info = env.step(action)
    env.render()
    i += 1
    if done or i == 100000:
      i = 0
      obs,_ = env.reset()

    
