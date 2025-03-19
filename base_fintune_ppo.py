import arenax_minigames.coop_puzzle
import arenax_sai
import gymnasium.envs.registration
from stable_baselines3 import PPO
from arenax_minigames import coop_puzzle
from stable_baselines3 import PPO
import gymnasium
import pickle
# Create the environment
env = gymnasium.make('CoopPuzzle-v0',grid_size="med",render_mode="human")

ppo_model = PPO.load("gail.zip")


obs,_ = env.reset()
i = 0
while True:

    action = ppo_model.predict(obs)
    # user_input = input("Enter action (integer): ")
    # if user_input.isdigit():
    #     action = int(user_input)
    obs, reward, done, trunc, info = env.step(action[0])
    env.render()
    i += 1
    if done or i == 100000:
      i = 0
      obs,_ = env.reset()

    
