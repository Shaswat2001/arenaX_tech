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

def load_model(name = "gc_policy"):
    with open(f"data/models/{name}.pkl", "rb") as f:
        loaded_policy = pickle.load(f)

    return loaded_policy

loaded_policy = load_model()

obs,_ = env.reset()
i = 0
while True:

    action = loaded_policy.predict(obs)
    obs, reward, done, trunc, info = env.step(action[0])
    env.render()
    i += 1
    if done or i == 100000:
      i = 0
      obs,_ = env.reset()

    
