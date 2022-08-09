import gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
env = make_vec_env("FrozenLake-v1", n_envs=4)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=25000)
model.save("ppo_frozenlake_25000")

print("finish learning")
# del model # remove to demonstrate saving and loading

model = PPO.load("ppo_frozenlake")

obs = env.reset()
env.render()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()