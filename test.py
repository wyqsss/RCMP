from concurrent.futures import BrokenExecutor
from turtle import done
import gym

from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
env = gym.make("FrozenLake-v1")

model = PPO("MlpPolicy", env, verbose=1)

model = PPO.load("ppo_frozenlake_1e6")

# model = DQN("MlpPolicy", env, verbose=1)
# model = DQN.load("dqn_frozenlake_1e6")

obs = env.reset()
sc = 0
ncount = 1000
step = 0
for i in range(ncount):
    obs = env.reset()
    env.render()
    count = 0
    while True:
        action, _states = model.predict(obs)

        obs, rewards, dones, info = env.step(action)
        count += 1
        print(f"action is {action}, reward is {rewards}, done is {done}, info is {info}, obs is {obs}")
        if rewards == 1:
            print("successfully")
            sc += 1
        print(dones)
        if dones:
            step += count
            print("done----")
            break

print(f"success rate is {sc/ncount}")
print(f"平均步长：{step/1000}")