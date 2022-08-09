import gym

from stable_baselines3 import DQN

env = gym.make("FrozenLake-v1")

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1e6, log_interval=4)
model.save("dqn_frozenlake_1e6")

del model # remove to demonstrate saving and loading

# model = DQN.load("dqn_cartpole")

# obs = env.reset()
# while True:
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     env.render()
#     if done:
#       obs = env.reset()