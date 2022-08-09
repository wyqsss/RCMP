"""
Deep Q-Network Q(a, s)
-----------------------
TD Learning, Off-Policy, e-Greedy Exploration (GLIE).
Q(S, A) <- Q(S, A) + alpha * (R + lambda * Q(newS, newA) - Q(S, A))
delta_w = R + lambda * Q(newS, newA)
See David Silver RL Tutorial Lecture 5 - Q-Learning for more details.
Reference
----------
original paper: https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
EN: https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0#.5m3361vlw
CN: https://zhuanlan.zhihu.com/p/25710327
Note: Policy Network has been proved to be better than Q-Learning, see tutorial_atari_pong.py
Environment
-----------
# The FrozenLake v0 environment
https://gym.openai.com/envs/FrozenLake-v0
The agent controls the movement of a character in a grid world. Some tiles of
the grid are walkable, and others lead to the agent falling into the water.
Additionally, the movement direction of the agent is uncertain and only partially
depends on the chosen direction. The agent is rewarded for finding a walkable
path to a goal tile.
SFFF       (S: starting point, safe)
FHFH       (F: frozen surface, safe)
FFFH       (H: hole, fall to your doom)
HFFG       (G: goal, where the frisbee is located)
The episode ends when you reach the goal or fall in a hole. You receive a reward
of 1 if you reach the goal, and zero otherwise.
Prerequisites
--------------
tensorflow>=2.0.0a0
tensorlayer>=2.0.0
To run
-------
python tutorial_DQN.py --train/test
"""
import argparse
import os
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorlayer as tl

# add arguments in command  --train/test
parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='test', action='store_true', default=True)
args = parser.parse_args()
tl.logging.set_verbosity(tl.logging.DEBUG)

#####################  hyper parameters  ####################
env_id = 'FrozenLake-v1'
alg_name = 'DQN'
lambd = .99  # decay factor
e = 0.01  # e-Greedy Exploration, the larger the more random
num_episodes = 2000
render = False  # display the game environment
rList = [] #记录奖励
##################### DQN ##########################


def to_one_hot(i, n_classes=None):
    a = np.zeros(n_classes, 'uint8')
    a[i] = 1
    return a

custom_map = [
        "HFGFFF",
        "HFFFHF",
        "HFFFFH",
        "FFFFFF",
        "FFHFFS"
    ]

## Define Q-network q(a,s) that ouput the rewards of 4 actions by given state, i.e. Action-Value Function.
# encoding for state: 4x4 grid can be represented by one-hot vector with 16 integers.
def get_model(inputs_shape, num_heads=1):
    ni = tl.layers.Input(inputs_shape, name='observation')
    nh1 = tl.layers.Dense(10, act=None, b_init=None, name='q_h_1')(ni)
    nn = tl.layers.Dense(4, act=None, b_init=None, name='q_a_s')(nh1)
    # models = []
    # for i in range(num_heads):
    #     models.append(tl.models.Model(inputs=ni, outputs=nn, name="Q-Network"))
    return tl.models.Model(inputs=ni, outputs=nn, name="Q-Network")


def save_ckpt(model):  # save trained weights
    path = os.path.join('model', '_'.join([alg_name, env_id]))
    if not os.path.exists(path):
        os.makedirs(path)
    tl.files.save_weights_to_hdf5(os.path.join(path, 'dqn_model.hdf5'), model)


def load_ckpt(model):  # load trained weights
    path = os.path.join('model', '_'.join([alg_name, env_id]))
    tl.files.load_hdf5_to_weights(os.path.join(path, 'demo.hdf5'), model)


if __name__ == '__main__':
    env = gym.make(env_id, desc=custom_map)
    nactions = env.action_space.n
    nobserves = env.observation_space.n
    t0 = time.time()

    qnetwork = get_model([None, nobserves])
    # qnetwork.train()
    # train_weights = qnetwork.trainable_weights

    # optimizer = tf.optimizers.SGD(learning_rate=0.1)

    t0 = time.time()
    # if args.train:
    #     all_episode_reward = []
    #     for i in range(num_episodes):
    #         ## Reset environment and get first new observation
    #         s = env.reset()  # observation is state, integer 0 ~ 15
    #         rAll = 0
    #         if render: env.render()
    #         for j in range(99):  # step index, maximum step is 99
    #             ## Choose an action by greedily (with e chance of random action) from the Q-network
    #             allQ = qnetwork(np.asarray([to_one_hot(s, 16)], dtype=np.float32)).numpy()
    #             a = np.argmax(allQ, 1)

    #             ## e-Greedy Exploration !!! sample random action
    #             if np.random.rand(1) < e:
    #                 a[0] = env.action_space.sample()
    #             ## Get new state and reward from environment
    #             s1, r, d, _ = env.step(a[0])
    #             if render: env.render()
    #             ## Obtain the Q' values by feeding the new state through our network
    #             Q1 = qnetwork(np.asarray([to_one_hot(s1, 16)], dtype=np.float32)).numpy()

    #             ## Obtain maxQ' and set our target value for chosen action.
    #             maxQ1 = np.max(Q1)  # in Q-Learning, policy is greedy, so we use "max" to select the next action.
    #             targetQ = allQ
    #             targetQ[0, a[0]] = r + lambd * maxQ1
    #             ## Train network using target and predicted Q values
    #             # it is not real target Q value, it is just an estimation,
    #             # but check the Q-Learning update formula:
    #             #    Q'(s,a) <- Q(s,a) + alpha(r + lambd * maxQ(s',a') - Q(s, a))
    #             # minimizing |r + lambd * maxQ(s',a') - Q(s, a)|^2 equals to force Q'(s,a) ≈ Q(s,a)
    #             with tf.GradientTape() as tape:
    #                 _qvalues = qnetwork(np.asarray([to_one_hot(s, 16)], dtype=np.float32))
    #                 # print(f"_qvalues is {_qvalues}, target Q is {targetQ}")
    #                 _loss = tl.cost.mean_squared_error(targetQ, _qvalues, is_mean=False)
    #             grad = tape.gradient(_loss, train_weights)
    #             optimizer.apply_gradients(zip(grad, train_weights))

    #             rAll += r
    #             s = s1
    #             ## Reduce chance of random action if an episode is done.
    #             if d ==True:
    #                 e = 1. / ((i / 50) + 10)  # reduce e, GLIE: Greey in the limit with infinite Exploration
    #                 break

    #         ## Note that, the rewards here with random action
    #         print('Training  | Episode: {}/{}  | Episode Reward: {:.4f} | Running Time: {:.4f}' \
    #               .format(i, num_episodes, rAll, time.time() - t0))

    #         if i == 0:
    #             all_episode_reward.append(rAll)
    #         else:
    #             all_episode_reward.append(all_episode_reward[-1] * 0.9 + rAll * 0.1)

    #     save_ckpt(qnetwork)  # save model
    #     plt.plot(all_episode_reward)
    #     if not os.path.exists('image'):
    #         os.makedirs('image')
    #     plt.savefig(os.path.join('image', '_'.join([alg_name, env_id])))

    if args.test:
        num_episodes = 1000
        load_ckpt(qnetwork)  # load model
        qnetwork.eval()
        step = 0
        for i in range(num_episodes):
            ## Reset environment and get first new observation
            s = env.reset()  # observation is state, integer 0 ~ 15
            rAll = 0
            if render: env.render()
            for j in range(99):  # step index, maximum step is 99
                ## Choose an action by greedily (with e chance of random action) from the Q-network
                allQ = qnetwork(np.asarray([to_one_hot(s, nobserves)], dtype=np.float32)).numpy()
                a = np.argmax(allQ, 1)  # no epsilon, only greedy for testing

                ## Get new state and reward from environment
                s1, r, d, _ = env.step(a[0])
                rAll += r
                s = s1
                if render: env.render()
                ## Reduce chance of random action if an episode is done.
                if d:
                    step += j 
                    break

            print('Testing  | Episode: {}/{}  | Episode Reward: {:.4f} | Running Time: {:.4f}' \
                  .format(i, num_episodes, rAll, time.time() - t0))
            rList.append(rAll)
        print("正确率: " + str(sum(rList) / num_episodes * 100) + "%")
        print(f"平均步数: {step/1000}")