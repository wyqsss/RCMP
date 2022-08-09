import argparse
from glob import glob
import os
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from stable_baselines3 import PPO
import pandas as pd
import random

os.environ['CUDA_VISIBLE_DEVICES']="0" # 指定哪块GPU训练
config=tf.compat.v1.ConfigProto() 
# 设置最大占有GPU不超过显存的80%（可选）
# config.gpu_options.per_process_gpu_memory_fraction=0.8
config.gpu_options.allow_growth = True  # 设置动态分配GPU内存
sess=tf.compat.v1.Session(config=config)

def setup_seed(seed):
    random.seed(seed)  # 为python设置随机种子
    np.random.seed(seed)  # 为numpy设置随机种子
    tf.random.set_seed(seed)  # tf cpu fix seed
    os.environ['TF_DETERMINISTIC_OPS'] = '1'  # tf gpu fix seed, please `pip install tensorflow-determinism` first


setup_seed(42)


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
num_heads = 5
uncertainty_limit = 0.11
help_remaining = 0
eval_ep = 2000
##################### DQN ##########################


def to_one_hot(i, n_classes=None):
    a = np.zeros(n_classes, 'uint8')
    a[i] = 1
    return a


## Define Q-network q(a,s) that ouput the rewards of 4 actions by given state, i.e. Action-Value Function.
# encoding for state: 4x4 grid can be represented by one-hot vector with 16 integers.
def get_model(inputs_shape, num_heads=1):
    models = []
    for i in range(num_heads):
        ni = tl.layers.Input(inputs_shape, name='observation')
        # nh1 = tl.layers.Dense(25, W_init=tl.initializers.RandomUniform(0, 0.1), act=None, b_init=None, name='q_h_1')(ni)
        # nh2 = tl.layers.Dense(25, W_init=tl.initializers.RandomUniform(0, 0.1), act=None, b_init=None, name='q_h_2')(nh1)
        nn = tl.layers.Dense(4, act=None, b_init=None, name='q_a_s')(ni)
        models.append(tl.models.Model(inputs=ni, outputs=nn, name=f"Q-Network-{i}"))
    return models


def save_ckpt(model):  # save trained weights
    path = os.path.join('model', '_'.join([alg_name, env_id]))
    if not os.path.exists(path):
        os.makedirs(path)
    tl.files.save_weights_to_hdf5(os.path.join(path, 'dqn_model.hdf5'), model)


def load_ckpt(model):  # load trained weights
    path = os.path.join('model', '_'.join([alg_name, env_id]))
    tl.files.save_weights_to_hdf5(os.path.join(path, 'dqn_model.hdf5'), model)

def helpAvailable():
    global help_remaining
    if help_remaining > 0:
      help_remaining -= 1
      return True
    return False

def demonstratorAction(model, state):
    return model.predict(state[0], deterministic=True)[0][0]

def eval(networks, env):
    num_episodes = 100
    rList = [] #记录奖励
    for i in range(num_episodes):
        ## Reset environment and get first new observation
        s = env.reset()  # observation is state, integer 0 ~ 15
        rAll = 0
        if render: env.render()
        for j in range(99):  # step index, maximum step is 99
            ## Choose an action by greedily (with e chance of random action) from the Q-network
            vals = [[]for i in range(4)]
            for heads in range(num_heads):
                allQ = networks[heads](np.asarray([to_one_hot(s, 16)], dtype=np.float32)).numpy()
                vals[1].append(allQ[0][1])
                vals[2].append(allQ[0][2])
                vals[3].append(allQ[0][3])
                vals[0].append(allQ[0][0])
            allQ = np.array([np.sum(val) for val in vals])
            # print(f"all Q is {allQ}")
            a = np.argmax(allQ)
            ## Get new state and reward from environment
            s1, r, d, _ = env.step(a)
            rAll += r
            s = s1
            if render: env.render()
            ## Reduce chance of random action if an episode is done.
            if d: break

        print('Testing  | Episode: {}/{}  | Episode Reward: {:.4f} | Running Time: {:.4f}' \
                .format(i, num_episodes, rAll, time.time() - t0))
        rList.append(rAll)
    print("正确率: " + str(sum(rList) / num_episodes * 100) + "%")


if __name__ == '__main__':

    qnetworks = get_model([None, 16], num_heads)
    train_weightes = []
    for qnetwork in qnetworks:
        qnetwork.train()
        train_weightes.append(qnetwork.trainable_weights)

    optimizer = tf.optimizers.Adam(learning_rate=0.01)
    env = gym.make(env_id)

    PPO_model_name = "ppo_frozenlake_1e6"
    model = PPO('MlpPolicy', env, verbose=1)
    model = PPO.load(PPO_model_name)
    model.set_env(env)

    t0 = time.time()
    if args.train:
        all_episode_reward = []
        cnt_win = 0
        for i in range(num_episodes*num_heads):
            ## Reset environment and get first new observation
            s = env.reset()  # observation is state, integer 0 ~ 15
            rAll = 0
            if render: env.render()
            
            for j in range(99):  # step index, maximum step is 99
                ## Choose an action by greedily (with e chance of random action) from the Q-network
                vals = [[]for i in range(4)]
                evalQ = []
                for heads in range(num_heads):
                    allQ = qnetworks[heads](np.asarray([to_one_hot(s, 16)], dtype=np.float32)).numpy()
                    # print(allQ)
                    evalQ.append(allQ)
                    # allQ = tf.nn.softmax(allQ)
                    # print(allQ)
                    vals[1].append(allQ[0][1])
                    vals[2].append(allQ[0][2])
                    vals[3].append(allQ[0][3])
                    vals[0].append(allQ[0][0])
                variances = np.array([np.var(np.array(val)) for val in vals])
                uncertainty = np.mean(variances) / num_heads
                # print(f"uncertainty is {uncertainty}")
                allQ = np.array([np.sum(val) for val in vals])
                # print(f"all Q is {allQ}")
                a = np.argmax(allQ)
                # print(f"action is {a}")
                if uncertainty > uncertainty_limit and helpAvailable():
                    a = demonstratorAction(model, s)
                ## e-Greedy Exploration !!! sample random action
                elif np.random.rand(1) < e:
                    a = env.action_space.sample()

                ## Get new state and reward from environment
                s1, r, d, _ = env.step(a)
                if render: env.render()
                ## Obtain the Q' values by feeding the new state through our network
                he = j % num_heads
                if he > 0:
                    Q1 = qnetworks[he](np.asarray([to_one_hot(s1, 16)], dtype=np.float32)).numpy()

                    ## Obtain maxQ' and set our target value for chosen action.
                    maxQ1 = np.max(Q1)  # in Q-Learning, policy is greedy, so we use "max" to select the next action.
                    targetQ = evalQ[he]
                    targetQ[0, a] = r + lambd * maxQ1
                    ## Train network using target and predicted Q values
                    # it is not real target Q value, it is just an estimation,
                    # but check the Q-Learning update formula:
                    #    Q'(s,a) <- Q(s,a) + alpha(r + lambd * maxQ(s',a') - Q(s, a))
                    # minimizing |r + lambd * maxQ(s',a') - Q(s, a)|^2 equals to force Q'(s,a) ≈ Q(s,a)
                    with tf.GradientTape() as tape:
                        _qvalues = qnetworks[he](np.asarray([to_one_hot(s, 16)], dtype=np.float32))
                        # print(f"_qvalues is {_qvalues}, target Q is {targetQ}")
                        _loss = tl.cost.mean_squared_error(targetQ, _qvalues, is_mean=False)
                        # print(f"loss is {_loss}")
                    grad = tape.gradient(_loss, train_weightes[he])
                    # print(f"grad is : {grad}")
                    optimizer.apply_gradients(zip(grad, train_weightes[he]))

                rAll += r
                s = s1
                ## Reduce chance of random action if an episode is done.
                if d ==True:
                    e = 1. / ((i / 50) + 10)  # reduce e, GLIE: Greey in the limit with infinite Exploration
                    break
            cnt_win += rAll
            if i % eval_ep == 0:
                eval(qnetworks, env)
            ## Note that, the rewards here with random action
            print('Training  | Episode: {}/{}  | Episode Reward: {:.4f} | Running Time: {:.4f}' \
                  .format(i, num_episodes*num_heads, rAll, time.time() - t0))

            if i == 0:
                all_episode_reward.append(rAll)
            else:
                all_episode_reward.append(all_episode_reward[-1] * 0.9 + rAll * 0.1)
            if (i % 50 == 0):
                print("period: ",i, ": ")
                if (cnt_win / 50 > 0.4):
                     e += 0.01
                elif (cnt_win / 50 > 0.2):
                      e += 0.005
                elif (cnt_win / 50 > 0.1):
                      e += 0.003
                elif (cnt_win / 50 > 0.05):
                      e += 0.001
                print("current accuracy: %.2f %%" %(cnt_win/50.0*100))
                cnt_win = 0

        save_ckpt(qnetworks[0])  # save model
        plt.plot(all_episode_reward)
        if not os.path.exists('image'):
            os.makedirs('image')
        plt.savefig(os.path.join('image', '_'.join([alg_name, env_id])))
        dataframe = pd.DataFrame({f"heads{num_heads}": all_episode_reward})
        dataframe.to_csv(f"heads{num_heads}.csv",sep=',')

    if args.test:
        eval(qnetworks, env=env)
