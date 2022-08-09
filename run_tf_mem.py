import argparse
import os
from pyexpat import model
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from collections import deque
import random
from stable_baselines3 import PPO
import pandas as pd
import math
from scipy.ndimage import gaussian_filter1d

def setup_seed(seed):
    random.seed(seed)  # 为python设置随机种子
    np.random.seed(seed)  # 为numpy设置随机种子
    tf.random.set_seed(seed)  # tf cpu fix seed
    os.environ['TF_DETERMINISTIC_OPS'] = '1'  # tf gpu fix seed, please `pip install tensorflow-determinism` first


setup_seed(42)

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
  assert tf.config.experimental.get_memory_growth(physical_devices[0])
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

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
e = 0.1  # e-Greedy Exploration, the larger the more random
num_episodes = 1001
render = False  # display the game environment
rList = [] #记录奖励
minibatch_size = 64
buffer_size = 2000
NUM_HEADS =  5
memory = deque([], maxlen = buffer_size)
stride_length = 150
uncertainty_limit = 5e-5
help_remaining = 700
importance_threshold = 0.3

custom_map = [
        "HFGFFF",
        "HFFFHF",
        "HFFFFH",
        "FFFFFF",
        "FFHFFS"
    ]
##################### DQN ##########################



def to_one_hot(i, n_classes=None):
    a = np.zeros(n_classes, 'uint8')
    a[i] = 1
    return a

def to_one_hots(i, n_classes=None):
    result = []
    for k in range(len(i)):
        a = np.zeros(n_classes, 'uint8')
        a[i[k]] = 1
        result.append(a)
    result = np.asarray(result, dtype=np.float32)
    return result


## Define Q-network q(a,s) that ouput the rewards of 4 actions by given state, i.e. Action-Value Function.
# encoding for state: 4x4 grid can be represented by one-hot vector with 16 integers.0.33773518 0.33773518 0.33773518 0.33795592
def get_model(inputs_shape, num_heads=1):
    models = []
    for i in range(num_heads):
        ni = tl.layers.Input(inputs_shape, name='observation')
        nh1 = tl.layers.Dense(10, act=None, W_init=tl.initializers.RandomUniform(0, 1.), b_init=None, name='q_h_1')(ni)
        # nh2 = tl.layers.Dense(25, W_init=tl.initializers.RandomUniform(0, 0.1), act=None, b_init=None, name='q_h_2')(nh1)
        nn = tl.layers.Dense(4, act=None, b_init=None, name='q_a_s')(nh1)
        models.append(tl.models.Model(inputs=ni, outputs=nn))
    return models

def get_demo_model(inputs_shape, num_heads=1):
    ni = tl.layers.Input(inputs_shape, name='observation')
    nh1 = tl.layers.Dense(10, act=None, b_init=None, name='q_h_1')(ni)
    # nh2 = tl.layers.Dense(25, W_init=tl.initializers.RandomUniform(0, 0.1), act=None, b_init=None, name='q_h_2')(nh1)
    nn = tl.layers.Dense(4, act=None, b_init=None, name='q_a_s')(nh1)
    return tl.models.Model(inputs=ni, outputs=nn)


def save_ckpt(model, name):  # save trained weights
    path = os.path.join('model', '_'.join([alg_name, env_id]))
    if not os.path.exists(path):
        os.makedirs(path)
    tl.files.save_weights_to_hdf5(os.path.join(path, f'{name}.hdf5'), model)


def load_ckpt(model):  # load trained weights
    path = os.path.join('model', '_'.join([alg_name, env_id]))
    tl.files.load_hdf5_to_weights(os.path.join(path, 'demo.hdf5'), model)

def helpAvailable():
    global help_remaining
    if help_remaining > 0:
      help_remaining -= 1
      return True
    return False

def demonstratorAction(model, state, nobs):
    allQ = model(np.asarray([to_one_hot(state, nobs)], dtype=np.float32)).numpy()
    a = np.argmax(allQ, 1)
    return a

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

            allQ = networks(np.asarray([to_one_hot(s, env.observation_space.n)], dtype=np.float32)).numpy()
            # print(f"all Q is {allQ}")
            a = np.argmax(allQ)
            ## Get new state and reward from environment
            s1, r, d, _ = env.step(a)
            rAll += r
            s = s1
            if render: env.render()
            ## Reduce chance of random action if an episode is done.
            if d: break

        # print('Testing  | Episode: {}/{}  | Episode Reward: {:.4f} | Running Time: {:.4f}' \
        #         .format(i, num_episodes, rAll, time.time() - t0))
        rList.append(rAll)
    print("yalid 正确率: " + str(sum(rList) / num_episodes * 100) + "%")

def mheads_eval(networks, env):
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
            for heads in range(NUM_HEADS):
                allQ = networks[heads](np.asarray([to_one_hot(s, env.observation_space.n)], dtype=np.float32)).numpy()
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

def discount_eval(networks, env):
    num_episodes = 100
    total_discountrewards = 0
    for i in range(num_episodes):
        ## Reset environment and get first new observation
        s = env.reset()  # observation is state, integer 0 ~ 15
        discount_reward = 0
        for j in range(99):  # step index, maximum step is 99
            ## Choose an action by greedily (with e chance of random action) from the Q-network
            vals = [[]for i in range(4)]
            for heads in range(NUM_HEADS):
                allQ = networks[heads](np.asarray([to_one_hot(s, env.observation_space.n)], dtype=np.float32)).numpy()
                vals[1].append(allQ[0][1])
                vals[2].append(allQ[0][2])
                vals[3].append(allQ[0][3])
                vals[0].append(allQ[0][0])
            allQ = np.array([np.sum(val) for val in vals])
            # print(f"all Q is {allQ}")
            a = np.argmax(allQ)
            ## Get new state and reward from environment
            s1, r, d, _ = env.step(a)
            if r > 0:
                discount_reward += math.pow(lambd, j)*r
            s = s1
            ## Reduce chance of random action if an episode is done.
            if d: break
        total_discountrewards += discount_reward
    print(f"discount rewards is {total_discountrewards / 100}")
    return total_discountrewards / 100

if __name__ == '__main__':
    masks = []
    for m in range(NUM_HEADS):
        mask = []
        for n in range(minibatch_size):
            if n % NUM_HEADS == m:
                mask.append([1, 1, 1, 1])
            else:
                mask.append([0, 0, 0, 0])

        masks.append(mask)
    masks = np.asarray(masks)
    env = gym.make(env_id, desc=custom_map)
    nactions = env.action_space.n
    nobserves = env.observation_space.n
    qnetworks = get_model([None, nobserves], NUM_HEADS)

    qnettargets = get_model([None, nobserves], NUM_HEADS)

    train_weightes = []
    for qnetwork in qnetworks:
        qnetwork.train()
        train_weightes.append(qnetwork.trainable_weights)
    
    for qnettarget in qnettargets:
        qnettarget.eval()

    optimizer = tf.optimizers.Adam(learning_rate=1e-3)
    

    # PPO_model_name = "ppo_frozenlake_1e6"
    # model = PPO('MlpPolicy', env, verbose=1)
    # model = PPO.load(PPO_model_name)
    # model.set_env(env)
    model = get_demo_model([None, nobserves])
    load_ckpt(model)
    model.eval()

    t0 = time.time()
    if args.train:
        accum_discountrewardslist = []
        accum_discountreward = 0
        discount_rewardslist = []
        all_episode_reward = []
        uncertainty_history = []
        total_step = 0
        cnt_win = 0
        used_advicelist = []
        used_advice = 0
        for i in range(num_episodes):
            ## Reset environment and get first new observation
            s = env.reset()  # observation is state, integer 0 ~ 15
            rAll = 0
            if render: env.render()
            ep_avg_uncertainty = 0
            for j in range(99):  # step index, maximum step is 99
                ## Choose an action by greedily (with e chance of random action) from the Q-network
                vals = [[]for i in range(4)]
                
                for heads in range(NUM_HEADS):
                    allQ = qnetworks[heads](np.asarray([to_one_hot(s, nobserves)], dtype=np.float32)).numpy()
                    # print(allQ)
                    allQ = tf.nn.softmax(allQ)
            
                    # allQ = tf.nn.softmax(allQ)
                    # print(allQ)
                    vals[1].append(allQ[0][1])
                    vals[2].append(allQ[0][2])
                    vals[3].append(allQ[0][3])
                    vals[0].append(allQ[0][0])
                variances = np.array([np.var(np.array(val)) for val in vals])
                uncertainty = np.mean(variances)
                ep_avg_uncertainty += uncertainty
                # print(f"uncertainty is {uncertainty}")
                allQ = np.array([np.sum(val) for val in vals])
                # print(f"all Q is {allQ}")
                a = np.argmax(allQ)

                ## e-Greedy Exploration !!! sample random action
                # impotQ = model(np.asarray([to_one_hot(s, nobserves)], dtype=np.float32)).numpy()
                # im_sig = np.max(impotQ) - np.min(impotQ)
                # # print(f"im_sig is {im_sig}")
                # if im_sig > importance_threshold and helpAvailable():
                #     a = np.argmax(impotQ, 1)[0]
                #     used_advice += 1
                # # if uncertainty > uncertainty_limit and helpAvailable():
                #     a = demonstratorAction(model, s, nobserves)[0]
                #     used_advice += 1
                if np.random.rand(1) < 0.5 and helpAvailable():
                    a = demonstratorAction(model, s, nobserves)[0]
                    used_advice += 1
                elif np.random.rand(1) < e:
                    a = env.action_space.sample()
                ## Get new state and reward from environment
                s1, r, d, _ = env.step(a)

                rAll += r
                # 加入经验池
                # if d and r == 0.0:
                #     r = -1
                memory.append((s, a, r, s1, d))
                if render: env.render()

                if d:
                    total_step += 1
                    break
                if len(memory) < minibatch_size:
                    total_step += 1
                    s = s1
                    continue
                ## Obtain the Q' values by feeding the new state through our network
                mini_batch = random.sample(memory, minibatch_size)
                rs = [val[0] for val in mini_batch]
                ma = np.array([val[1] for val in mini_batch])
                mr = np.array([val[2] for val in mini_batch])
                ns = [val[3] for val in mini_batch]
                # d = np.array([val[4] for val in mini_batch])

                for head in range(NUM_HEADS):
                    # mini_batch = random.sample(memory, minibatch_size)
                    # rs = [val[0] for val in mini_batch]
                    # ma = np.array([val[1] for val in mini_batch])
                    # mr = np.array([val[2] for val in mini_batch])
                    # ns = [val[3] for val in mini_batch]

                    targetQ = qnettargets[head](to_one_hots(ns, nobserves)).numpy()
                    Q1 = qnetworks[head](to_one_hots(rs, nobserves)).numpy()
                    # print(targetQ.shape)
                    ## Obtain maxQ' and set our target value for chosen action.
                    # maxQ1 = np.max(targetQ)  # in Q-Learning, policy is greedy, so we use "max" to select the next action.
                    maxtarQ = [np.max(row) for row in targetQ]
                    # print(maxQ1)[ 1.3577476e-02  1.4822565e-02  3.9894313e-02 -1.3673261e-02]
                    # targetQ = Q1
                    # targetQ[0, ma[0]] = mr + lambd * maxQ1
                    for b in range(len(Q1)):
                        Q1[b][ma[b]] = mr[b] + lambd * maxtarQ[b]
                    ## Train network using target and predicted Q values
                    # it is not real target Q value, it is just an estimation,
                    # but check the Q-Learning update formula:
                    #    Q'(s,a) <- Q(s,a) + alpha(r + lambd * maxQ(s',a') - Q(s, a))
                    # minimizing |r + lambd * maxQ(s',a') - Q(s, a)|^2 equals to force Q'(s,a) ≈ Q(s,a)
                    
                    with tf.GradientTape() as tape:
                        _qvalues = qnetworks[head](to_one_hots(rs, nobserves))
                        # print(f"_qvalues is {_qvalues}, target Q is {Q1}, print masks is {masks[head]}")
                        # _qvalues = _qvalues*masks[head]
                        # Q1 = Q1*masks[head]
                        # print(f"_qvalues is {_qvalues}, target Q is {Q1}")
                        _loss = tl.cost.mean_squared_error(_qvalues, Q1, is_mean=False)
                    # print(f"loss is {_loss}")
                    grad = tape.gradient(_loss, train_weightes[head])
                    # print(f"grad is : {grad}")
                    optimizer.apply_gradients(zip(grad, train_weightes[head]))

                s = s1
                total_step += 1
                ## Reduce chance of random action if an episode is done.
                # if d ==True:
                #     e = 1. / ((i / 50) + 10)  # reduce e, GLIE: Greey in the limit with infinite Exploration
                #     break

                if total_step % stride_length == 0:
                    tl.files.assign_weights(qnetwork.all_weights, qnettarget)
                    # for eval_layer, target_layer in zip(qnetwork.all_layers, qnettarget.all_layers):
                    #     target_layer.set_weights(eval_layer.get_weights())

            cnt_win += rAll
            uncertainty_history.append(ep_avg_uncertainty/j)
            ## Note that, the rewards here with random action
            print('Training  | Episode: {}/{} | step: {}  | Episode Reward: {:.4f} | Running Time: {:.4f}' \
                  .format(i, num_episodes, j, rAll, time.time() - t0))

            if i == 0:
                all_episode_reward.append(rAll)
            else:
                all_episode_reward.append(all_episode_reward[-1] * 0.9 + rAll * 0.1)
            if (i % 50 == 0):
                
                print("current accuracy: %.2f %%" %(cnt_win/50.0*100))
                cnt_win = 0
            if i % 40 == 0:
                discount_reward = discount_eval(qnetworks, env)
                discount_rewardslist.append(discount_reward)
                accum_discountreward += discount_reward
                accum_discountrewardslist.append(accum_discountreward)
            used_advicelist.append(used_advice)
        save_ckpt(qnetworks[0], "big_random")  # save model
        # plt.plot(all_episode_reward)
        disrewardata = pd.DataFrame({f"heads{NUM_HEADS}_discount_rewards": discount_rewardslist, f"heads{NUM_HEADS}_acc_rewards": accum_discountrewardslist})
        disrewardata.to_csv(f"5x6_mapdata/heads{NUM_HEADS}_disrewards_random.csv",sep=',')

        used_advice_data = pd.DataFrame({ f"heads{NUM_HEADS}_acc_usedadvices": used_advicelist})
        used_advice_data.to_csv(f"5x6_mapdata/heads{NUM_HEADS}_usedadvice_random.csv",sep=',')
        rewardata = pd.DataFrame({f"heads{NUM_HEADS}": all_episode_reward})
        rewardata.to_csv(f"5x6_mapdata/heads{NUM_HEADS}_rewards_random.csv",sep=',')
        # if not os.path.exists('image'):
        #     os.makedirs('image')
        # plt.savefig(os.path.join('image', '_'.join([alg_name, env_id])))
        dataframe = pd.DataFrame({f"heads{NUM_HEADS}": uncertainty_history})
        dataframe.to_csv(f"5x6_mapdata/heads{NUM_HEADS}_random.csv",sep=',')
        print("plot------")
        # for ep_uncertainty in ep_uncertainties:
        plt.plot(list(range(num_episodes)), gaussian_filter1d(uncertainty_history, sigma=2))

        plt.title("Uncertainty by epoch")
        plt.savefig(f"5x6_mapdata/heads{NUM_HEADS}_ep_uncertainty_random")
    mheads_eval(qnetworks, env)
    # if args.test:
    #     num_episodes = 100
    #     load_ckpt(qnetwork)  # load model
    #     for i in range(num_episodes):
    #         ## Reset environment and get first new observation
    #         s = env.reset()  # observation is state, integer 0 ~ 15
    #         rAll = 0
    #         if render: env.render()
    #         for j in range(99):  # step index, maximum step is 99
    #             ## Choose an action by greedily (with e chance of random action) from the Q-network
    #             allQ = qnetwork(np.asarray([to_one_hot(s, 16)], dtype=np.float32)).numpy()
    #             a = np.argmax(allQ, 1)  # no epsilon, only greedy for testing

    #             ## Get new state and reward from environment
    #             s1, r, d, _ = env.step(a[0])
    #             rAll += r
    #             s = s1
    #             if render: env.render()
    #             ## Reduce chance of random action if an episode is done.
    #             if d: break

    #         print('Testing  | Episode: {}/{}  | Episode Reward: {:.4f} | Running Time: {:.4f}' \
    #               .format(i, num_episodes, rAll, time.time() - t0))
    #         rList.append(rAll)
    #     print("正确率: " + str(sum(rList) / num_episodes * 100) + "%")
