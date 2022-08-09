from matplotlib.pyplot import axis
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import copy
from collections import deque
import random
import statistics
from stable_baselines3 import PPO
import pandas as pd


def weights_init(m):
    """custom weights initialization"""
    classtype = m.__class__
    if classtype == nn.Linear or classtype == nn.Conv2d:
        print("default init")
        # nn.init.xavier_uniform_(m.weight)
        m.weight.data.normal_(0.0, 0.1)
        # m.bias.data.fill_(0)
    elif classtype == nn.BatchNorm2d:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    else:
        print('%s is not initialized.' %classtype)

class Net(nn.Module):
    def __init__(self, ):
      super(Net, self).__init__()
      self.fc1 = nn.Linear(16, 10)
      # self.fc2 = nn.Linear(64, 64)
      self.fc1.apply(weights_init)
      # self.fc2.apply(weights_init)

    def forward(self, x):
      x = F.relu(self.fc1(x))
      # print(x.shape)
      # x = self.bn(x)
      # x = F.relu(self.fc2(x))
      return x

class HeadNet(nn.Module):
    def __init__(self, n_actions=4):
      super(HeadNet, self).__init__()
      self.fc1 = nn.Linear(10, n_actions, bias=False)
      # self.fc1.apply(weights_init)

    def forward(self, x):
        x = self.fc1(x)
        return x

class EnsembleNet(nn.Module):
    def __init__(self, n_ensemble, n_actions):
        super(EnsembleNet, self).__init__()
        self.core_net = Net()
        self.net_list = nn.ModuleList([HeadNet(n_actions=n_actions) for k in range(n_ensemble)])

    def _core(self, x):
        return self.core_net(x)

    def _heads(self, x):
        return [net(x) for net in self.net_list]

    def forward(self, x, k):
        if k is not None:
            return self.net_list[k](self.core_net(x))
        else:
            core_cache = self._core(x)
            net_heads = self._heads(core_cache)
            return net_heads

class RCMP: 
  def __init__(self, env, obs_continuous = True, num_heads = 4, minibatch_size = 8, head_chance = 0.3, \
               stride_length = 50, uncertainty_limit = 0.5, help_limit = 700, alpha = 0.01,\
               epsilon=0.2, gamma=0.8, buffer_size=2500, DEVICE=None):
    # Hyperparameters
    # --------------------
    # Environment (Assume box obs and discrete action)
    self.obs_continuous = obs_continuous
    self.device = DEVICE
    if self.obs_continuous:  #Continuous Obs
      self.obs_size = env.observation_space.shape[0]
    else: #Discrete obs
      self.obs_size = env.observation_space.n
    self.num_actions = env.action_space.n
    self.num_heads = num_heads
    # Experience Replay / DQN Training
    self.minibatch_size = minibatch_size
    self.stride_length = stride_length
    self.alpha = alpha
    self.epsilon = epsilon # Set to 0 for bootstrapped exploration, >0 is epsilon greedy + bootstrapped exploration
    self.gamma = gamma
    self.buffer_size = buffer_size
    self.memory = deque([], maxlen = buffer_size)
    # RCMP / U3A
    self.env = env
    self.reward_history = list()
    self.uncertainty_history = list()
    self.ep_uncertainty_history = list()
    self.uncertainty_limit = uncertainty_limit
    self.help_remaining = help_limit
    self.head_chance = head_chance
    self.policy_net = EnsembleNet(self.num_heads, self.num_actions)
    self.target_net = EnsembleNet(self.num_heads, self.num_actions)
    self.target_net.eval()
    self.opt = torch.optim.Adam(self.policy_net.parameters(), lr=self.alpha)

  def helpAvailable(self):
    if self.help_remaining > 0:
      self.help_remaining -= 1
      return True
    return False

  def setDemonstrator(self, model):
    # param model: StableBaselines 3 Model
    self.demonstrator = copy.deepcopy(model)

  def demonstratorAction(self, state):
    # param self.demonstrator: StableBaselines 3 Model
    # Returns only action, not next hidden stat
    if self.obs_continuous:
      return self.demonstrator.predict(state, deterministic=True)[0][0]
    return self.demonstrator.predict(state[0], deterministic=True)[0][0]

  def train(self, num_epochs, num_steps):
    #set masks
    masks = []
    for m in range(self.num_heads):
      mask = []
      for n in range(self.minibatch_size):
        if random.random() > -1:
          mask.append(1)
        else:
          mask.append(0)
      masks.append(mask)
    masks = torch.Tensor(masks)
    # Reset network weights
    # Data Collection
    total_steps = 0
    # Reset history lists
    self.reward_history = list()
    self.uncertainty_history = list()
    self.ep_uncertainty_history = list()
    flag = True
    cnt_win = 0
    for ep in range(num_epochs):
      # print("Start of Epoch help remaining", self.help_remaining)
      # print("Epoch = ", ep, end = "|")
      reward_total = 0
      ep_steps = 0
      ep_avg_uncertainty = 0
      # Set random head as the "core" decider for this epoch
      # coreN, targetN = random.choice(self.networks)
      # Initialize state
      state = self.env.reset()
      state = np.resize(state, (self.obs_size))
      for step in range(num_steps):
        # ------------------------------RCMP--------------------------------
        # Calculate uncertainty mu
        variance_sum = 0
        state = torch.FloatTensor(state)
        self.policy_net.eval()
        vals = self.policy_net(state, None)
        # print("got eval")
        # Sum over actions
        for a in range(self.num_actions):
          q_values = list()
          # Create list of Q(s,a) values over heads
          for h in range(self.num_heads):
            q_value = vals[h][a].detach().numpy()
            q_values.append(q_value)
          # Add variance to total variance
          variance_sum += np.var(q_values)
        # Divide summed variance by num_actions
        mu = variance_sum / self.num_actions
        ep_avg_uncertainty+= mu
        # Data Collection
        self.uncertainty_history.append(mu)
        # Action Advising
        used_help = False
        if mu > self.uncertainty_limit and self.helpAvailable(): # Take action demonstrator would take
          action =  self.demonstratorAction(state)
          print("use demostration")
          used_help = True
        else: # Take action from normal policy (eg. Epsilon Greedy)
          if random.random() < self.epsilon:
            action = self.env.action_space.sample()
          else:
            q_head = torch.zeros(1, self.num_actions)
            for qval in vals:
              q_head += qval.detach()
            # print(q_head)
            # Core Action
            action = np.argmax(q_head.numpy()) # 论文采取的是平均
        # -----------------------Step The Environment------------------------
        next_state, reward, done, _ = self.env.step(action)
        ep_steps += 1
        next_state = np.resize(next_state, (self.obs_size))
        
        cnt_win += reward
        # Store the transition
        # print(f" next_state type is {type(next_state)}")
        if done and reward == 0.:
          reward = -1
        reward_total += reward
        self.memory.append((state, action, reward, next_state, done))

        if done :
          self.epsilon = 1. / ((ep / 50) + 10)
          total_steps+=1
          break
        # Experience Replay Buffer too small
        if len(self.memory) < self.minibatch_size:
          total_steps += 1
          state = next_state
          continue
        # ------------------------Experience Replay--------------------------
        # Sample a minibatch of transitions
        mini_batch = random.sample(self.memory, self.minibatch_size)
        states = torch.stack([batch[0] for batch in mini_batch])
        actions = np.array([batch[1] for batch in mini_batch])
        rewards = np.array([batch[2] for batch in mini_batch])
        next_states = np.array([batch[3] for batch in mini_batch])
        dones = np.array([batch[4] for batch in mini_batch])

        # states = states.requires_grad_(True).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        dones = torch.Tensor(dones).to(self.device)
        states.requires_grad = True
        next_states.requires_grad = False
        # losses = [0.0 for _ in range(self.num_heads)]
        self.policy_net.train()
        # sprint(f"states shape is {states.shape}")
        q_policy_vals = self.policy_net(states, None)
        # print(f"qp val {q_policy_vals}")
        next_q_target_vals = self.target_net(next_states, None)
        cnt_losses = []
        for k in range(self.num_heads):
            #TODO finish masking
            total_used = torch.sum(masks[k])
            if total_used > 0.0:
                next_q_vals = next_q_target_vals[k].data

                next_qs = next_q_vals.max(1)[0] # max returns a pair

                preds = q_policy_vals[k].gather(1, actions[:,None]).squeeze(1)
                # print(f"preds {preds}")
                targets = rewards + self.gamma * next_qs
                # print(f"pred is {preds}, targets is {targets}")
                l1loss = F.smooth_l1_loss(preds, targets, reduction='mean')
                # full_loss = masks[k]*l1loss
                loss = torch.sum(l1loss/total_used)
                cnt_losses.append(loss)
                # losses[k] = loss.cpu().detach().item()
                # print(f"head {k} loss is : {loss.cpu().detach().item()}")

        loss = sum(cnt_losses)/self.num_heads
        # print(f"epoch {ep}, step {step} total loss : {loss}")
        self.opt.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            # print(param.data)
            if param.grad is not None:
                print(f"grad is {param.grad.data}")
                # divide grads in core
                param.grad.data *=1.0/float(self.num_heads)
        # nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.opt.step()
        # self.target_net.load_state_dict(self.policy_net.state_dict())
      if ep % self.stride_length == 0:
        self.target_net.load_state_dict(self.policy_net.state_dict())
      
      if ep % 50 == 0:
        print("current accuracy: %.2f %%" %(cnt_win/50.0*100))
        cnt_win = 0
      # END FOR STATE
      # -------------------------------------------
      # Data Collection
      self.reward_history.append(reward_total)
      self.ep_uncertainty_history.append(ep_avg_uncertainty/step)
      # print(f"epoch step is {ep_steps}")
      # Print the reward
      # print("Reward = {}, Average 15 Reward = {:.2f}".format(reward_total, statistics.mean(self.reward_history[-15:])))
    # END FOR EPOCH
      print('Training  | Episode: {}/{}  | Episode Reward: {:.4f}'.format(ep, num_epochs, reward_total))
      # if ep == 155:
      #   break
    # ------------------------------------------
    self.env.reset()
    print("test----")
    self.policy_net.eval()
    sc = 0
    ncount = 100
    for i in range(ncount):
        obs = env.reset()
        # env.render()
        while True:
            vals = self.policy_net(state, None)
            q_head = torch.zeros(1, self.num_actions)
            for qval in vals:
              q_head += qval.detach()
            # print(q_head)
            # Core Action
            action = np.argmax(q_head.numpy()) # 论文采取的是平均
            obs, rewards, dones, info = env.step(action)
            # print(f"action is {action}, reward is {rewards}, done is {done}, info is {info}, obs is {obs}")
            if rewards == 1:
                print("successfully")
                sc += 1
            # print(dones)
            if dones:
                print("done----")
                break

    print(f"success rate is {sc/ncount}")
    return reward_total

if __name__ == '__main__':
  env_name = "FrozenLake-v1"
  env = gym.make(env_name)
  # Load the demonstrator model
  PPO_model_name = "ppo_frozenlake_1e6"
  model = PPO('MlpPolicy', env, verbose=1)
  model = PPO.load(PPO_model_name)
  model.set_env(env)

  RCMP_model = RCMP(env = env, obs_continuous=False, num_heads=1, help_limit=0, uncertainty_limit=0.11, gamma=0.9999, minibatch_size=64, epsilon=0.08, alpha = 0.0012, buffer_size=2000, stride_length=150)
  RCMP_model.setDemonstrator(model)
  num_ep = 2000
  num_steps = 500
  num_runs = 1
  run_rewards = list()
  ep_uncertainties = []
  for run in range(num_runs):
    RCMP_model.train(num_ep, num_steps)
    print("finish trian")