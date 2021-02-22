import gym
import numpy as np
import itertools
from gym import spaces
from collections.abc import Iterable 
from MiniGridEnv.utils import assign_env_config


class MiniGridEnv(gym.Env):
  # """Custom Environment that follows gym interface"""
  #metadata = {'render.modes': ['human']}

  def __init__(self, *args, **kwargs):
    super(MiniGridEnv, self).__init__()
    self.resource = 500
    self.resource_max = 1000
    self.n_agent = 5
    self.coop_alpha = 0.1
    self.battery_capacity = np.array([10, 10, 10, 10, 10])
    self.max_order_quantity = 5
    self.discount_gamma = 1
    self.step_limit = 100
    self.p_max = 1
    self.k_max = 0.1
    self.mu_max = 0.5
    assign_env_config(self, kwargs)

    self.obs_dim = (self.n_agent * 2) +  4
    # Define action and observation space
    self.action_space = spaces.Discrete(11)
    # Example for using image as input:
    self.observation_space = spaces.Box(
        low=np.zeros(self.obs_dim),
        high=np.array([self.p_max, self.p_max, self.k_max, self.resource_max] + 
        [self.mu_max] * self.n_agent + 
        [self.max_order_quantity] * self.n_agent),
        dtype=np.float32)
        #255, shape=
        #            (HEIGHT, WIDTH, N_CHANNELS), dtype=np.float32)
    self.reset()

  def _STEP(self, agent_outputs):
    # Execute one time step within the environment
    done = False
    m = 3 + self.n_agent
    actions = self.process_agents_outputs(agent_outputs)
    total_order_quantity = actions.sum(-1)
    excess = total_order_quantity - self.resource

    self.battery_level = self.state[4:m]

    rewards = []
    for agent in range(self.n_agent):
      demand = np.random.poisson(self.mu[agent])
      battery = self.battery_level[agent]

      # Reward for Suplying Energy to User
      supplied = min(demand, battery) * self.p # satisfaction
      # Penalty for Inability to Supply Sufficient Energy from Battery
      mismatch = max(0, demand - battery) * self.k # dissatisfaction
      # Proportional Cost of Exceeding Renewable Supply
      proportion_of_excess = max(0, (excess/total_order_quantity)*actions[agent])* self.c
      # Discharge of Battery modelled as a Holding Cost
      #discharge = max(0, battery - demand) * self.battery_discharge
      reward = supplied - (mismatch+proportion_of_excess)
      if isinstance(reward, Iterable):
        reward = sum(reward)

      rewards[agent] = reward

      # Update State
      self.battery_level[agent] = max(0, battery - demand) + actions[agent]
    
    self.resource = 500
    self.state = np.hstack([self.state[:3], self.resource, self.state[4:m], self.battery_level])

    self.step_count += 1
    if self.step_count >= self.step_limit:
      done = True
    
    return self.state, reward, done, {}

  def _RESET(self):
    m = 4 #+ self.n_agent
    self.p = max(1, np.random.rand() * self.p_max)
    self.c = max(1, np.random.rand() * self.p_max)
    # self.h = np.random.rand() * min(self.cost, self.h_max)
    self.k = np.random.rand() * self.k_max
    self.resource = np.random.rand() * self.resource_max
    self.mu = np.zeros(self.n_agent)
    #mu_s = []

    self.state = np.zeros(self.obs_dim)
    self.state[:m] = np.array([self.p, self.c, self.k, self.resource]) #, mu_s])

    for agent in range(self.n_agent):
      self.mu[agent] = np.random.rand() * self.mu_max
      j = agent+m
      self.state[:j+1] = np.hstack([self.state[:j], self.mu[agent]])
      #print(self.state)
      
    #self.state = np.zeros(self.obs_dim)
    #self.state[:m] = np.array([self.p, self.c, self.k, self.resource, mu_s])

    self.step_count = 0

    return self.state

  def reset(self):
    return self._RESET()

  def step(self, agent_outputs):
    return self._STEP(agent_outputs)

  def process_agents_outputs(self, agent_outputs):
    m = 3 + self.n_agent
    actions = np.zeros(self.n_agent)
    for agent in range(self.n_agent):
      spare_capacity = self.battery_capacity - self.state[m + agent]
      actions[agent] = min(agent_outputs[agent], self.max_order_quantity, spare_capacity)
    
    return actions

