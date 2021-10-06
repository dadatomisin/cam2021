import gym
import scipy.io
#import MiniGridEnv
import numpy as np
from scipy.special import expit
from util_crc import crc, basic_crc, psuedo_reward_v1, psuedo_reward_v2

num = 5   # number of agents
p = 1.0
c = 1.0
k = 0.1
Q = 4e1
action = np.zeros(num)
battery = np.zeros(num)
mu = np.array([5, 10, 15, 5, 8]) #np.random.randint(20, size=num) + 1

N = 10000
cost_record = np.zeros([1,N])
target_record = np.zeros([num,N])
action_record = np.zeros([num,N])
demand_record = np.zeros([num,N])
reward_record = np.zeros([num,N])
battery_record = np.zeros([num,N])

for i in range(N):
    #battery = battery + action
    action, target, effective_c = basic_crc(p, c, k, mu, num, battery, Q, capacity=1e2)
    battery = battery + action
    demand = np.random.poisson(mu)
    #import pdb; pdb.set_trace()
    action_sum = action.sum()
    cost = expit(action_sum - Q)
    #reward = psuedo_reward_v1(p, c, k, Q, num, demand, battery, action)
    reward = psuedo_reward_v2(p, cost, k, Q, num, demand, battery, action)
    battery = battery - demand
    battery[battery < 0] = 0
    cost_record[:,i] = cost#effective_c
    target_record[:,i] = target
    action_record[:,i] = action
    demand_record[:,i] = demand
    reward_record[:,i] = reward
    battery_record[:,i] = battery
    
import pdb; pdb.set_trace()
    
mdic = {"target" : target_record, "action" : action_record, "demand" : demand_record, "reward" : reward_record, "battery" : battery_record, "cost" : cost_record}
scipy.io.savemat("mixed_crc_test_2.mat", mdic)
