import gym
import scipy.io
import MiniGridEnv
import numpy as np
from util_crc import crc, psuedo_reward_v1, psuedo_reward_v2

num = 5   # number of agents
p = 1.0
c = 1.0
k = 0.1
Q = 4e1
action = np.zeros(num)
battery = np.zeros(num)
mu = np.array([5, 10, 15, 5, 8]) #np.random.randint(20, size=num) + 1

N = 100000
cost_record = np.zeros([1,N])
target_record = np.zeros([num,N])
action_record = np.zeros([num,N])
demand_record = np.zeros([num,N])
reward_record = np.zeros([num,N])
battery_record = np.zeros([num,N])
error_record = np.zeros([num,N])

err = np.zeros(num)
for i in range(N):
    battery = battery + action
    action, target, effective_c, err = crc(p, c, k, mu, num, battery, Q, i_err=err, Kp=0.85530119, Ki=0.38346481, capacity=1e2)
    demand = np.random.poisson(mu)
    reward = psuedo_reward_v1(p, c, k, Q, num, demand, battery, action)
    battery = battery - demand
    battery[battery < 0] = 0
    cost_record[:,i] = effective_c
    target_record[:,i] = target
    action_record[:,i] = action
    demand_record[:,i] = demand
    reward_record[:,i] = reward
    battery_record[:,i] = battery
    error_record[:,i] = err
    
mdic = {"target" : target_record, "action" : action_record, "demand" : demand_record, "reward" : reward_record, "battery" : battery_record, "cost" : cost_record, "error" : error_record}
scipy.io.savemat("PI_5_test0v2.mat", mdic)