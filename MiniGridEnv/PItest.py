import gym
import scipy.io
import MiniGridEnv
import numpy as np
#from CRC import CRC, CR2, psuedo_reward
from CRC import psuedo_reward
from scratchpad import crc

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
    action, target, effective_c, err = crc(p, c, k, mu, num, battery, Q, err, capacity=1e2)
    demand = np.random.poisson(mu)
    reward = psuedo_reward(p, c, k, Q, num, demand, battery, action)
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
#from CRC import sumdemand
#i = 0
#d = np.zeros(9999)
#for kprice in range(1, 9999):
#    price = kprice/10000
#    tmp = (p - price + k)/(p - price + k + (0.1*price))
#    d[i] = sumdemand(tmp,mu)
#    i+=1
#mdic = {"d" : d}
#scipy.io.savemat("demand.mat", mdic)
#import pdb; pdb.set_trace()
