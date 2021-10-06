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
#print(mu)

#env = gym.make('MiniGrid-v0')
#env.reset()
# demand = np.random.poisson(mu)

N = 10000
cost_record = np.zeros([1,N])
target_record = np.zeros([num,N])
action_record = np.zeros([num,N])
demand_record = np.zeros([num,N])
reward_record = np.zeros([num,N])
battery_record = np.zeros([num,N])

for i in range(N):
    battery = battery + action
    #battery[battery>100] = 100
    #print("previous action: {}".format(action))
    #print("battery + previous action: {}".format(battery))
    #battery_record[:,j] = battery
    action, target, effective_c = CRC(p, c, k, mu, num, battery, Q, capacity=1e2)
    #action, target, effective_c, _ = CR2(p, c, k, mu, num, battery, Q, capacity=1e2)
    #action = np.random.randint(100, size=num) #+ 1
    #action, target, effective_c = crc(p, c, k, mu, num, battery, Q, capacity=1e2)
    demand = np.random.poisson(mu)
    reward = psuedo_reward(p, c, k, Q, num, demand, battery, action)
    battery = battery - demand
    battery[battery < 0] = 0
    #action, target, effective_c, _ = CR2(p, c, k, mu, num, battery, Q, capacity=1e2)
    #print("action: {}".format(action))
    #print("target: {}".format(target))
    #print("effective cost: {}".format(effective_c))
    #print("demand: {}".format(demand))
    #print("reward: {}".format(reward))
    #print("battery after supplying demand: {}".format(battery))
    #env.step()
    cost_record[:,i] = effective_c
    target_record[:,i] = target
    action_record[:,i] = action
    demand_record[:,i] = demand
    reward_record[:,i] = reward
    battery_record[:,i] = battery
    #battery_record[:,j+1] = battery
    #import pdb; pdb.set_trace()
    
mdic = {"target" : target_record, "action" : action_record, "demand" : demand_record, "reward" : reward_record, "battery" : battery_record, "cost" : cost_record}
scipy.io.savemat("mixed_v3_quadratic_0504.mat", mdic)
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
