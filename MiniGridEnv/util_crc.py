import math
from scipy import special
import numpy as np
from scipy.stats import poisson
from collections.abc import Iterable

# PI controller that takes as input kp, ki, current error and previous integral error
def PI(Kp, Ki, err, prev_i_err):
    i_err = prev_i_err + err
    u = Kp*err + Ki*i_err
    return u, i_err

# Aggregate demand from agents assuming proportional control 
def aggregate_demand(CR, mu, battery):
    demand = 0
    for a,b in zip(mu, battery):
        tmp = poisson.ppf(CR, a) - b
        demand += max(0, tmp)
    return demand

# Aggregate demand from agents assuming PI control
def aggregate_demand_PI(CR, mu, battery, i_err, Kp, Ki):
    demand = 0
    for a,b,c in zip(mu, battery, i_err):
        err = poisson.ppf(CR, a) - b
        tmp, _ = PI(Kp, Ki, err, c)
        demand += max(0, tmp)
    return demand

#  Define Supply Curve approximation function
def supply_curve(Q, price, function='sigmoid'):
    supply = 0
    error = True
    if function == 'sigmoid':
        error = False
        supply = Q + special.logit(price)
    elif function == 'linear':
        error = False
        supply = Q * price
    elif function == 'quadratic':
        error = False
        supply = Q * (price ** 1/2)
    if error:
        print('Function Type not not specified')
    return supply

# Bisection search to find intersection of demand and supply curves
def bisection_search(p, k, h, Q, mu, battery, i_err, Kp, Ki, sf='sigmoid'):
    tol = 1e-5
    if sf == 'sigmoid':
        lb = 1e-20
        ub = 1 - lb
    else: 
        lb = 1e-20
        ub = 1e2
    iter_limit = 10000
    for _ in range(iter_limit):
        mp = (ub + lb)/2
        tmp = (p - mp + k)/(p - mp + k + (0.1*mp) + h)
        var1 = supply_curve(Q, mp, function=sf)
        #var2 = aggregate_demand(tmp, mu, battery)
        var2 = aggregate_demand_PI(tmp, mu, battery, i_err, Kp, Ki)
        var3 = var1 - var2
        if abs(var3) < 1 or (ub - lb)/2 < tol:
            #print('converged')
            break
        if var3 > 0:
            ub = mp
        else:
            lb = mp
    return mp

def crc(p, c, k, mu, n, battery, Q, i_err=0, Kp=1, Ki=0, gamma=1, h=0, capacity=1e2):
    z = np.zeros(n)
    a1 = np.zeros(n)
    new_i_err = np.zeros(n)
    space = capacity - battery

    cost = bisection_search(p, k, h, Q, mu, battery, i_err, Kp, Ki, sf='sigmoid')
    CR = (p - cost + k)/(p - cost + k + (0.1*cost) + h)

    for i in range(n):
        z[i] = min(poisson.ppf(CR, mu[i]), capacity)
        err = poisson.ppf(CR, mu[i]) - battery[i]
        u, new_i_err[i] = PI(Kp, Ki, err, i_err[i])
        u = max(0, u)
        a1[i] = min(u, space[i])
    return a1, z, cost, new_i_err

# Psuedo Reward v1 uses area under supply curve as effective cost and assumes ideal step function
def psuedo_reward_v1(p, c, k, Q, n, demands, batteries, actions):
    rewards = np.zeros(n)
    total_order_quantity = actions.sum(-1)
    excess = max(0, total_order_quantity - Q)
    for agent in range(n):
        demand = demands[agent]
        battery = batteries[agent]
        supplied = min(demand, battery) * p
        # Penalty for Inability to Supply Sufficient Energy from Battery
        mismatch = max(0, demand - battery) * k
        if total_order_quantity == 0:
            # Proportional Cost of Exceeding Renewable Supply
            proportion_of_excess = 0
            # Discharge of Battery modelled as a Holding Cost
            discharge = 0
        else:
            # Proportional Cost of Exceeding Renewable Supply
            proportion_of_excess = max(0, (excess/total_order_quantity)*actions[agent]) * c
            # Discharge of Battery modelled as a Holding Cost
            discharge = max(0, battery - demand) * 0.1 * c * (excess/total_order_quantity)
        reward = supplied - (mismatch+proportion_of_excess+discharge)
        if isinstance(reward, Iterable):
            reward = sum(reward)
        rewards[agent] = reward 
    return rewards

# Pseudo Reward v2 uses the cost price found using bisection search
def psuedo_reward_v2(p, c, k, Q, n, demands, batteries, actions):
    rewards = np.zeros(n)
    total_order_quantity = actions.sum(-1)
    excess = max(0, total_order_quantity - Q)
    for agent in range(n):
        demand = demands[agent]
        battery = batteries[agent]
        # Reward for Suplying Energy to User
        supplied = min(demand, battery) * p
        # Penalty for Inability to Supply Sufficient Energy from Battery
        mismatch = max(0, demand - battery) * k
        # Cost of Purchasing Energy
        cost = actions[agent]*c
        # Discharge Modelled as Holding Cost
        discharge = max(0, battery - demand) * 0.1 * c
        reward = supplied - (mismatch+cost+discharge)
        if isinstance(reward, Iterable):
            reward = sum(reward)
        rewards[agent] = reward
    return rewards
