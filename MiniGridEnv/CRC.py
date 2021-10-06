import math
import scipy
import numpy as np
from scipy.stats import poisson
from collections.abc import Iterable


def sigmoid(x):
  return 1.0 / (1.0 + math.exp(-x))

def sumdemand(CR, mu):
    demand = 0
    for x in mu:
        demand += poisson.ppf(CR, x)
    return demand

def CR3(p, c, k, mu, n, battery, Q, gamma=1, h=0, capacity=1e2):
    tol = 1e-10
    lb = 1e-20
    ub = 1 - lb
    z = np.zeros(n)
    a1 = np.zeros(n)
    space = capacity - battery

    for mp in [lb, ub]:
        var1 = scipy.special.logit(mp)
        tmp = (p - mp + k)/(p - mp + k + (0.1*mp) + h)
        var2 = sumdemand(tmp,mu) - Q
        #print("Var 1: {}, Var 2: {}, Critical Ratio: {}".format(var1,var2,tmp))
    
    iter_limit = 10000
    converged = False

    for _ in range(iter_limit):
        mp = (ub + lb)/2
        var1 = scipy.special.logit(mp)
        tmp = (p - mp + k)/(p - mp + k + (0.1*mp) + h)
        #var2 = poisson.ppf(tmp, sum_mu) - Q
        var2 = sumdemand(tmp, mu) - Q
        var3 = var1 - var2
        if var3 == 0 or (ub - lb)/2 < tol:
            converged = True
            break
        if var3 > 0:
            ub = mp
        else:
            lb = mp
        #print("Var 1: {}, Var 2: {}, Upper Bound: {}, Lower Bound: {}, Mid Point: {}, Critical Ratio: {}".format(var1,var2,ub,lb,mp,tmp))

    cost = mp
    CR = (p - cost + k)/(p - cost + k + (0.1*cost) + h)

    #test = 0
    for i in range(n):
        #test += poisson.ppf(CR, mu[i])
        z[i] = min(poisson.ppf(CR, mu[i]), capacity)
        a1[i] = min(poisson.ppf(CR, mu[i]) - battery[i], space[i])
        #import pdb; pdb.set_trace()
    #print("Target: {}, Actions: {}".format(z,a1))
    return a1, z, cost, converged

def CR2(p, c, k, mu, n, battery, Q, gamma=1, h=0, capacity=1e2):
    tol = 1e-10
    lb = 1e-20
    ub = 1 - lb
    z = np.zeros(n)
    a1 = np.zeros(n)
    space = capacity - battery

    for mp in [lb, ub]:
        var1 = scipy.special.logit(mp)
        tmp = (p - mp + k)/(p - mp + k + (0.1*mp) + h)
        var2 = sumdemand(tmp,mu) - Q
        #print("Var 1: {}, Var 2: {}, Critical Ratio: {}".format(var1,var2,tmp))
    
    iter_limit = 10000
    converged = False

    for _ in range(iter_limit):
        mp = (ub + lb)/2
        var1 = scipy.special.logit(mp)
        tmp = (p - mp + k)/(p - mp + k + (0.1*mp) + h)
        #var2 = poisson.ppf(tmp, sum_mu) - Q
        var2 = sumdemand(tmp, mu) - Q
        var3 = var1 - var2
        if var3 == 0 or (ub - lb)/2 < tol:
            converged = True
            break
        if var3 > 0:
            ub = mp
        else:
            lb = mp
        #print("Var 1: {}, Var 2: {}, Upper Bound: {}, Lower Bound: {}, Mid Point: {}, Critical Ratio: {}".format(var1,var2,ub,lb,mp,tmp))

    cost = mp
    CR = (p - cost + k)/(p - cost + k + (0.1*cost) + h)

    #test = 0
    for i in range(n):
        #test += poisson.ppf(CR, mu[i])
        z[i] = min(poisson.ppf(CR, mu[i]), capacity)
        a1[i] = min(poisson.ppf(CR, mu[i]) - battery[i], space[i])
        #import pdb; pdb.set_trace()
    #print("Target: {}, Actions: {}".format(z,a1))
    return a1, z, cost, converged


def CRC(p, c, k, mu, n, battery, Q, gamma=1, h=0, capacity=1e2):
    z = np.zeros(n)
    a1 = np.zeros(n)

    spare_capacity = capacity*np.ones(n) - battery

    CR = np.zeros(n)
    for i in range(n):
        tmp1 = (p - k)/(p - (k + h))
        CR[i] = tmp1
        tmp2 = poisson.ppf(tmp1, mu[i])
        z[i] = min(capacity, tmp2)
        tmp3 = min(tmp2, spare_capacity[i])
        a1[i] = max(0, tmp3)

    iterations = 0
    iterations_limit = 1000
    converged = False
    while not converged:
        #import pdb; pdb.set_trace()
        check = 0
        order_total = np.sum(a1)
        excess = order_total - Q #max(0, order_total - Q)
        for i in range(n):
            effective_c = sigmoid(excess) 
            #if excess > 0:
            #    prop_mult = (excess/order_total)
            #else:
            #    prop_mult = 0
            #effective_c = gamma * c * prop_mult
            h = 0.1 * effective_c
            CR[i] = (p - effective_c + k)/(p - effective_c + k + h)
            tmp = poisson.ppf(CR[i], mu[i])
            if abs(tmp - z[i]) < 0.001:
                check += 1
            else:
                #z[i] = min(capacity, tmp)
                z[i] = (z[i] + min(capacity, tmp))/2
                tmp = min(tmp, spare_capacity[i])
                #a1[i] = max(0, tmp)
                a1[i] = (a1[i] + max(0, tmp))/2
        print("Critical Ratio: {}".format(CR))
        print("Effective Cost: {}, Order Total: {}, Available Resource: {}, Targets: {}".format(effective_c,order_total,Q, z))
        iterations += 1
        if check == n or iterations >= iterations_limit:
            converged = True
    return a1, z, effective_c

def psuedo_reward(p, c, k, Q, n, demands, batteries, actions):
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
    
    # import pdb; pdb.set_trace() 
    return rewards

#def CRC(p, c, k, mu, n, battery, Q, gamma=1, h=0):
#    z = np.zeros(n)
#    a1 = np.zeros(n)
#    #a2 = []
#    CR = np.zeros(n)
#    #prop_mult = []
#    #effective_c = []
#    for i in range(n):
#        #prop_mult.append(0) 
#        #effective_c.append(0)
#        tmp1 = (p - k)/(p - (k + h))
#        tmp1 = min(0.999, tmp1)
#        #CR.append(tmp1)
#        CR[i] = tmp1
#        tmp2 = poisson.ppf(tmp1, mu[i])
#        #z.append(tmp2)
#        z[i] = tmp2
#        #a1.append(max(0, (tmp2 - battery[i])))
#        a1[i] = max(0, (tmp2 - battery[i]))
#        #a2.append(0)
#        # new_a.append(max(z - battery[i], 0))
    
#    iterations = 0
#    iterations_limit = 200
#    #import pdb; pdb.set_trace()
#    converged = False
#    while not converged:
#        check = 0
#        order_total = np.sum(a1)
#        excess = max(0, order_total - Q)
#        for i in range(n):
#            if excess > 0:
#                prop_mult = (excess/order_total)
#                #prop_mult[i] = (excess/order_total)
#            else:
#                prop_mult = 0
#                #prop_mult[i] = 0
#            effective_c = gamma * c * prop_mult
#            #effective_c[i] = gamma * c * prop_mult[i]
#            #h = 0.1 * effective_c[i]
#            #CR[i] = (p - effective_c[i] + k)/(p - effective_c[i] + 0.1*effective_c[i] + k + h)
#            h = 0.1 * effective_c
#            CR[i] = (p - effective_c + k)/(p - effective_c + k + h)
#            #print(CR[i])
#            #import pdb; pdb.set_trace()
#            CR[i] = min(0.999, CR[i])
#            tmp = poisson.ppf(CR[i], mu[i])
#            #import pdb; pdb.set_trace()
#            #a[i] = max(0, z[i])
#            if abs(tmp - z[i]) < 0.001:
#                check += 1
#            else:
#                z[i] = tmp
#                a1[i]= max(0, (tmp - battery[i]))
#        #print(CR)
#        #print(z)
#        #print(a1)
#        iterations += 1
#        if check == n or iterations >= iterations_limit:
#            converged = True
    
#    return a1
            
