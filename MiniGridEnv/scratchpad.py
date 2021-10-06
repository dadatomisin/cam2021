import math
import scipy
import numpy as np
from scipy.stats import poisson
from collections.abc import Iterable

# PI controller that takes as input kp, ki, current error and previous integral error
# TO DO:    Test on solitary agent and tune then deploy for multiple agents, modify function call 
#           to provide necessary inputs to function
#           Implement BO to tune controller
def PI(Kp, Ki, err, prev_i_err):
    #Kp = 0.8
    #Ki = 0.8
    #Kp = 1
    #Ki = 0
    i_err = prev_i_err + err
    u = Kp*err + Ki*i_err
    return u, i_err


def aggregate_demand(CR, mu, battery):
    demand = 0
    for a,b in zip(mu, battery):
        tmp = poisson.ppf(CR, a) - b
        demand += max(0, tmp)
    return demand

def aggregate_demand_PI(CR, mu, battery, i_err, Kp, Ki):
    demand = 0
    # make new_i_err a numpy array
    for a,b,c in zip(mu, battery, i_err):
        err = poisson.ppf(CR, a) - b
        tmp, _ = PI(Kp, Ki, err, c)
        demand += max(0, tmp)
    return demand

def supply_curve(Q, price, function='sigmoid'):
    supply = 0
    error = True
    if function == 'sigmoid':
        error = False
        supply = Q + scipy.special.logit(price)
    elif function == 'linear':
        error = False
        supply = Q * price
    elif function == 'quadratic':
        error = False
        supply = Q * (price ** 1/2)
    if error:
        print('Function Type not not specified')
    return supply

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

def crc(p, c, k, mu, n, battery, Q, i_err, Kp, Ki, gamma=1, h=0, capacity=1e2):
    z = np.zeros(n)
    a1 = np.zeros(n)
    new_i_err = np.zeros(n)
    space = capacity - battery

    cost = bisection_search(p, k, h, Q, mu, battery, i_err, Kp, Ki, sf='sigmoid')
    CR = (p - cost + k)/(p - cost + k + (0.1*cost) + h)
    #import pdb; pdb.set_trace()

    #test = 0
    for i in range(n):
        #test += poisson.ppf(CR, mu[i])
        z[i] = min(poisson.ppf(CR, mu[i]), capacity)
        err = poisson.ppf(CR, mu[i]) - battery[i]
        u, new_i_err[i] = PI(Kp, Ki, err, i_err[i])
        u = max(0, u)
        #a1[i] = min(poisson.ppf(CR, mu[i]) - battery[i], space[i])
        a1[i] = min(u, space[i])
        #import pdb; pdb.set_trace()
    #print("Target: {}, Actions: {}".format(z,a1))
    return a1, z, cost, new_i_err #, converged
