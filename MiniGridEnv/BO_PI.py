import gym
import scipy.io
import MiniGridEnv
import numpy as np
from CRC import psuedo_reward
from scratchpad import crc



from tqdm import trange

def target_function(Kp, Ki):
    num = 5   # number of agents
    p = 1.0
    c = 1.0
    k = 0.1
    Q = 4e1
    N = 10000
    err = np.zeros(num)
    action = np.zeros(num)
    battery = np.zeros(num)
    reward_sum = np.zeros(N)
    mu = np.array([5, 10, 15, 5, 8])

    for i in range(N):
        battery = battery + action
        action, _, _, _ = crc(p, c, k, mu, num, battery, Q, err, Kp, Ki, capacity=1e2)
        demand = np.random.poisson(mu)
        reward = psuedo_reward(p, c, k, Q, num, demand, battery, action)
        battery = battery - demand
        battery[battery < 0] = 0
        reward_sum[i] = reward.sum()
    avg_reward_sum = reward_sum.mean()
    return -avg_reward_sum
    

def dummy_function(X):
    sample_num = X[:,0].size
    output = np.empty(sample_num)
    for i in trange(sample_num):
        input = {'Kp': X[i,0], 'Ki': X[i,1]}
        result = target_function(**input)
        output[i] = result
    return output
        
from emukit.core.loop.user_function import UserFunctionWrapper
from emukit.core import ContinuousParameter, ParameterSpace

domain = (-10, 10)
space = ParameterSpace([ContinuousParameter('Kp', *domain), ContinuousParameter('Ki', *domain)])

import GPy
from GPy.models import GPRegression
from emukit.model_wrappers import GPyModelWrapper
from emukit.core.initial_designs import RandomDesign

design = RandomDesign(space)

x_init = design.get_samples(50)
y_init = dummy_function(x_init)

gpy_model = GPRegression(x_init, y_init, GPy.kern.RBF(1, lengthscale=0.08, variance=20), noise_var=1e-10)
emukit_model = GPyModelWrapper(gpy_model)
emukit_model.optimize()

from emukit.bayesian_optimization.acquisitions import ExpectedImprovement

expected_improvement = ExpectedImprovement(model = emukit_model)

from emukit.bayesian_optimization.loops import BayesianOptimizationLoop

BO_loop = BayesianOptimizationLoop(model = model_emukit,
                                         space = parameter_space,
                                         acquisition = expected_improvement,
                                         batch_size = 100)

num_iterations = 500

BO_loop.run_loop(dummy_function, num_iterations)

results = bayesopt_loop.get_results()

import pdb; pdb.set_trace()


