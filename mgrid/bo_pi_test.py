import scipy.io
import numpy as np
from tqdm import trange
import numpy as np
from util_crc import crc, psuedo_reward_v1, psuedo_reward_v2

def target_function(Kp, Ki):
    num = 5   # number of agents
    p = 1.0
    c = 1.0
    k = 0.1
    Q = 4e1
    N = 1000
    err = np.zeros(num)
    action = np.zeros(num)
    battery = np.zeros(num)
    reward_sum = np.zeros(N)
    mu = np.array([5, 10, 15, 5, 8])
    #tmp = np.random.randint(2, high=11, size=1)
    #num = tmp[0]
    #mu = np.random.randint(1, high=21, size=num)
    #Q = 0.9*mu.sum()

    for i in range(N):
        battery = battery + action
        action, _, effective_c, err = crc(p, c, k, mu, num, battery, Q, i_err=err, Kp=Kp, Ki=Ki, capacity=1e2)
        demand = np.random.poisson(mu)
        #reward = psuedo_reward_v1(p, c, k, Q, num, demand, battery, action)
        reward = psuedo_reward_v2(p, effective_c, k, Q, num, demand, battery, action)
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
    return output[:, np.newaxis]
       
from emukit.core.loop.user_function import UserFunctionWrapper
from emukit.core import ContinuousParameter, ParameterSpace

domain = (-10, 10)
space = ParameterSpace([ContinuousParameter('Kp', *domain), ContinuousParameter('Ki', *domain)])

from GPy.models import GPRegression
from emukit.model_wrappers import GPyModelWrapper
from emukit.core.initial_designs import RandomDesign

design = RandomDesign(space)

x_init = design.get_samples(500)
y_init = dummy_function(x_init)

gpy_model = GPRegression(x_init, y_init)
emukit_model = GPyModelWrapper(gpy_model)
emukit_model.optimize()

from emukit.bayesian_optimization.acquisitions import ExpectedImprovement

expected_improvement = ExpectedImprovement(model = emukit_model)

from emukit.bayesian_optimization.loops import BayesianOptimizationLoop

BO_loop = BayesianOptimizationLoop(model = emukit_model,
                                         space = space,
                                         acquisition = expected_improvement,
                                         batch_size = 1)

num_iterations = 50
BO_loop.run_loop(dummy_function, num_iterations)

results = BO_loop.get_results()
print(results.minimum_value, results.minimum_location)