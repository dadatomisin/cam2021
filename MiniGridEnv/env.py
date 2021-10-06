import torch
from torch.multiprocessing import Pool
from collections import OrderedDict

class TensorConditioner(): # NB, this does not allow for dynamic type, device, etc changes! This is possible but the code will need to be changed!
    def __init__(self, device=None, dtype=None, requires_grad=False):
        self._initialised = False

        self.set_device(device)
        self.set_dtype(dtype)
        self.set_grad(requires_grad)

        self._initialised = True

        self._update_kwargs()

    def _update_kwargs(self):
        if self._initialised:
            self.set_tensor_kwargs()
            self.set_convert_kwargs()

    def set_dtype(self, dtype):
        if dtype:
            self._dtype = dtype
        else:
            self._dtype = torch.float16 if "cuda" in str(self._device) else torch.float32

        self._update_kwargs()

    def set_device(self, device):
        if device:
            if isinstance(device, torch.device):
                self._device = device
            elif isinstance(device, str):
                self._device = torch.device(device)
            else:
                raise TypeError("The 'TensorConditioner' class needs either a 'torch.device' type or a 'string' type for the 'device' argument.")

        else:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._update_kwargs()

    def set_grad(self, requires_grad):
        self._requires_grad = requires_grad

        self._update_kwargs()

    def set_tensor_kwargs(self):
        self._tensor_kwargs = {
            "dtype": self._dtype,
            "device": self._device,
            "requires_grad": self._requires_grad
        }

    def set_convert_kwargs(self):
        self._convert_kwargs = {
            "dtype": self._dtype,
            "device": self._device
        }

    def get_tensor(self, data):
        return torch.tensor(data, **self._tensor_kwargs)

    def convert_tensor(self, tensor):
        new_tensor = tensor.to(**self._convert_kwargs)
        new_tensor.requires_grad = self._requires_grad
        return new_tensor

    def ones(self, shape):
        return torch.ones(shape, **self._tensor_kwargs)

    def zeros(self, shape):
        return torch.zeros(shape, **self._tensor_kwargs)

    def arange(self, *args, **kwargs):
        return torch.arange(*args, **kwargs, **self._tensor_kwargs)

    def empty(self, shape):
        return torch.empty(shape, **self._tensor_kwargs)

class Discrete(TensorConditioner):
    def __init__(self, num_options, device=None):
        assert num_options > 1, "'num_options' must be bigger than 1"

        super(Discrete, self).__init__(device=device)

        self._dist = torch.distributions.categorical.Categorical(
            probs=self.ones(num_options)
        )

    def sample(self, shape=(1,)):
        return self._dist.sample(shape)

class Continuous(TensorConditioner):
    def __init__(self, low=0.0, high=1.0, device=None):
        super(Continuous, self).__init__(device=device)

        self._dist = torch.distributions.uniform.Uniform(
            self.get_tensor(low), self.get_tensor(high)
        )

    def sample(self, shape=(1,)):
        return self._dist.sample(shape)

class MiniGrid(TensorConditioner):
    def __init__(self, batch=1, done=False, r=1e3, num=5, p=1.0, c=1.0, k=0.1, mu=20, N=1000, seed=0, *args, **kwargs):
        super(MiniGrid, self).__init__()
        self.batch_size = batch
        self.done = done
        self.resource = torch.ones(batch,num)*r
        self.resource_max = 1e8
        self.n_agent = num
        #self.coop_alpha = coop
        self.battery_capacity = torch.ones(batch,num) * 100
        self.max_order_quantity = torch.ones(batch,num) * 1e2
        self.discount_gamma = 1
        self.step_limit = N
        self.p_max = 1
        self.k_max = 0.1
        self.mu_max = 50 #0.5
        self.p = torch.ones(batch,num) * p #None
        self.c = torch.ones(batch,num) * c #None
        self.k = torch.ones(batch,num) * k #None
        #self.resource = None
        self.mu = torch.randn(batch,num) + mu #None
        self.mu[self.mu <= 0] = 5
        self.battery_level = torch.zeros(batch, num) #None
        self.step_count = 0
        self.seed = seed
        torch.manual_seed(self.seed)

        #assign_env_config(self, kwargs)

        self.obs_dim = [self.n_agent, 6]# * 2) +  4
        self.action_space = Discrete(101)

    def step(self, agent_outputs):
        return self._STEP(agent_outputs)
    
    def _STEP(self, agent_outputs):
        with torch.no_grad():
            actions = self.process_agents_outputs(agent_outputs)
            total_action = actions.sum(-1).reshape(self.batch_size, 1)
            excess = torch.max(torch.zeros_like(total_action), total_action - self.resource)
            total_action[total_action == 0] = 1
            z = excess/total_action
            demand = torch.poisson(self.mu)
            supplied = torch.min(demand, self.battery_level) * self.p
            mismatch = torch.max(torch.zeros_like(demand), demand-self.battery_level) * self.k
            discharge = torch.max(torch.zeros_like(demand), self.battery_level-demand) * 0.1 * self.c * z
            cost = torch.max(torch.zeros_like(actions), actions) * self.c * z
            rewards = supplied - (cost + mismatch + discharge)
            
            self.battery_level = self.battery_level - demand
            self.battery_level[self.battery_level < 0] = 0
            self.battery_level = self.battery_level + actions
            
            self.resource = torch.ones(self.batch_size,self.n_agent) * 500
            self.state = torch.stack([self.p, self.c, self.k, self.resource, self.mu, self.battery_level], dim=2)

            self.step_count += 1
            if self.step_count >= self.step_limit:
                self.done = True
            
            return self.state, rewards, self.done, {}
    
    def _RESET(self):
        self.p = torch.ones(self.batch_size, self.n_agent) * self.p_max
        self.c = torch.ones(self.batch_size, self.n_agent) * self.p_max
        self.k = torch.ones(self.batch_size, self.n_agent) * self.k_max
        self.resource = torch.ones(self.batch_size, self.n_agent) * self.resource_max
        self.mu = torch.ones(self.batch_size, self.n_agent) * self.mu_max
        self.battery_level = torch.zeros(self.batch_size, self.n_agent)
        
        self.state = torch.stack([self.p, self.c, self.k, self.resource, self.mu, self.battery_level], dim=2)
        self.step_count = 0
        return self.state

    def reset(self):
        return self._RESET()

    def process_agents_outputs(self, agent_outputs):
        agent_outputs = self.ensure_action_shape(agent_outputs)
        spare_capacity = self.battery_capacity - self.battery_level
        max_order = torch.min(self.max_order_quantity, spare_capacity)
        actions = torch.min(agent_outputs, max_order)
        actions = torch.max(torch.zeros_like(actions),actions)
        return actions

    def ensure_action_shape(self, agent_outputs):
        if len(agent_outputs.shape) < 2:
            agent_outputs = agent_outputs.unsqueeze(0)
        return agent_outputs