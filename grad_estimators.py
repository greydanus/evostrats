# Evostrats: a Minimal PyTorch Benchmark for Evolutionary Strategies
# Sam Greydanus

import numpy as np
import torch

from .utils import get_params, set_params, get_grads, clear_grads, jacobian_of_params

def get_es_args(as_dict=False):
  arg_dict = {'popsize': 100,
              'sigma': 1e-2,
              'sigma_learn_rate': 0,
              'use_antithetic': True,
              'use_fitness_shaping': True,
              'use_safe_mutation': False,
              'alpha': 1.,                # put this between 0 and 1 to do guided ES
              'beta': 100.,                # gradient scaling coefficient (from ES paper)
              'device': 'cuda'}
  return arg_dict if as_dict else ObjectView(arg_dict)

class Backprop():
  def __init__(self, bias=None):
    self.bias = bias

  def step(self, model, fitness_fn, x, is_training=True):
    fitness = fitness_fn(model)
    fitness.backward()
    grad = get_grads(model)
    grad = grad - self.bias * grad.norm() if self.bias is not None else grad
    clear_grads(model)
    return fitness.item(), grad


class Evostrat():
  def __init__(self, num_params,          # number of parameters
               popsize=100,               # population size / # of perturbations
               sigma=1e-2,                # standard deviation of perturbations
               sigma_learn_rate=0,        # adaptive sigma normalization (as in PEPG paper)
               use_antithetic=True,       # whether to use antithetic sampling
               use_fitness_shaping=True,  # whether to use fitness shaping (as in NES paper)
               use_safe_mutation=False,   # whether to use safe mutation of output gradients
               alpha=1.,                  # guided evolution coeff: range:[0,1], 1=vanilla es
               beta=50.,                  # constant multiplicative factor for gradient
               device='cpu'):             # 'cpu' for CPU, 'cuda' for GPU

    self.num_params = num_params
    self.popsize = popsize
    self.sigma = sigma * torch.ones(1, num_params, device=device)
    self.sigma_learn_rate = sigma_learn_rate
    self.use_antithetic = use_antithetic
    self.use_fitness_shaping = use_fitness_shaping
    self.use_safe_mutation = use_safe_mutation
    self.alpha = alpha
    self.beta = beta
    self.device = device
    self.prev_grad_est = None

  def step(self, model, fitness_fn, x, is_training=True):
    '''Use evolution strategies to estimate fitness gradient.'''
    fitness, epsilons = self.eval_population(model, fitness_fn, x)
    current_fitness = fitness_fn(model).item()

    if self.use_fitness_shaping:
      fitness = self.fitness_shaping(fitness)

    if self.sigma_learn_rate > 0 and self.use_antithetic:
      # note: this is only implemented for antithetic sampling
      self.update_adaptive_sigma(fitness, epsilons)

    grad = self.estimate_grad(fitness, epsilons, current_fitness)
    if is_training:
      self.prev_grad_est = grad
    return current_fitness, grad

  def eval_population(self, model, fitness_fn, x):
        '''Evaluate the fitness of a "population" of perturbations. If you squint,
    you can see that evolutionary strategies are glorified guess-and-check.'''
    params = get_params(model)
    epsilons, fitness = self.sample(model, x), np.zeros(self.popsize)

    for i in range(self.popsize):
      set_params(model, params + epsilons[i])
      fitness[i] = fitness_fn(model)

    set_params(model, params)
    return fitness, epsilons

  def sample(self, model, x):
    '''Sample perturbations to the model parameters, using various sampling strategies.'''

    eps = torch.randn(self.popsize, self.num_params).to(self.device)
    if self.use_antithetic:
      eps = eps[:self.popsize//2]

    if self.alpha < 1. and self.prev_grad_est is not None: # guided es
      eps = self.guided_es_sample(eps)

    if self.use_safe_mutation:  # safe mutation with output gradients
      eps = self.safe_mutation(eps, model, x)

    if self.use_antithetic:
      eps = torch.cat([eps, -eps], dim=0).to(self.device) # antithetic sampling

    return self.sigma * eps

  def estimate_grad(self, fitness, epsilons, current_fitness):
    '''Given fitness scores of a population, estimate the fitness gradient.'''

    if self.use_antithetic:
      diffs = 0.5 * (fitness[:self.popsize//2] - fitness[self.popsize//2:])
      epsilons = epsilons[:self.popsize//2]
    else:
      diffs = fitness - current_fitness

    diffs = self.beta * torch.Tensor(diffs.reshape(-1,1)).to(self.device)
    if not self.use_fitness_shaping:
      diffs /= (self.sigma.mean()**2 * self.num_params) # finite differences denominator

    return (diffs * epsilons).mean(0)

  def guided_es_sample(self, eps):
    '''See "guided evolutionary strategies" (arxiv.org/abs/1806.10230)
    here we use a stale gradient to guide search, as in (arxiv.org/abs/1910.05268)'''
    U = (self.prev_grad_est / self.prev_grad_est.norm()).reshape(1,-1)
    subspace_sample = np.sqrt(self.num_params) * U
    return np.sqrt(self.alpha)*eps + np.sqrt(1-self.alpha)*subspace_sample

  def safe_mutation(self, eps, model, x):
    '''See "safe mutation via output gradients..." (arxiv.org/abs/1712.06563)'''
    J = jacobian_of_params(model, x)  # J is of dimension [num_outputs x num_inputs]
    mutation_scale = J.abs().mean(0).reshape(1,-1)
    mutation_scale = mutation_scale.clamp(0.01, 5)  # for stability, put in range [0.01, 5]
    return eps / mutation_scale  # yay, gradients won't blow up as much

  def fitness_shaping(self, fitness):
    '''See "natural evolution strategies" (arxiv.org/abs/1106.4487).'''
    ranks = np.empty_like(fitness) # next line ranks, then puts in [-.5, 0.5]
    ranks[np.argsort(fitness)] = np.linspace(-.5, 0.5, len(fitness))
    fitness = ranks  # rank-based fitness shaping
    return fitness

  def update_adaptive_sigma(self, fitness, epsilons):
    '''See "parameter exploring policy gradients" (paper: bit.ly/3dBw3RX).'''
    epsilons = epsilons[:self.popsize//2]
    S = (epsilons.pow(2) - self.sigma.pow(2)) / self.sigma  # [popsize/2, num_params]
    est_current_fitness = (fitness[:self.popsize//2] + fitness[self.popsize//2:]) / 2.0
    fitness_err = est_current_fitness - est_current_fitness.mean()
    fitness_err = torch.Tensor(fitness_err).reshape(-1,1).to(self.device) # [popsize/2, 1]
    sigma_grad = (fitness_err * S).mean(0, keepdim=True) \
                  / (self.popsize * fitness.std()) # [1, num_params]
    self.sigma += self.sigma_learn_rate * sigma_grad.squeeze()