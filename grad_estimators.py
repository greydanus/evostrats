# Evostrats: a Minimal PyTorch Benchmark for Evolutionary Strategies
# Sam Greydanus

import numpy as np
import torch

from .utils import get_params, set_params, get_grads, clear_grads, jacobian_of_params


class Backprop():
  def __init__(self):
    pass

  def step(self, model, fitness_fn, x):
    fitness = fitness_fn(model)
    fitness.backward()
    grad = get_grads(model)
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

  def step(self, model, fitness_fn, x):
    # use evolution strategies to estimate fitness gradient

    fitness, epsilons = self.eval_population(model, fitness_fn, x)
    mean_fitness = np.mean(fitness)

    if self.use_fitness_shaping:
      fitness = self.fitness_shaping(fitness)

    if self.sigma_learn_rate > 0 and self.use_antithetic:
      # note: this is only implemented for antithetic sampling
      self.update_adaptive_sigma(fitness, epsilons)

    grad = self.estimate_grad(fitness, epsilons)
    return mean_fitness, grad

  def sample(self, model, x):
    # sample perturbations to the model parameters, using various sampling strategies

    if self.use_antithetic:
      eps = self.sigma * torch.randn(self.popsize//2, self.num_params).to(self.device)
      eps = torch.cat([eps, -eps], dim=0).to(self.device) # antithetic sampling
    else:
      eps = self.sigma * torch.randn(self.popsize, self.num_params).to(self.device)

    if self.alpha < 1. and self.prev_grad_est is not None: # guided es
      eps = self.guided_es_sample(eps)

    if self.use_safe_mutation:  # safe mutation with output gradients
      eps = self.safe_mutation(eps, model, x)

    return eps

  def eval_population(self, model, fitness_fn, x):
    # evaluate the fitness of a "population" of perturbations to model params

    params = get_params(model)
    epsilons, fitness = self.sample(model, x), np.zeros(self.popsize)

    for i in range(self.popsize):
      set_params(model, params + epsilons[i])
      fitness[i] = fitness_fn(model)

    set_params(model, params)
    return fitness, epsilons

  def estimate_grad(self, fitness, epsilons):
    # given fitness scores of a population, estimate the fitness gradient

    if self.use_antithetic:
      diffs = (fitness[:self.popsize//2] - fitness[self.popsize//2:])
      epsilons = epsilons[:self.popsize//2]
    else:
      diffs = fitness - fitness.mean()

    diffs = self.beta * torch.Tensor(diffs.reshape(-1,1)).to(self.device)
    if not self.use_fitness_shaping:
      diffs /= (self.sigma.mean()**2 * len(params)) # finite differences denominator

    self.prev_grad_est = grad = (diffs * epsilons).mean(0)
    return grad

  def guided_es_sample(self, eps):
    # see "guided evolutionary strategies" (arxiv.org/abs/1806.10230)
    #    here we use a stale gradient to guide search, as in (arxiv.org/abs/1910.05268)
    U = (self.prev_grad_est / self.prev_grad_est.norm()).reshape(1,-1)
    subspace_sample = self.sigma * np.sqrt(self.num_params) * U
    return np.sqrt(self.alpha)*eps + np.sqrt(1-self.alpha)*subspace_sample

  def safe_mutation(self, eps, model, x):
    # see "safe mutation via output gradients..." (arxiv.org/abs/1712.06563)
    J = jacobian_of_params(model, x)
    scale = J.abs().mean(0).reshape(1,-1)
    scale[scale < 0.01] = 0.01 # minimum value is 0.01
    scale[scale > 5.] = 5. # max value is 5
    eps /= scale
    return eps

  def fitness_shaping(self, fitness):
    # see "natural evolution strategies" (arxiv.org/abs/1106.4487)
    ranks = np.empty_like(fitness) # next line ranks, then puts in [-.5, 0.5]
    ranks[np.argsort(fitness)] = np.linspace(-.5, 0.5, len(fitness))
    fitness = ranks  # rank-based fitness shaping
    return fitness

  def update_adaptive_sigma(self, fitness, epsilons):
    # see "parameter exploring policy gradients" (paper: bit.ly/3dBw3RX)
    epsilons = epsilons[:self.popsize//2]
    S = (epsilons.pow(2) - self.sigma.pow(2)) / self.sigma  # [popsize/2, num_params]
    est_current_fitness = (fitness[:self.popsize//2] + fitness[self.popsize//2:]) / 2.0
    fitness_err = est_current_fitness - est_current_fitness.mean()
    fitness_err = torch.Tensor(fitness_err).reshape(-1,1).to(self.device) # [popsize/2, 1]
    sigma_grad = (fitness_err * S).mean(0, keepdim=True) \
                  / (self.popsize * fitness.std()) # [1, num_params]
    self.sigma += self.sigma_learn_rate * sigma_grad.squeeze()