# Evostrats: a Minimal PyTorch Benchmark for Evolutionary Strategies
# Sam Greydanus

import numpy as np
import time

import torch
import torch.nn as nn

from .utils import get_params, set_params, angle_between, relative_norm, ObjectView
from .grad_estimators import Backprop


def get_evostrat_args(as_dict=False):
  arg_dict = {'popsize': 5,
              'sigma': 1e-3,
              'sigma_learn_rate': 0,
              'use_antithetic': True,
              'use_fitness_shaping': False,
              'use_safe_mutation': False,
              'sigma_learn_rate': 0,
              'alpha': 1.,                # put this between 0 and 1 to do guided ES
              'beta': 100.,                # gradient scaling coefficient (from ES paper)
              'device': 'cpu'}
  return arg_dict if as_dict else ObjectView(arg_dict)

def get_quad_args(as_dict=False):
  arg_dict = {'quad_n': 1000,
              'quad_m': 2000,
              'total_steps': 1000,
              'learn_rate': 2e-2,
              'lr_anneal_coeff': 1.,
              'use_evolution': True,
              'test_every': 100,
              'print_every': 100,
              'device': 'cpu',
              'seed': 2,}
  return arg_dict if as_dict else ObjectView(arg_dict)


def generate_problem(n, m, seed=None):
  rs = np.random.RandomState(seed=seed)
  A, b = rs.randn(m, n), rs.randn(m, 1)
  grad_bias = rs.randn(n, 1)
  return A, b, grad_bias


def get_fitness_fn(A, b): # make loss look like fitness
  def fitness_fn(model):
    return -.5*torch.norm(model(A) - b)**2 / len(b)
  return fitness_fn



# train an mnist model + lots of custom logging utilities
def train_quadratic(model, grad_estimator, args):
  model.train()

  # construct the problem
  A_np, b_np, gbias_np = generate_problem(args.quad_m, args.quad_n, seed=args.seed)
  A, b, gbias = torch.Tensor(A_np), torch.Tensor(b_np), torch.Tensor(gbias_np)
  gbias = 1.05 * gbias / gbias.norm()
  fitness_fn = get_fitness_fn(A, b)

  results = {'train_loss':[], 'angle':[], 'rnorm':[], 'sigma_mean':[], 'sigma_std':[], 'dt':[]}
  t0 = time.time()
  for i in range(args.total_steps):

      # update the model
      # print(fitness_fn(model))
      fitness, grad_est = grad_estimator.step(model, fitness_fn, A.to(args.device))
      loss = -fitness
      new_params = get_params(model) + args.learn_rate * grad_est
      set_params(model, new_params)

      # bookkeeping
      results['train_loss'].append(loss)
      if i % args.test_every == 0:
        _, grad_est = grad_estimator.step(model, fitness_fn, A.to(args.device))
        _, grad_true = Backprop().step(model, fitness_fn, A.to(args.device))
        angle, rnorm = angle_between(grad_est, grad_true), relative_norm(grad_est, grad_true)

        t1 = time.time()
        results['angle'].append(angle) ; results['rnorm'].append(rnorm)
        results['dt'].append(t1-t0)
        s_mu, s_std = 0, 0
        if hasattr(grad_estimator, 'sigma'):
          s_mu, s_std = grad_estimator.sigma.mean(), grad_estimator.sigma.std()
        results['sigma_mean'].append(s_mu) ; results['sigma_mean'].append(s_std)
        print(('step {}, dt {:.0f}s, train {:.2e}, angle {:.1e}, rel_norm {:.1e}, s_mu {:.1e}, s_std {:.1e}')
              .format(i, t1-t0, loss, angle, rnorm, s_mu, s_std))
        t0 = t1
  return results
