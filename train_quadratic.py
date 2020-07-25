# Evostrats: a Minimal PyTorch Benchmark for Evolutionary Strategies
# Sam Greydanus

import numpy as np
import time

import torch
import torch.nn as nn

from .utils import get_params, set_params, angle_between, relative_norm, ObjectView
from .grad_estimators import Backprop, get_es_args
from .optimizers import get_sgd_state, get_adam_state, sgd_fn, adam_fn

def get_quad_args(as_dict=False):
  arg_dict = {'quad_m': 2000,
              'quad_n': 1000,
              'total_steps': 1000,
              'learn_rate': 2e-2,
              'lr_anneal_coeff': 1.,
              'use_adam': False,
              'test_every': 10,
              'print_every': 100,
              'device': 'cpu',
              'seed': 0,}
  return arg_dict if as_dict else ObjectView(arg_dict)


def generate_problem(m, n, seed=None):
  rs = np.random.RandomState(seed=seed)
  A, b = rs.randn(m, n), rs.randn(m, 1)
  grad_bias = rs.randn(n, 1)
  return A, b, grad_bias


def get_fitness_fn(A, b): # make loss look like fitness
  def fitness_fn(model):
    return -.5*torch.norm(model(A) - b)**2 / len(b)
  return fitness_fn



# train a model + lots of custom logging utilities
def train_quadratic(model, grad_estimator, args):
  model.train()

  # construct the problem
  A_np, b_np, gbias_np = generate_problem(args.quad_m, args.quad_n, seed=args.seed)
  A, b, gbias = torch.Tensor(A_np), torch.Tensor(b_np), torch.Tensor(gbias_np)
  gbias = 1.05 * gbias / gbias.norm()
  fitness_fn = get_fitness_fn(A, b)

  # set up the optimizer
  optimizer_state = get_adam_state() if args.use_adam else get_sgd_state()
  optimizer_fn = adam_fn if args.use_adam else sgd_fn

  results = {'train_loss':[], 'angle':[], 'rnorm':[], 'sigma_mean':[], 'sigma_std':[], 'dt':[]}
  t0 = time.time()
  for i in range(args.total_steps):

      # update the model
      # print(fitness_fn(model))
      fitness, grad_est = grad_estimator.step(model, fitness_fn, A.to(args.device))
      loss = -fitness
      grad_est, optimizer_state = optimizer_fn(grad_est, optimizer_state)  # this lets us swap out sgd for adam
      new_params = get_params(model) + args.learn_rate * grad_est
      set_params(model, new_params)

      # bookkeeping
      results['train_loss'].append(loss)
      if i % args.test_every == 0:
        _, grad_est = grad_estimator.step(model, fitness_fn, A.to(args.device), is_training=False)
        _, grad_true = Backprop().step(model, fitness_fn, A.to(args.device))
        angle, rnorm = angle_between(grad_est, grad_true), relative_norm(grad_est, grad_true)

        t1 = time.time()
        results['angle'].append(angle) ; results['rnorm'].append(rnorm)
        results['dt'].append(t1-t0)
        s_mu, s_std = 0, 0
        if hasattr(grad_estimator, 'sigma'):
          s_mu, s_std = grad_estimator.sigma.mean(), grad_estimator.sigma.std()
        results['sigma_mean'].append(s_mu) ; results['sigma_mean'].append(s_std)
      if i % args.print_every == 0:
        print(('step {}, dt {:.0f}s, train {:.2e}, angle {:.1e}, rel_norm {:.1e}, s_mu {:.1e}, s_std {:.1e}')
              .format(i, t1-t0, loss, angle, rnorm, s_mu, s_std))
        t0 = t1
  return results
