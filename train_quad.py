# Evostrats: a Minimal PyTorch Benchmark for Evolutionary Strategies
# Sam Greydanus

import numpy as np
import time
import torch

from .models import QuadraticModel
from .utils import ObjectView


# a set of default hyperparameters
def get_quad_args(as_dict=False):
  arg_dict = {'m': 2000,
              'n': 1000,
              'total_steps': 1000,
              'learn_rate': 5e-2,
              'lr_anneal_coeff': 1,
              'use_adam': False,
              'test_every': 10,
              'print_every': 100,
              'device': 'cpu',
              'seed': 0}
  return arg_dict if as_dict else ObjectView(arg_dict)


# inspired by the toy problem in the Guided ES paper (arxiv.org/abs/1806.10230)
def quad_init(m, n, seed=None):
  A = np.random.randn(m, n)
  b = np.random.randn(m, 1)
  grad_bias = np.random.randn(n, 1)
  return A, b, grad_bias


def quad_fitness_fn(model, inputs, targets): # make loss look like fitness
  def fitness_fn(model):
    return -.5*torch.norm(model(inputs) - targets)**2 / len(targets)
  return fitness_fn


class QuadraticDataloader():
  def __init__(self, m, n, seed=0):
    # construct the problem
    A, b, gbias = quad_init(m, n, seed=seed)
    A, b, gbias = [torch.Tensor(v) for v in [A, b, gbias]]
    gbias = 1.05 * gbias / gbias.norm()
    (self.A, self.b, self.gbias) = (A, b, gbias)

  def step(self):
    return self.A, self.b