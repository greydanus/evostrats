# Evostrats: a Minimal PyTorch Benchmark for Evolutionary Strategies
# Sam Greydanus

import numpy as np
import time
import torch

from .models import RastriginModel
from .utils import ObjectView


# a set of default hyperparameters
def get_rast_args(as_dict=False):
  arg_dict = {'d': 2,
              'total_steps': 1000,
              'learn_rate': 1e-3,
              'lr_anneal_coeff': 1,
              'use_adam': False,
              'test_every': 10,
              'print_every': 50,
              'device': 'cpu',
              'seed': 0}
  return arg_dict if as_dict else ObjectView(arg_dict)


# inspired by the toy problem in the Guided ES paper (arxiv.org/abs/1806.10230)
def rast_init(N):
  return np.ones((1,2))

def rast_fitness_fn(model, inputs, targets): # make loss look like fitness
  def fitness_fn(model):
    return rastrigin(model(inputs)).sum()
  return fitness_fn

def rastrigin(x):
  """Rastrigin test objective function, shifted by 10. units away from origin"""
  z = x - 10.0
  z = z.reshape(-1, z.shape[-1])
  dim = z.shape[-1]
  cosine = np.cos if type(x) is np.ndarray else torch.cos
  return -(10*z.shape[-1] + (z**2 - 10 * cosine(2*np.pi*z)).sum(-1, keepdims=True))


class RastriginDataloader():
  def __init__(self):
    pass
  def step(self):
    return torch.Tensor([]), torch.Tensor([])