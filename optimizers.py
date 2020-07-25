# Evostrats: a Minimal PyTorch Benchmark for Evolutionary Strategies
# Sam Greydanus

import torch


def get_adam_state():
  return {'m': None, 'v': None, 'b1': 0.9,
          'b2': 0.999, 'eps': 10**-8, 'step': 0}

def adam_fn(grad, adam_state):
  # see https://github.com/HIPS/autograd/blob/master/autograd/misc/optimizers.py
  adam = adam_state
  if adam['m'] is None:
    adam['m'], adam['v'] = torch.zeros_like(grad), torch.zeros_like(grad)
  adam['m'] = (1 - adam['b1']) * grad      + adam['b1'] * adam['m']  # First  moment estimate.
  adam['v'] = (1 - adam['b2']) * (grad**2) + adam['b2'] * adam['v']  # Second moment estimate.
  mhat = adam['m'] / (1 - adam['b1']**(adam['step'] + 1))    # Bias correction.
  vhat = adam['v'] / (1 - adam['b2']**(adam['step'] + 1))
  adam_grad = mhat / (torch.sqrt(vhat) + adam['eps'])
  adam['step'] += 1
  return adam_grad, adam

def get_sgd_state():
  return {}

def sgd_fn(grad, sgd_state):
  return grad, sgd_state