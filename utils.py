# Evostrats: a Minimal PyTorch Benchmark for Evolutionary Strategies
# Sam Greydanus

import numpy as np
import torch

# Some people are haters, eg stackoverflow.com/questions/21752259/python-why-pickle
import pickle  # but I will import and cherish you nonetheless <3


def to_pickle(thing, path): # save something
    with open(path, 'wb') as handle:
        pickle.dump(thing, handle, protocol=pickle.HIGHEST_PROTOCOL)

def from_pickle(path): # load something
    thing = None
    with open(path, 'rb') as handle:
        thing = pickle.load(handle)
    return thing

class ObjectView(object):
    def __init__(self, d): self.__dict__ = d


def get_params(model):
  return torch.cat([p.reshape(-1) for p in model.parameters()])

def count_params(model):
  return len(get_params(model))

def set_params(model, pv):
  pointer = 0
  for p in model.parameters():
    start = pointer
    stop = pointer + p.reshape(-1).shape[0]
    p.data = pv[start:stop].reshape(p.data.shape).clone()
    pointer = stop

def get_grads(model):
  return torch.cat([p.grad.reshape(-1) for p in model.parameters()])

def clear_grads(model):
  for p in model.parameters(): p.grad = None

def angle_between(a, b, eps=1e-10):
  return (a @ b) / (a.norm()*b.norm() + eps)

def relative_norm(a, b, eps=1e-10):
  return a.norm() / (b.norm() + eps)

# I modified this from someone else's repo on GitHub, but it seems valid
#    -> the dude=github.com/crisbodnar, the line of code=https://bit.ly/2Btikye
def jacobian_of_params(model, x):
  output = model(x) # assume model and x are on same device
  grad_output = torch.zeros(*output.shape).to(x.device)
  jacobian = torch.zeros(output.shape[-1], count_params(model)).to(x.device)

  clear_grads(model)
  for i in range(output.shape[-1]): # note: inefficient if there are many outputs
    grad_output[..., i] = 1.0
    output.backward(grad_output, retain_graph=True)
    jacobian[i] = get_grads(model)
    clear_grads(model)
  return jacobian