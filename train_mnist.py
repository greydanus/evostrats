# Evostrats: a Minimal PyTorch Benchmark for Evolutionary Strategies
# Sam Greydanus

import numpy as np
import time

import torch, torchvision
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from .utils import ObjectView
from .models import MNISTModel


# a set of default hyperparameters
def get_mnist_args(as_dict=False):
  arg_dict = {'channels': 6,
              'kernel_size': 5,
              'output_size': 10,
              'total_steps': 12000,
              'batch_size': 100,
              'learn_rate': 1e-1,
              'lr_anneal_coeff': 0.75,
              'use_adam': False,
              'test_every': 100,
              'print_every': 100,
              'device': 'cuda',
              'seed': 0}
  return arg_dict if as_dict else ObjectView(arg_dict)


def mnist_fitness_fn(model, inputs, targets): # make loss look like fitness
  criterion = nn.CrossEntropyLoss()
  def fitness_fn(model):
    return -criterion(model(inputs), targets)
  return fitness_fn


# utility for loading MNIST data. Because you can never train too many MNIST classifiers.
class MNISTDataloader():
  def __init__(self, batch_size, seed=0, root='./data'):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    trainset = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)
    (self.trainloader, self.testloader) = (trainloader, testloader)
    self.trainiter = iter(self.trainloader)

  def step(self):
    try:
      return next(self.trainiter)
    except:
      self.trainiter = iter(self.trainloader)
      return next(self.trainiter)


# loop over test set to measure loss and accuracy
def evaluate_mnist(model, dataloader):
  criterion = nn.CrossEntropyLoss()
  device = list(model.parameters())[0].device # what device is the model on?
  model.eval()
  accs, losses = [], []
  for i, (inputs, targets) in enumerate(dataloader):
    logits = model(inputs.to(device))
    loss = criterion(logits, targets.to(device))
    preds = logits.argmax(-1).cpu().numpy()
    targets = targets.cpu().numpy().astype(np.float32)
    losses.append(loss.item())
    accs.append( 100.*sum(preds==targets)/targets.shape[0] )
  model.train()
  return np.mean(losses), np.mean(accs)