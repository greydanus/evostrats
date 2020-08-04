# Evostrats: a Minimal PyTorch Benchmark for Evolutionary Strategies
# Sam Greydanus

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import count_params

class QuadraticModel(torch.nn.Module):
  def __init__(self, x):
    super(QuadraticModel, self).__init__()
    self.x = torch.nn.Parameter(x)
    print("Initialized QuadraticModel with {} parameters".format(count_params(self)))

  def forward(self, A):
    return A @ self.x

class MNISTModel(torch.nn.Module):
  def __init__(self, channels=7, kernel_size=5, output_size=10):
    super(MNISTModel, self).__init__()
    self.conv1 = nn.Conv2d(1, channels, kernel_size, padding=2)
    self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=2)
    self.linear_input_size = H = channels*7*7
    self.linear1 = nn.Linear(H, output_size)
    print("Initialized MNISTModel with {} parameters".format(count_params(self)))

  def forward(self, x):
    x = F.max_pool2d(self.conv1(x).relu(), 2)
    x = F.max_pool2d(self.conv2(x).relu(), 2)
    x = x.view(-1, self.linear_input_size)   # reshape variable
    return self.linear1(x)