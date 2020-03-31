# Evostrats: a Minimal PyTorch Benchmark for Evolutionary Strategies
# Sam Greydanus

from .utils import *
from .models import SimpleCNN, QuadraticModel
from .train_mnist import train_mnist
from .train_quadratic import train_quadratic
from .grad_estimators import Backprop, Evostrat