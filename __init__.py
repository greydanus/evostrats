# Evostrats: a Minimal PyTorch Benchmark for Evolutionary Strategies
# Sam Greydanus

from .utils import *
from .models import SimpleCNN, QuadraticModel
from .optimizers import get_sgd_state, get_adam_state, sgd_fn, adam_fn
from .train_mnist import train_mnist
from .train_quadratic import train_quadratic
from .grad_estimators import Backprop, Evostrat