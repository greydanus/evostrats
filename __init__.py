# Evostrats: a Minimal PyTorch Benchmark for Evolutionary Strategies
# Sam Greydanus

from .utils import *
from .grad_estimators import Backprop, Evostrat, get_es_args
from .train_generic import train_generic
from .model import MNISTModel, QuadraticModel