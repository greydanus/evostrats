# Evostrats: a Minimal PyTorch Benchmark for Evolutionary Strategies
# Sam Greydanus

import numpy as np
import time

import torch, torchvision
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from .optimizers import get_sgd_state, get_adam_state, sgd_fn, adam_fn
from .utils import get_params, set_params, angle_between, relative_norm
from .grad_estimators import Backprop, get_es_args


# your generic training loop + lots of custom logging utilities
def train_generic(args, model, grad_estimator, dataloader, fitness_fn_getter, evaluate_fn=None):
  np.random.seed(args.seed)
  model.train()

  # set up the optimizer
  optimizer_state = get_adam_state() if args.use_adam else get_sgd_state()
  optimizer_fn = adam_fn if args.use_adam else sgd_fn

  results = {'train_loss':[], 'test_loss':[], 'test_acc': [], 'angle':[], \
             'rnorm':[], 'sigma_mean':[], 'sigma_std':[], 'dt':[]}
  (s_mu, s_std, test_loss, test_acc) = (0, 0, 0, 0)
  t0 = time.time()
  for step in range(args.total_steps):

    # run data through model and estimate gradient
    inputs, targets = dataloader.step()
    (inputs, targets) = (inputs.to(args.device), targets.to(args.device))
    fitness_fn = fitness_fn_getter(model, inputs, targets)
    fitness, grad_est = grad_estimator.step(model, fitness_fn, inputs, is_training=True)
    loss = -fitness ; results['train_loss'].append(loss)
    
    # update trainable parameters
    grad_est, optimizer_state = optimizer_fn(grad_est, optimizer_state)
    new_params = get_params(model) + args.learn_rate * grad_est
    set_params(model, new_params)

    # learning rate schedule
    if step % 1000 == 0 and step > 0:
      args.learn_rate *= args.lr_anneal_coeff

    # bookkeeping
    if step % args.test_every == 0:
      t1 = time.time()
      if evaluate_fn is not None:
        test_loss, test_acc = evaluate_model(model, dataloader.testloader)

      _, grad_est = grad_estimator.step(model, fitness_fn, inputs)
      _, grad_true = Backprop().step(model, fitness_fn, inputs)
      angle, rnorm = angle_between(grad_est, grad_true), relative_norm(grad_est, grad_true)

      if hasattr(grad_estimator, 'sigma'):
        s_mu, s_std = grad_estimator.sigma.mean(), grad_estimator.sigma.std()

      results['dt'].append(t1-t0)
      results['test_loss'].append(test_loss) ; results['test_acc'].append(test_acc)
      results['angle'].append(angle) ; results['rnorm'].append(rnorm)
      results['sigma_mean'].append(s_mu) ; results['sigma_mean'].append(s_std)

    if step % args.print_every == 0:
      print(('step {}, dt {:.0f}s, train {:.2e}, test {:.1e}, acc {:.1f}, ' + \
             'angle {:.1e}, rel_norm {:.1e}, s_mu {:.1e}, s_std {:.1e}')
            .format(step, t1-t0, loss, test_loss, test_acc, angle, rnorm, s_mu, s_std))
      t0 = t1
  return results
