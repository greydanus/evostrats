# Evostrats: a Minimal PyTorch Benchmark for Evolutionary Strategies
# Sam Greydanus

import numpy as np
import time

import torch, torchvision
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from .optimizers import get_sgd_state, get_adam_state, sgd_fn, adam_fn
from .utils import get_params, set_params, angle_between, relative_norm, ObjectView
<<<<<<< HEAD
from .grad_estimators import Backprop, get_es_args
=======
from .grad_estimators import Backprop


def get_evostrat_args(as_dict=False):
  arg_dict = {'popsize': 100,
              'sigma': 1e-2,
              'sigma_learn_rate': 0,
              'use_antithetic': True,
              'use_fitness_shaping': True,
              'use_safe_mutation': False,
              'sigma_learn_rate': 0,
              'alpha': 1.,                # put this between 0 and 1 to do guided ES
              'beta': 100.,                # gradient scaling coefficient (from ES paper)
              'device': 'cuda'}
  return arg_dict if as_dict else ObjectView(arg_dict)
>>>>>>> parent of 04a74ef... Update hyperparams in pivot_states


# a set of default hyperparameters
def get_mnist_args(as_dict=False):
  arg_dict = {'channels': 6,
              'kernel_size': 5,
              'output_size': 10,
              'epochs': 20,
              'batch_size': 100,
              'learn_rate': 1e-1,
              'lr_anneal_coeff': 0.75,
              'use_adam': False,
              'test_every': 100,
              'print_every': 100,
              'device': 'cuda',
              'seed': 0,}
  return arg_dict if as_dict else ObjectView(arg_dict)


# utility for loading MNIST data. Because you can never train too many MNIST classifiers.
def get_dataloaders(args):
  transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

  trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
  trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

  testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
  testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=2)
  return trainloader, testloader


# loop over test set to measure loss and accuracy
def evaluate_model(model, dataloader, criterion):
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


# make a loss function look like a fitness function
def get_fitness_fn(criterion, inputs, targets):
  def fitness_fn(model):
    return -criterion(model(inputs), targets)
  return fitness_fn


# train an mnist model + lots of custom logging utilities
def train_mnist(model, data, grad_estimator, args):
  trainloader, testloader = data
  criterion = nn.CrossEntropyLoss()
  model.train() ; model.to(args.device)

  # set up the optimizer
  optimizer_state = get_adam_state() if args.use_adam else get_sgd_state()
  optimizer_fn = adam_fn if args.use_adam else sgd_fn

  results = {'train_loss':[], 'test_loss':[], 'test_acc':[], 'global_step':0, \
             'angle':[], 'rnorm':[], 'sigma_mean':[], 'sigma_std':[], 'dt':[]}
  t0 = time.time()
  for epoch in range(args.epochs):
    for (inputs, targets) in trainloader:

      # update the model
      inputs, targets = inputs.to(args.device), targets.to(args.device)  # move data to GPU
      fitness_fn = get_fitness_fn(criterion, inputs, targets)
      fitness, grad_est = grad_estimator.step(model, fitness_fn, inputs)
      loss = -fitness  # semantics are weird
      grad_est, optimizer_state = optimizer_fn(grad_est, optimizer_state)  # lets us swap out sgd, adam
      new_params = get_params(model) + args.learn_rate * grad_est
      set_params(model, new_params)

      # learning rate schedule
      if results['global_step'] % 1000 == 0 and results['global_step'] > 0:
        args.learn_rate *= args.lr_anneal_coeff

      # bookkeeping
      results['global_step'] += 1
      results['train_loss'].append(loss)
      if results['global_step'] % args.test_every == 0:
        test_loss, test_acc = evaluate_model(model, testloader, criterion)

        fitness_fn = get_fitness_fn(criterion, inputs, targets)
        _, grad_est = grad_estimator.step(model, fitness_fn, inputs, is_training=False)
        _, grad_true = Backprop().step(model, fitness_fn, inputs)
        angle, rnorm = angle_between(grad_est, grad_true), relative_norm(grad_est, grad_true)

        t1 = time.time()
        results['test_loss'].append(test_loss) ; results['test_acc'].append(test_acc)
        results['angle'].append(angle.item()) ; results['rnorm'].append(rnorm.item())
        results['dt'].append(t1-t0)
        s_mu, s_std = 0, 0
        if hasattr(grad_estimator, 'sigma'):
          s_mu, s_std = grad_estimator.sigma.mean().item(), grad_estimator.sigma.std().item()
        results['sigma_mean'].append(s_mu) ; results['sigma_mean'].append(s_std)

      # logging everything because we're trying to do SCIENCE
      if results['global_step'] % args.print_every == 0:
        print(('epoch {}, global_step {}, dt {:.0f}s, train {:.1e}, test {:.1e}, ' + \
              'acc {:.1f}, angle {:.1e}, rel_norm {:.1e}, s_mu {:.1e}, s_std {:.1e}')
              .format(epoch, results['global_step'], t1-t0, loss, test_loss, test_acc,
                      angle, rnorm, s_mu, s_std))
        t0 = t1
  return results
