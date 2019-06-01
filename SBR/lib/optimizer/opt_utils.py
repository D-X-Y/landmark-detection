# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch

def obtain_optimizer(params, config, logger):
  assert hasattr(config, 'optimizer'), 'Must have the optimizer attribute'
  optimizer = config.optimizer.lower()
  if optimizer == 'sgd':
    opt = torch.optim.SGD(params, lr=config.LR, momentum=config.momentum,
                          weight_decay=config.Decay, nesterov=config.nesterov)
  elif optimizer == 'rmsprop':
    opt = torch.optim.RMSprop(params, lr=config.LR, momentum=config.momentum,
                          alpha = config.alpha, eps=config.epsilon,
                          weight_decay = config.weight_decay)
  elif optimizer == 'adam':
    opt = torch.optim.Adam(params, lr=config.LR, amsgrad=config.amsgrad)
  else:
    raise TypeError('Does not know this optimizer : {:}'.format(config))

  scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=config.schedule, gamma=config.gamma)

  strs = config.criterion.split('-')
  assert len(strs) == 2, 'illegal criterion : {:}'.format(config.criterion)
  if strs[0].lower() == 'mse':
    size_average = strs[1].lower() == 'avg'
    criterion = torch.nn.MSELoss(size_average)
    message = 'Optimizer : {:}, MSE Loss with size-average={:}'.format(opt, size_average)
    if logger is not None: logger.log(message)
    else                 : print(message)
  else:
    raise TypeError('Does not know this optimizer : {:}'.format(config.criterion))

  return opt, scheduler, criterion
