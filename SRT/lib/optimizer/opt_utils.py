# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
from bisect import bisect_right
from .loss_module import MaskedLoss

class MultiStepLRS(torch.optim.lr_scheduler._LRScheduler):

  def __init__(self, optimizer, milestones, gammas, last_epoch=-1):
    if not list(milestones) == sorted(milestones):
      raise ValueError('Milestones should be a list of'
                       ' increasing integers. Got {:}', milestones)
    assert len(milestones) == len(gammas), '{:} vs {:}'.format(milestones, gammas)
    self.milestones = milestones
    self.gammas = gammas
    super(MultiStepLRS, self).__init__(optimizer, last_epoch)

  def get_lr(self):
    LR = 1
    for x in self.gammas[:bisect_right(self.milestones, self.last_epoch)]: LR = LR * x
    return [base_lr * LR for base_lr in self.base_lrs]


def obtain_optimizer(params, config, logger):
  assert hasattr(config, 'optimizer'), 'Must have the optimizer attribute'
  optimizer = config.optimizer.lower()
  if optimizer == 'sgd':
    opt = torch.optim.SGD(params, lr=config.LR, momentum=config.momentum,
                          weight_decay=config.weight_decay, nesterov=config.nesterov)
  elif optimizer == 'rmsprop':
    opt = torch.optim.RMSprop(params, lr=config.LR, momentum=config.momentum,
                          alpha = config.alpha, eps=config.epsilon,
                          weight_decay = config.weight_decay)
  elif optimizer == 'adam':
    opt = torch.optim.Adam(params, lr=config.LR, amsgrad=config.amsgrad, betas=(config.betas[0], config.betas[1]), weight_decay=config.weight_decay)
  else:
    raise TypeError('Does not know this optimizer : {:}'.format(config))

  if config.scheduler == "multistep":
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=config.schedule, gamma=config.gamma)
  elif config.scheduler == "multisteps":
    scheduler = MultiStepLRS(opt, milestones=config.schedule, gammas=config.gammas)
  elif config.scheduler == "cosine":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, float(config.epochs), eta_min=config.LR_min)
  else:
    raise ValueError('invalid scheduler : {:}'.format(config.scheduler))
    

  strs = config.criterion.split('-')
  assert len(strs) == 2, 'illegal criterion : {:}'.format(config.criterion)

  if strs[0].lower() == 'mse':
    criterion = MaskedLoss('mse', reduction=strs[1])
    message = 'Optimizer : {:}, MSE Loss with reduction={:}'.format(opt, strs)
  elif strs[0].lower() == 'l1':
    criterion = MaskedLoss('L1', reduction=strs[1])
    message = 'Optimizer : {:}, L1 Loss with reduction={:}'.format(opt, strs)
  elif strs[0].lower() == 'smooth_l1':
    criterion = MaskedLoss('smoothL1', reduction=strs[1])
    message = 'Optimizer : {:}, L1 Loss with reduction={:}'.format(opt, strs)
  else:
    raise TypeError('Does not know this optimizer : {:}'.format(config.criterion))

  if logger is not None: logger.log(message)
  else                 : print(message)

  return opt, scheduler, criterion
