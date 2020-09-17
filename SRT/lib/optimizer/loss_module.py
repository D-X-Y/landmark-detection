# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedLoss(nn.Module):
  def __init__(self, xtype, reduction):
    super(MaskedLoss, self).__init__()
    reductions = ['none', 'mean', 'sum', 'batch']
    assert reduction in reductions, 'Invalid {:}, not in {:}'.format(reduction, reductions)
    self.reduction = reduction
    if self.reduction == 'batch':
      self.reduction = 'sum'
      self.divide_by_batch = True
    else:
      self.divide_by_batch = False
    if xtype == 'mse':
      self.func = F.mse_loss
    elif xtype == 'smoothL1':
      self.func = F.smooth_l1_loss
    elif xtype == 'L1':
      self.func = F.l1_loss
    else: raise ValueError('Invalid Type : {:}'.format(xtype))
  
    self.loss_l1_func  = F.l1_loss
    self.loss_mse_func = F.mse_loss
    self.xtype = xtype

  def extra_repr(self):
    return ('type={xtype}, reduction={reduction}, divide_by_batch={divide_by_batch}'.format(**self.__dict__))

  def forward(self, inputs, targets, masks):
    batch_size = inputs.size(0)
    if masks is not None:
      inputs  = torch.masked_select(inputs, masks.bool())
      targets = torch.masked_select(targets, masks.bool())
    x_loss = self.func(inputs, targets, reduction=self.reduction)
    if self.divide_by_batch:
      x_loss = x_loss / batch_size
    return x_loss

