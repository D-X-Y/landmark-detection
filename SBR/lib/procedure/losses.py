# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np
import numbers, torch
import torch.nn.functional as F

def compute_stage_loss(criterion, targets, outputs, masks):
  assert isinstance(outputs, list), 'The ouputs type is wrong : {:}'.format(type(outputs))
  total_loss = 0
  each_stage_loss = []
  
  for output in outputs:
    stage_loss = 0
    output = torch.masked_select(output , masks)
    target = torch.masked_select(targets, masks)

    stage_loss = criterion(output, target)
    total_loss = total_loss + stage_loss
    each_stage_loss.append(stage_loss.item())
  return total_loss, each_stage_loss


def show_stage_loss(each_stage_loss):
  if each_stage_loss is None:            return 'None'
  elif isinstance(each_stage_loss, str): return each_stage_loss
  answer = ''
  for index, loss in enumerate(each_stage_loss):
    answer = answer + ' : L{:1d}={:7.4f}'.format(index+1, loss)
  return answer


def sum_stage_loss(losses):
  total_loss = None
  each_stage_loss = []
  for loss in losses:
    if total_loss is None:
      total_loss = loss
    else:
      total_loss = total_loss + loss
    each_stage_loss.append(loss.data[0])
  return total_loss, each_stage_loss
