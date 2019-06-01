##############################################################
### Copyright (c) 2018-present, Xuanyi Dong                ###
### Style Aggregated Network for Facial Landmark Detection ###
### Computer Vision and Pattern Recognition, 2018          ###
##############################################################
import numpy as np
import numbers, torch
import torch.nn.functional as F

def compute_stage_loss(criterion, target_var, outputs, mask_var, total_labeled_cpm, weight_of_idt):

  total_loss = 0
  each_stage_loss = []
  mask_outputs = []
  for output_var in outputs:
    stage_loss = 0
    output = torch.masked_select(output_var, mask_var)
    target = torch.masked_select(target_var, mask_var)
    mask_outputs.append(output)

    stage_loss = criterion(output, target) / (total_labeled_cpm*2)
    total_loss = total_loss + stage_loss
    each_stage_loss.append(stage_loss.item())
  if weight_of_idt is not None and weight_of_idt > 0:
    pair_loss_a = torch.sum( torch.abs(mask_outputs[0] - mask_outputs[1]) )
    pair_loss_b = torch.sum( torch.abs(mask_outputs[0] - mask_outputs[2]) )
    pair_loss_c = torch.sum( torch.abs(mask_outputs[1] - mask_outputs[2]) )
    identity_loss = weight_of_idt * (pair_loss_a + pair_loss_b + pair_loss_c) / 3
    each_stage_loss.append(identity_loss.item())
    total_loss = total_loss + identity_loss
  return total_loss, each_stage_loss

def show_stage_loss(each_stage_loss):
  if each_stage_loss is None:            return 'None'
  elif isinstance(each_stage_loss, str): return each_stage_loss
  answer = ''
  for index, loss in enumerate(each_stage_loss):
    answer = answer + ' : L{:1d}={:6.3f}'.format(index+1, loss)
  return answer

def sum_stage_loss(losses):
  total_loss = None
  each_stage_loss = []
  for loss in losses:
    if total_loss is None:
      total_loss = loss
    else:
      total_loss = total_loss + loss
    each_stage_loss.append(loss.item())
  return total_loss, each_stage_loss
