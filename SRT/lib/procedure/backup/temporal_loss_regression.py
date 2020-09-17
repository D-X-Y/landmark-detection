# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import models
import torch
import torch.nn.functional as F
import numpy as np
import math, numbers


def get_in_map(locs):
  assert locs.dim() == 4, 'locs : {:}'.format(locs.shape)
  return torch.sum((locs > -1) + (locs < 1), dim=-1) == 4


def calculate_temporal_loss(criterion, locs, past2now, future2now, FBcheck, mask, config):
  # return the calculate target from the first frame to the whole sequence.
  batch, frames, num_pts, _ = locs.size()
  assert batch == past2now.size(0) == future2now.size(0) == FBcheck.size(0), '{:} vs {:} vs {:} vs {:}'.format(locs.size(), past2now.size(), future2now.size(), FBcheck.size())
  assert num_pts == past2now.size(2) == future2now.size(2) == FBcheck.size(1), '{:} vs {:} vs {:} vs {:}'.format(locs.size(), past2now.size(), future2now.size(), FBcheck.size())
  assert frames-1 == past2now.size(1) == future2now.size(1), '{:} vs {:} vs {:} vs {:}'.format(locs.size(), past2now.size(), future2now.size(), FBcheck.size())
  assert mask.dim() == 4 and mask.size(0) == batch and mask.size(1) == num_pts, 'mask : {:}'.format(mask.size())

  locs, past2now, future2now = locs.contiguous(), past2now.contiguous(), future2now.contiguous()
  FBcheck, mask = FBcheck.contiguous(), mask.view(batch, num_pts).contiguous()
  if config.sbr_loss_type == 'L1':
    past_loss, future_loss = criterion.loss_l1_func(locs[:,1:], past2now, reduction='none'), criterion.func(locs[:,:-1], future2now, reduction='none')
  elif config.sbr_loss_type == 'MSE':
    past_loss, future_loss = criterion.loss_mse_func(locs[:,1:], past2now, reduction='none'), criterion.func(locs[:,:-1], future2now, reduction='none')
  else:
    raise ValueError('invalid sbr loss type : {:}'.format(config.sbr_loss_type))
  
  temporal_loss = past_loss + future_loss

  # check oks
  with torch.no_grad():
    inmap_ok = get_in_map( locs ).sum(1) == frames
    check_ok = torch.sqrt(FBcheck[:,:,0]**2 + FBcheck[:,:,1]**2) < config.fb_thresh
    distc_ok = (past_loss.sum(-1) + future_loss.sum(-1))/4 < config.dis_thresh
    distc_ok = distc_ok.sum(1) == frames-1
    data_ok  = (inmap_ok.view(batch, 1, num_pts, 1) + check_ok.view(batch, 1, num_pts, 1) + distc_ok.view(batch, 1, num_pts, 1) + mask.view(batch, 1, num_pts, 1)) == 4
  
  nums     = torch.sum(data_ok).item()
  if nums == 0: return 0, nums
  else:
    selected_loss = torch.masked_select(temporal_loss, data_ok)
    return torch.mean(selected_loss), nums
