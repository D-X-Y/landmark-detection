# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#
import torch
import torch.nn.functional as F
from multiview import TriangulateDLT_BatchPoints
from multiview import ProjectKRT_Batch
from multiview import ProjectKRT


def get_in_map(locs):
  assert locs.dim() == 4, 'locs : {:}'.format(locs.shape)
  return torch.sum((locs > -1) & (locs < 1), dim=-1) == 2


def calculate_multiview_loss(criterion, mv_locs, proj_locs, masks, config):
  assert mv_locs.dim() == 4 and mv_locs.size(-1) == 2, 'invalid mv-locs size : {:}'.format(mv_locs.shape)
  assert mv_locs.size() == proj_locs.size(), '{:} vs {:}'.format(mv_locs.shape, proj_locs.shape)
  batch, cameras, num_pts, _ = mv_locs.size()

  if config.sbt_loss_type == 'L1':
    stm_losses = criterion.loss_l1_func(mv_locs, proj_locs, reduction='none')
  elif config.sbt_loss_type == 'MSE':
    stm_losses = criterion.loss_mse_func(mv_locs, proj_locs, reduction='none')
  else:
    raise ValueError('invalid SBT loss type : {:}'.format( config.sbt_loss_type ))

  with torch.no_grad():
    inmap_ok1 = get_in_map(mv_locs).sum(1)   == cameras
    inmap_ok2 = get_in_map(proj_locs).sum(1) == cameras
    inmap_ok  = (inmap_ok1.int() + inmap_ok2.int()) == 2
    distc_ok  = stm_losses.sum(-1)/2 < config.stm_dis_thresh
    #distc_ok  = distc_ok.sum(1) == cameras
    data_ok   = (inmap_ok.view(batch, 1, num_pts, 1).int() + distc_ok.view(batch, -1, num_pts, 1).int() + masks.view(batch, 1, num_pts, 1).int()) == 3
    
  nums     = torch.sum(data_ok).item() * 1.0 / cameras
  if nums == 0: return 0, nums
  else:
    selected_loss = torch.masked_select(stm_losses, data_ok)
    return torch.mean(selected_loss), nums
