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

from xvision import identity2affine


def get_in_map(locs):
  assert locs.dim() == 4, 'locs : {:}'.format(locs.shape)
  return torch.sum((locs > -1) + (locs < 1), dim=-1) == 4


def calculate_temporal_loss(criterion, heatmaps, locs, past2now, future2now, FBcheck, mask, config):
  # return the calculate target from the first frame to the whole sequence.
  batch, frames, num_pts, _ = locs.size()
  assert batch == past2now.size(0) == future2now.size(0) == FBcheck.size(0), '{:} vs {:} vs {:} vs {:}'.format(locs.size(), past2now.size(), future2now.size(), FBcheck.size())
  assert num_pts == past2now.size(2) == future2now.size(2) == FBcheck.size(1), '{:} vs {:} vs {:} vs {:}'.format(locs.size(), past2now.size(), future2now.size(), FBcheck.size())
  assert frames-1 == past2now.size(1) == future2now.size(1), '{:} vs {:} vs {:} vs {:}'.format(locs.size(), past2now.size(), future2now.size(), FBcheck.size())
  assert mask.dim() == 4 and mask.size(0) == batch and mask.size(1) == num_pts, 'mask : {:}'.format(mask.size())

  locs, past2now, future2now = locs.contiguous(), past2now.contiguous(), future2now.contiguous()
  FBcheck, mask = FBcheck.contiguous(), mask.view(batch, num_pts).contiguous()
  with torch.no_grad():
    past2now_l1_dis = criterion.loss_l1_func(locs[:,1:], past2now, reduction='none')
    futu2now_l1_dis = criterion.loss_l1_func(locs[:,:-1], future2now, reduction='none')

    inmap_ok = get_in_map( locs ).sum(1) == frames
    check_ok = torch.sqrt(FBcheck[:,:,0]**2 + FBcheck[:,:,1]**2) < config.fb_thresh
    distc_ok = (past2now_l1_dis.sum(-1) + futu2now_l1_dis.sum(-1))/4 < config.dis_thresh
    distc_ok = distc_ok.sum(1) == frames-1
    data_ok  = (inmap_ok.view(batch, 1, num_pts, 1) + check_ok.view(batch, 1, num_pts, 1) + distc_ok.view(batch, 1, num_pts, 1) + mask.view(batch, 1, num_pts, 1)) == 4

  if config.sbr_loss_type == 'L1':
    past_loss     = criterion.loss_l1_func(locs[:,1:], past2now, reduction='none')
    future_loss   = criterion.loss_l1_func(locs[:,:-1], future2now, reduction='none')
    temporal_loss = past_loss + future_loss
    final_loss    = torch.masked_select(temporal_loss, data_ok)
    final_loss    = torch.mean( final_loss )
    loss_string   = ''
  elif config.sbr_loss_type == 'MSE':
    past_loss     = criterion.loss_mse_func(locs[:,1:], past2now, reduction='none')
    future_loss   = criterion.loss_mse_func(locs[:,:-1], future2now, reduction='none')
    temporal_loss = past_loss + future_loss
    final_loss    = torch.masked_select(temporal_loss, data_ok)
    final_loss    = torch.mean( final_loss )
    loss_string   = ''
  elif config.sbr_loss_type == 'HEAT':
    H, W               = heatmaps[0].size(-2), heatmaps[0].size(-1)
    identity_grid      = F.affine_grid(identity2affine().cuda().view(1, 2, 3), torch.Size([1, 1, H, W]))
    identity_grid      = identity_grid.view(1, H, W, 2)
    past2now_grid      = identity_grid - (past2now-locs[:,:-1]).view(batch*(frames-1)*num_pts, 1, 1, 2)
    past2now_heatmaps  = [x[:,:-1].contiguous().view(batch*(frames-1)*num_pts, 1, H, W) for x in heatmaps]
    past2now_targets   = [x[:,1: ].contiguous() for x in heatmaps]
    past2now_predicts  = [F.grid_sample(x, past2now_grid, align_corners=True).view(batch, frames-1, num_pts, H, W) for x in past2now_heatmaps]

    futu2now_grid      = identity_grid + (future2now-locs[:,1:]).view(batch*(frames-1)*num_pts, 1, 1, 2)
    futu2now_heatmaps  = [x[:,1: ].contiguous().view(batch*(frames-1)*num_pts, 1, H, W) for x in heatmaps]
    futu2now_targets   = [x[:,:-1].contiguous() for x in heatmaps]
    futu2now_predicts  = [F.grid_sample(x, futu2now_grid).view(batch, frames-1, num_pts, H, W) for x in futu2now_heatmaps]

    data_ok            = data_ok.view(batch, 1, num_pts, 1, 1)
    loss_list, loss_string = [], ''
    for index in range( len(past2now_targets) ):
      past_loss  = criterion(past2now_predicts[index], past2now_targets[index], data_ok)
      futu_loss  = criterion(futu2now_predicts[index], futu2now_targets[index], data_ok)
      if index != 0: loss_string += ' '
      loss_string += 'S{:}[P={:.6f}, F={:.6f}]'.format(index, past_loss.item(), futu_loss.item())
      loss_list.append( past_loss )
      loss_list.append( futu_loss )
      #final_loss += past_loss + futu_loss
    final_loss = sum(loss_list)
  else:
    raise ValueError('invalid SBR loss type : {:}'.format( config.sbr_loss_type ))

  nums     = torch.sum(data_ok).item()
  if nums == 0: return 0         , nums, loss_string
  else        : return final_loss, nums, loss_string
