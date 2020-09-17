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

from xvision import identity2affine


def get_in_map(locs):
  assert locs.dim() == 4, 'locs : {:}'.format(locs.shape)
  return torch.sum((locs > -1) & (locs < 1), dim=-1) == 2


def calculate_multiview_loss(criterion, mv_heatmaps, mv_locs, proj_locs, masks, config):
  assert mv_locs.dim() == 4 and mv_locs.size(-1) == 2, 'invalid mv-locs size : {:}'.format(mv_locs.shape)
  assert mv_locs.size() == proj_locs.size(), '{:} vs {:}'.format(mv_locs.shape, proj_locs.shape)
  batch, cameras, num_pts, _ = mv_locs.size()

  with torch.no_grad():
    inmap_ok1 = get_in_map(mv_locs).sum(1)   == cameras
    inmap_ok2 = get_in_map(proj_locs).sum(1) == cameras
    inmap_ok  = (inmap_ok1.int() + inmap_ok2.int()) == 2
    stm_dis   = criterion.loss_l1_func(mv_locs, proj_locs, reduction='none')
    distc_ok  = stm_dis.sum(-1)/2 < config.stm_dis_thresh
    data_ok   = (inmap_ok.view(batch, 1, num_pts, 1).int() + distc_ok.view(batch, -1, num_pts, 1).int() + masks.view(batch, 1, num_pts, 1).int()) == 3

  if config.sbt_loss_type == 'L1':
    stm_losses = criterion.loss_l1_func(mv_locs, proj_locs, reduction='none')
    final_loss = torch.masked_select(stm_losses, data_ok)
    final_loss = torch.mean( final_loss )
  elif config.sbt_loss_type == 'MSE':
    stm_losses = criterion.loss_mse_func(mv_locs, proj_locs, reduction='none')
    final_loss = torch.masked_select(stm_losses, data_ok)
    final_loss = torch.mean( final_loss )
  elif config.sbt_loss_type == 'HEAT':
    H, W               = mv_heatmaps[0].size(-2), mv_heatmaps[0].size(-1)
    identity_grid      = F.affine_grid(identity2affine().cuda().view(1, 2, 3), torch.Size([1, 1, H, W]), align_corners=True)
    multiview_grid     = identity_grid + (proj_locs-mv_locs).view(batch*cameras*num_pts, 1, 1, 2)
    multiview_heatmaps = [x.contiguous().view(batch*cameras*num_pts, 1, H, W) for x in mv_heatmaps]
    multiview_predicts = [F.grid_sample(x, multiview_grid, mode='bilinear', padding_mode='border', align_corners=True).view(batch, cameras, num_pts, H, W) for x in multiview_heatmaps]
  
    data_ok, loss_list = data_ok.view(batch, cameras, num_pts, 1, 1), []
    for index in range( len(multiview_predicts) ):
      mv_loss = criterion(multiview_predicts[index], mv_heatmaps[index], data_ok)
      loss_list.append( mv_loss )
    final_loss = sum(loss_list)
  else:
    raise ValueError('invalid SBT loss type : {:}'.format( config.sbt_loss_type ))
    
  nums     = torch.sum(data_ok).item() * 1.0 / cameras
  if nums == 0: return 0, nums
  else        : return final_loss, nums
