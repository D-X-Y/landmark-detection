# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers, math
import numpy as np
import models.model_utils as MU
from .basic_utils import SobelConv, Generate_Weight
from .basic_utils_batch import torch_inverse_batch, warp_feature_batch

"""
peak_config = {}
def obtain_config(heatmap, radius):
  identity_str = '{}-{}'.format(radius, heatmap.get_device() if heatmap.is_cuda else -1 )
  if identity_str not in peak_config:
    if heatmap.is_cuda:
      with torch.cuda.device_of(heatmap):
        X = MU.np2variable(torch.arange(-radius, radius+1), heatmap.is_cuda, False).view(1, 1, radius*2+1)
        Y = MU.np2variable(torch.arange(-radius, radius+1), heatmap.is_cuda, False).view(1, radius*2+1, 1)
    else:
      X = MU.np2variable(torch.arange(-radius, radius+1), heatmap.is_cuda, False).view(1, 1, radius*2+1)
      Y = MU.np2variable(torch.arange(-radius, radius+1), heatmap.is_cuda, False).view(1, radius*2+1, 1)
    peak_config[ identity_str ] = [X, Y]
  return peak_config[ identity_str ]
"""


def lk_tensor_track_batch(feature_old, feature_new, pts_locations, patch_size, max_step, feature_template=None):
  # feature[old,new] : 4-D tensor [1, C, H, W]
  # pts_locations is a 2-D tensor [Num-Pts, (Y,X)]
  if feature_new.dim() == 3:
    feature_new = feature_new.unsqueeze(0)
  if feature_old is not None and feature_old.dim() == 3:
    feature_old = feature_old.unsqueeze(0)
  assert feature_new.dim() == 4, 'The dimension of feature-new is not right : {}.'.format(feature_new.dim())
  BB, C, H, W = list(feature_new.size())
  if feature_old is not None:
    assert 1 == feature_old.size(0) and 1 == BB, 'The first dimension of feature should be one not {}'.format(feature_old.size())
    assert C == feature_old.size(1) and H == feature_old.size(2) and W == feature_old.size(3), 'The size is not right : {}'.format(feature_old.size())
  assert isinstance(patch_size, int), 'The format of lk-parameters are not right : {}'.format(patch_size)
  num_pts = pts_locations.size(0)
  device = feature_new.device

  weight_map = Generate_Weight( [patch_size*2+1, patch_size*2+1] ) # [H, W]
  with torch.no_grad():
    weight_map = torch.tensor(weight_map).view(1, 1, 1, patch_size*2+1, patch_size*2+1).to(device)

    sobelconvx = SobelConv('x', feature_new.dtype).to(device)
    sobelconvy = SobelConv('y', feature_new.dtype).to(device)
  
  # feature_T should be a [num_pts, C, patch, patch] tensor
  if feature_template is None:
    feature_T = warp_feature_batch(feature_old, pts_locations, patch_size)
  else:
    assert feature_old is None, 'When feature_template is not None. feature_old must be None'
    feature_T = feature_template
  assert feature_T.size(2) == patch_size * 2 + 1 and feature_T.size(3) == patch_size * 2 + 1, 'The size of feature-template is not ok : {}'.format(feature_T.size())
  gradiant_x = sobelconvx(feature_T)
  gradiant_y = sobelconvy(feature_T)
  J = torch.stack([gradiant_x, gradiant_y], dim=1)
  weightedJ = J * weight_map
  H = torch.bmm( weightedJ.view(num_pts,2,-1), J.view(num_pts, 2, -1).transpose(2,1) )
  inverseH = torch_inverse_batch(H)

  #print ('PTS : {}'.format(pts_locations))
  for step in range(max_step):
    # Step-1 Warp I with W(x,p) to compute I(W(x;p))
    feature_I = warp_feature_batch(feature_new, pts_locations, patch_size)
    # Step-2 Compute the error feature
    r = feature_I - feature_T
    # Step-7 Compute sigma
    sigma = torch.bmm(weightedJ.view(num_pts,2,-1), r.view(num_pts,-1, 1))
    # Step-8 Compute delta-p
    deltap = torch.bmm(inverseH, sigma).squeeze(-1)
    pts_locations = pts_locations - deltap

  return pts_locations


def lk_forward_backward_batch(features, locations, window, steps):
  sequence, C, H, W = list(features.size())
  seq, num_pts, _ = list(locations.size())
  assert seq == sequence, '{:} vs {:}'.format(features.size(), locations.size())

  previous_pts = [ locations[0] ]
  for iseq in range(1, sequence):
    feature_old = features.narrow(0, iseq-1, 1)
    feature_new = features.narrow(0, iseq  , 1)
    nextPts = lk_tensor_track_batch(feature_old, feature_new, previous_pts[iseq-1], window, steps, None)
    previous_pts.append(nextPts)

  fback_pts = [None] * (sequence-1) + [ previous_pts[-1] ]
  for iseq in range(sequence-2, -1, -1):
    feature_old = features.narrow(0, iseq+1, 1)
    feature_new = features.narrow(0, iseq  , 1)
    backPts = lk_tensor_track_batch(feature_old, feature_new, fback_pts[iseq+1]   , window, steps, None)
    fback_pts[iseq] = backPts

  back_pts = [None] * (sequence-1) + [ locations[-1] ]
  for iseq in range(sequence-2, -1, -1):
    feature_old = features.narrow(0, iseq+1, 1)
    feature_new = features.narrow(0, iseq  , 1)
    backPts = lk_tensor_track_batch(feature_old, feature_new, back_pts[iseq+1]    , window, steps, None)
    back_pts[iseq] = backPts

  return torch.stack(previous_pts), torch.stack(fback_pts), torch.stack(back_pts)
