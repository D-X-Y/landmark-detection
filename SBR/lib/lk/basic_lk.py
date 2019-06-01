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
from .basic_utils import SobelConv
from .basic_utils import ComputeGradient, Generate_Weight, warp_feature
from .basic_utils import torch_inverse

def lk_tensor_track(feature_old, feature_new, pts_locations, patch_size, max_step, threshold=0.0001):
  # feature[old,new] : 3-D tensor [C, H, W]
  # pts_locations is a 2-D point [ Y, X ]
  assert feature_old.dim() == 3 and feature_new.dim() == 3, 'The dimension is not right : {} and {}'.format(feature_old.dim(), feature_new.dim())
  C, H, W = feature_old.size(0), feature_old.size(1), feature_old.size(2)
  assert C == feature_new.size(0) and H == feature_new.size(1) and W == feature_new.size(2), 'The size is not right : {}'.format(feature_new.size())
  assert pts_locations.dim() == 1 and pts_locations.size(0) == 2, 'The location is not right : {}'.format(pts_locations)
  if isinstance(patch_size, int): patch_size = (patch_size, patch_size)
  assert isinstance(patch_size, tuple) and len(patch_size) == 2 and isinstance(max_step, int), 'The format of lk-parameters are not right : {}, {}'.format(patch_size, max_step)
  assert isinstance(patch_size[0], int) and isinstance(patch_size[1], int), 'The format of lk-parameters are not right : {}'.format(patch_size)

  def abserror(deltap):
    deltap = MU.variable2np(deltap)
    return float(np.sqrt(np.sum(deltap*deltap)))
  
  weight_map = Generate_Weight( [patch_size[0]*2+1, patch_size[1]*2+1] ) # [H, W]
  with torch.cuda.device_of(feature_old):
    weight_map = MU.np2variable(weight_map, feature_old.is_cuda, False).unsqueeze(0)

    feature_T = warp_feature(feature_old, pts_locations, patch_size)
    gradiant_x = ComputeGradient(feature_T, 'x')
    gradiant_y = ComputeGradient(feature_T, 'y')
    J = torch.stack([gradiant_x, gradiant_y])
    weightedJ = J*weight_map
    H = torch.mm( weightedJ.view(2,-1), J.view(2, -1).transpose(1,0) )
    inverseH = torch_inverse(H)

    for step in range(max_step):
      # Step-1 Warp I with W(x,p) to compute I(W(x;p))
      feature_I = warp_feature(feature_new, pts_locations, patch_size)
      # Step-2 Compute the error feature
      r = feature_I - feature_T
      # Step-7 Compute sigma
      sigma = torch.mm(weightedJ.view(2,-1), r.view(-1, 1))
      # Step-8 Compute delta-p
      deltap = torch.mm(inverseH, sigma).squeeze(1)
      pts_locations = pts_locations - deltap
      if abserror(deltap) < threshold: break

  return pts_locations
