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

#### The utils for LK
def torch_inverse_batch(deltp):
  # deltp must be [K,2]
  assert deltp.dim() == 3 and deltp.size(1) == 2 and deltp.size(2) == 2, 'The deltp format is not right : {}'.format( deltp.size() )
  a, b, c, d = deltp[:,0,0], deltp[:,0,1], deltp[:,1,0], deltp[:,1,1]
  a = a + np.finfo(float).eps
  d = d + np.finfo(float).eps
  divide = a*d-b*c+np.finfo(float).eps
  inverse = torch.stack([d, -b, -c, a], dim=1) / divide.unsqueeze(1)
  return inverse.view(-1,2,2)


def warp_feature_batch(feature, pts_location, patch_size):
  # feature must be [1,C,H,W] and pts_location must be [Num-Pts, (x,y)]
  _, C, H, W = list(feature.size())
  num_pts = pts_location.size(0)
  assert isinstance(patch_size, int) and feature.size(0) == 1 and pts_location.size(1) == 2, 'The shapes of feature or points are not right : {} vs {}'.format(feature.size(), pts_location.size())
  assert W > 1 and H > 1, 'To guarantee normalization {}, {}'.format(W, H)

  def normalize(x, L):
    return -1. + 2. * x / (L-1)

  crop_box = torch.cat([pts_location-patch_size, pts_location+patch_size], 1)
  crop_box[:, [0,2]] = normalize(crop_box[:, [0,2]], W)
  crop_box[:, [1,3]] = normalize(crop_box[:, [1,3]], H)
 
  affine_parameter = [(crop_box[:,2]-crop_box[:,0])/2, crop_box[:,0]*0, (crop_box[:,2]+crop_box[:,0])/2,
                      crop_box[:,0]*0, (crop_box[:,3]-crop_box[:,1])/2, (crop_box[:,3]+crop_box[:,1])/2]
  #affine_parameter = [(crop_box[:,2]-crop_box[:,0])/2, MU.np2variable(torch.zeros(num_pts),feature.is_cuda,False), (crop_box[:,2]+crop_box[:,0])/2,
  #                    MU.np2variable(torch.zeros(num_pts),feature.is_cuda,False), (crop_box[:,3]-crop_box[:,1])/2, (crop_box[:,3]+crop_box[:,1])/2]
  theta = torch.stack(affine_parameter, 1).view(num_pts, 2, 3)
  feature = feature.expand(num_pts,C, H, W)
  grid_size = torch.Size([num_pts, 1, 2*patch_size+1, 2*patch_size+1])
  grid = F.affine_grid(theta, grid_size)
  sub_feature = F.grid_sample(feature, grid)
  return sub_feature
