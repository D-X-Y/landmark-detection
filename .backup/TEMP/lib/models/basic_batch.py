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

def find_tensor_peak_batch(heatmap, radius, downsample, threshold = 0.000001):
  assert heatmap.dim() == 3, 'The dimension of the heatmap is wrong : {}'.format(heatmap.size())
  assert radius > 0 and isinstance(radius, numbers.Number), 'The radius is not ok : {}'.format(radius)
  num_pts, H, W = heatmap.size(0), heatmap.size(1), heatmap.size(2)
  assert W > 1 and H > 1, 'To avoid the normalization function divide zero'
  # find the approximate location:
  score, index = torch.max(heatmap.view(num_pts, -1), 1)
  index_w = (index % W).float()
  index_h = (index / W).float()
  
  def normalize(x, L):
    return -1. + 2. * x.data / (L-1)
  boxes = [index_w - radius, index_h - radius, index_w + radius, index_h + radius]
  boxes[0] = normalize(boxes[0], W)
  boxes[1] = normalize(boxes[1], H)
  boxes[2] = normalize(boxes[2], W)
  boxes[3] = normalize(boxes[3], H)
  #affine_parameter = [(boxes[2]-boxes[0])/2, boxes[0]*0, (boxes[2]+boxes[0])/2,
  #                   boxes[0]*0, (boxes[3]-boxes[1])/2, (boxes[3]+boxes[1])/2]
  #theta = torch.stack(affine_parameter, 1).view(num_pts, 2, 3)

  affine_parameter = torch.zeros((num_pts, 2, 3))
  affine_parameter[:,0,0] = (boxes[2]-boxes[0])/2
  affine_parameter[:,0,2] = (boxes[2]+boxes[0])/2
  affine_parameter[:,1,1] = (boxes[3]-boxes[1])/2
  affine_parameter[:,1,2] = (boxes[3]+boxes[1])/2
  # extract the sub-region heatmap
  theta = affine_parameter.to(heatmap.device)
  grid_size = torch.Size([num_pts, 1, radius*2+1, radius*2+1])
  grid = F.affine_grid(theta, grid_size)
  sub_feature = F.grid_sample(heatmap.unsqueeze(1), grid).squeeze(1)
  sub_feature = F.threshold(sub_feature, threshold, np.finfo(float).eps)

  X = torch.arange(-radius, radius+1).to(heatmap).view(1, 1, radius*2+1)
  Y = torch.arange(-radius, radius+1).to(heatmap).view(1, radius*2+1, 1)
  
  sum_region = torch.sum(sub_feature.view(num_pts,-1),1)
  x = torch.sum((sub_feature*X).view(num_pts,-1),1) / sum_region + index_w
  y = torch.sum((sub_feature*Y).view(num_pts,-1),1) / sum_region + index_h
     
  x = x * downsample + downsample / 2.0 - 0.5
  y = y * downsample + downsample / 2.0 - 0.5
  return torch.stack([x, y],1), score
