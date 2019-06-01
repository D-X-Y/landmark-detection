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

def torch_inverse(deltp):
  assert deltp.dim() == 2 and deltp.size(0) == 2 and deltp.size(1) == 2, 'The deltp format is not right : {}'.format( deltp.size() )
  a, b, c, d = deltp[0,0], deltp[0,1], deltp[1,0], deltp[1,1]
  a = a + np.finfo(float).eps
  d = d + np.finfo(float).eps
  divide = a*d-b*c
  inverse = torch.cat([d, -b, -c, a]).view(2,2)
  return inverse / divide

class SobelConv(nn.Module):
  def __init__(self, tag, dtype):
    super(SobelConv, self).__init__()
    if tag == 'x':
      Sobel = np.array([ [-1./8, 0, 1./8], [-2./8, 0, 2./8], [ -1./8, 0, 1./8] ])
      #Sobel = np.array([ [ 0, 0, 0], [-0.5,0,0.5], [ 0, 0, 0] ])
    elif tag == 'y':
      Sobel = np.array([ [ -1./8, -2./8, -1./8], [ 0, 0, 0], [ 1./8, 2./8, 1./8] ])
      #Sobel = np.array([ [ 0,-0.5, 0], [ 0, 0, 0], [ 0, 0.5, 0] ])
    else:
      raise NameError('Do not know this tag for Sobel Kernel : {}'.format(tag))
    Sobel = torch.from_numpy(Sobel).type(dtype)
    Sobel = Sobel.view(1, 1, 3, 3) 
    self.register_buffer('weight', Sobel)
    self.tag = tag

  def forward(self, input):
    weight = self.weight.expand(input.size(1), 1, 3, 3).contiguous()
    return F.conv2d(input, weight, groups=input.size(1), padding=1)

  def __repr__(self):
    return ('{name}(tag={tag})'.format(name=self.__class__.__name__, **self.__dict__))

def ComputeGradient(feature, tag):
  if feature.dim() == 3:
    feature = feature.unsqueeze(0)
    squeeze = True
  else:
    squeeze = False
  assert feature.dim() == 4, 'feature must be [batch x C x H x W] not {}'.format(feature.size())
  sobel = SobelConv(tag)
  if feature.is_cuda: sobel.cuda()
  if squeeze: return sobel(feature).squeeze(0)
  else:       return sobel(feature)

def Generate_Weight(patch_size, sigma=None):
  assert isinstance(patch_size, list) or isinstance(patch_size, tuple)
  assert patch_size[0] > 0 and patch_size[1] > 0, 'the patch size must > 0 rather :{}'.format(patch_size)
  center = [(patch_size[0]-1.)/2, (patch_size[1]-1.)/2]
  maps = np.fromfunction( lambda x, y: (x-center[0])**2 + (y-center[1])**2, (patch_size[0], patch_size[1]), dtype=int)
  if sigma is None: sigma = min(patch_size[0], patch_size[1])/2.
  maps = np.exp(maps / -2.0 / sigma / sigma)
  maps[0, :] = maps[-1, :] = maps[:, 0] = maps[:, -1] = 0
  return maps.astype(np.float32)

def warp_feature(feature, pts_location, patch_size):
  # pts_location is [X,Y], patch_size is [H,W]
  C, H, W = feature.size(0), feature.size(1), feature.size(2)
  def normalize(x, L):
    return -1. + 2. * x / (L-1)

  crop_box = [pts_location[0]-patch_size[1], pts_location[1]-patch_size[0], pts_location[0]+patch_size[1], pts_location[1]+patch_size[0]]
  crop_box[0] = normalize(crop_box[0], W)
  crop_box[1] = normalize(crop_box[1], H)
  crop_box[2] = normalize(crop_box[2], W)
  crop_box[3] = normalize(crop_box[3], H)
  affine_parameter = [(crop_box[2]-crop_box[0])/2, MU.np2variable(torch.zeros(1),feature.is_cuda,False), (crop_box[0]+crop_box[2])/2,
                      MU.np2variable(torch.zeros(1),feature.is_cuda,False), (crop_box[3]-crop_box[1])/2, (crop_box[1]+crop_box[3])/2]

  affine_parameter = torch.cat(affine_parameter).view(2, 3)
 
  theta = affine_parameter.unsqueeze(0)
  feature = feature.unsqueeze(0)
  grid_size = torch.Size([1, 1, 2*patch_size[0]+1, 2*patch_size[1]+1])
  grid = F.affine_grid(theta, grid_size)
  sub_feature = F.grid_sample(feature, grid).squeeze(0)
  return sub_feature
