# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#
# Stacked Hourglass Networks for Human Pose Estimation (https://arxiv.org/abs/1603.06937)
from __future__ import division
import time, math, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from .basic_batch import find_tensor_peak_batch

class Residual(nn.Module):
  def __init__(self, numIn, numOut):
    super(Residual, self).__init__()
    self.numIn = numIn
    self.numOut = numOut
    middle = self.numOut // 2

    self.conv_A = nn.Sequential(
                    nn.BatchNorm2d(numIn), nn.ReLU(inplace=True),
                    nn.Conv2d(numIn, middle, kernel_size=1, dilation=1, padding=0, bias=True))
    self.conv_B = nn.Sequential(
                    nn.BatchNorm2d(middle), nn.ReLU(inplace=True),
                    nn.Conv2d(middle, middle, kernel_size=3, dilation=1, padding=1, bias=True))
    self.conv_C = nn.Sequential(
                    nn.BatchNorm2d(middle), nn.ReLU(inplace=True),
                    nn.Conv2d(middle, numOut, kernel_size=1, dilation=1, padding=0, bias=True))

    if self.numIn != self.numOut:
      self.branch = nn.Sequential(
                      nn.BatchNorm2d(self.numIn), nn.ReLU(inplace=True),
                      nn.Conv2d(self.numIn, self.numOut, kernel_size=1, dilation=1, padding=0, bias=True))

  def forward(self, x):
    residual = x
    main = self.conv_A(x)
    main = self.conv_B(main)
    main = self.conv_C(main)
    if hasattr(self, 'branch'):
      residual = self.branch( residual )
    return main + residual


class HierarchicalPMS(nn.Module):
  def __init__(self, numIn, numOut):
    super(HierarchicalPMS, self).__init__()
    self.numIn = numIn
    self.numOut = numOut
    cA, cB, cC = self.numOut//2, self.numOut//4, self.numOut-self.numOut//2-self.numOut//4
    assert cA + cB + cC == numOut, '({:}, {:}, {:}) = {:}'.format(cA, cB, cC, numOut)

    self.conv_A = nn.Sequential(
                    nn.BatchNorm2d(numIn), nn.ReLU(inplace=True),
                    nn.Conv2d(numIn, cA, kernel_size=3, dilation=1, padding=1, bias=True))
    self.conv_B = nn.Sequential(
                    nn.BatchNorm2d(cA), nn.ReLU(inplace=True),
                    nn.Conv2d(cA, cB, kernel_size=3, dilation=1, padding=1, bias=True))
    self.conv_C = nn.Sequential(
                    nn.BatchNorm2d(cB), nn.ReLU(inplace=True),
                    nn.Conv2d(cB, cC, kernel_size=3, dilation=1, padding=1, bias=True))

    if self.numIn != self.numOut:
      self.branch = nn.Sequential(
                      nn.BatchNorm2d(self.numIn), nn.ReLU(inplace=True),
                      nn.Conv2d(self.numIn, self.numOut, kernel_size=1, dilation=1, padding=0, bias=True))

  def forward(self, x):
    residual = x
    A = self.conv_A(x)
    B = self.conv_B(A)
    C = self.conv_C(B)
    main = torch.cat((A, B, C), dim=1)
    if hasattr(self, 'branch'):
      residual = self.branch( residual )
    return main + residual



class Hourglass(nn.Module):
  def __init__(self, n, nModules, nFeats, module):
    super(Hourglass, self).__init__()
    self.n = n
    self.nModules = nModules
    self.nFeats = nFeats
    
    self.res = nn.Sequential(*[module(nFeats, nFeats) for _ in range(nModules)])

    down = [nn.MaxPool2d(kernel_size = 2, stride = 2)]
    down += [module(nFeats, nFeats) for _ in range(nModules)]
    self.down = nn.Sequential(*down)

    if self.n > 1:
      self.mid = Hourglass(n - 1, self.nModules, self.nFeats, module)
    else:
      self.mid = nn.Sequential(*[module(nFeats, nFeats) for _ in range(nModules)])
    
    up = [module(nFeats, nFeats) for _ in range(nModules)]
    #up += [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)]
    self.up = nn.Sequential(*up)
    
  def forward(self, x):
    res  = self.res(x)
    down = self.down(res)
    mid  = self.mid(down)
    up   = self.up(mid)
    up   = torch.nn.functional.interpolate(up, [res.size(2), res.size(3)], mode='bilinear', align_corners=True)
    return res + up



class HourGlassNet(nn.Module):
  def __init__(self, config, points, sigma, input_dim):
    super(HourGlassNet, self).__init__()
    self.downsample = 4
    self.sigma      = sigma

    self.config     = copy.deepcopy( config )
    if self.config.module == 'Residual':
      module = Residual
    elif self.config.module == 'HierarchicalPMS':
      module = HierarchicalPMS
    else:
      raise ValueError('Invaliad module for HG : {:}'.format(self.config.module))

    self.pts_num  = points
    self.nStack   = self.config.stages
    self.nModules = self.config.nModules
    self.nFeats   = self.config.nFeats
    self.recursive = self.config.recursive

    #self.conv = nn.Sequential(
    #              nn.Conv2d(input_dim, 64, kernel_size = 7, stride = 2, padding = 3, bias = True), 
    #              nn.BatchNorm2d(64), nn.ReLU(inplace = True))
    self.conv = nn.Sequential(
                   nn.Conv2d(input_dim, 32, kernel_size = 3, stride = 2, padding = 1, bias = True), 
                   nn.BatchNorm2d(32), nn.ReLU(inplace = True),
                   nn.Conv2d(       32, 32, kernel_size = 3, stride = 1, padding = 1, bias = True), 
                   nn.BatchNorm2d(32), nn.ReLU(inplace = True),
                   nn.Conv2d(       32, 64, kernel_size = 3, stride = 1, padding = 1, bias = True), 
                   nn.BatchNorm2d(64), nn.ReLU(inplace = True))
    self.ress = nn.Sequential(
                  module(64, 128),
                  nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1),
                  module(128, 128), module(128, self.nFeats))
    
    _features, _tmpOut, _ll_, _tmpOut_ = [], [], [], []

    for i in range(self.nStack):
      feature = Hourglass(self.recursive, self.nModules, self.nFeats, module)
      feature = [feature] + [module(self.nFeats, self.nFeats) for _ in range(self.nModules)]
      feature += [nn.Conv2d(self.nFeats, self.nFeats, kernel_size = 1, stride = 1, bias = True),
                  nn.BatchNorm2d(self.nFeats), nn.ReLU(inplace = True)]
      feature = nn.Sequential(*feature)
      _features.append(feature)
      _tmpOut.append(nn.Conv2d(self.nFeats, self.pts_num, kernel_size = 1, stride = 1, bias = True))
      if i < self.nStack - 1:
        _ll_.append(nn.Conv2d(self.nFeats, self.nFeats, kernel_size = 1, stride = 1, bias = True))
        _tmpOut_.append(nn.Conv2d(self.pts_num, self.nFeats, kernel_size = 1, stride = 1, bias = True))
        
    self.features = nn.ModuleList(_features)
    self.tmpOuts = nn.ModuleList(_tmpOut)
    self.trsfeas = nn.ModuleList(_ll_)
    self.trstmps = nn.ModuleList(_tmpOut_)
    if self.config.sigmoid:
      self.sigmoid = nn.Sigmoid()
    else:
      self.sigmoid = None


  def extra_repr(self):
    return ('{name}(sigma={sigma}, downsample={downsample})'.format(name=self.__class__.__name__, **self.__dict__))

  
  def forward(self, inputs):
    assert inputs.dim() == 4, 'This model accepts 4 dimension input tensor: {}'.format(inputs.size())
    batch_size, feature_dim = inputs.size(0), inputs.size(1)
    x = self.conv(inputs)
    x = self.ress(x)
    
    features, heatmaps, batch_locs, batch_scos = [], [], [], []
    
    for i in range(self.nStack):
      feature = self.features[i](x)

      features.append(feature)

      tmpOut = self.tmpOuts[i](feature)
      if self.sigmoid is not None:
        tmpOut = self.sigmoid(tmpOut)
      heatmaps.append(tmpOut)
      if i < self.nStack - 1:
        ll_ = self.trsfeas[i](feature)
        tmpOut_ = self.trstmps[i](tmpOut)
        x = x + ll_ + tmpOut_

    # The location of the current batch
    for ibatch in range(batch_size):
      batch_location, batch_score = find_tensor_peak_batch(heatmaps[-1][ibatch], self.sigma, self.downsample)
      batch_locs.append( batch_location )
      batch_scos.append( batch_score )
    batch_locs, batch_scos = torch.stack(batch_locs), torch.stack(batch_scos)
    
    return features, heatmaps, batch_locs, batch_scos


def ProHourGlass(config, points, sigma, use_gray):
  print ('Initialize hourglass with configure : {}'.format(config))
  idim = 1 if use_gray else 3
  model = HourGlassNet(config, points, sigma, idim)
  return model
