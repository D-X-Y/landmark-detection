# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#
# Rethinking on Multi-Stage Networks for Human Pose Estimation
from __future__ import division
import time, math, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import BN_ReLU_C1x1
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



class Hourglass(nn.Module):
  def __init__(self, nDepth, nModules, nFeats):
    super(Hourglass, self).__init__()
    self.nDepth   = nDepth
    self.nModules = nModules
    self.nFeats   = nFeats
    
    self.down_layers   = nn.ModuleList()
    self.skip_connects = nn.ModuleList()
    self.up_layers     = nn.ModuleList()
    for i in range(nDepth):
      iC, oC = nFeats*(2**i), min(nFeats*(2**(i+1)), nFeats*(2**(nDepth-1)))
      self.down_layers.append  ( self.get_seq(iC, oC, nModules) )
      self.skip_connects.append( BN_ReLU_C1x1(oC, nFeats) )
      self.up_layers.append    ( self.get_seq(nFeats, nFeats, nModules) )
    self.middle = self.get_seq(nFeats*(2**(nDepth-1)), nFeats, nModules)
    self.norm   = nn.Sequential(nn.BatchNorm2d(nFeats), nn.ReLU(inplace=True))
    
  
  def get_seq(self, ichannel, ochannel, n):
    layers = []
    for i in range(n):
      if i+1<n: layer = Residual(ichannel, ichannel)
      else    : layer = Residual(ichannel, ochannel)
      layers.append( layer )
    return nn.Sequential(*layers)
    
  def forward(self, x, cache_features):
    down_outputs, up_outputs = [None] * self.nDepth, [None] * self.nDepth
    assert cache_features is None or len(cache_features) == self.nDepth, 'invalid length : {:}'.format(len(cache_features), self.nDepth)
    cache_shapes = []
    for i in range(self.nDepth):
      cache_shapes.append( (x.size(2), x.size(3)) )
      temp = self.down_layers[i]( x )
      if cache_features is not None: temp = temp + cache_features[i]
      down_outputs[i] = temp
      x = F.interpolate(down_outputs[i], [x.size(2)//2, x.size(3)//2], mode='bilinear', align_corners=True)

    x = self.middle( x )

    for i in range(self.nDepth-1, -1, -1):
      x = F.interpolate(x, cache_shapes[i], mode='bilinear', align_corners=True)
      temp = self.skip_connects[i]( down_outputs[i] ) + x
      x = self.up_layers[i]( temp )
      up_outputs[i] = x
    out = self.norm(x)
    return out, down_outputs, up_outputs



class MSPNet(nn.Module):
  def __init__(self, config, points, sigma, input_dim):
    super(MSPNet, self).__init__()
    self.downsample = 4
    self.sigma      = sigma

    self.config     = copy.deepcopy( config )

    self.pts_num  = points
    self.nStack   = self.config.stages
    self.nModules = self.config.nModules
    self.nFeats   = self.config.nFeats
    self.recursive = self.config.recursive

    self.basic = nn.Sequential(
                   nn.Conv2d(input_dim, 64, kernel_size = 7, stride = 2, padding = 3),
                   Residual(64, 128),
                   BN_ReLU_C1x1(128, 128), nn.MaxPool2d(kernel_size=2, stride=2)
        )
    self.res_block = nn.Sequential(
            nn.Conv2d(128, 64, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 1, bias=False)
        )
    self.conv = Residual(128, self.nFeats)
   
    self.HGs  = nn.ModuleList()
    _features, _tmpOut, _ll_, _tmpOut_ = [], [], [], []

    for i in range(self.nStack):
      self.HGs.append( Hourglass(self.recursive, self.nModules, self.nFeats) )
      feature = [Residual(self.nFeats, self.nFeats) for _ in range(self.nModules)]
      feature += [nn.Conv2d(self.nFeats, self.nFeats, kernel_size = 1, stride = 1, bias = True),
                  nn.BatchNorm2d(self.nFeats), nn.ReLU(inplace = True)]
      feature = nn.Sequential(*feature)
      _features.append(feature)
      _tmpOut.append(nn.Conv2d(self.nFeats, self.pts_num, kernel_size = 1, stride = 1, bias = True))
      if i < self.nStack - 1:
        _ll_.append(nn.Conv2d(self.nFeats, self.nFeats, kernel_size = 1, stride = 1, bias = True))
        _tmpOut_.append(nn.Conv2d(self.pts_num, self.nFeats, kernel_size = 1, stride = 1, bias = True))
        
    self.features = nn.ModuleList(_features)
    self.tmpOuts  = nn.ModuleList(_tmpOut)
    self.trsfeas  = nn.ModuleList(_ll_)
    self.trstmps  = nn.ModuleList(_tmpOut_)

    self.cross_stages = nn.ModuleList()
    for i in range(self.nStack-1):
      cross_layer = nn.ModuleList()
      for j in range(self.recursive):
        iC, oC = self.nFeats*(2**j), min(self.nFeats*(2**(j+1)), self.nFeats*(2**(self.recursive-1)))
        cross_layer.append(nn.ModuleList([BN_ReLU_C1x1(oC, oC), BN_ReLU_C1x1(self.nFeats, oC)])) 
        #print ('{:}-th stage : {:}-th layer : {:}, {:}'.format(i, j, iC, oC))
      self.cross_stages.append(cross_layer)


  def extra_repr(self):
    return ('{name}(sigma={sigma}, downsample={downsample})'.format(name=self.__class__.__name__, **self.__dict__))

  
  def forward(self, inputs):
    assert inputs.dim() == 4, 'This model accepts 4 dimension input tensor: {}'.format(inputs.size())
    batch_size, feature_dim = inputs.size(0), inputs.size(1)
    pool = self.basic(inputs)
    ress = self.res_block(pool) + pool
    x    = self.conv(ress)
    
    features, heatmaps, batch_locs, batch_scos = [], [], [], []
    
    previous_features = None
    for i in range(self.nStack):
      hg_fea, temp_downs, temp_ups = self.HGs[i](x, previous_features)
      if i + 1 < self.nStack:
        previous_features = [None] * self.recursive
        for istep in range(self.recursive):
          previous_features[istep] = self.cross_stages[i][istep][0](temp_downs[istep]) + self.cross_stages[i][istep][1](temp_ups[istep])

      feature = self.features[i]( hg_fea )

      features.append(feature)

      tmpOut = self.tmpOuts[i](feature)
      heatmaps.append(tmpOut)
      if i < self.nStack - 1:
        ll_ = self.trsfeas[i](feature)
        tmpOut_ = self.trstmps[i](tmpOut)
        x = x + ll_ + tmpOut_

    # The location of the current batch
    final_heatmap = heatmaps[-1]
    B, NP, H, W = final_heatmap.size()
    batch_locs, batch_scos = find_tensor_peak_batch(final_heatmap.view(B*NP, H, W), self.sigma, self.downsample)
    batch_locs, batch_scos = batch_locs.view(B, NP, 2), batch_scos.view(B, NP)
    
    return features, heatmaps, batch_locs, batch_scos


def ProMSPNet(config, points, sigma, use_gray):
  print ('Initialize hourglass with configure : {}'.format(config))
  idim = 1 if use_gray else 3
  model = MSPNet(config, points, sigma, idim)
  return model
