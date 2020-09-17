# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#
# CU-Net: Coupled U-Nets
from __future__ import division
import time, math, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from .modules import BN_ReLU_C1x1, DenseLayer
from .basic_batch import find_tensor_peak_batch


class CU_Block(nn.Module):
  def __init__(self, channels, adapC, neck_size, growth_rate, num_blocks):
    super(CU_Block, self).__init__()
    self.num_blocks = num_blocks
    down_blocks, up_blocks, adapters_skip = [], [], []
    adapters_down, adapters_up = [], []
    self.cache_features = []
    for i in range(self.num_blocks):
      down_blocks.append  ( DenseLayer(channels, growth_rate, neck_size, 0) )
      up_blocks.append    ( DenseLayer(channels*2, growth_rate, neck_size, 0) )
      adapters_skip.append( BN_ReLU_C1x1(channels+growth_rate, adapC) )
      adapters_down.append( BN_ReLU_C1x1(channels+growth_rate, adapC) )
      adapters_up.append  ( BN_ReLU_C1x1(channels*2+growth_rate, adapC) )
    self.down_blocks    = nn.ModuleList( down_blocks )
    self.adapters_skip  = nn.ModuleList( adapters_skip )
    self.adapters_down  = nn.ModuleList( adapters_down )
    self.neck_block     = DenseLayer(channels, growth_rate, neck_size, 0)
    self.adapter_neck   = BN_ReLU_C1x1(channels+growth_rate, channels)
    self.up_blocks      = nn.ModuleList( up_blocks )
    self.adapters_up    = nn.ModuleList( adapters_up )
  
  def forward(self, x, previous_features):
    skip_list = [None] * self.num_blocks
    down_outputs, up_outputs = [None] * self.num_blocks, [None] * self.num_blocks
    cache_features, cache_shapes = [], []
    for i in range(self.num_blocks):
      #print ('D {:}/{:} size : {:}'.format(i, self.num_blocks, x.size()))
      cache_shapes.append( (x.size(2), x.size(3)) )
      tinputs = [x] + previous_features[i]
      temp = self.down_blocks[i]( tinputs )
      #print ('D- {:}'.format(temp.size()))
      cache_features.append( temp )
      down_outputs[i] = self.adapters_down[i]( tinputs + [temp] )
      skip_list[i]    = self.adapters_skip[i]( tinputs + [temp] )
      x = F.interpolate(down_outputs[i], [x.size(2)//2, x.size(3)//2], mode='bilinear', align_corners=True)

    temp = self.neck_block( [x] + previous_features[self.num_blocks] )
    cache_features.append( temp )
    x = self.adapter_neck( [x] + previous_features[self.num_blocks] + [temp] )

    for i in range(self.num_blocks-1, -1, -1):
      #print ('U {:}/{:} size : {:}'.format(i, self.num_blocks, x.size()))
      x = F.interpolate(x, cache_shapes[i], mode='bilinear', align_corners=True)
      tinputs = [x] + previous_features[2*self.num_blocks-i] + [skip_list[i]]
      temp = self.up_blocks[i]( tinputs )
      cache_features.append( temp )
      up_outputs[i] = self.adapters_up[i]( tinputs + [temp] )
    outputs = up_outputs[0]
    #print ('output shape : {:}'.format( outputs.size() ))
    return outputs, cache_features
    
      
  


class CUNet(nn.Module):
  def __init__(self, config, points, sigma, input_dim):
    super(CUNet, self).__init__()
    self.downsample = 4
    self.sigma      = sigma

    self.config     = copy.deepcopy( config )

    self.pts_num      = points
    self.init_channel = self.config.init_channel
    self.neck_size    = self.config.neck_size
    self.growth_rate  = self.config.growth_rate
    self.layer_num    = self.config.layer_num
    self.num_blocks   = self.config.num_blocks
    self.conv = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(input_dim, self.init_channel, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(self.init_channel)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=2, stride=2)),
        ]))
    num_chans = self.init_channel

    blocks = []
    for i in range(self.layer_num):
      blocks.append( CU_Block(num_chans + i*self.growth_rate, num_chans, self.neck_size, self.growth_rate, self.num_blocks) )
    self.blocks = nn.ModuleList( blocks )

    self.intermedia = nn.ModuleList()
    for i in range(1, self.layer_num):
      self.intermedia.append( BN_ReLU_C1x1(num_chans*(i+1), num_chans) )

    self.linears = nn.ModuleList()
    for i in range(0, self.layer_num):
      self.linears.append( BN_ReLU_C1x1(num_chans, self.pts_num) )
    

  def extra_repr(self):
    return ('{name}(sigma={sigma}, downsample={downsample})'.format(name=self.__class__.__name__, **self.__dict__))

  
  def forward(self, inputs):
    assert inputs.dim() == 4, 'This model accepts 4 dimension input tensor: {}'.format(inputs.size())
    batch_size, feature_dim = inputs.size(0), inputs.size(1)
    x = self.conv(inputs)
    
    intermediates = [[] for i in range(2*self.num_blocks+1)]
    features, heatmaps = [], []
    for i in range(self.layer_num):
      if i == 0: xinputs = x
      else     : xinputs = self.intermedia[i-1]( features + [x] )
      feature, cache_intermediate = self.blocks[i]( x, intermediates )
      heatmap = self.linears[i]( feature )

    
      features.append(feature)
      for idx, tempF in enumerate(cache_intermediate):
        intermediates[idx].append( tempF )
      heatmaps.append(heatmap)
    final_heatmap = heatmaps[-1]
    B, NP, H, W = final_heatmap.size()
    batch_locs, batch_scos = find_tensor_peak_batch(final_heatmap.view(B*NP, H, W), self.sigma, self.downsample)
    batch_locs, batch_scos = batch_locs.view(B, NP, 2), batch_scos.view(B, NP)
    
    return features, heatmaps, batch_locs, batch_scos


def ProCUNet(config, points, sigma, use_gray):
  print ('Initialize hourglass with configure : {}'.format(config))
  idim = 1 if use_gray else 3
  model = CUNet(config, points, sigma, idim)
  return model
