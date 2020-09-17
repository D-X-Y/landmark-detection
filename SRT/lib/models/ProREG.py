# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#
import torch
import torch.nn as nn
from .initialization import weights_init_reg
from .modules import CoordConv

def conv_bn(inp, oup, kernels, stride, pad):
  return nn.Sequential(
    nn.Conv2d(inp, oup, kernels, stride, pad, bias=False),
    nn.BatchNorm2d(oup),
    nn.ReLU6(inplace=True)
  )


def conv_1x1_bn(inp, oup):
  return nn.Sequential(
    nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
    nn.BatchNorm2d(oup),
    nn.ReLU6(inplace=True)
  )


class InvertedResidual(nn.Module):
  def __init__(self, inp, oup, stride, expand_ratio):
    super(InvertedResidual, self).__init__()
    self.stride = stride
    assert stride in [1, 2]

    hidden_dim = round(inp * expand_ratio)
    self.use_res_connect = self.stride == 1 and inp == oup

    if expand_ratio == 1:
      self.conv = nn.Sequential(
        # dw
        nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
        nn.BatchNorm2d(hidden_dim),
        nn.ReLU6(inplace=True),
        # pw-linear
        nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
      )
    else:
      self.conv = nn.Sequential(
        # pw
        nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
        nn.BatchNorm2d(hidden_dim),
        nn.ReLU6(inplace=True),
        # dw
        nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
        nn.BatchNorm2d(hidden_dim),
        nn.ReLU6(inplace=True),
        # pw-linear
        nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
      )

  def forward(self, x):
    if self.use_res_connect:
      return x + self.conv(x)
    else:
      return self.conv(x)


class MobileNetV2REG(nn.Module):
  def __init__(self, input_dim, input_channel, width_mult, pts_num):
    super(MobileNetV2REG, self).__init__()
    self.pts_num = pts_num
    block = InvertedResidual
    interverted_residual_setting = [
      # t, c, n, s
      [1, 48 , 1, 1],
      [2, 48 , 5, 2],
      [2, 96 , 1, 2],
      [4, 96 , 6, 1],
      [2, 16 , 1, 1],
    ]

    input_channel = int(input_channel * width_mult)
    features      = [conv_bn(input_dim, input_channel, (3,3), 2, 1)]
    # building inverted residual blocks
    for t, c, n, s in interverted_residual_setting:
      output_channel = int(c * width_mult)
      for i in range(n):
        if i == 0: stride = s
        else     : stride = 1
        features.append( block(input_channel, output_channel, stride, expand_ratio=t) )
        input_channel = output_channel
    features.append( nn.AdaptiveAvgPool2d( (14,14) ) )
    self.features = nn.Sequential(*features)
  
    self.S1 = nn.Sequential(
                  CoordConv(input_channel  , input_channel*2, True, kernel_size=3, padding=1),
                  conv_bn(input_channel*2, input_channel*2, (3,3), 2, 1))
    self.S2 = nn.Sequential(
                  CoordConv(input_channel*2, input_channel*4, True, kernel_size=3, padding=1),
                  conv_bn(input_channel*4, input_channel*8, (7,7), 1, 0))

    output_neurons  = 14*14*input_channel + 7*7*input_channel*2 + input_channel*8
    self.locator    = nn.Sequential(
                         nn.Linear(output_neurons, pts_num*2))

    #self.classifier = nn.Linear(output_neurons, pts_num)
    #self.classifier = nn.Sequential(
    #                     block(input_channel*1, input_channel*4, 1, 2),
    #                     nn.AdaptiveAvgPool2d( (16,12) ),
    #                    block(input_channel*4, input_channel*4, 1, 2),
    #                     nn.AdaptiveAvgPool2d( (8,6) ),
    #                     nn.Conv2d(input_channel*4, pts_num, (8,6)))
    self.apply( weights_init_reg )

  def forward(self, xinputs):
    if xinputs.dim() == 5:
      batch, seq, C, H, W = xinputs.shape
      batch_locs = self.forward_x( xinputs.view(batch*seq, C, H, W) )
      _, N, _ = batch_locs.shape
      return batch_locs.view(batch, seq, N, 2)
    else:
      return self.forward_x( xinputs )

  def forward_x(self, x):
    batch, C, H, W = x.size()
    features = self.features(x)
    S1 = self.S1( features )
    S2 = self.S2( S1 )
    tensors = torch.cat((features.view(batch, -1), S1.view(batch, -1), S2.view(batch, -1)), dim=1)
    batch_locs = self.locator(tensors).view(batch, self.pts_num, 2)
    #batch_scos = self.classifier(tensors).view(batch, self.pts_num, 1)
    return batch_locs


def ProRegression(config, points, use_gray):
  idim = 1 if use_gray else 3
  model = MobileNetV2REG(idim, config.ichannel, config.width_mult, points) 
  model.downsample = config.downsample
  return model
