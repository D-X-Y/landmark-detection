# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#
import torch
import torch.nn as nn

'''
An alternative implementation for PyTorch with auto-infering the x-y dimensions.
'''
class AddCoords(nn.Module):

  def __init__(self, with_r=False):
    super().__init__()
    self.with_r = with_r

  def forward(self, input_tensor):
    """
    Args:
      input_tensor: shape(batch, channel, x_dim, y_dim)
    """
    batch_size, _, x_dim, y_dim = input_tensor.size()

    xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
    yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

    xx_channel = xx_channel.float() / (x_dim - 1)
    yy_channel = yy_channel.float() / (y_dim - 1)

    xx_channel = xx_channel * 2 - 1
    yy_channel = yy_channel * 2 - 1

    xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
    yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

    xx_channel, yy_channel = xx_channel.type_as(input_tensor), yy_channel.type_as(input_tensor)
    ret = torch.cat([
      input_tensor,
      xx_channel,
      yy_channel], dim=1)

    if self.with_r:
      rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2))
      ret = torch.cat([ret, rr], dim=1)

    return ret


class CoordConv(nn.Module):

  def __init__(self, in_channels, out_channels, with_r=False, **kwargs):
    super().__init__()
    self.addcoords = AddCoords(with_r=with_r)
    in_size = in_channels+2
    if with_r:
      in_size += 1
    self.conv = nn.Conv2d(in_size, out_channels, **kwargs)

  def forward(self, x):
    ret = self.addcoords(x)
    ret = self.conv(ret)
    return ret


class BN_ReLU_C1x1(nn.Module):
  def __init__(self, in_num, out_num):
    super(BN_ReLU_C1x1, self).__init__()
    layers = [nn.BatchNorm2d(in_num),
              nn.ReLU(inplace=True),
              nn.Conv2d(in_num, out_num, kernel_size=1, stride=1, bias=False)]
    self.layers = nn.Sequential(*layers)

  def forward(self, ifeatures):
    if isinstance(ifeatures, list): ifeatures = torch.cat(ifeatures, dim=1)
    features = self.layers(ifeatures)
    return features


class DenseLayer(nn.Module):
  def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
    super(DenseLayer, self).__init__()
    layers = [nn.BatchNorm2d(num_input_features),
              nn.ReLU(inplace=True),
              nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False),
              nn.BatchNorm2d(bn_size * growth_rate),
              nn.ReLU(inplace=True),
              nn.Conv2d(bn_size * growth_rate, growth_rate,  kernel_size=3, stride=1, padding=1, bias=False)]
    if drop_rate > 0:
      layers += [nn.Dropout(drop_rate)]
    self.layers = nn.Sequential(*layers)

  def forward(self, ifeatures):
    if isinstance(ifeatures, list): ifeatures = torch.cat(ifeatures, dim=1)
    features = self.layers(ifeatures)
    return features
