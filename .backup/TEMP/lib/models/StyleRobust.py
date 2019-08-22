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


class Hourglass(nn.Module):
  def __init__(self, n, nModules, nFeats):
    super(Hourglass, self).__init__()
    self.n = n
    self.nModules = nModules
    self.nFeats = nFeats
    
    self.res = nn.Sequential(*[Residual(nFeats, nFeats) for _ in range(nModules)])

    down = [nn.MaxPool2d(kernel_size = 2, stride = 2)]
    down += [Residual(nFeats, nFeats) for _ in range(nModules)]
    self.down = nn.Sequential(*down)

    if self.n > 1:
      self.mid = Hourglass(n - 1, self.nModules, self.nFeats)
    else:
      self.mid = nn.Sequential(*[Residual(nFeats, nFeats) for _ in range(nModules)])
    
    up = [Residual(nFeats, nFeats) for _ in range(nModules)]
    up += [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)]
    self.up = nn.Sequential(*up)
    
  def forward(self, x):
    res  = self.res(x)
    down = self.down(res)
    mid  = self.mid(down)
    up   = self.up(mid)
    return res + up


class HourglassNet(nn.Module):
  def __init__(self, config, points):
    super(HourglassNet, self).__init__()
    self.downsample = 4

    self.config   = config

    self.pts_num  = points
    self.nStack   = self.config.stages
    self.nModules = self.config.nModules
    self.nFeats   = self.config.nFeats
    self.recursive = self.config.recursive

    self.conv = nn.Sequential(
                  nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = True), 
                  nn.BatchNorm2d(64), nn.ReLU(inplace = True))
    self.ress = nn.Sequential(
                  Residual(64, 128),
                  nn.MaxPool2d(kernel_size = 2, stride = 2),
                  Residual(128, 128), Residual(128, self.nFeats))
    
    _features, _tmpOut, _ll_, _tmpOut_ = [], [], [], []

    for i in range(self.nStack):
      feature = Hourglass(self.recursive, self.nModules, self.nFeats)
      feature = [feature] + [Residual(self.nFeats, self.nFeats) for _ in range(self.nModules)]
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
  
  def forward(self, inputs):
    assert inputs.dim() == 4, 'This model accepts 4 dimension input tensor: {}'.format(inputs.size())
    batch_size, feature_dim = inputs.size(0), inputs.size(1)
    x = self.conv(inputs)
    x = self.ress(x)
    
    heatmaps, batch_locs, batch_scos = [], [], []
    features = []

    for i in range(self.nStack):
      feature = self.features[i](x)
      features.append(feature)

      tmpOut = self.tmpOuts[i](feature)
      heatmaps.append(tmpOut)
      if i < self.nStack - 1:
        ll_ = self.trsfeas[i](feature)
        tmpOut_ = self.trstmps[i](tmpOut)
        x = x + ll_ + tmpOut_

    # The location of the current batch
    for ibatch in range(batch_size):
      batch_location, batch_score = find_tensor_peak_batch(heatmaps[-1][ibatch], self.config.argmax, self.downsample)
      batch_locs.append( batch_location )
      batch_scos.append( batch_score )
    batch_locs, batch_scos = torch.stack(batch_locs), torch.stack(batch_scos)
    
    if self.training:
      return features, heatmaps, batch_locs, batch_scos
    else:
      return heatmaps, batch_locs, batch_scos

def StyleLandmarkNet(config, points):
  model = HourglassNet(config, points)
  return model
