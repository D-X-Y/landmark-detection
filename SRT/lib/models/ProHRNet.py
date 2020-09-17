# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#
# Deep High-Resolution Representation Learning for Human Pose Estimation, CVPR 2019
from __future__ import division
import time, math, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from .basic_batch import find_tensor_peak_batch


BN_MOMENTUM = 0.01


def conv3x3(in_planes, out_planes, stride=1):
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride, downsample):
    super(BasicBlock, self).__init__()
    self.conv1 = conv3x3(inplanes, planes, stride)
    self.bn1   = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
    self.relu  = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes)
    self.bn2   = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)
    return out


class Bottleneck(nn.Module):
  expansion = 4

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(Bottleneck, self).__init__()
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
    self.bn1   = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn2   = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
    self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
    self.bn3   = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
    self.relu  = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)
    return out



class HighResolutionModule(nn.Module):
  def __init__(self, num_branches, block, num_blocks, num_inchannels, num_channels, fuse_method, multi_scale_output=True):
    super(HighResolutionModule, self).__init__()
    self._check_branches(num_branches, num_blocks, num_inchannels, num_channels)

    self.num_inchannels = num_inchannels
    self.fuse_method = fuse_method
    self.num_branches = num_branches

    self.multi_scale_output = multi_scale_output

    self.branches    = self._make_branches(num_branches, block, num_blocks, num_channels)
    self.fuse_layers = self._make_fuse_layers()
    self.relu = nn.ReLU(inplace=True)

  def _check_branches(self, num_branches, num_blocks, num_inchannels, num_channels):
    if num_branches != len(num_blocks):
      error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(num_branches, len(num_blocks))
      raise ValueError(error_msg)

    if num_branches != len(num_channels):
      error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(num_branches, len(num_channels))
      raise ValueError(error_msg)

    if num_branches != len(num_inchannels):
      error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(num_branches, len(num_inchannels))
      raise ValueError(error_msg)

  def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride):
    downsample = None
    if stride != 1 or self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.num_inchannels[branch_index], num_channels[branch_index] * block.expansion,
              kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(num_channels[branch_index] * block.expansion, momentum=BN_MOMENTUM),
      )

    layers = []
    layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index], stride, downsample))
    self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
    for i in range(1, num_blocks[branch_index]):
      layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index], 1, None))
    return nn.Sequential(*layers)

  def _make_branches(self, num_branches, block, num_blocks, num_channels):
    branches = []
    for i in range(num_branches):
      branches.append(self._make_one_branch(i, block, num_blocks, num_channels, 1))
    return nn.ModuleList(branches)

  def _make_fuse_layers(self):
    if self.num_branches == 1:
      return None

    num_branches = self.num_branches
    num_inchannels = self.num_inchannels
    fuse_layers = []
    for i in range(num_branches if self.multi_scale_output else 1):
      fuse_layer = []
      for j in range(num_branches):
        if j > i:
          fuse_layer.append(nn.Sequential(
            nn.Conv2d(num_inchannels[j], num_inchannels[i], 1, 1, 0, bias=False),
            nn.BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM)))
          # nn.Upsample(scale_factor=2**(j-i), mode='nearest')))
        elif j == i:
          fuse_layer.append(None)
        else:
          conv3x3s = []
          for k in range(i - j):
            if k == i - j - 1:
              num_outchannels_conv3x3 = num_inchannels[i]
              conv3x3s.append(nn.Sequential(
                nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False),
                nn.BatchNorm2d(num_outchannels_conv3x3, momentum=BN_MOMENTUM)))
            else:
              num_outchannels_conv3x3 = num_inchannels[j]
              conv3x3s.append(nn.Sequential(
                nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False),
                nn.BatchNorm2d(num_outchannels_conv3x3, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)))
          fuse_layer.append(nn.Sequential(*conv3x3s))
      fuse_layers.append(nn.ModuleList(fuse_layer))

    return nn.ModuleList(fuse_layers)

  def get_num_inchannels(self):
    return self.num_inchannels

  def forward(self, x):
    if self.num_branches == 1:
      return [self.branches[0](x[0])]

    for i in range(self.num_branches):
      x[i] = self.branches[i](x[i])

    x_fuse = []
    for i in range(len(self.fuse_layers)):
      y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
      for j in range(1, self.num_branches):
        if i == j:
          y = y + x[j]
        elif j > i:
          y = y + F.interpolate(self.fuse_layers[i][j](x[j]), size=[x[i].shape[2], x[i].shape[3]], mode='bilinear', align_corners=False)
        else:
          y = y + self.fuse_layers[i][j](x[j])
      x_fuse.append(self.relu(y))

    return x_fuse


class HighResolutionNet(nn.Module):

  def __init__(self, config, points, sigma, input_dim):
    super(HighResolutionNet, self).__init__()
    self.inplanes = 64
    self.config   = copy.deepcopy( config )
    self.sigma    = sigma
    self.downsample = 4
    # stem net
    self.stem  = nn.Sequential(
                   nn.Conv2d(input_dim, 64, kernel_size=3, stride=2, padding=1, bias=False),
                   nn.BatchNorm2d(64, momentum=BN_MOMENTUM), nn.ReLU(inplace=True),
                   nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
                   nn.BatchNorm2d(64, momentum=BN_MOMENTUM), nn.ReLU(inplace=True))

    self.layer1 = self._make_layer(Bottleneck, 64, 64, 4)

    num_channels = self.config.S2_NUM_CHANNELS
    assert self.config.S2_BLOCK in ["BASIC", "BOTTLENECK"]
    block = BasicBlock if self.config.S2_BLOCK == "BASIC" else Bottleneck
    num_channels = [ num_channels[i] * block.expansion for i in range(len(num_channels)) ]
    self.transition1 = self._make_transition_layer([256], num_channels)
    self.stage2, pre_stage_channels = self._make_stage('S2', num_channels)

    num_channels = self.config.S3_NUM_CHANNELS
    block = BasicBlock if self.config.S3_BLOCK == "BASIC" else Bottleneck
    num_channels = [ num_channels[i] * block.expansion for i in range(len(num_channels)) ]
    self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
    self.stage3, pre_stage_channels = self._make_stage('S3', num_channels)

    num_channels = self.config.S4_NUM_CHANNELS
    block = BasicBlock if self.config.S4_BLOCK == "BASIC" else Bottleneck
    num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
    self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
    self.stage4, pre_stage_channels = self._make_stage('S4', num_channels, multi_scale_output=True)

    final_inp_channels = sum(pre_stage_channels)

    self.head = nn.Sequential(
      nn.Conv2d(final_inp_channels, final_inp_channels, kernel_size=self.config.FINAL_CONV_KERNEL, stride=1, padding=1 if self.config.FINAL_CONV_KERNEL == 3 else 0),
      nn.BatchNorm2d(final_inp_channels, momentum=BN_MOMENTUM),
      nn.ReLU(inplace=True),
      nn.Conv2d(final_inp_channels,             points, kernel_size=self.config.FINAL_CONV_KERNEL, stride=1, padding=1 if self.config.FINAL_CONV_KERNEL == 3 else 0)
    )
    if self.config.sigmoid: self.sigmoid = nn.Sigmoid()
    else                  : self.sigmoid = None

  def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
    num_branches_cur = len(num_channels_cur_layer)
    num_branches_pre = len(num_channels_pre_layer)

    transition_layers = []
    for i in range(num_branches_cur):
      if i < num_branches_pre:
        if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
          transition_layers.append(
            nn.Sequential(
              nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], 3, 1, 1, bias=False),
              nn.BatchNorm2d(num_channels_cur_layer[i], momentum=BN_MOMENTUM), nn.ReLU(inplace=True)
            ))
        else:
          transition_layers.append(None)
      else:
        conv3x3s = []
        for j in range(i + 1 - num_branches_pre):
          inchannels  = num_channels_pre_layer[-1]
          outchannels = num_channels_cur_layer[i] if j == i - num_branches_pre else inchannels
          conv3x3s.append(
            nn.Sequential(
              nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False),
              nn.BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
              nn.ReLU(inplace=True)
            ))
        transition_layers.append(nn.Sequential(*conv3x3s))
    return nn.ModuleList(transition_layers)

  def _make_layer(self, block, inplanes, planes, blocks, stride=1):
    if stride != 1 or inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
      )
    else: downsample = None
    layers = [ block(inplanes, planes, stride, downsample) ]
    inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(inplanes, planes))
    return nn.Sequential(*layers)

  def _make_stage(self, prefix, num_inchannels, multi_scale_output=True):
    num_modules  = getattr(self.config, '{:}_NUM_MODULES'.format(prefix))
    num_branches = getattr(self.config, '{:}_NUM_BRANCHES'.format(prefix))
    num_blocks   = getattr(self.config, '{:}_NUM_BLOCKS'.format(prefix))
    num_channels = getattr(self.config, '{:}_NUM_CHANNELS'.format(prefix))
    block        = BasicBlock if getattr(self.config, '{:}_BLOCK'.format(prefix)) == "BASIC" else Bottleneck
    fuse_method  = getattr(self.config, '{:}_FUSE_METHOD'.format(prefix))

    modules = []
    for i in range(num_modules):
      # multi_scale_output is only used last module
      if not multi_scale_output and i == num_modules - 1:
        reset_multi_scale_output = False
      else:
        reset_multi_scale_output = True
      modules.append(
        HighResolutionModule(num_branches, block, num_blocks,
                   num_inchannels, num_channels, fuse_method,
                   reset_multi_scale_output)
      )
      num_inchannels = modules[-1].get_num_inchannels()

    return nn.Sequential(*modules), num_inchannels

  def extra_repr(self):
    return ('{name}(sigma={sigma}, downsample={downsample})'.format(name=self.__class__.__name__, **self.__dict__))

  def forward(self, x):
    x = self.stem(x)
    x = self.layer1(x)

    x_list = []
    for i in range(self.config.S2_NUM_BRANCHES):
      if self.transition1[i] is not None:
        x_list.append(self.transition1[i](x))
      else:
        x_list.append(x)
    y_list = self.stage2(x_list)

    x_list = []
    for i in range(self.config.S3_NUM_BRANCHES):
      if self.transition2[i] is not None:
        x_list.append(self.transition2[i](y_list[-1]))
      else:
        x_list.append(y_list[i])
    y_list = self.stage3(x_list)

    x_list = []
    for i in range(self.config.S4_NUM_BRANCHES):
      if self.transition3[i] is not None:
        x_list.append(self.transition3[i](y_list[-1]))
      else:
        x_list.append(y_list[i])
    feats = self.stage4(x_list)

    # Head Part
    batch_size, _, height, width = feats[0].shape
    feat1 = F.interpolate(feats[1], size=(height, width), mode='bilinear', align_corners=False)
    feat2 = F.interpolate(feats[2], size=(height, width), mode='bilinear', align_corners=False)
    feat3 = F.interpolate(feats[3], size=(height, width), mode='bilinear', align_corners=False)
    cat_f = torch.cat([feats[0], feat1, feat2, feat3], 1)
    heatmaps = self.head(cat_f)

    # The location of the current batch
    batch_locs, batch_scos = [], []
    for ibatch in range(batch_size):
      batch_location, batch_score = find_tensor_peak_batch(heatmaps[ibatch], self.sigma, self.downsample)
      batch_locs.append( batch_location )
      batch_scos.append( batch_score )
    batch_locs, batch_scos = torch.stack(batch_locs), torch.stack(batch_scos)
    return feats, [heatmaps], batch_locs, batch_scos

  def init_weights(self):
    logger.info('=> init weights from normal distribution')
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        nn.init.normal_(m.weight, std=0.001)
        # nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def ProHRNet(config, points, sigma, use_gray):
  print ('Initialize HR-Net with configure : {}'.format(config))
  idim = 1 if use_gray else 3
  model = HighResolutionNet(config, points, sigma, idim)
  return model
