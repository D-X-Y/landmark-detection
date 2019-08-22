# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn
from torch.nn import init

def weights_init_cpm(m):
  classname = m.__class__.__name__
  # print(classname)
  if classname.find('Conv') != -1:
    m.weight.data.normal_(0, 0.01)
    if m.bias is not None: m.bias.data.zero_()
  elif classname.find('BatchNorm2d') != -1:
    m.weight.data.fill_(1)
    m.bias.data.zero_()

def weights_init_xavier(m):
  classname = m.__class__.__name__
  # print(classname)
  if classname.find('Conv') != -1:
    init.xavier_normal_(m.weight.data, gain=1)
  elif classname.find('Linear') != -1:
    init.xavier_normal_(m.weight.data, gain=1)
    if m.bias is not None:
      init.constant(m.bias.data, 0.0)
  elif classname.find('BatchNorm2d') != -1:
    if m.weight is not None:
      init.uniform(m.weight.data, 1.0, 0.02)
    if m.bias is not None:
      init.constant(m.bias.data, 0.0)

def weights_init_wgan(m):
  if isinstance(m, nn.Conv2d):
    m.weight.data.normal_(0, 0.02)
    m.bias.data.zero_()
  elif isinstance(m, nn.ConvTranspose2d):
    m.weight.data.normal_(0, 0.02)
    m.bias.data.zero_()
  elif isinstance(m, nn.Linear):
    m.weight.data.normal_(0, 0.02)
    m.bias.data.zero_()
  elif isinstance(m, nn.BatchNorm2d):
    if m.weight is not None:
      nn.init.constant_(m.weight, 1)
    if m.bias is not None:
      nn.init.constant_(m.bias, 0)
