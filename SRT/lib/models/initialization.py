# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import math, torch
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

def weights_init_reg(m):
  if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    m.weight.data.normal_(0, math.sqrt(2. / n))
    if m.bias is not None:
      m.bias.data.zero_()
  elif isinstance(m, nn.BatchNorm2d):
    m.weight.data.fill_(1)
    m.bias.data.zero_()
  elif isinstance(m, nn.Linear):
    n = m.weight.size(1)
    m.weight.data.normal_(0, 0.01)
    m.bias.data.zero_()
