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
