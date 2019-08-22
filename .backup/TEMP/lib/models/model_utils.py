# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from scipy.ndimage.interpolation import zoom
from collections import OrderedDict
import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy, numbers, numpy as np

def get_parameters(model, bias):
  for m in model.modules():
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
      if bias:
        yield m.bias
      else:
        yield m.weight
    elif isinstance(m, nn.BatchNorm2d):
      if bias:
        yield m.bias
      else:
        yield m.weight

def remove_module_dict(state_dict, is_print=False):
  new_state_dict = OrderedDict()
  for k, v in state_dict.items():
    if k[:7] == 'module.':
      name = k[7:] # remove `module.`
    else:
      name = k
    new_state_dict[name] = v
  if is_print: print(new_state_dict.keys())
  return new_state_dict
