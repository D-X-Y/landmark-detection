# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import copy, numbers, numpy as np
from collections import OrderedDict


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


def count_parameters_in_MB(model):
  if isinstance(model, nn.Module):
    return sum(v.numel() for v in model.parameters()) / 1e6
  else:
    return sum(v.numel() for v in model) / 1e6


def load_checkpoint(model_path):
  model_path = Path('{:}'.format(model_path))
  assert model_path.exists(), 'model_path {:} does not exist'.format(model_path)
  checkpoint = torch.load( model_path )
  if 'last_checkpoint' in checkpoint:
    last_checkpoint = checkpoint['last_checkpoint']
    assert last_checkpoint.exists(), 'The last checkpoint in model-path does not exist : {:}'.format(last_checkpoint)
    last_checkpoint = torch.load(str(last_checkpoint))
  else:
    last_checkpoint = checkpoint
  return last_checkpoint
