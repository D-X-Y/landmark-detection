# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch

def save_checkpoint(state, filename, logger):
  torch.save(state, filename)
  logger.log('save checkpoint into {}'.format(filename))
  return filename
