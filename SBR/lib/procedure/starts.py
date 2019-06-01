# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os, sys, time
import numpy as np
import torch
import random

def prepare_seed(rand_seed):
  np.random.seed(rand_seed)
  random.seed(rand_seed)
  torch.manual_seed(rand_seed)
  torch.cuda.manual_seed_all(rand_seed)
