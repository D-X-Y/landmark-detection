# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""Set up paths."""            
  
import sys, os  
from os import path as osp

def add_path(path):            
  if path not in sys.path:   
    sys.path.insert(0, path)
this_dir = osp.dirname(osp.abspath(__file__))

# Add lib to PYTHONPATH  
lib_path = osp.abspath(osp.join(this_dir, '..', 'lib'))
add_path(lib_path)
