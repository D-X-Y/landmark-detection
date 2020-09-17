# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from os import path as osp

def load_file_lists(file_paths):
  if isinstance(file_paths, str):
    file_paths = [ file_paths ]
  print ('Function [load_lists] input {:} files'.format(len(file_paths)))
  all_strings = []
  for file_idx, file_path in enumerate(file_paths):
    assert osp.isfile(file_path), 'The {:}-th path : {:} is not a file.'.format(file_idx, file_path)
    listfile = open(file_path, 'r')
    listdata = listfile.read().splitlines()
    listfile.close()
    print ('Load [{:d}/{:d}]-th list : {:} with {:} images'.format(file_idx, len(file_paths), file_path, len(listdata)))
    all_strings += listdata
  return all_strings
